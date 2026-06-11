"""
Fine-Tune GPT-2 on Classical Arabic Poetry
============================================
Uses aubmindlab/aragpt2-base — an Arabic GPT-2 pre-trained on 77GB of Arabic text.
This is the correct base model for Arabic generation (NOT the English gpt2).

Why aragpt2 over gpt2?
  • Pre-trained on Arabic script — vocabulary already covers Arabic tokens
  • Uses a SentencePiece tokenizer tuned for Arabic morphology
  • Much lower perplexity baseline on Arabic text

Run:
  python finetune/train.py
  python finetune/train.py --epochs 5 --batch_size 2 --lr 3e-5
"""

import os
import math
import argparse
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",  default="aubmindlab/aragpt2-base")
    p.add_argument("--data_path",   default="data/poetry_corpus.txt")
    p.add_argument("--output_dir",  default="outputs/aragpt2-poetry")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--block_size",  type=int,   default=256)
    p.add_argument("--max_samples", type=int,   default=None)
    p.add_argument("--warmup_ratio",type=float, default=0.1)
    return p.parse_args()


# ── Data ──────────────────────────────────────────────────────────────────────

def load_corpus(path: str, max_samples: int | None) -> Dataset:
    """
    Load the poetry corpus.
    Each poem (separated by blank lines) becomes one training example.
    """
    text = Path(path).read_text(encoding="utf-8")
    poems = [p.strip() for p in text.split('\n\n') if p.strip()]
    if max_samples:
        poems = poems[:max_samples]
    print(f"  Loaded {len(poems):,} poems from {path}")
    return Dataset.from_dict({"text": poems})


def tokenize_and_chunk(dataset: Dataset, tokenizer, block_size: int) -> Dataset:
    """
    Tokenize poems and chunk into fixed-length blocks.

    Why chunking?
    GPT models need fixed-length inputs. We:
    1. Tokenize each poem (no truncation yet)
    2. Concatenate all token ids with EOS separator between poems
    3. Split the long sequence into chunks of `block_size`

    This ensures no context is wasted — even partial poems contribute.
    """
    eos = tokenizer.eos_token_id

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=False, padding=False)

    def chunk_into_blocks(examples):
        # Add EOS token between documents
        all_ids = []
        for ids in examples["input_ids"]:
            all_ids.extend(ids + [eos])

        # Chunk
        total = (len(all_ids) // block_size) * block_size
        chunks = [all_ids[i:i+block_size] for i in range(0, total, block_size)]

        return {
            "input_ids":      chunks,
            "attention_mask": [[1] * block_size] * len(chunks),
            "labels":         chunks,   # for causal LM, labels = input_ids
        }

    tokenized = dataset.map(tokenize_batch, batched=True,
                            remove_columns=["text"],
                            desc="Tokenizing")
    chunked   = tokenized.map(chunk_into_blocks, batched=True,
                              desc="Chunking into blocks")
    return chunked


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Arabic Poetry GPT-2 Fine-Tuning")
    print(f"  Base model : {args.model_name}")
    print(f"  Data       : {args.data_path}")
    print(f"{'='*60}\n")

    # 1. Tokenizer
    print("[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"      Vocab size  : {tokenizer.vocab_size:,}")
    print(f"      EOS token   : {tokenizer.eos_token!r}")

    # 2. Data
    print("\n[2/5] Preparing dataset...")
    raw = load_corpus(args.data_path, args.max_samples)
    split = raw.train_test_split(test_size=0.05, seed=42)
    train_ds = tokenize_and_chunk(split["train"], tokenizer, args.block_size)
    eval_ds  = tokenize_and_chunk(split["test"],  tokenizer, args.block_size)
    print(f"      Train blocks : {len(train_ds):,}")
    print(f"      Eval blocks  : {len(eval_ds):,}")

    # 3. Model
    print("\n[3/5] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Total params     : {n_params:,}")
    print(f"      Trainable params : {n_trainable:,}")

    # 4. Trainer setup
    print("\n[4/5] Configuring Trainer...")
    steps_per_epoch = len(train_ds) // args.batch_size
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = int(total_steps * args.warmup_ratio)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        weight_decay=0.01,          # L2 regularization via AdamW
        lr_scheduler_type="cosine", # cosine decay — better than linear for LMs
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=max(1, steps_per_epoch // 5),
        logging_dir=os.path.join(args.output_dir, "logs"),
        fp16=torch.cuda.is_available(),  # mixed precision when a CUDA GPU is available
        report_to="none",
        save_total_limit=2, # keep only last 2 checkpoints to save disk
    )

    # DataCollator for Causal LM:
    # mlm=False means NO masking — we predict next token, not masked tokens
    # This is the fundamental difference from BERT training
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 5. Train
    print("\n[5/5] Training...\n")
    train_result = trainer.train()

    # Save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Report
    eval_results = trainer.evaluate()
    ppl = math.exp(eval_results["eval_loss"])

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Train loss  : {train_result.training_loss:.4f}")
    print(f"  Eval loss   : {eval_results['eval_loss']:.4f}")
    print(f"  Perplexity  : {ppl:.2f}")
    print(f"  Model saved : {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
