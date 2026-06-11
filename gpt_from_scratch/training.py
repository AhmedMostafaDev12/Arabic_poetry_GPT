"""
GPT From Scratch — Module 5: Training Loop
============================================
Train the mini-GPT on Arabic poetry from scratch.
Every line is explained. No Trainer API — raw PyTorch.

What you'll learn:
  • AdamW optimizer and why not plain Adam
  • Learning rate warmup + cosine decay schedule
  • Gradient clipping — what it is and why GPT needs it
  • Gradient accumulation — simulate large batches on small GPU
  • Training loop structure: forward → loss → backward → step
  • How to track and visualize training progress

Run:  python -m gpt_from_scratch.training
      python -m gpt_from_scratch.training --max_steps 500 --batch_size 4
"""

import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from gpt_from_scratch.tokenizer_bpe import BPETokenizer
from gpt_from_scratch.gpt_model import GPT, GPTConfig


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class PoetryDataset(Dataset):
    """
    Converts a tokenized corpus into (input, target) pairs for causal LM.

    For causal language modeling:
      input  = tokens[0 : block_size]
      target = tokens[1 : block_size+1]   ← shifted by 1

    This way, for each position i, we predict token i+1 from tokens 0..i.
    This is the core of the GPT training objective.
    """

    def __init__(self, token_ids: list[int], block_size: int):
        self.ids        = token_ids
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.ids) - self.block_size)

    def __getitem__(self, idx):
        chunk  = self.ids[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)   # input
        y = torch.tensor(chunk[1:],  dtype=torch.long)   # target (shifted)
        return x, y


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Optimizer — AdamW
# ═══════════════════════════════════════════════════════════════════════════════

"""
WHY ADAMW INSTEAD OF ADAM?
──────────────────────────
Adam updates each parameter with adaptive learning rates:
  m_t = β1·m_{t-1} + (1-β1)·g_t          ← momentum
  v_t = β2·v_{t-1} + (1-β2)·g_t²         ← velocity (RMS)
  θ_t = θ_{t-1} - lr · m_t / (√v_t + ε)

Weight decay in Adam: θ_{t+1} = (1 - lr·λ)·θ_t - lr·g_t / (√v_t + ε)
The problem: decay is divided by √v_t → inconsistent regularization.

AdamW fixes this (Loshchilov & Hutter, 2019):
  θ_{t+1} = (1 - lr·λ)·θ_t - lr·m_t / (√v_t + ε)
Decay is applied directly, not divided by the adaptive factor.
Result: cleaner weight regularization → better generalization.

WHICH PARAMETERS GET WEIGHT DECAY?
  • YES: weight matrices (W_q, W_k, W_v, W_o, FFN weights)
  • NO:  biases, LayerNorm scale/shift, embeddings
  This is the standard GPT-2 configuration.
"""

def configure_adamw(model: GPT, lr: float, weight_decay: float, betas: tuple):
    """
    Configure AdamW with selective weight decay.
    Only apply decay to 2D+ parameters (weight matrices), not biases/norms.
    """
    decay_params    = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Apply decay to matrices (2D), not vectors (1D: biases, norms, embeddings)
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=1e-8)
    n_decay    = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)

    print(f"  AdamW: {n_decay:,} params with decay, {n_no_decay:,} without")
    return optimizer


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Learning Rate Schedule
# ═══════════════════════════════════════════════════════════════════════════════

"""
LEARNING RATE WARMUP + COSINE DECAY
─────────────────────────────────────
Starting with a high LR causes instability (large gradients blow up weights).
Starting with a low LR and ramping up (warmup) stabilizes early training.
Then cosine decay gradually reduces LR as training converges.

Schedule:
  steps < warmup_steps → LR linearly increases 0 → max_lr
  steps ≥ warmup_steps → LR follows cosine curve from max_lr → min_lr

The cosine shape is smoother than linear decay and empirically better for LMs.
"""

def get_lr(step: int, warmup_steps: int, max_steps: int,
           max_lr: float, min_lr: float) -> float:
    # Warmup phase
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Cosine decay phase
    if step >= max_steps:
        return min_lr

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Mini-GPT Training from Scratch")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── 1. Build tokenizer ────────────────────────────────────────────────────
    print("[1/5] Building tokenizer...")
    corpus_path = Path(args.data_path)
    if not corpus_path.exists():
        print(f"  Warning: {corpus_path} not found. Using sample corpus.")
        from gpt_from_scratch.tokenizer_bpe import SAMPLE_CORPUS as corpus_text
    else:
        corpus_text = corpus_path.read_text(encoding="utf-8")

    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train(corpus_text, verbose=False)
    print(f"  Vocab size: {len(tokenizer.vocab):,}")

    # ── 2. Tokenize corpus ────────────────────────────────────────────────────
    print("\n[2/5] Tokenizing corpus...")
    all_ids = tokenizer.encode(corpus_text)
    print(f"  Total tokens: {len(all_ids):,}")

    split    = int(0.9 * len(all_ids))
    train_ds = PoetryDataset(all_ids[:split],  block_size=args.block_size)
    val_ds   = PoetryDataset(all_ids[split:],  block_size=args.block_size)
    print(f"  Train examples: {len(train_ds):,}")
    print(f"  Val   examples: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              drop_last=True)

    # ── 3. Build model ────────────────────────────────────────────────────────
    print("\n[3/5] Building model...")
    config = GPTConfig(
        vocab_size  = len(tokenizer.vocab),
        max_seq_len = args.block_size,
        d_model     = args.d_model,
        n_layers    = args.n_layers,
        n_heads     = args.n_heads,
        dropout     = 0.1,
    )
    model = GPT(config).to(device)
    counts = model.param_count()
    print(f"  Total parameters: {counts['TOTAL']:,}")

    # ── 4. Optimizer + LR schedule ────────────────────────────────────────────
    print("\n[4/5] Configuring optimizer...")
    optimizer = configure_adamw(
        model,
        lr=args.max_lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    # ── 5. Training loop ──────────────────────────────────────────────────────
    print("\n[5/5] Training...\n")

    warmup_steps = args.max_steps // 10
    history = {"step": [], "train_loss": [], "val_loss": [], "lr": []}

    model.train()
    step       = 0
    epoch      = 0
    best_val   = float('inf')
    t0         = time.time()

    # Gradient accumulation: simulate large batch without extra memory
    # effective_batch = batch_size × accum_steps
    accum_steps = args.grad_accum

    while step < args.max_steps:
        epoch += 1
        for x, y in train_loader:
            if step >= args.max_steps:
                break

            x, y = x.to(device), y.to(device)

            # ── Forward + Loss ──────────────────────────────────────────────
            logits, loss = model(x, targets=y)

            # Scale loss for gradient accumulation
            # (accumulate gradients over N micro-steps, then update once)
            loss = loss / accum_steps
            loss.backward()

            # ── Gradient accumulation step ──────────────────────────────────
            if (step + 1) % accum_steps == 0 or step == args.max_steps - 1:

                # Gradient clipping
                # WHY? Large gradients can destabilize training.
                # Clip the global gradient norm to max_norm=1.0
                # This is crucial for transformer training stability.
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                # Update learning rate
                lr = get_lr(step, warmup_steps, args.max_steps,
                            args.max_lr, args.max_lr / 10)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # Optimizer step + zero grad
                optimizer.step()
                optimizer.zero_grad()

            step += 1

            # ── Logging ─────────────────────────────────────────────────────
            if step % args.log_every == 0:
                # Validation loss
                val_loss = evaluate(model, val_loader, device,
                                    max_batches=10)
                train_loss = loss.item() * accum_steps
                elapsed = time.time() - t0

                history["step"].append(step)
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["lr"].append(lr)

                ppl = math.exp(min(val_loss, 20))   # cap to avoid overflow
                print(f"  step {step:5d}/{args.max_steps}"
                      f"  train={train_loss:.4f}"
                      f"  val={val_loss:.4f}"
                      f"  ppl={ppl:.1f}"
                      f"  lr={lr:.2e}"
                      f"  t={elapsed:.1f}s")

                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(model, tokenizer, config, args.output_dir, step)

    print(f"\n  Training complete. Best val loss: {best_val:.4f}")
    return model, tokenizer, history


def evaluate(model, loader, device, max_batches=None):
    """Compute average validation loss."""
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            total_loss += loss.item()
            n_batches  += 1
            if max_batches and n_batches >= max_batches:
                break
    model.train()
    return total_loss / max(n_batches, 1)


def save_checkpoint(model, tokenizer, config, output_dir, step):
    """Save model + tokenizer."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config":      config,
        "vocab":       tokenizer.vocab,
        "merges":      tokenizer.merges,
        "step":        step,
    }, f"{output_dir}/checkpoint_best.pt")


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",  default="data/poetry_corpus.txt")
    p.add_argument("--output_dir", default="outputs/minigpt")
    p.add_argument("--vocab_size", type=int,   default=1000)
    p.add_argument("--block_size", type=int,   default=128)
    p.add_argument("--d_model",    type=int,   default=128)
    p.add_argument("--n_layers",   type=int,   default=4)
    p.add_argument("--n_heads",    type=int,   default=4)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--max_steps",  type=int,   default=1000)
    p.add_argument("--max_lr",     type=float, default=3e-4)
    p.add_argument("--grad_accum", type=int,   default=4)
    p.add_argument("--log_every",  type=int,   default=50)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, tokenizer, history = train(args)

    # Quick generation demo
    print("\n── Generation sample ──────────────────────────────────")
    prompt = "الخيل"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    device = next(model.parameters()).device
    generated = model.generate(
        prompt_ids.to(device),
        max_new_tokens=50,
        temperature=0.9,
        top_p=0.92,
    )
    output_ids = generated[0].tolist()
    text = tokenizer.decode(output_ids)
    print(f"Prompt : {prompt}")
    print(f"Output : {text}")
