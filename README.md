# Arabic Poetry Generator — Fine-tuned GPT + GPT from Scratch

Generate classical Arabic poetry (فصحى) in the style of Al-Mutanabbi and Ibn Arabi.  
This project has two parts: a **fine-tuned production model** and a **full educational GPT implementation from scratch**.

---

## Project Structure

```
arabic-poetry-gpt/
│
├── data/
│   └── prepare.py              # Load & clean arbml/ashaar dataset (180k poems)
│
├── finetune/
│   └── train.py                # Fine-tune aragpt2-base on classical Arabic poetry
│
├── gpt_from_scratch/           ← EDUCATIONAL MODULE
│   ├── tokenizer_bpe.py        # Module 1 — BPE tokenizer built from scratch
│   ├── attention.py            # Module 2 — Scaled dot-product + multi-head attention
│   ├── transformer_block.py    # Module 3 — Full transformer decoder block (Pre-LN)
│   ├── gpt_model.py            # Module 4 — Complete GPT model with generation
│   └── training.py             # Module 5 — Training loop: AdamW, warmup, cosine decay
│
├── generate.py                 # Inference for fine-tuned model
├── requirements.txt
└── README.md
```

---

## Part 1 — Fine-Tuned Model

### Why `aragpt2-base` and not `gpt2`?

`gpt2` was pre-trained on English text. Its tokenizer has almost no Arabic vocabulary — Arabic words would break down into single bytes or unknown tokens, making fine-tuning ineffective.

`aubmindlab/aragpt2-base` was pre-trained on 77GB of Arabic text. Its SentencePiece tokenizer handles Arabic morphology, making it the correct starting point for Arabic generation.

### Quick Start

```bash
pip install -r requirements.txt

# Step 1: Download and prepare the dataset
python data/prepare.py

# Step 2: Fine-tune
python finetune/train.py --epochs 3 --batch_size 2

# Step 3: Generate poetry
python generate.py --prompt "الخيل والليل والبيداء" --strategy top_p
python generate.py --prompt "أنا من أهوى" --strategy beam --num_beams 5
```

### Generation Strategies

| Strategy | Diversity | Coherence | Best For |
|---|---|---|---|
| Greedy | None | Highest | Deterministic output |
| Beam search | Low | Very high | Formal, structured verse |
| Top-k | Medium | Medium | Creative exploration |
| Top-p (nucleus) | High | Good | Default — best balance |

---

## Part 2 — GPT From Scratch (Educational Module)

Run each module independently. Each is self-contained with inline explanations.

### Module 1 — BPE Tokenizer
```bash
python -m gpt_from_scratch.tokenizer_bpe
```
Builds a Byte-Pair Encoding tokenizer from scratch on Arabic text.  
Covers: character vocab → pair counting → merging → encode/decode.

**Key concept:**
```
"العزائم" → ['الع', 'زا', 'ئم', '</w>'] → [234, 189, 67, 1]
```

### Module 2 — Attention Mechanism
```bash
python -m gpt_from_scratch.attention
```
Implements scaled dot-product attention and multi-head attention from scratch.

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

Covers: Q/K/V intuition, √d_k scaling, causal masking, multi-head parallelism.

### Module 3 — Transformer Block
```bash
python -m gpt_from_scratch.transformer_block
```
Assembles a complete GPT decoder block.

```
x → LayerNorm → MultiHeadAttention → + x (residual)
  → LayerNorm → FFN(4× expand, GELU) → + x (residual)
```

Covers: positional encoding (learned vs sinusoidal), GELU vs ReLU, Pre-LN vs Post-LN, residual connections.

### Module 4 — Complete GPT Model
```bash
python -m gpt_from_scratch.gpt_model
```
Full GPT model with autoregressive generation.

```
tokens → Embeddings → TransformerBlock × N → LayerNorm → LM Head → logits
```

Covers: weight tying, top-p sampling, generation loop, parameter breakdown.

### Module 5 — Training Loop
```bash
python -m gpt_from_scratch.training --max_steps 500
```
Raw PyTorch training loop — no Trainer API.

Covers: AdamW with selective weight decay, LR warmup + cosine decay, gradient clipping, gradient accumulation.

---

## Architecture Comparison

| | Mini-GPT (scratch) | aragpt2-base |
|---|---|---|
| Layers | 4 | 12 |
| d_model | 128 | 768 |
| Heads | 4 | 12 |
| Parameters | ~2M | 135M |
| Context | 256 tokens | 1024 tokens |
| Training data | Your corpus | 77GB Arabic |
| Purpose | Education | Production |

---

## Dataset

`arbml/ashaar` — 180,000+ classical Arabic poems from HuggingFace.  
Poets include: Al-Mutanabbi, Ibn Arabi, Al-Buhtiri, Abu Nuwas, Al-Farazdaq.

The `data/prepare.py` script:
- Normalizes Arabic letter forms (أ إ آ → ا)
- Optionally removes tashkeel (diacritics)
- Filters by poem length
- Saves a clean plain-text corpus

---

## Evaluation

Training tracks **perplexity** (PP):

```
PP = exp(cross-entropy loss)
```

| Stage | Expected PP |
|---|---|
| Random init on Arabic | ~1000–5000 |
| Pre-trained aragpt2 on Arabic text | ~30–60 |
| Fine-tuned on classical poetry | ~15–35 |

Lower perplexity = the model is less "surprised" by the text.

---

## References

- [GPT-2 Paper — Radford et al. (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Attention Is All You Need — Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)
- [Neural Machine Translation of Rare Words with Subword Units (BPE) — Sennrich et al. (2016)](https://arxiv.org/abs/1508.07909)
- [Decoupled Weight Decay Regularization (AdamW) — Loshchilov & Hutter (2019)](https://arxiv.org/abs/1711.05101)
- [AraGPT2 — Antoun et al. (2020)](https://arxiv.org/abs/2012.15520)
- [arbml/ashaar dataset](https://huggingface.co/datasets/arbml/ashaar)
