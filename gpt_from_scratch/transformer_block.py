"""
GPT From Scratch — Module 3: Transformer Block
================================================
Build the complete transformer decoder block from scratch.
Each GPT-2 layer is one of these blocks stacked on top of each other.

What you'll learn:
  • Positional encoding — how transformers know word order
  • The Feed-Forward Network (FFN) and why it uses 4× expansion
  • GELU activation — why not ReLU?
  • LayerNorm and residual connections — why they're critical
  • Pre-LN vs Post-LN and why GPT-2 uses Pre-LN

Run:  python -m gpt_from_scratch.transformer_block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from gpt_from_scratch.attention import MultiHeadAttention, make_causal_mask


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Positional Encoding
# ═══════════════════════════════════════════════════════════════════════════════

"""
THE PROBLEM: Attention has no sense of order.
─────────────────────────────────────────────
"الخيل والليل والبيداء" and "والبيداء الخيل والليل" 
produce IDENTICAL attention computations without positional information.

SOLUTION: Add position information to each token embedding.

TWO APPROACHES:

1. Sinusoidal Positional Encoding (original "Attention is All You Need"):
   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
   Properties:
     • Deterministic, no parameters to learn
     • Can extrapolate to longer sequences than seen in training
     • Each position has a unique fingerprint

2. Learned Positional Embeddings (GPT-2, BERT):
   A simple embedding table: nn.Embedding(max_seq_len, d_model)
   Properties:
     • More flexible, can learn task-specific patterns
     • Cannot extrapolate beyond max_seq_len
     • Slightly better in practice for fixed-length tasks

GPT-2 uses LEARNED embeddings.
"""

class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings used in GPT-2.
    Simply an embedding lookup: position → d_model vector.
    """
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model] — token embeddings
        Returns token_embeddings + positional_embeddings
        """
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device)        # [T]
        pos_emb   = self.embedding(positions)                # [T, d_model]
        return x + pos_emb.unsqueeze(0)                      # broadcast over batch


class SinusoidalPositionalEncoding(nn.Module):
    """
    Original sinusoidal PE from 'Attention is All You Need'.
    Included for comparison and educational purposes.
    """
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(max_seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Feed-Forward Network (FFN)
# ═══════════════════════════════════════════════════════════════════════════════

"""
WHAT IS THE FFN?
────────────────
After attention mixes information across positions,
the FFN processes each position INDEPENDENTLY.

Think of it as:
  • Attention  = "gather information from other tokens"
  • FFN        = "think about what I just gathered"

Architecture:
  Linear(d_model → 4×d_model) → GELU → Linear(4×d_model → d_model)

WHY 4× EXPANSION?
  Empirically shown to work well. The wider hidden layer gives the network
  more capacity to store factual associations.
  (Geva et al., 2021 showed FFN layers act as "key-value memories")

WHY GELU INSTEAD OF RELU?
  ReLU: f(x) = max(0, x)          — hard zero cutoff
  GELU: f(x) = x·Φ(x)             — smooth, probabilistic gating
  GELU performs better on NLP tasks (used in BERT, GPT-2, GPT-3).
  The smooth gradient allows better learning near zero.
"""

class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Applied identically to every position.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        d_ff = d_model * 4   # standard 4× expansion

        self.fc1     = nn.Linear(d_model, d_ff)
        self.fc2     = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        x = self.fc1(x)               # → [B, T, 4×d_model]
        x = F.gelu(x)                 # GELU activation
        x = self.dropout(x)
        x = self.fc2(x)               # → [B, T, d_model]
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Transformer Block (GPT-style, Pre-LN)
# ═══════════════════════════════════════════════════════════════════════════════

"""
RESIDUAL CONNECTIONS
────────────────────
output = LayerNorm(x + sublayer(x))

Why residuals?
  • Allow gradients to flow directly back through the network (skip connection)
  • Prevent vanishing gradients in deep networks
  • The network learns the RESIDUAL (correction) rather than full transformation
  • In theory: a 12-layer GPT can still learn as well as a 1-layer if needed

PRE-LN vs POST-LN
──────────────────
Original paper (Post-LN):
  output = LayerNorm(x + Attention(x))

GPT-2 (Pre-LN):
  output = x + Attention(LayerNorm(x))

Pre-LN benefits:
  • More stable gradients at training start
  • Allows training without learning rate warmup
  • Better for very deep models (e.g. GPT-3 with 96 layers)
"""

class TransformerBlock(nn.Module):
    """
    One GPT-style transformer decoder block (Pre-LN).

    Structure:
      x → LayerNorm → MultiHeadAttention → + x  (residual 1)
        → LayerNorm → FFN                → + x  (residual 2)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = FeedForwardNetwork(d_model, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    [batch, seq_len, d_model]
            mask: causal mask

        Returns:
            x:       [batch, seq_len, d_model]
            weights: attention weights from this block
        """
        # Sub-layer 1: Causal Self-Attention (with Pre-LN + residual)
        residual    = x
        x_norm      = self.ln1(x)
        attn_out, weights = self.attn(x_norm, mask=mask)
        x           = residual + self.drop(attn_out)

        # Sub-layer 2: Feed-Forward Network (with Pre-LN + residual)
        residual    = x
        x_norm      = self.ln2(x)
        ffn_out     = self.ffn(x_norm)
        x           = residual + self.drop(ffn_out)

        return x, weights


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Demo
# ═══════════════════════════════════════════════════════════════════════════════

def demo():
    print("\n" + "="*60)
    print("  TRANSFORMER BLOCK — DEMO")
    print("="*60)

    d_model = 64
    n_heads = 4
    seq_len = 10
    batch   = 2

    block = TransformerBlock(d_model=d_model, n_heads=n_heads, dropout=0.0)
    mask  = make_causal_mask(seq_len, device=torch.device("cpu"))
    x     = torch.randn(batch, seq_len, d_model)

    print(f"\nConfig: d_model={d_model}, n_heads={n_heads}, seq_len={seq_len}")

    with torch.no_grad():
        out, weights = block(x, mask=mask)

    print(f"\nInput  shape : {x.shape}")
    print(f"Output shape : {out.shape}   (unchanged — residual connection)")

    # Verify residual didn't collapse
    diff = (out - x).abs().mean().item()
    print(f"Mean |output - input|: {diff:.4f}  (non-zero means block learned something)")

    # Parameter count
    def count_params(module):
        return sum(p.numel() for p in module.parameters())

    print(f"\nParameter breakdown:")
    print(f"  LayerNorm ×2     : {count_params(block.ln1) + count_params(block.ln2):,}")
    print(f"  MultiHeadAttn    : {count_params(block.attn):,}")
    print(f"  FFN (4× expand)  : {count_params(block.ffn):,}")
    print(f"  Total per block  : {count_params(block):,}")

    # Positional encoding comparison
    print(f"\nPositional Encoding comparison (d_model={d_model}, seq_len={seq_len}):")
    learned = LearnedPositionalEmbedding(max_seq_len=1024, d_model=d_model)
    sinus   = SinusoidalPositionalEncoding(max_seq_len=1024, d_model=d_model)

    dummy = torch.zeros(1, seq_len, d_model)
    with torch.no_grad():
        out_l = learned(dummy)
        out_s = sinus(dummy)

    print(f"  Learned  output[0,0,:4] : {out_l[0,0,:4].tolist()}")
    print(f"  Sinus    output[0,0,:4] : {out_s[0,0,:4].tolist()}")
    print(f"  Learned params: {count_params(learned):,}   (trainable)")
    print(f"  Sinus   params: {count_params(sinus):,}     (fixed buffer)")

    print("\n  Key takeaways:")
    print("  1. Block output shape = input shape (critical for stacking)")
    print("  2. Pre-LN normalizes BEFORE each sublayer (GPT-2 style)")
    print("  3. FFN uses 4× hidden expansion: Linear(64→256→64)")
    print("  4. Residual connections let gradients bypass each block")


if __name__ == "__main__":
    demo()
