"""
GPT From Scratch — Module 2: Attention Mechanism
==================================================
Build scaled dot-product attention and multi-head attention from scratch.
This is the core operation of every transformer — the "secret sauce".

What you'll learn:
  • What attention actually computes and why
  • Q, K, V matrices — what they represent
  • Why we scale by √d_k
  • How causal masking enforces left-to-right generation
  • Multi-head attention — why multiple heads?

Run:  python -m gpt_from_scratch.attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Intuition
# ═══════════════════════════════════════════════════════════════════════════════

"""
THE ATTENTION INTUITION
───────────────────────
Consider the Arabic sentence:
  "الخيل والليل والبيداء تعرفني"
   (The horses, the night, and the desert know me)

When processing "تعرفني" (know me), the model must decide:
  → How much should I look at "الخيل"?   (horses — subject)
  → How much should I look at "الليل"?   (night  — subject)
  → How much should I look at "والبيداء"? (desert — subject)

Attention computes exactly this: a weighted average over all previous tokens,
where the weights are learned from the content of those tokens.

THE Q, K, V FRAMEWORK
──────────────────────
Think of it like a search engine:
  • Query  (Q): "What information am I looking for?"
  • Key    (K): "What information does each token advertise?"
  • Value  (V): "What information does each token actually contain?"

Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V

Step by step:
  1. Q · Kᵀ         → similarity scores between each query-key pair
  2. / √d_k         → scale to prevent vanishing gradients (see below)
  3. + mask          → set future positions to -∞ (causal attention)
  4. softmax(...)    → convert scores to probabilities (sum to 1)
  5. · V             → weighted sum of values
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Scaled Dot-Product Attention
# ═══════════════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention (Vaswani et al., 2017).

    Args:
        Q: Query  tensor  [batch, heads, seq_len, d_k]
        K: Key    tensor  [batch, heads, seq_len, d_k]
        V: Value  tensor  [batch, heads, seq_len, d_v]
        mask: Boolean mask [batch, 1, seq_len, seq_len] — True = MASKED (ignored)
        dropout: dropout probability on attention weights

    Returns:
        output:  [batch, heads, seq_len, d_v]
        weights: [batch, heads, seq_len, seq_len] — attention probabilities
    """
    d_k = Q.size(-1)

    # Step 1: Compute raw attention scores
    # Q: [B, H, T, d_k]  ·  Kᵀ: [B, H, d_k, T]  →  scores: [B, H, T, T]
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Step 2: Scale by √d_k
    # WHY? Without scaling, dot products grow large as d_k increases.
    # Large values push softmax into regions with near-zero gradients.
    # Example: if d_k=64, raw scores could be ~8x larger → scale by 1/√64 = 1/8
    scores = scores / math.sqrt(d_k)

    # Step 3: Apply causal mask (for GPT — decoder-only)
    # The mask is a lower-triangular matrix:
    #   token at position i can only attend to positions 0..i
    #   positions i+1, i+2, ... are set to -∞ → softmax gives 0
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    # Step 4: Softmax → attention probabilities
    # Each row sums to 1.0 — this is the "attention distribution"
    weights = F.softmax(scores, dim=-1)

    # Handle -inf columns (fully masked positions → NaN after softmax → zero)
    weights = torch.nan_to_num(weights, nan=0.0)

    # Optional dropout on attention weights (regularization)
    if dropout > 0.0 and torch.is_grad_enabled():
        weights = F.dropout(weights, p=dropout)

    # Step 5: Weighted sum of values
    # weights: [B, H, T, T]  ·  V: [B, H, T, d_v]  →  [B, H, T, d_v]
    output = torch.matmul(weights, V)

    return output, weights


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Multi-Head Attention
# ═══════════════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (Vaswani et al., 2017).

    WHY MULTIPLE HEADS?
    ───────────────────
    A single attention head learns one type of relationship.
    Multiple heads run in parallel, each in a lower-dimensional space.
    Each head can specialize in different relationships:
      Head 1 → syntactic dependencies (subject → verb)
      Head 2 → long-range references  (pronoun → antecedent)
      Head 3 → positional patterns    (adjacent tokens)
      Head 4 → semantic similarity    (synonyms)
    ...etc. The heads are NOT pre-assigned — they learn specialization.

    Architecture:
      d_model → split into h heads of size d_k = d_model / h
      Each head: Q_i, K_i, V_i = linear projections of input
      Concatenate all head outputs → project back to d_model
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads   # dimension per head
        self.dropout  = dropout

        # Single linear layers for all heads combined (efficient)
        # Instead of h separate Wq_i, Wk_i, Wv_i matrices,
        # we use one big matrix and split afterward
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)   # output projection

        self._init_weights()

    def _init_weights(self):
        # Scaled initialization — prevents signal from exploding/vanishing
        for layer in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    [batch, seq_len, d_model]
            mask: [batch, 1, seq_len, seq_len] causal mask

        Returns:
            output:  [batch, seq_len, d_model]
            weights: [batch, n_heads, seq_len, seq_len]
        """
        B, T, _ = x.shape

        # 1. Linear projections
        Q = self.W_q(x)   # [B, T, d_model]
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Split into heads: [B, T, d_model] → [B, n_heads, T, d_k]
        # Reshape: d_model = n_heads × d_k
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 3. Attention for all heads in parallel
        attn_out, weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )

        # 4. Concatenate heads: [B, n_heads, T, d_k] → [B, T, d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # 5. Final output projection
        output = self.W_o(attn_out)

        return output, weights


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Causal Mask Utility
# ═══════════════════════════════════════════════════════════════════════════════

def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (autoregressive) mask.

    Returns a [1, 1, seq_len, seq_len] boolean tensor where:
      True  = this position is MASKED (set to -inf before softmax)
      False = this position is VISIBLE

    Example for seq_len=4:
      [[False  True  True  True ]
       [False False  True  True ]
       [False False False  True ]
       [False False False False]]

    Token 0 can only see itself.
    Token 3 can see tokens 0, 1, 2, 3.
    """
    # Upper triangular matrix (above diagonal) = future positions
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).to(device)   # [1, 1, T, T]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Demo
# ═══════════════════════════════════════════════════════════════════════════════

def demo():
    print("\n" + "="*60)
    print("  MULTI-HEAD ATTENTION — DEMO")
    print("="*60)

    # GPT-2 small config
    d_model  = 64
    n_heads  = 4
    seq_len  = 8
    batch    = 2

    device = torch.device("cpu")

    # Random input (simulating token embeddings)
    x = torch.randn(batch, seq_len, d_model)

    print(f"\nConfig:")
    print(f"  d_model  = {d_model}  (embedding dimension)")
    print(f"  n_heads  = {n_heads}  (attention heads)")
    print(f"  d_k      = {d_model // n_heads}  (dimension per head = d_model / n_heads)")
    print(f"  seq_len  = {seq_len}")
    print(f"  batch    = {batch}")

    # Build causal mask
    mask = make_causal_mask(seq_len, device)
    print(f"\nCausal mask (seq_len={seq_len}):")
    print("  T = masked (future), F = visible (past/present)")
    for row in mask[0, 0]:
        print("  " + " ".join("T" if v else "F" for v in row))

    # Forward pass
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    with torch.no_grad():
        output, weights = mha(x, mask=mask)

    print(f"\nForward pass:")
    print(f"  Input  shape : {x.shape}")
    print(f"  Output shape : {output.shape}  (same as input — d_model preserved)")
    print(f"  Weights shape: {weights.shape}  [batch, heads, seq, seq]")

    print(f"\nAttention weights for batch=0, head=0:")
    print("  (row = query token, col = key token, value = how much attention)")
    w = weights[0, 0].detach()
    for i, row in enumerate(w):
        vals = " ".join(f"{v:.2f}" for v in row)
        print(f"  token {i}: [{vals}]")
    print("  Note: upper triangle is 0 (causal mask in effect)")

    # Count parameters
    n_params = sum(p.numel() for p in mha.parameters())
    print(f"\nParameters in MHA: {n_params:,}")
    print(f"  = 4 × (d_model × d_model) = 4 × {d_model} × {d_model} = {4*d_model*d_model}")

    print("\n  Key takeaways:")
    print("  1. Output shape = Input shape (attention is a mixing operation)")
    print("  2. Each head sees a different projection of the same input")
    print("  3. Causal mask ensures token i cannot peek at token i+1, i+2, ...")
    print("  4. Softmax makes each row of weights sum to 1.0")


if __name__ == "__main__":
    demo()
