"""
GPT From Scratch — Module 4: Complete GPT Model
=================================================
Assemble all pieces into a complete GPT model.
This is architecturally equivalent to GPT-2 small.

What you'll learn:
  • How all components stack together
  • Token + positional embedding combination
  • The language model head (LM head)
  • How to generate text autoregressively
  • How to count and understand model parameters

Run:  python -m gpt_from_scratch.gpt_model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from gpt_from_scratch.attention import make_causal_mask
from gpt_from_scratch.transformer_block import TransformerBlock


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Config
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GPTConfig:
    """
    GPT model hyperparameters.

    These determine the model size:
      • More layers (n_layers) → deeper, more abstract reasoning
      • More heads (n_heads)   → more relationship types per layer
      • More d_model           → richer token representations
      • More vocab_size        → larger language coverage

    GPT-2 variants for reference:
    ┌──────────────┬────────┬─────────┬────────┬─────────────┐
    │ Model        │ Layers │ d_model │ Heads  │ Parameters  │
    ├──────────────┼────────┼─────────┼────────┼─────────────┤
    │ GPT-2 small  │   12   │   768   │   12   │   117M      │
    │ GPT-2 medium │   24   │  1024   │   16   │   345M      │
    │ GPT-2 large  │   36   │  1280   │   20   │   762M      │
    │ GPT-2 XL     │   48   │  1600   │   25   │  1558M      │
    │ Our mini-GPT │    4   │   128   │    4   │    ~2M      │
    └──────────────┴────────┴─────────┴────────┴─────────────┘
    """
    vocab_size   : int   = 5000    # tokenizer vocabulary size
    max_seq_len  : int   = 256     # maximum context window
    d_model      : int   = 128     # embedding dimension
    n_layers     : int   = 4       # number of transformer blocks
    n_heads      : int   = 4       # attention heads per block
    dropout      : float = 0.1
    tie_weights  : bool  = True    # share token embedding ↔ LM head weights


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Complete GPT Model
# ═══════════════════════════════════════════════════════════════════════════════

class GPT(nn.Module):
    """
    Complete GPT decoder-only language model.

    Full forward pass:
      tokens → token_embeddings + positional_embeddings
             → dropout
             → TransformerBlock × n_layers
             → LayerNorm
             → Linear (LM head)
             → logits [vocab_size]
             → (optional) softmax → next-token probabilities
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # ── Embedding layers ──────────────────────────────────────────────────
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb   = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_drop  = nn.Dropout(config.dropout)

        # ── Transformer blocks ────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])

        # ── Final LayerNorm (before LM head) ──────────────────────────────────
        self.ln_f = nn.LayerNorm(config.d_model)

        # ── Language Model Head ───────────────────────────────────────────────
        # Maps d_model → vocab_size (predicts next token)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share weights between token_emb and lm_head
        # This is used in GPT-2. Benefits:
        #   • Reduces parameters by vocab_size × d_model
        #   • Encodes the intuition that "predicting token X" and
        #     "representing token X" should use similar vectors
        if config.tie_weights:
            self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        """GPT-2 style weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            input_ids: [batch, seq_len] — integer token ids
            targets:   [batch, seq_len] — next-token targets (for training)

        Returns:
            logits: [batch, seq_len, vocab_size]
            loss:   scalar cross-entropy loss (None if targets not provided)
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"

        device = input_ids.device

        # 1. Token + Positional Embeddings
        tok_emb = self.token_emb(input_ids)                    # [B, T, d_model]
        positions = torch.arange(T, device=device)
        pos_emb = self.pos_emb(positions)                      # [T, d_model]
        x = self.emb_drop(tok_emb + pos_emb)                   # [B, T, d_model]

        # 2. Causal mask
        mask = make_causal_mask(T, device)                     # [1, 1, T, T]

        # 3. Pass through all transformer blocks
        all_attn_weights = []
        for block in self.blocks:
            x, attn_w = block(x, mask=mask)
            all_attn_weights.append(attn_w)

        # 4. Final layer norm
        x = self.ln_f(x)                                       # [B, T, d_model]

        # 5. LM head: project to vocabulary
        logits = self.lm_head(x)                               # [B, T, vocab_size]

        # 6. Compute loss if targets provided (training mode)
        loss = None
        if targets is not None:
            # Cross-entropy over all positions
            # Reshape: [B, T, vocab_size] → [B×T, vocab_size]
            #           [B, T]            → [B×T]
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1,   # ignore padding positions
            )

        return logits, loss

    # ── Autoregressive Generation ──────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.9,
        top_p: float = 0.92,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation with nucleus (top-p) sampling.

        THE GENERATION LOOP:
          At each step:
          1. Forward pass → logits for all positions
          2. Take logits at the LAST position only (next token prediction)
          3. Apply temperature scaling
          4. Apply top-p filtering
          5. Sample from the filtered distribution
          6. Append sampled token to sequence
          7. Repeat

        Args:
            prompt_ids:     [1, prompt_len] initial token ids
            max_new_tokens: how many tokens to generate
            temperature:    > 1 = more random, < 1 = more greedy
            top_p:          nucleus threshold (0.0 = greedy, 1.0 = unrestricted)
            eos_token_id:   stop generation if this token is produced

        Returns:
            [1, prompt_len + max_new_tokens] generated token ids
        """
        self.eval()
        ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            ids_cond = ids[:, -self.config.max_seq_len:]

            # Forward pass
            logits, _ = self(ids_cond)

            # Focus on last position → next token distribution
            next_logits = logits[:, -1, :]   # [1, vocab_size]

            # Temperature scaling: divide logits before softmax
            # High temp → flat distribution (more random)
            # Low  temp → peaked distribution (more greedy)
            next_logits = next_logits / max(temperature, 1e-8)

            # Top-p (nucleus) filtering
            next_logits = self._top_p_filter(next_logits, top_p)

            # Sample from the distribution
            probs      = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)   # [1, 1]

            # Append to sequence
            ids = torch.cat([ids, next_token], dim=1)

            # Early stop on EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return ids

    def _top_p_filter(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        Nucleus (top-p) filtering.
        Keep only the smallest set of tokens whose cumulative probability ≥ top_p.
        Set all other logits to -∞.
        """
        if top_p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Tokens whose cumulative prob exceeds top_p get removed.
        # Shift the mask right by one so the FIRST token crossing the
        # threshold is still kept (otherwise we may drop everything when
        # the top token alone already exceeds top_p).
        remove_mask = cumulative_probs > top_p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(remove_mask, float('-inf'))

        # Scatter back to original ordering
        filtered = torch.full_like(logits, float('-inf'))
        filtered.scatter_(1, sorted_indices, sorted_logits)
        return filtered

    # ── Utility ───────────────────────────────────────────────────────────────

    def param_count(self) -> dict:
        """Detailed parameter breakdown by component."""
        def count(m): return sum(p.numel() for p in m.parameters())
        return {
            "token_embedding":   count(self.token_emb),
            "pos_embedding":     count(self.pos_emb),
            "transformer_blocks": count(self.blocks),
            "final_layernorm":   count(self.ln_f),
            "lm_head": 0 if self.config.tie_weights else count(self.lm_head),
            "TOTAL": count(self),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Demo
# ═══════════════════════════════════════════════════════════════════════════════

def demo():
    print("\n" + "="*60)
    print("  COMPLETE GPT MODEL — DEMO")
    print("="*60)

    config = GPTConfig(
        vocab_size=5000,
        max_seq_len=256,
        d_model=128,
        n_layers=4,
        n_heads=4,
        dropout=0.0,
        tie_weights=True,
    )

    model = GPT(config)
    model.eval()

    print(f"\nModel config:")
    for k, v in config.__dict__.items():
        print(f"  {k:<15} = {v}")

    print(f"\nParameter breakdown:")
    for component, count in model.param_count().items():
        marker = "←" if component == "TOTAL" else ""
        print(f"  {component:<25} : {count:>10,} {marker}")

    # Forward pass (training mode — with loss)
    B, T = 2, 32
    ids     = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    with torch.no_grad():
        logits, loss = model(ids, targets=targets)

    print(f"\nForward pass (training mode):")
    print(f"  Input  : {ids.shape}")
    print(f"  Logits : {logits.shape}  [batch, seq_len, vocab_size]")
    print(f"  Loss   : {loss.item():.4f}  (random init → should be ~log({config.vocab_size}) = {torch.log(torch.tensor(float(config.vocab_size))):.2f})")

    # Generation
    prompt = torch.tensor([[1, 42, 17, 89]])   # fake prompt token ids
    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.9,
        top_p=0.92,
    )

    print(f"\nAutoregressive generation:")
    print(f"  Prompt length    : {prompt.shape[1]}")
    print(f"  Generated length : {generated.shape[1]}")
    print(f"  New tokens       : {generated[0, prompt.shape[1]:].tolist()}")

    print(f"\n  Key takeaways:")
    print(f"  1. logits shape is always [B, T, vocab_size]")
    print(f"  2. For generation, we only care about logits[:, -1, :]")
    print(f"  3. Loss = cross_entropy(predicted_logit, true_next_token)")
    print(f"  4. Weight tying: lm_head.weight IS token_emb.weight")
    print(f"     → saves {config.vocab_size * config.d_model:,} parameters")


if __name__ == "__main__":
    demo()
