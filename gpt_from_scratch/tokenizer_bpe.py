"""
GPT From Scratch — Module 1: Tokenizer & BPE
==============================================
Build a Byte-Pair Encoding (BPE) tokenizer from scratch.
This is the same algorithm used by GPT-2, GPT-3, GPT-4, and LLaMA.

What you'll learn:
  • Why we don't tokenize by character or word
  • How BPE learns a vocabulary from raw text
  • How encoding/decoding works
  • Why Arabic needs special tokenization treatment

Run:  python -m gpt_from_scratch.tokenizer_bpe
"""

import re
from collections import defaultdict, Counter


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Why Tokenization Matters
# ═══════════════════════════════════════════════════════════════════════════════

"""
A language model operates on INTEGERS, not text.
We need a reversible mapping:  text ↔ sequence of integers

Three naive approaches and their problems:
─────────────────────────────────────────────────────────────────────────────
1. Character-level tokenization
   "مرحبا" → ['م','ر','ح','ب','ا'] → [40, 41, 42, 43, 44]
   Problem: Very long sequences. No sense of morphology.
   Arabic word = ~4 chars → context window fills with short info

2. Word-level tokenization
   "مرحبا بالعالم" → ['مرحبا', 'بالعالم'] → [1234, 5678]
   Problem: Huge vocabulary. "مرحبا" and "مرحباً" are different tokens.
   Arabic has rich morphology — same root, dozens of surface forms.
   Unknown words become [UNK] → information loss.

3. Byte-Pair Encoding (BPE) — what GPT uses ✓
   Learns frequent subword units from data.
   "مرحبا" → ['مرح', '##با'] → [890, 234]
   Benefits:
     • Fixed vocabulary size (e.g. 50,000)
     • No unknown words (worst case = individual bytes)
     • Captures morphological patterns
     • Balances sequence length vs vocabulary size
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: BPE Algorithm Implementation
# ═══════════════════════════════════════════════════════════════════════════════

class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer trained from scratch.

    Algorithm (Sennrich et al., 2016):
    ──────────────────────────────────
    TRAINING:
      1. Start with character-level vocabulary
      2. Count all adjacent symbol pairs in the corpus
      3. Merge the most frequent pair into a new symbol
      4. Repeat until vocabulary reaches target size

    ENCODING (at inference):
      Apply the learned merges greedily, left to right.
    """

    def __init__(self, vocab_size: int = 500):
        self.vocab_size   = vocab_size
        self.merges       = {}          # (a, b) → ab
        self.vocab        = {}          # token → id
        self.id_to_token  = {}          # id → token
        self._trained     = False

    # ── Training ──────────────────────────────────────────────────────────────

    def _get_word_freqs(self, corpus: str) -> dict:
        """
        Split corpus into words and count frequencies.
        Each word is represented as a tuple of characters + end-of-word marker.
        The </w> marker lets the model distinguish word-internal from word-final tokens.
        e.g. "مرحبا" → ('م', 'ر', 'ح', 'ب', 'ا', '</w>')
        """
        word_freqs = defaultdict(int)
        for word in corpus.split():
            if word:
                # Represent word as space-separated characters with </w> at end
                word_freqs[' '.join(list(word)) + ' </w>'] += 1
        return dict(word_freqs)

    def _get_pair_freqs(self, word_freqs: dict) -> Counter:
        """Count all adjacent symbol pairs across the corpus."""
        pairs = Counter()
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def _merge_pair(self, pair: tuple, word_freqs: dict) -> dict:
        """Apply one merge rule to all words in the corpus."""
        a, b = pair
        merged = a + b   # new merged symbol (no space between them)
        pattern = re.compile(r'(?<!\S)' + re.escape(a + ' ' + b) + r'(?!\S)')
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            new_word = pattern.sub(merged, word)
            new_word_freqs[new_word] = freq
        return new_word_freqs

    def train(self, corpus: str, verbose: bool = True):
        """
        Train BPE on a text corpus.

        Args:
            corpus: raw text string
            verbose: print merge operations
        """
        print(f"\n{'─'*60}")
        print(f"  Training BPE Tokenizer")
        print(f"  Target vocab size : {self.vocab_size}")
        print(f"  Corpus length     : {len(corpus):,} chars")
        print(f"{'─'*60}")

        # Step 1: Build initial character vocabulary
        word_freqs = self._get_word_freqs(corpus)
        chars = set()
        for word in word_freqs:
            for symbol in word.split():
                chars.add(symbol)

        # Initial vocab: all unique characters + special tokens
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        vocab_list = special_tokens + sorted(chars)
        self.vocab        = {tok: i for i, tok in enumerate(vocab_list)}
        self.id_to_token  = {i: tok for tok, i in self.vocab.items()}

        if verbose:
            print(f"\n  Initial character vocab: {len(self.vocab)} tokens")

        # Step 2: Learn merges
        n_merges = self.vocab_size - len(self.vocab)
        for step in range(n_merges):
            pair_freqs = self._get_pair_freqs(word_freqs)
            if not pair_freqs:
                break

            # Pick the most frequent pair
            best_pair  = max(pair_freqs, key=pair_freqs.get)
            best_freq  = pair_freqs[best_pair]

            # Stop if no pair appears more than once (no benefit to merging)
            if best_freq < 2:
                break

            # Record and apply the merge
            new_symbol = best_pair[0] + best_pair[1]
            self.merges[best_pair] = new_symbol

            new_id = len(self.vocab)
            self.vocab[new_symbol]       = new_id
            self.id_to_token[new_id]     = new_symbol

            word_freqs = self._merge_pair(best_pair, word_freqs)

            if verbose and step < 20:
                print(f"  Merge {step+1:3d}: {best_pair[0]!r} + {best_pair[1]!r}"
                      f" → {new_symbol!r}  (freq={best_freq})")

        self._trained = True
        print(f"\n  Final vocab size: {len(self.vocab)}")
        print(f"  Total merges learned: {len(self.merges)}")

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _tokenize_word(self, word: str) -> list[str]:
        """Apply learned BPE merges to a single word."""
        symbols = list(word) + ['</w>']

        # Apply merges in the order they were learned
        for (a, b), merged in self.merges.items():
            i = 0
            new_symbols = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i+1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def encode(self, text: str) -> list[int]:
        """Convert text → list of token ids."""
        assert self._trained, "Call .train() first"
        ids = [self.vocab.get('<BOS>', 2)]
        for word in text.split():
            tokens = self._tokenize_word(word)
            for tok in tokens:
                ids.append(self.vocab.get(tok, self.vocab.get('<UNK>', 1)))
        ids.append(self.vocab.get('<EOS>', 3))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Convert list of token ids → text."""
        tokens = [self.id_to_token.get(i, '<UNK>') for i in ids]
        # Remove special tokens, join, restore word boundaries
        text = ' '.join(
            tok for tok in tokens
            if tok not in ('<PAD>', '<BOS>', '<EOS>', '<UNK>')
        )
        text = text.replace('</w> ', ' ').replace('</w>', '')
        return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Demo
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_CORPUS = """
على قدر أهل العزم تأتي العزائم وتأتي على قدر الكرام المكارم
وتعظم في عين الصغير صغارها وتصغر في عين العظيم العظائم
الخيل والليل والبيداء تعرفني والسيف والرمح والقرطاس والقلم
أعز مكان في الدنيا سرج سابح وخير جليس في الزمان كتاب
أنا من أهوى ومن أهوى أنا نحن روحان حللنا بدنا
لا تحزن على ما فات فإن الحزن لا يعيد ما فات
العلم نور والجهل ظلام والعاقل من يطلب النور
"""


def demo():
    print("\n" + "="*60)
    print("  BPE TOKENIZER — DEMO")
    print("="*60)

    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(SAMPLE_CORPUS, verbose=True)

    test_phrases = [
        "العزائم تأتي",
        "الخيل والليل",
        "العلم نور",
    ]

    print(f"\n{'─'*60}")
    print("  Encoding test phrases:")
    print(f"{'─'*60}")

    for phrase in test_phrases:
        ids      = tokenizer.encode(phrase)
        decoded  = tokenizer.decode(ids)
        tokens   = [tokenizer.id_to_token[i] for i in ids]
        print(f"\n  Input   : {phrase!r}")
        print(f"  Tokens  : {tokens}")
        print(f"  IDs     : {ids}")
        print(f"  Decoded : {decoded!r}")

    print(f"\n{'─'*60}")
    print("  Vocabulary sample (first 30 non-special tokens):")
    print(f"{'─'*60}")
    shown = 0
    for token, idx in sorted(tokenizer.vocab.items(), key=lambda x: x[1]):
        if token not in ('<PAD>', '<UNK>', '<BOS>', '<EOS>') and shown < 30:
            print(f"  {idx:4d}  {token!r}")
            shown += 1

    print("\n  Key insight:")
    print("  Common subwords get their own token → efficient sequence length")
    print("  Rare combinations fall back to character pieces → no unknown words")


if __name__ == "__main__":
    demo()
