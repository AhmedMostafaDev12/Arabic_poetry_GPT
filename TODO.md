# Arabic Poetry GPT — Project Checklist

Tracking remaining work to take this from a working prototype to a
**CV-ready, deployed portfolio project**.

---

## ✅ Phase 1 — Bug fixes (DONE)

- [x] Renamed digit-prefixed modules in `gpt_from_scratch/` so Python can import them
  (`01_tokenizer_bpe.py` → `tokenizer_bpe.py`, etc.)
- [x] Updated all internal imports + docstring `Run:` examples to the new names
- [x] Fixed top-p (nucleus) filter bug in `gpt_from_scratch/gpt_model.py`
      (now correctly keeps the first token crossing the threshold)
- [x] Fixed `finetune/train.py`:
      `evaluation_strategy` → `eval_strategy`,
      `tokenizer=` → `processing_class=`,
      auto-enable `fp16` on CUDA
- [x] Fixed hardcoded `pad_token_id=2` in `generate.py` — now reads from tokenizer
- [x] Updated `README.md` to reference the new file names

---

## 🧪 Phase 2 — Train and collect real numbers

- [ ] Run `python data/prepare.py` and confirm `data/poetry_corpus.txt` exists
- [ ] Fine-tune on a Colab/Kaggle GPU:
      `python finetune/train.py --epochs 3 --batch_size 2`
- [ ] **Record actual final perplexity** and replace the `~15–35` placeholder in `README.md`
- [ ] Save 5–10 prompt → output samples (one per strategy: greedy / beam / top-k / top-p)
      to `examples/generations.md`
- [ ] Add `matplotlib` plot of train/val loss + LR curve →
      save to `outputs/training_curve.png` and embed in README
- [ ] (Optional) Train the from-scratch mini-GPT for ~500 steps so you can report
      **two perplexity numbers** (mini-GPT vs fine-tuned)

---

## 📦 Phase 3 — Repo hygiene (must-have for CV)

- [ ] `git init` → first commit
- [ ] Create a **public GitHub repo** named `arabic-poetry-gpt`
- [ ] Add `.gitignore`:
  ```gitignore
  __pycache__/
  *.py[cod]
  *.pt
  *.bin
  outputs/
  data/poetry_corpus.txt
  .venv/
  .env
  .ipynb_checkpoints/
  ```
- [ ] Add `LICENSE` (MIT — one click in GitHub's "Add file" menu)
- [ ] Update `requirements.txt` — add `numpy`, `matplotlib`, `tqdm`,
      and pin major versions
- [ ] Add at least 3 sanity tests in `tests/`:
  - [ ] `test_attention.py` — shape check on `MultiHeadAttention`
  - [ ] `test_bpe.py` — encode → decode roundtrip on Arabic text
  - [ ] `test_gpt.py` — `GPT` forward pass returns `[B, T, vocab_size]`
- [ ] Add a `Makefile` or `scripts/` directory (`make train`, `make generate`)
- [ ] Delete the messy `arabic_poetry_finetuning (1).ipynb` or rename it cleanly

---

## 📝 Phase 4 — README polish

- [ ] Add **badges** at top (Python version, license, HF Hub link, HF Spaces link)
- [ ] Add a **Results** section with:
  - [ ] Real final perplexity number
  - [ ] `outputs/training_curve.png` embedded
  - [ ] 2–3 sample generations inline
- [ ] Add a **Quickstart with the deployed model**:
  ```python
  from transformers import pipeline
  pipe = pipeline("text-generation", model="ahmedmostafagamal8/aragpt2-poetry")
  print(pipe("الخيل والليل والبيداء")[0]["generated_text"])
  ```
- [ ] Add **dataset citation** for `arbml/ashaar`
- [ ] Add **training resources** (Colab/Kaggle? T4? training time?)
- [ ] Link the HF Space demo + Hub model card at the very top

---

## 🚀 Phase 5 — Deployment

### Push the model to HuggingFace Hub

- [ ] `pip install huggingface_hub` and `huggingface-cli login`
- [ ] Push model + tokenizer:
  ```python
  from transformers import AutoTokenizer, AutoModelForCausalLM
  model     = AutoModelForCausalLM.from_pretrained("outputs/aragpt2-poetry")
  tokenizer = AutoTokenizer.from_pretrained("outputs/aragpt2-poetry")
  model.push_to_hub("ahmedmostafagamal8/aragpt2-poetry")
  tokenizer.push_to_hub("ahmedmostafagamal8/aragpt2-poetry")
  ```
- [ ] Write a proper **model card** (README inside the model repo):
      base model, dataset, hyperparams, final perplexity, examples,
      intended use, limitations, bias notes (classical vs modern register)

### Gradio demo on HF Spaces

- [ ] Create a new Space → SDK: Gradio → free CPU
- [ ] Add `app.py` (see template in phase 5 docs / previous chat)
- [ ] Add Space `requirements.txt`: `torch`, `transformers`, `gradio`
- [ ] Test the live URL → bookmark it for your CV

---

## 💎 Phase 6 — Stretch (optional, each adds CV weight)

- [ ] Add a **FastAPI + Dockerfile** to the GitHub repo (code only — no need to host)
- [ ] Quantize model with `bitsandbytes` (int8) or export to **ONNX**
- [ ] Add a **distinct-n** diversity metric alongside perplexity
- [ ] Add a **GitHub Actions** CI workflow running `pytest` on every push
- [ ] Add `ruff` lint step in the same CI workflow
- [ ] (Ambitious) Add an Arabic meter/rhythm checker on output

---

## ✏️ Phase 7 — CV writeup

- [ ] Update the 3-line CV brief with the real perplexity number
- [ ] Add links: GitHub repo + HF Hub model + HF Space demo
- [ ] Mention concrete numbers in the brief:
      180K poems · 135M params · final PP · training time · free-tier deployment

---

## Recommended order of work

1. **This weekend:** Phases 2 + 3 (train, init git, push to GitHub with hygiene files)
2. **Next week:** Phases 4 + 5 (polish README, push to Hub, build Gradio Space)
3. **Free weekend after:** Phase 6 (only what you find fun)
4. **When done:** Phase 7 — update the CV

Last meaningful update to this file: Phase 1 completed on 2026-06-11.
