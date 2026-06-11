"""
Arabic Poetry Generation — Inference
======================================
Generate classical Arabic poetry using the fine-tuned aragpt2 model.

Run:
  python generate.py --prompt "الخيل والليل"
  python generate.py --prompt "أنا من أهوى" --strategy beam --num_beams 5
  python generate.py --prompt "إذا غامرت" --temperature 1.2 --max_new_tokens 200
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",   default="outputs/aragpt2-poetry",
                   help="Fine-tuned model dir or HuggingFace model ID")
    p.add_argument("--prompt",       default="الخيل والليل والبيداء",
                   help="Arabic prompt to start generation")
    p.add_argument("--strategy",     default="top_p",
                   choices=["greedy", "beam", "top_k", "top_p"])
    p.add_argument("--max_new_tokens", type=int,   default=120)
    p.add_argument("--temperature",    type=float, default=0.9)
    p.add_argument("--top_p",          type=float, default=0.92)
    p.add_argument("--top_k",          type=int,   default=50)
    p.add_argument("--num_beams",      type=int,   default=5)
    p.add_argument("--num_outputs",    type=int,   default=1)
    return p.parse_args()


def get_generation_kwargs(args, tokenizer) -> dict:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    base = {
        "max_new_tokens":       args.max_new_tokens,
        "num_return_sequences": args.num_outputs,
        "pad_token_id":         pad_id,
        "repetition_penalty":   1.3,  # discourage repeating same phrases
    }
    if args.strategy == "greedy":
        return {**base, "do_sample": False}
    if args.strategy == "beam":
        return {**base, "do_sample": False,
                "num_beams": args.num_beams, "early_stopping": True}
    if args.strategy == "top_k":
        return {**base, "do_sample": True,
                "temperature": args.temperature, "top_k": args.top_k}
    if args.strategy == "top_p":
        return {**base, "do_sample": True,
                "temperature": args.temperature, "top_p": args.top_p, "top_k": 0}


def generate_poetry(model_path: str, prompt: str, args) -> list[str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading model from: {model_path}")
    print(f"  Device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()

    gen_kwargs = get_generation_kwargs(args, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        for out in outputs
    ]


def format_as_verse(text: str) -> str:
    """Split generated text into verse lines for display."""
    # Split on common Arabic punctuation or every ~50 chars
    import re
    lines = re.split(r'[،؟\n]', text)
    lines = [l.strip() for l in lines if l.strip()]
    return '\n'.join(lines)


def main():
    args = parse_args()

    print(f"\n{'═'*60}")
    print(f"  Arabic Poetry Generator — Fine-tuned aragpt2")
    print(f"  Strategy : {args.strategy}")
    print(f"  Prompt   : {args.prompt}")
    print(f"{'═'*60}\n")

    results = generate_poetry(args.model_path, args.prompt, args)

    for i, text in enumerate(results, 1):
        print(f"── قصيدة {i} {'─'*48}")
        print(f"{args.prompt} {format_as_verse(text)}")
        print()


if __name__ == "__main__":
    main()
