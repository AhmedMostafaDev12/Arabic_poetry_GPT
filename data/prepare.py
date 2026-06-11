"""
Arabic Poetry Dataset — Preparation
======================================
Loads classical Arabic poetry from HuggingFace, cleans it,
and saves a plain-text corpus ready for fine-tuning.

Dataset: "arbml/ashaar" — 180k+ classical Arabic poems
         includes Mutanabbi, Ibn Arabi, Al-Buhtiri, and more.

Run:  python data/prepare.py
"""

import re
import unicodedata
from pathlib import Path
from datasets import load_dataset


# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_FILE  = Path("data/poetry_corpus.txt")
MAX_POEMS    = 5000      # keep manageable for fine-tuning on a single GPU
MIN_LINES    = 2         # skip very short fragments
MAX_LINES    = 30        # skip epics that are too long for block_size


# ── Arabic text cleaning ──────────────────────────────────────────────────────

# Diacritics (tashkeel) — optional: keep them for classical فصحى
TASHKEEL = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670]')

def clean_arabic(text: str, remove_tashkeel: bool = False) -> str:
    """
    Clean Arabic text for language model training.

    Steps:
    1. Normalize Unicode (NFC)
    2. Normalize Arabic letter forms (e.g. أ إ آ → ا)
    3. Optionally remove tashkeel (diacritics)
    4. Remove non-Arabic characters except punctuation & newlines
    5. Collapse multiple spaces/newlines
    """
    # 1. Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Normalize Alef variants → bare Alef (ا)
    text = re.sub(r'[أإآ]', 'ا', text)
    # Normalize Teh Marbuta → Heh
    text = re.sub(r'ة', 'ه', text)
    # Normalize Yeh variants
    text = re.sub(r'[ىئ]', 'ي', text)

    # 3. Remove tashkeel if requested
    if remove_tashkeel:
        text = TASHKEEL.sub('', text)

    # 4. Keep Arabic letters, Arabic punctuation, spaces, newlines
    text = re.sub(r'[^\u0600-\u06FF\s\n،؟!.]', ' ', text)

    # 5. Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ── Load & process ────────────────────────────────────────────────────────────

def load_poetry_dataset() -> list[str]:
    """
    Load arbml/ashaar from HuggingFace.
    Falls back to a small hardcoded sample if offline.
    """
    print("Loading arbml/ashaar dataset from HuggingFace...")
    try:
        ds = load_dataset("arbml/ashaar", split="train", trust_remote_code=True)
        print(f"  Loaded {len(ds):,} poems")
        return [row["poem"] for row in ds if row.get("poem")]
    except Exception as e:
        print(f"  Could not load dataset: {e}")
        print("  Using built-in sample poems instead.")
        return SAMPLE_POEMS


def process_poems(raw_poems: list[str]) -> list[str]:
    """Clean and filter poems."""
    processed = []
    for poem in raw_poems:
        cleaned = clean_arabic(poem, remove_tashkeel=False)
        lines = [l.strip() for l in cleaned.split('\n') if l.strip()]
        if MIN_LINES <= len(lines) <= MAX_LINES:
            processed.append('\n'.join(lines))
        if len(processed) >= MAX_POEMS:
            break
    return processed


def save_corpus(poems: list[str], path: Path):
    """Save poems to a plain-text file, separated by blank lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for poem in poems:
            f.write(poem + '\n\n')
    print(f"  Saved {len(poems):,} poems → {path}")
    print(f"  File size: {path.stat().st_size / 1024:.1f} KB")


# ── Sample poems (offline fallback) ──────────────────────────────────────────
# A selection from Al-Mutanabbi and Ibn Arabi for testing without internet

SAMPLE_POEMS = [
    """على قدر أهل العزم تأتي العزائم
وتأتي على قدر الكرام المكارم
وتعظم في عين الصغير صغارها
وتصغر في عين العظيم العظائم""",

    """أنا من أهوى ومن أهوى أنا
نحن روحان حللنا بدنا
فإذا أبصرتني أبصرته
وإذا أبصرته أبصرتنا""",

    """إذا غامرت في شرف مروم
فلا تقنع بما دون النجوم
فطعم الموت في أمر حقير
كطعم الموت في أمر عظيم""",

    """الخيل والليل والبيداء تعرفني
والسيف والرمح والقرطاس والقلم
صحبت في الفلوات الوحش منفردا
حتى تعجب مني القور والأكم""",

    """أعز مكان في الدنى سرج سابح
وخير جليس في الزمان كتاب
وخير أخ لم تلده أمك إنه
أخو ثقة لا تشتكي منه غياب""",

    """لا أبعد الله عنا من إذا ذكروا
خير الملوك أتوا بالبدر والقمر
عجبت للناس إذ لا يشكرون له
وكيف يكفر بالشمس من هو مبصر""",

    """يا من يرى عمر تكنيفه كلفا
أحمل عليك بما لا تستطيع حملا
كأنك لم تسمع بليلى ولا عمرا
ولم تعش عيشة من أحب وأكملا""",

    """وما الدهر إلا من رواة قصائدي
إذا قلت شعرا أصبح الدهر منشدا
فسار به من لا يسير مشمرا
وغنى به من لا يغني مغردا""",
]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw   = load_poetry_dataset()
    poems = process_poems(raw)

    print(f"\nDataset stats:")
    print(f"  Raw poems loaded  : {len(raw):,}")
    print(f"  After filtering   : {len(poems):,}")
    avg_lines = sum(len(p.split('\n')) for p in poems) / max(len(poems), 1)
    print(f"  Avg lines/poem    : {avg_lines:.1f}")

    save_corpus(poems, OUTPUT_FILE)
    print("\nSample poem:")
    print("─" * 40)
    print(poems[0] if poems else "(none)")
