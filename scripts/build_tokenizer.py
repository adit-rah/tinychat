"""Train the frozen 4k BPE tokenizer on the TinyStories train split.

Run once; commit the resulting artifacts/tokenizer/tokenizer.json. CPU-only.

    python scripts/build_tokenizer.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nanofable.tokenizer import train_tokenizer  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "tokenizer")
OUT_PATH = os.path.join(OUT_DIR, "tokenizer.json")
VOCAB_SIZE = 4096


def text_iter():
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split="train")
    for row in ds:
        yield row["text"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Training BPE (vocab={VOCAB_SIZE}) on TinyStories train split...")
    train_tokenizer(text_iter(), save_path=OUT_PATH, vocab_size=VOCAB_SIZE)
    print(f"Saved tokenizer to {OUT_PATH}")


if __name__ == "__main__":
    main()
