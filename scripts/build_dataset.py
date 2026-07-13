"""Tokenize the TinyStories train + validation splits into uint16 memmaps.

Run after build_tokenizer.py. Outputs go under artifacts/data/ (gitignored — large).

    python scripts/build_dataset.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nanofable.data import build_token_memmap  # noqa: E402
from nanofable.tokenizer import load_tokenizer  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
TOK_PATH = os.path.join(ROOT, "artifacts", "tokenizer", "tokenizer.json")
DATA_DIR = os.path.join(ROOT, "artifacts", "data")


def main():
    from datasets import load_dataset

    os.makedirs(DATA_DIR, exist_ok=True)
    tok = load_tokenizer(TOK_PATH)
    for split, fname in [("train", "train.bin"), ("validation", "val.bin")]:
        ds = load_dataset("roneneldan/TinyStories", split=split)
        out = os.path.join(DATA_DIR, fname)
        n = build_token_memmap((row["text"] for row in ds), tok, out)
        print(f"{split}: wrote {n:,} tokens -> {out}")


if __name__ == "__main__":
    main()
