import numpy as np
import pytest

from nanofable.data import build_token_memmap
from nanofable.tokenizer import load_tokenizer, train_tokenizer

_CORPUS = [
    "Once upon a time there was a little cat named Tom who loved the sun.",
    "Tom liked to play in the garden every day with his good friend Lily.",
    "One day they found a shiny red ball under the big old oak tree nearby.",
    "They laughed and played until the sky turned orange in the evening.",
] * 200


@pytest.fixture
def data_paths(tmp_path):
    """Build a small tokenizer + train/val memmaps; return (train_path, val_path)."""
    tok_path = str(tmp_path / "tok.json")
    train_tokenizer(_CORPUS, save_path=tok_path, vocab_size=512)
    tok = load_tokenizer(tok_path)
    train_path = str(tmp_path / "train.bin")
    val_path = str(tmp_path / "val.bin")
    build_token_memmap(_CORPUS, tok, train_path)
    build_token_memmap(_CORPUS[: len(_CORPUS) // 2], tok, val_path)
    return train_path, val_path


@pytest.fixture
def small_cfg():
    # A genuinely small config so CPU smoke tests are fast (ctx well below the real 512).
    from nanofable.config import ModelConfig

    return ModelConfig(n_layer=2, n_embd=64, n_head=4, ctx=32, vocab=512)
