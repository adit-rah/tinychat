import os

import numpy as np
import pytest

from tinychat.data import batch_iterator, build_token_memmap
from tinychat.tokenizer import load_tokenizer, train_tokenizer

CORPUS = [
    "Once upon a time there was a little cat named Tom.",
    "Tom liked to play in the garden every day with his friend Lily.",
    "One day they found a shiny red ball under the big oak tree.",
] * 100


def _make_tokenizer(tmp_path):
    p = str(tmp_path / "tok.json")
    train_tokenizer(CORPUS, save_path=p, vocab_size=512)
    return load_tokenizer(p)


def test_memmap_dtype_and_range(tmp_path):
    tok = _make_tokenizer(tmp_path)
    out = str(tmp_path / "data.bin")
    n = build_token_memmap(CORPUS, tok, out)
    arr = np.memmap(out, dtype=np.uint16, mode="r")
    assert arr.dtype == np.uint16
    assert arr.shape[0] == n
    assert int(arr.max()) < 512  # all token ids fit the vocab


def test_interrupted_build_leaves_no_output_file(tmp_path):
    # A build killed mid-corpus must not leave a truncated file at out_path — callers use
    # the file's existence as the "already built" check.
    tok = _make_tokenizer(tmp_path)
    out = str(tmp_path / "data.bin")

    def texts_then_crash():
        yield CORPUS[0]
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        build_token_memmap(texts_then_crash(), tok, out)
    assert not os.path.exists(out)


def test_batch_iterator_deterministic(tmp_path):
    tok = _make_tokenizer(tmp_path)
    out = str(tmp_path / "data.bin")
    build_token_memmap(CORPUS, tok, out)
    it1 = batch_iterator(out, ctx=8, tokens_per_step=32, seed=0)
    it2 = batch_iterator(out, ctx=8, tokens_per_step=32, seed=0)
    x1, y1 = next(it1)
    x2, y2 = next(it2)
    assert (x1 == x2).all() and (y1 == y2).all()


def test_batch_shapes_and_shift(tmp_path):
    tok = _make_tokenizer(tmp_path)
    out = str(tmp_path / "data.bin")
    build_token_memmap(CORPUS, tok, out)
    it = batch_iterator(out, ctx=8, tokens_per_step=32, seed=1)
    x, y = next(it)
    assert x.shape == (4, 8) and y.shape == (4, 8)
    # y is x shifted by one position
    assert (y[:, :-1] == x[:, 1:]).all()


def test_start_step_matches_advanced_iterator(tmp_path):
    # Resuming at start_step=3 yields what the 4th batch of a fresh run would be.
    tok = _make_tokenizer(tmp_path)
    out = str(tmp_path / "data.bin")
    build_token_memmap(CORPUS, tok, out)
    it = batch_iterator(out, ctx=8, tokens_per_step=32, seed=0)
    for _ in range(3):
        next(it)
    x_fresh, _ = next(it)  # the 4th batch (step index 3)
    x_resume, _ = next(batch_iterator(out, ctx=8, tokens_per_step=32, seed=0, start_step=3))
    assert (x_fresh == x_resume).all()


def test_different_seed_differs(tmp_path):
    tok = _make_tokenizer(tmp_path)
    out = str(tmp_path / "data.bin")
    build_token_memmap(CORPUS, tok, out)
    x0, _ = next(batch_iterator(out, ctx=8, tokens_per_step=32, seed=0))
    x1, _ = next(batch_iterator(out, ctx=8, tokens_per_step=32, seed=7))
    assert not (x0 == x1).all()
