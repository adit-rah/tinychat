import torch

from nanofable.generate import generate, next_token
from nanofable.model import build_model
from nanofable.tokenizer import load_tokenizer, train_tokenizer

CORPUS = ["Once upon a time there was a little cat who liked to play."] * 100


def _tok(tmp_path):
    p = str(tmp_path / "tok.json")
    train_tokenizer(CORPUS, save_path=p, vocab_size=512)
    return load_tokenizer(p)


def test_generate_returns_decodable_string(small_cfg, tmp_path):
    tok = _tok(tmp_path)
    model = build_model(small_cfg, "fp16")
    out = generate(model, tok, "Once upon a time", max_new_tokens=10, seed=0)
    assert isinstance(out, str)
    # continuation is at most max_new_tokens tokens
    assert len(tok.encode(out).ids) <= 10


def test_generate_deterministic_under_seed(small_cfg, tmp_path):
    tok = _tok(tmp_path)
    model = build_model(small_cfg, "fp16")
    a = generate(model, tok, "Once upon a time", max_new_tokens=15, seed=42)
    b = generate(model, tok, "Once upon a time", max_new_tokens=15, seed=42)
    assert a == b


def test_next_token_returns_valid_id(small_cfg, tmp_path):
    tok = _tok(tmp_path)
    model = build_model(small_cfg, "fp16")
    ids = tok.encode("Once upon a time").ids
    nxt = next_token(model, ids)
    assert isinstance(nxt, int)
    assert 0 <= nxt < small_cfg.vocab


def test_next_token_matches_generate_first_step(small_cfg, tmp_path):
    # Stepping manually with next_token must reproduce generate() exactly —
    # same sampling path, same generator stream.
    tok = _tok(tmp_path)
    model = build_model(small_cfg, "fp16")
    full = generate(model, tok, "Once upon a time", max_new_tokens=15, seed=7)

    gen = torch.Generator(device="cpu").manual_seed(7)
    ids = tok.encode("Once upon a time").ids
    out = []
    for _ in range(15):
        nxt = next_token(model, ids + out, generator=gen)
        if nxt == 1:  # EOS_ID
            break
        out.append(nxt)
    assert tok.decode(out) == full
