import os

import torch

from nanofable.config import ModelConfig
from nanofable.model import build_model
from nanofable.train import _make_scheduler, load_latest, save_checkpoint


def _make_state():
    cfg = ModelConfig(n_layer=2, n_embd=64, n_head=4, ctx=32, vocab=512)
    model = build_model(cfg, "fp16")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = _make_scheduler(opt, total_steps=100)
    return model, opt, sched


def test_save_then_load_restores_state(tmp_path):
    model, opt, sched = _make_state()
    save_checkpoint(str(tmp_path), step=7, tokens_seen=700, model=model, opt=opt, sched=sched)
    ckpt = load_latest(str(tmp_path))
    assert ckpt["step"] == 7 and ckpt["tokens_seen"] == 700
    assert torch.equal(ckpt["model"]["tok_emb.weight"], model.tok_emb.weight)


def test_load_latest_none_when_absent(tmp_path):
    assert load_latest(str(tmp_path)) is None


def test_leftover_tmp_is_ignored(tmp_path):
    model, opt, sched = _make_state()
    save_checkpoint(str(tmp_path), step=3, tokens_seen=300, model=model, opt=opt, sched=sched)
    # Simulate a kill-9 mid-write: a stale .tmp left behind, plus garbage in it.
    with open(os.path.join(tmp_path, "ckpt_latest.pt.tmp"), "wb") as f:
        f.write(b"corrupt partial write")
    ckpt = load_latest(str(tmp_path))  # must read the good ckpt, not the .tmp
    assert ckpt["step"] == 3


def test_periodic_checkpoint_written(tmp_path):
    model, opt, sched = _make_state()
    save_checkpoint(str(tmp_path), step=5, tokens_seen=500, model=model, opt=opt,
                    sched=sched, periodic=True)
    assert os.path.exists(os.path.join(tmp_path, "ckpt_step5.pt"))
    assert os.path.exists(os.path.join(tmp_path, "ckpt_latest.pt"))
