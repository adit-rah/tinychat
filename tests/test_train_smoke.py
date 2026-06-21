import csv
import os

from tinychat.train import train_run


def _steps(csv_path):
    with open(csv_path) as f:
        return [int(r["step"]) for r in csv.DictReader(f)]


def _common(small_cfg, data_paths, run_dir, total_tokens):
    train_path, val_path = data_paths
    tps = small_cfg.ctx  # 1 row per step -> fast
    train_run(
        small_cfg, "fp16", seed=0, run_dir=str(run_dir),
        train_path=train_path, val_path=val_path,
        total_tokens=total_tokens, tokens_per_step=tps,
        eval_every=1, eval_batches=2, ckpt_every=100,
    )


def test_smoke_train_writes_csv_and_ckpt(tmp_path, small_cfg, data_paths):
    _common(small_cfg, data_paths, tmp_path, total_tokens=small_cfg.ctx * 3)
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "ckpt_latest.pt").exists()
    assert (tmp_path / "meta.json").exists()


def test_completion_marker_written(tmp_path, small_cfg, data_paths):
    _common(small_cfg, data_paths, tmp_path, total_tokens=small_cfg.ctx * 2)
    assert (tmp_path / "DONE").exists()


def test_resume_continues_from_last_step(tmp_path, small_cfg, data_paths):
    _common(small_cfg, data_paths, tmp_path, total_tokens=small_cfg.ctx * 2)
    steps1 = _steps(tmp_path / "metrics.csv")
    assert max(steps1) == 2

    # Resume to 4 steps: must not restart at 0; steps strictly increasing.
    _common(small_cfg, data_paths, tmp_path, total_tokens=small_cfg.ctx * 4)
    steps2 = _steps(tmp_path / "metrics.csv")
    assert max(steps2) == 4
    assert steps2 == sorted(steps2)
    assert steps2.count(0) == 0
    assert steps2[: len(steps1)] == steps1  # earlier rows preserved


def test_resume_drops_csv_rows_ahead_of_checkpoint(tmp_path, small_cfg, data_paths):
    # Run to 2 steps, then simulate a kill where the CSV got ahead of the checkpoint by
    # appending a bogus future row. On resume it must be dropped (no duplicate/out-of-order).
    _common(small_cfg, data_paths, tmp_path, total_tokens=small_cfg.ctx * 2)
    with open(tmp_path / "metrics.csv", "a") as f:
        f.write("99,99,0,0,1,0,0,0,0\n")  # a step far ahead of the checkpoint (step 2)
    assert 99 in _steps(tmp_path / "metrics.csv")

    _common(small_cfg, data_paths, tmp_path, total_tokens=small_cfg.ctx * 4)
    steps = _steps(tmp_path / "metrics.csv")
    assert 99 not in steps
    assert steps == [1, 2, 3, 4]


def test_meta_json_has_total_bytes(tmp_path, small_cfg, data_paths):
    import json
    _common(small_cfg, data_paths, tmp_path, total_tokens=small_cfg.ctx * 2)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["total_bytes"] > 0 and meta["precision"] == "fp16"
