import os

import nanofable.sweep as sweep

# The conftest `data_paths` fixture provides real (tiny) train/val memmaps for the
# end-to-end parallel test; the fast kwargs keep a real tiny-tier run to a few CPU steps.
FAST_KWARGS = dict(total_tokens=1024, tokens_per_step=512, eval_every=1000,
                   eval_batches=2, ckpt_every=1000)  # ctx is frozen at 512 -> 1 row/step


def test_matrix_is_4x2x2():
    m = sweep.sweep_matrix()
    assert len(m) == 16
    assert len(set(m)) == 16
    tiers = {t for t, _, _ in m}
    assert tiers == {"tiny", "small", "medium", "large"}
    assert {p for _, p, _ in m} == {"fp16", "ternary"}
    assert {s for _, _, s in m} == {0, 1}


def test_run_sweep_skips_done(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sweep, "train_run",
                        lambda cfg, prec, seed, run_dir, *a, **k: calls.append(run_dir))

    # Pre-mark one combo as DONE; it must not be trained.
    done_dir = sweep.run_dir_for(str(tmp_path), "tiny", "fp16", 0)
    os.makedirs(done_dir)
    open(os.path.join(done_dir, "DONE"), "w").close()

    sweep.run_sweep(str(tmp_path), "train.bin", "val.bin")

    assert done_dir not in calls
    assert len(calls) == 15  # 16 - 1 skipped


def test_run_sweep_only_subset(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sweep, "train_run",
                        lambda cfg, prec, seed, run_dir, *a, **k: calls.append(run_dir))
    sweep.run_sweep(str(tmp_path), "t.bin", "v.bin", only=[("small", "ternary", 1)])
    assert len(calls) == 1
    assert calls[0].endswith("small_ternary_1")


def test_run_sweep_calls_eval_after_train(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "train_run", lambda *a, **k: None)
    evaled = []
    sweep.run_sweep(str(tmp_path), "t.bin", "v.bin",
                    only=[("tiny", "fp16", 0)], eval_fn=evaled.append)
    assert len(evaled) == 1 and evaled[0].endswith("tiny_fp16_0")


def test_claim_run_is_exclusive(tmp_path):
    run_dir = str(tmp_path / "tiny_fp16_0")
    assert sweep.claim_run(run_dir) is True
    assert sweep.claim_run(run_dir) is False  # second claimant loses


def test_claim_run_markers_are_independent(tmp_path):
    # Training leaves CLAIM files behind; eval claims must not collide with them.
    run_dir = str(tmp_path / "tiny_fp16_0")
    assert sweep.claim_run(run_dir) is True
    assert sweep.claim_run(run_dir, name="EVAL_CLAIM") is True
    assert sweep.claim_run(run_dir, name="EVAL_CLAIM") is False


def test_run_sweep_with_claim_skips_claimed_runs(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sweep, "train_run",
                        lambda cfg, prec, seed, run_dir, *a, **k: calls.append(run_dir))

    claimed = sweep.run_dir_for(str(tmp_path), "tiny", "fp16", 0)
    sweep.claim_run(claimed)  # another worker holds this run

    sweep.run_sweep(str(tmp_path), "t.bin", "v.bin", claim=True)
    assert claimed not in calls
    assert len(calls) == 15


def test_run_sweep_parallel_trains_all_and_clears_stale_claims(tmp_path, data_paths):
    train_path, val_path = data_paths
    runs_root = str(tmp_path / "runs")
    only = [("tiny", "fp16", 0), ("tiny", "fp16", 1)]

    # A CLAIM left behind by a killed session must not block the run.
    stale = sweep.run_dir_for(runs_root, "tiny", "fp16", 0)
    sweep.claim_run(stale)

    sweep.run_sweep_parallel(runs_root, train_path, val_path, only=only,
                             workers=2, poll_s=1, **FAST_KWARGS)

    for tier, precision, seed in only:
        run_dir = sweep.run_dir_for(runs_root, tier, precision, seed)
        assert os.path.exists(os.path.join(run_dir, "DONE")), run_dir
    assert os.path.exists(os.path.join(runs_root, "worker0.log"))
    assert os.path.exists(os.path.join(runs_root, "worker1.log"))


def test_run_sweep_parallel_raises_on_worker_failure(tmp_path, data_paths):
    import pytest

    # Nonexistent data paths make both workers crash; the parent must surface it.
    with pytest.raises(RuntimeError, match="worker"):
        sweep.run_sweep_parallel(str(tmp_path / "runs"), "missing.bin", "missing.bin",
                                 only=[("tiny", "fp16", 0)], workers=2, poll_s=1,
                                 **FAST_KWARGS)
