"""End-to-end evaluate_run: train a tiny run, judge it with a fake judge, audit eval.json."""

import json
import os

import pytest

import eval.run_eval as run_eval
from nanofable.train import train_run


class RecordingJudge:
    """Scores 4s; flags any empty completion as a parse failure for the failure test."""

    def __init__(self, fail_on: str | None = None):
        self.fail_on = fail_on
        self.seen = []

    def score(self, prefix: str, completion: str) -> dict:
        self.seen.append((prefix, completion))
        if self.fail_on is not None and self.fail_on in completion:
            return {"grammar": 0, "consistency": 0, "completes": 0,
                    "parsed": False, "raw": "I refuse."}
        return {"grammar": 4, "consistency": 4, "completes": 4,
                "parsed": True, "raw": '{"grammar": 4, ...}'}


@pytest.fixture
def trained_run(tmp_path, small_cfg, data_paths, monkeypatch):
    """A 2-step trained run + patched frozen-artifact paths for the small test tokenizer."""
    train_path, val_path = data_paths
    run_dir = str(tmp_path / "tiny_fp16_0")
    train_run(small_cfg, "fp16", seed=0, run_dir=run_dir,
              train_path=train_path, val_path=val_path,
              total_tokens=small_cfg.ctx * 2, tokens_per_step=small_cfg.ctx,
              eval_every=1000, eval_batches=2, ckpt_every=1000)

    prefixes_path = str(tmp_path / "prefixes.jsonl")
    with open(prefixes_path, "w") as f:
        for i in range(3):
            f.write(json.dumps({"id": i, "prefix": "Once upon a time there was",
                                "gold_continuation": " a cat."}) + "\n")
    monkeypatch.setattr(run_eval, "PREFIXES", prefixes_path)
    monkeypatch.setattr(run_eval, "TOKENIZER",
                        os.path.join(os.path.dirname(train_path), "tok.json"))
    return run_dir


def test_evaluate_run_stores_completions_and_raw_judge_output(trained_run):
    judge = RecordingJudge()
    result = run_eval.evaluate_run(trained_run, judge=judge, max_new_tokens=8)

    saved = json.loads(open(os.path.join(trained_run, "eval.json")).read())
    assert saved["n"] == 3 and saved["n_parse_failures"] == 0
    for row in saved["per_prefix"]:
        assert isinstance(row["completion"], str)
        assert row["parsed"] is True
        assert row["raw"] == '{"grammar": 4, ...}'
        assert set(row["axes"]) == {"grammar", "consistency", "completes"}
        assert row["score"] == 4.0
    # The judge was shown exactly the generated completions that were stored.
    assert [c for _, c in judge.seen] == [r["completion"] for r in saved["per_prefix"]]


def test_evaluate_run_counts_parse_failures(trained_run):
    judge = RecordingJudge(fail_on="")  # every completion "fails" to parse
    result = run_eval.evaluate_run(trained_run, judge=judge, max_new_tokens=8)
    assert result["n_parse_failures"] == 3


def test_evaluate_pending_scores_done_runs_and_respects_claims(tmp_path, small_cfg,
                                                               data_paths, monkeypatch):
    from nanofable.sweep import claim_run, run_dir_for

    train_path, val_path = data_paths
    runs_root = str(tmp_path / "runs")
    # Two DONE runs (real tiny checkpoints), one of them already eval-claimed elsewhere.
    for seed in (0, 1):
        rd = run_dir_for(runs_root, "tiny", "fp16", seed)
        train_run(small_cfg, "fp16", seed=seed, run_dir=rd,
                  train_path=train_path, val_path=val_path,
                  total_tokens=small_cfg.ctx * 2, tokens_per_step=small_cfg.ctx,
                  eval_every=1000, eval_batches=2, ckpt_every=1000)
        claim_run(rd)  # leftover *training* claim must not block eval
    claimed = run_dir_for(runs_root, "tiny", "fp16", 1)
    claim_run(claimed, name="EVAL_CLAIM")  # another eval worker owns this one

    prefixes_path = str(tmp_path / "prefixes.jsonl")
    with open(prefixes_path, "w") as f:
        f.write(json.dumps({"id": 0, "prefix": "Once upon a time",
                            "gold_continuation": " a cat."}) + "\n")
    monkeypatch.setattr(run_eval, "PREFIXES", prefixes_path)
    monkeypatch.setattr(run_eval, "TOKENIZER",
                        os.path.join(os.path.dirname(train_path), "tok.json"))

    done = run_eval.evaluate_pending(runs_root, judge=RecordingJudge(), claim=True,
                                     max_new_tokens=8)

    assert [os.path.basename(d) for d in done] == ["tiny_fp16_0"]
    assert os.path.exists(os.path.join(runs_root, "tiny_fp16_0", "eval.json"))
    assert not os.path.exists(os.path.join(claimed, "eval.json"))
    # Idempotent: nothing left for this worker on a second pass.
    assert run_eval.evaluate_pending(runs_root, judge=RecordingJudge(), claim=True,
                                     max_new_tokens=8) == []


def test_evaluate_run_passes_decoding_params_to_generate(trained_run, monkeypatch):
    import nanofable.generate as gen_mod

    seen_kwargs = {}
    real_generate = gen_mod.generate

    def spy(model, tok, prefix, **kwargs):
        seen_kwargs.update(kwargs)
        return real_generate(model, tok, prefix, **kwargs)

    monkeypatch.setattr(run_eval, "_generate", spy)
    run_eval.evaluate_run(trained_run, judge=RecordingJudge(), max_new_tokens=8,
                          temperature=0.001, top_k=0)
    assert seen_kwargs["temperature"] == 0.001 and seen_kwargs["top_k"] == 0
