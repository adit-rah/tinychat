"""Pure helpers behind the reference ladder (scripts/run_calibration.py)."""

import importlib.util
import os

_SPEC = importlib.util.spec_from_file_location(
    "run_calibration",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "run_calibration.py"))
cal = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cal)


def test_reference_line_greedy_and_sampled():
    line = cal.reference_line("roneneldan/TinyStories-1M", False, 200, 3.1234, 0.0567)
    assert line == ("- reference (TinyStories-1M, greedy, n=200) mean: 3.123  "
                    "(95% CI ±0.057)  (addendum)")
    line = cal.reference_line("roneneldan/TinyStories-33M", True, 50, 3.687, 0.11)
    assert "sampled temp1.0/topk40" in line and "n=50" in line


def test_score_set_detailed_keeps_axes_and_writes_jsonl(tmp_path):
    import json

    class FakeJudge:
        def score(self, prefix, completion):
            return {"grammar": 5, "consistency": 3, "completes": 1,
                    "parsed": True, "raw": "{}"}

    prefixes = [{"id": i, "prefix": "Once", "gold_continuation": " upon"}
                for i in range(3)]
    detail = str(tmp_path / "gold.axes.jsonl")
    rows = cal.score_set_detailed(FakeJudge(), prefixes, lambda it: "x",
                                  detail_path=detail)
    assert [r["grammar"] for r in rows] == [5, 5, 5]
    assert rows[0]["score"] == 3.0  # mean of 5/3/1
    on_disk = [json.loads(l) for l in open(detail)]
    assert on_disk == rows


def test_round_robin_split_partitions_the_ladder():
    from nanofable.kaggle import LADDER

    for workers in (1, 2, 3):
        shards = [LADDER[i::workers] for i in range(workers)]
        flat = [m for s in shards for m in s]
        assert sorted(flat) == sorted(LADDER)  # every model scored exactly once
