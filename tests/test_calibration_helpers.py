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


def test_round_robin_split_partitions_the_ladder():
    from tinychat.kaggle import LADDER

    for workers in (1, 2, 3):
        shards = [LADDER[i::workers] for i in range(workers)]
        flat = [m for s in shards for m in s]
        assert sorted(flat) == sorted(LADDER)  # every model scored exactly once
