import json
import os

from nanofable.plotting import collect_results, plot_frontier


def _make_run(runs_dir, name, total_bytes, val_ppl, judge_mean):
    d = os.path.join(runs_dir, name)
    os.makedirs(d)
    json.dump({"total_bytes": total_bytes, "precision": name.split("_")[1]},
              open(os.path.join(d, "meta.json"), "w"))
    with open(os.path.join(d, "metrics.csv"), "w") as f:
        f.write("step,val_ppl\n100," + str(val_ppl) + "\n")
    json.dump({"mean": judge_mean, "ci_low": judge_mean - 0.1, "ci_high": judge_mean + 0.1},
              open(os.path.join(d, "eval.json"), "w"))


def test_collect_results_reads_runs(tmp_path):
    runs = str(tmp_path)
    _make_run(runs, "tiny_fp16_0", 2_750_000, 12.0, 3.2)
    _make_run(runs, "tiny_ternary_0", 1_220_000, 18.0, 2.9)
    rows = collect_results(runs)
    assert len(rows) == 2
    by = {(r["tier"], r["precision"]): r for r in rows}
    assert by[("tiny", "fp16")]["total_bytes"] == 2_750_000
    assert by[("tiny", "ternary")]["judge_mean"] == 2.9
    assert by[("tiny", "fp16")]["val_ppl"] == 12.0


def test_plot_writes_nonempty_png(tmp_path):
    runs = str(tmp_path / "runs")
    os.makedirs(runs)
    # one capable point (judge >= 4.0, ppl within gate) and one not
    _make_run(runs, "small_fp16_0", 11_730_000, 10.0, 4.3)
    _make_run(runs, "tiny_ternary_0", 1_220_000, 13.0, 3.5)
    out = str(tmp_path / "frontier.png")
    plot_frontier(collect_results(runs), out)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_plot_headline_picks_smallest_capable(tmp_path):
    runs = str(tmp_path / "runs")
    os.makedirs(runs)
    # both capable; ternary is smaller in bytes -> it should be the headline
    _make_run(runs, "small_fp16_0", 11_730_000, 10.0, 4.5)
    _make_run(runs, "small_ternary_0", 3_050_000, 12.0, 4.2)  # 12 <= 1.5*10
    headline = plot_frontier(collect_results(runs), str(tmp_path / "f.png"))
    assert headline["precision"] == "ternary" and headline["tier"] == "small"
