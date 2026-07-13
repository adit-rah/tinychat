"""CLI for the §9 sweep. Idempotent: re-run after a session timeout to resume.

    python scripts/run_sweep.py            # full 16-run matrix, train + judge eval
    python scripts/run_sweep.py --no-eval  # train only (judge later)

Expects artifacts/data/{train,val}.bin (build_dataset.py) and, for eval, the committed
tokenizer + eval/prefixes.jsonl + a GPU judge.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nanofable.sweep import run_sweep  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "artifacts", "data")
RUNS = os.path.join(ROOT, "runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-eval", action="store_true", help="skip judge eval")
    args = ap.parse_args()

    eval_fn = None
    if not args.no_eval:
        from eval.run_eval import evaluate_run

        eval_fn = lambda run_dir: evaluate_run(run_dir)  # noqa: E731

    trained = run_sweep(
        RUNS,
        os.path.join(DATA, "train.bin"),
        os.path.join(DATA, "val.bin"),
        eval_fn=eval_fn,
    )
    print(f"Trained {len(trained)} run(s) this invocation.")


if __name__ == "__main__":
    main()
