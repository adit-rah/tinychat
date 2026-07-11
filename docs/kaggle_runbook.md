# Kaggle runbook (GPU execution)

The notebook [`notebooks/tinychat-training.ipynb`](../notebooks/tinychat-training.ipynb) runs the GPU-side
of the study. This doc is the short companion: what runs, in what order, and the gotchas.

## What runs (in order)
1. **Build data** — tokenize TinyStories → `train.bin`/`val.bin` memmaps (CPU, minutes).
2. **Calibration (once, before the sweep)** — score good / mediocre / degenerate references
   through the frozen rubric + Qwen2.5-7B judge → `eval/calibration.md`. **Inspect it.**
3. **Sweep (Phase A)** — train the 16 runs (4 tiers × fp16/ternary × 2 seeds) to 500M tokens.
4. **Eval (Phase B)** — load the judge once, score the 200 frozen prefixes per run → `eval.json`.
5. **Plot** — `frontier.png` (coherence vs total bytes) + the smallest-capable headline.
6. **Writeup** — fill `docs/writeup.md`.

## Kaggle settings (right-hand panel)
- **Accelerator:** P100 (single 16GB) or T4. The harness uses one GPU.
- **Persistence:** *Files only* — keeps `/kaggle/working/runs` across sessions so the sweep
  resumes instead of restarting.
- **Internet:** On (pip + TinyStories + Qwen judge downloads).

## Resume / session budget
- Sessions cap at 12h; quota ~30 GPU-h/week. Full sweep ≈ 10–20 GPU-h (~2 sessions).
- The sweep is **idempotent**: each run checkpoints (`ckpt_latest.pt`) and finished runs carry
  a `DONE` marker that is skipped. If a session times out, just re-run the sweep cell next
  session — it continues from the last checkpoint. No run exceeds ~2h, so none is un-resumable.
- Cap a session by setting `ONLY` in the sweep cell to a subset of the matrix.

## Gotchas
- **Pulling code fixes mid-session requires a kernel restart.** The sync cell updates files
  on disk, but Python's import cache keeps already-imported modules — and a crashed cell's
  traceback pins any loaded model in GPU memory. Restart the session (files in
  `/kaggle/working` survive with persistence on), then re-run the sync cell.
- **Run interactively (editor draft session), NOT "Save & Run All".** A commit/batch run
  (papermill) starts from a clean disk every time — no persisted `runs/`, no HF cache, so it
  re-downloads everything and can't resume — and it aborts the whole notebook on the first
  error. The workflow assumes a live editor session with *Files only* persistence.
- **Disk budget: `/kaggle/working` has a ~19.5GiB quota** (the 57.6GB figure is total scratch
  disk, mostly outside it). It holds `data/` (~1GB) + `runs/` (~2.5GB of `ckpt_latest`
  checkpoints) — comfortable. The ~15GB Qwen judge deliberately lives in the *ephemeral* HF
  cache and re-downloads in sessions that judge (calibration / eval), a few minutes each;
  never point `HF_HOME` into `/kaggle/working` or the sweep will hit the quota.
- **Never `pip install -r requirements.txt` on Kaggle.** `requirements.txt` pins `torch>=2.12`
  for the local dev env; installing it on Kaggle upgrades Kaggle's torch (2.10) and breaks the
  preinstalled `transformers` with `ImportError: cannot import name '_maybe_view_chunk_cat'`.
  `tinychat.kaggle.install_deps()` (called by `bootstrap()`) instead installs only
  `bitsandbytes` and leaves Kaggle's torch/transformers/datasets/tokenizers stack as shipped.
  Our code runs on Kaggle's torch.
- **Calibration is a gate, not a formality.** Before trusting any capability claim, confirm in
  `eval/calibration.md`: good refs clear ~4.0, garbage well below, and **intra-judge std <
  good−bad gap**. If the judge is too noisy, swap/upgrade it (an Anthropic-API pass via
  `AnthropicJudge` for borderline configs is permitted) before sweeping.
- **Frozen artifacts must predate the sweep.** The tokenizer, 200 prefixes, rubric, and judge
  prompt are committed in the repo; do not regenerate them on Kaggle.
- **OOM on the large tier?** Lower `micro_rows` in the sweep cell (smaller grad-accum
  micro-batch). fp16 autocast is already on for GPU runs.
- **Don't hand-edit results.** `metrics.csv` / `eval.json` are produced by the harness; the
  plot reads them.

## Monitoring
The sweep cell prints a banner per run (params, bytes, resume step) and a progress line at
every eval interval (250 steps): train loss, val PPL, tokens/sec, session ETA. Watch the
first lines of each tier for throughput — if the ETA doesn't fit the session, cap it with
`ONLY`. A non-finite train loss (ternary collapse, spec §12) **raises immediately** rather
than training on silently; re-running the sweep retries that run from its last checkpoint.
