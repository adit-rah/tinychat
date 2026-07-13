# nanofable

How small can an English language model get, measured in **bytes** and still write a coherent story? And do ternary (1.58-bit) weights push that floor down, or does the quality hit at tiny scale eat the savings?

This repo is a harness that tries to answer that. It trains 4 model sizes × 2 precisions (fp16 vs ternary) × 2 seeds on TinyStories, holds everything else fixed, and scores all of them through a frozen LLM judge against a capability gate that was written down *before* any of it ran.

The short version of what happened: **nothing cleared the bar.** More on that below, because the null is the interesting part.

## why bytes

Everyone reports parameters. Parameters aren't the most accurate proxy for "how big is this thing, actually" the moment you start changing precision. A 27M-param fp16 model and a 27M-param ternary model are the same size on paper and 6× apart on disk after packing. 

In addition, BitNet b1.58 showed ternary reaching fp16 parity — but only at ≥3B params, the opposite end of the scale. TernaryLM trained a native ternary model on TinyStories and got ~2× worse PPL than its fp32 baseline (58.4 vs ~28). One size, PPL only, no judge, no byte accounting.

Nobody's drawn the frontier: coherence vs total bytes, both precisions, across a sweep. That's the contribution, and it's why the x-axis is bytes.

## the vocab sacrifice

**We use a small custom vocab.** Not GPT-2's 50k.

Embeddings don't get ternarized; they stay fp16 in both arms. So at tiny scale, with a 50k vocab, your model ends up mostly being embedding table, ternarizing the blocks saves you almost nothing, and the headline plot collapses into two lines on top of each other. A "tiny" model with GPT-2's vocab would be ~6.4M embedding params before a single layer of actual compute.

So we train a 4,096-token BPE tokenizer on TinyStories. That keeps the blocks dominant and lets the ternary savings actually show up:

| tier   | params | fp16     | ternary | embed+head | ternary/fp16 |
|--------|-------:|---------:|--------:|-----------:|-------------:|
| tiny   | 1.38 M |  2.75 MB | 1.22 MB |    1.05 MB |        0.442 |
| small  | 5.87 M | 11.73 MB | 3.05 MB |    2.10 MB |        0.260 |
| medium | 15.7 M | 31.46 MB | 5.94 MB |    3.15 MB |        0.189 |
| large  | 27.8 M | 55.57 MB | 9.27 MB |    4.19 MB |        0.167 |

Two things fall out of this table. At the bottom, **vocab is the floor-setter** — the tiny ternary model is 86% embedding table by bytes. And **ternary's compression ratio improves with size** (0.44 → 0.17), because only the body ternarizes. So ternary bends the byte curve hardest exactly where the models are biggest, which is not where you want it if you're hunting for a floor.

The byte accounting itself is one tested function, `count_bytes()` in `src/nanofable/bytes.py`, pinned to the spec's formula literally: block bytes at 1.58/8 per weight, embeddings + tied head at fp16, per-layer scales counted honestly. The whole result rides on it being right, so it has its own test file. 

## the gate, frozen before anything ran

"Smallest capable model" is unfalsifiable unless you say what *capable* means before you look. So, the rubric became a frozen artifact, with the 200 held-out story prefixes, the judge model (`Qwen2.5-7B-Instruct`), and the judge prompt defined.

We set the bar of "capabilitt" at iff **both**:

- **Coherence:** mean judge score ≥ **4.0 / 5**, averaged over grammar / consistency / completes-sensibly, over the fixed 200 prefixes.
- **Perplexity:** val PPL ≤ **1.5 × best fp16 val PPL** in the sweep. (The *rule* is frozen; the number resolves once the fp16 runs land. That's pre-registration — you fix the policy, not the answer.)

And a tier-separation claim only counts if the 95% CI doesn't straddle the threshold. Otherwise the tiers are reported as indistinguishable, not as a crossing.

Everything else is held rigid across all 16 runs: same architecture (pre-norm RMSNorm, RoPE, SwiGLU, tied embeddings), same ctx=512, same 500M-token budget, same AdamW at peak LR 3e-4 with cosine decay, same 65,536 tokens per optimizer step. Only **size** and **precision** move. If a ternary run collapses, that's a finding, not a reason to retune one arm — retuning ternary's LR without retuning fp16's would quietly break the whole comparison.

## what happened: the gate is a null

All eight configs fail. Not "ternary fails and fp16 squeaks by" — **all of them**, both arms, every tier.

The best model in the sweep is `large_fp16`, and it tops out at a grammar-axis mean of **3.83 ±0.06**. To put that in context, we scored the published TinyStories checkpoints through the exact same judge, greedy, same 200 prefixes:

| model            | judge mean | grammar / consistency / completes |
|------------------|-----------:|-----------------------------------|
| TinyStories-1M   |      2.423 | 3.31 / 2.46 / 1.50                |
| TinyStories-3M   |      3.330 | 3.92 / 3.50 / 2.58                |
| TinyStories-8M   |      4.232 | **4.55** / 4.42 / 3.73            |
| TinyStories-28M  |      4.447 | 4.67 / 4.58 / 4.08                |
| TinyStories-33M  |      4.378 | 4.64 / 4.54 / 3.95                |
| gold (real text) |          — | 4.74 / 4.68 / 4.29                |

Our 27.8M-param best is statistically indistinguishable from *TinyStories-3M*. The smallest published checkpoint that clears 4.0 is the 8M, so that's the external anchor, and we're well under it. The PPL gate resolved to T = 8.089 (from `large_fp16` seed 1's val PPL of 5.393); `medium_fp16` and `large_fp16` clear it and still fail coherence. Which is the gate working as designed: **perplexity is necessary but nowhere near sufficient** for judged coherence.

So the headline question: where does the ternary frontier cross the fp16 frontier, doesn't get a concrete answer, because neither curve reaches the capable region. 

### what *did* show up

Two things survive the null, and both are about structure rather than thresholds.

**The axes separate, universally.** grammar > consistency > completes-sensibly. For every one of the 8 sweep configs, every one of the 5 published references, and the gold human text. Models learn to produce well-formed sentences long before they learn to hold a story's state across those sentences, and they finish a story sensibly last of all. The ceiling isn't flat, with even real TinyStories models' text scoring 4.74 grammar and 4.29 completion.

**Ternary taxes state harder than syntax.** At the large tier, going ternary costs −0.85 on grammar but −1.43 and −1.44 on consistency and completes — roughly **1.7× the penalty on the state-holding axes**. Ternary weights don't degrade a model uniformly; they take the structured, longer-range stuff first. That's a more useful thing to know than a crossing point, and I don't think it's been reported at this scale.

### why the sweep undershot

The obvious suspects are the 500M-token budget, the 4k vocab making the task harder per-token than the published models' 50k, and the shared 3e-4 LR being right for one arm and not the other. Untangling that is the next thing!

## the instrument bug (worth reading if you build eval harnesses)

Midway through, the sampled judge scores came back absurdly low, and the audit found the reason: **decoding policy was never in the freeze list.**

The calibration references were scored decoding-free (gold text, synthetic garbage) or greedy. The sweep models were scored on temperature-1.0 / top-k-40 samples. Scoring the known-good TinyStories-33M under the *sampled* policy gives 3.687 — below the 4.0 bar. So under sampled decoding the gate sat above what a known-good 33M model achieves, and had no discriminative power at all. The gate was never valid, and had the sweep happened to clear it, that result would have been garbage.

Fixed by freezing **greedy decoding, uniformly, for every model and every reference** — the only policy the original calibration actually validated. Set blind, before any greedy sweep-model score had been computed, which is the only thing that makes it an instrument correction rather than tuning the gate until you like the answer. The sampled scores are retained and reported alongside.

Two smaller ones from the same audit: judge parse failures were scoring 0 indistinguishably from a genuine verdict (now flagged and counted separately), and the judge's 32-token budget could truncate its own JSON mid-verdict (now 64). Also, the "intra-judge stability" check was vacuous, the judge decodes greedily, so re-scoring the same text gives std 0.0 by construction. It proves nothing. The n=200 CI is the real noise estimate.

## running it in a browser

The models ship as `.tpack` files (magic `TCP1`) — a self-describing format where ternary weights are actually packed at **5 trits/byte**, which is the only way the byte win is real. Torch-native formats have no sub-byte dtype, so a "ternary" safetensors file stores each trit in a full fp16 slot and is exactly as big as the fp16 model. The compression only exists if you pack it:

| tier   | `model.safetensors` (either arm) | ternary `.tpack` |
|--------|---------------------------------:|-----------------:|
| tiny   |                      2,758,256 B |      1,225,952 B |
| medium |                     31,477,256 B |      5,999,152 B |

Same bits, different storage. The HF release ships both — safetensors for standard tooling, `.tpack` as the honest small artifact.

`web/` is a static site with no server and no build step: it loads a `.tpack`, unpacks the trits once, and runs the ternary kernel client-side. You type a story opening and it continues it, streaming, in your browser. It's a **story completer, not a chatbot** — it was trained on TinyStories and it will not answer your questions.

```sh
python scripts/pack_model.py --ckpt local/tiny_ternary_0.pt --tier tiny \
    --precision ternary --out web/models/tiny_ternary.tpack
python3 -m http.server 8000 -d web    # ES modules + Cache API need http, not file://
```

Python and JS produce **bit-identical greedy output** — verified token-for-token, which is the only reason I trust the JS kernel. That required rounding the absmean scales to fp16 *before* writing them, so both sides dequantize from the same values.

A fun one: the models generate **mojibake**, and it's the dataset's fault. TinyStories stores curly punctuation CP1252-double-encoded (a `“` is literally the three characters `â€œ`), those sequences earned their own BPE merges, and so the model correctly learned to emit them. `web/js/display.js` maps them back at render time only — the tokenizer, dataset, and token ids are untouched, because cleaning the corpus would invalidate every completed run and the judge sees the same mojibake for every config anyway.

## setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest        # 106 tests, offline, no GPU
```

## the pipeline

| # | step | command | output |
|---|------|---------|--------|
| 1 | tokenizer *(frozen)* | `scripts/build_tokenizer.py` | `artifacts/tokenizer/tokenizer.json` |
| 2 | data | `scripts/build_dataset.py` | `artifacts/data/{train,val}.bin` |
| 3 | prefixes *(frozen)* | `scripts/make_prefixes.py` | `eval/prefixes.jsonl` (200) |
| 4 | calibration *(frozen)* | `scripts/run_calibration.py` | `eval/calibration.md` |
| 5 | sweep | `scripts/run_sweep.py` | `runs/<tier>_<prec>_<seed>/` |
| 6 | plot | `scripts/plot_frontier.py` | `docs/frontier.png` |

Steps 1–3 are one-time artifacts committed *before* the sweep — that's what makes "frozen" enforceable by version control rather than by my good intentions. Steps 4–5 want a GPU; the whole 16-run sweep is about 10–20 GPU-hours and fits in a week of Kaggle's free quota. Sessions hard-cap at 12h, so every run checkpoints and the sweep is idempotent — re-run it and finished runs are skipped.

## the compute caveat

Ternary is smaller to ship and *more* expensive to train — you carry a full-precision latent weight and quantize it on every forward pass. "Smaller in bytes" and "cheaper to train" point in opposite directions, and it would be dishonest to report the first without the second. Training compute is logged per run (FLOPs ≈ 6·N·T, plus wall-clock).

## layout

- `src/nanofable/` — model (`model.py`, `bitlinear.py`, `rope.py`), byte accounting (`bytes.py`), data, training, sweep, `.tpack` packing.
- `eval/` — the frozen rubric, judge prompt, and 200 prefixes; judge backends, capability gate, eval runner.
- `specs/` — the frozen spec. Source of truth.
- `docs/` — why the frozen config is what it is, byte accounting, the emergence-floor estimate, related work.
- `web/` — the static browser demo.
- `tests/` — mirrors `src/` and `eval/`. Offline, no GPU.

`runs/` and `artifacts/data/` are generated and gitignored. Results are never hand-edited — runs log to CSV and the plotting script reads the CSVs.

## what's next

The null is a result, but there's more to get. In rough order: A 2-bit arm is a sharper stretch goal than the spec's original 4-bit suggestion, ParetoQ finds 2-bit and ternary roughly tied at large scale, which makes the tiny-scale comparison the interesting one. And the deeper question the whole thing points at: the emergence floor is a property of the *distribution*, not of English. TinyStories engineered a distribution simple enough for ~1M params. Constrain it further and the floor should slide down with it.
