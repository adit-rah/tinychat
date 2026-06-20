# Project Spec — How Small Can English Get?

**The bytes-and-compute frontier of ternary vs. full-precision language models at the emergence threshold.**

---

## 1. One-line thesis

Coherent English emerges in tiny language models at a measurable scale; ternary
(1.58-bit) weights change *where that threshold sits when you measure size in bytes
rather than parameters*. This project maps the fp16-vs-ternary coherence frontier
across tiny model sizes and finds whether ternary pushes the bytes-for-emergence
frontier **down** (a real compression win) or whether the small-scale quality penalty
eats the byte savings.

## 2. Why this is worth doing

- **The emergence anchor (TinyStories):** With a vocabulary/domain constrained to simple
  children's stories, grammatical, coherent English emerges in models around the
  ~1–10M parameter range. The lever was the *data*, not raw scale.
- **The ternary penalty is largest exactly here.** BitNet-style parity with full
  precision only appears at the billion-parameter scale. Below ~100M there is a real
  per-parameter quality gap. A recent from-scratch ternary LM ("TernaryLM") on
  TinyStories reported roughly 2× worse validation perplexity than an identical-architecture
  FP32 baseline (~58 vs ~28 PPL) — but framed the gap as ternary acting like an implicit
  regularizer. So the trade-off is real and non-obvious.
- **The gap in the literature = the curve.** Prior work gives *single-point* comparisons
  (one ternary model vs one FP model). Nobody has cleanly drawn the **frontier**: coherence
  vs. *total bytes* across a sweep of tiny sizes, for fp16 and ternary side by side. That
  frontier is the contribution.

## 3. The measurable question

> Plotting a coherence metric against **total model size in bytes** (packed, honest
> accounting) for fp16 vs. ternary across tiny model sizes on TinyStories: does the
> ternary frontier sit below the fp16 frontier (net compression win), and if so, over
> what size range does it hold before the small-scale penalty makes it cross back?

The **smallest capable model in bytes** — the leftmost point on either curve that clears
the capability gate (§8) — is the headline read-off: we report its byte size and which
precision (fp16 or ternary) achieves it. The frontier curve is how we find it; the minimum
is the answer it yields.

Secondary question (compute axis): how does **training compute** (FLOPs / GPU-hours)
differ between the two? Expectation: ternary trains *more* expensively per step, so
"smaller in bytes" and "cheaper to train" point in opposite directions — worth stating
explicitly and measuring.

## 4. Experiment design

### Controlled variables (hold fixed across the entire sweep)
- Architecture family (decoder-only transformer; same layer recipe everywhere)
- Dataset and token budget (same tokens seen by every run)
- Tokenizer
- Optimizer, schedule, context length, seed protocol (≥2 seeds per config)

### Independent variables
- **Model size:** a sweep of ~4 sizes (see §5)
- **Precision:** fp16 baseline vs. ternary (BitLinear + STE)

### Dependent variables
- Held-out validation **perplexity** (primary, cheap, run every eval step)
- **Coherence score** from a judge model (grammar / consistency / completion quality),
  following the TinyStories evaluation idea
- **Total bytes** (packed — see §6) — this is the x-axis of the headline plot
- **Training compute**: FLOPs (≈ 6 · N_params · N_tokens) and wall-clock GPU-hours

### The headline deliverable
A single plot: **coherence (y) vs. total bytes, log-scale (x)**, two curves
(fp16, ternary), one marker per model size. The plot **annotates the smallest-bytes capable
point on each curve** (per the capability gate, §8), and the writeup names the global minimum
across both as the headline result: *"smallest capable English model = X bytes, achieved by
{ternary | fp16} at tier T."* Plus a short writeup stating where the ternary curve wins, where
it crosses, and why.

## 5. Concrete starting configs

Decoder-only transformer, modern components optional but **fixed across the sweep**
(RMSNorm, RoPE, SwiGLU are reasonable; or stay vanilla — just don't vary it). Rough
parameter count for the transformer blocks: `≈ 12 · n_layer · n_embd²`; embeddings add
`vocab · n_embd` (tied head).

| Tier   | n_layer | n_embd | n_head | ctx | ~block params |
|--------|---------|--------|--------|-----|---------------|
| tiny   | 4       | 128    | 4      | 256 | ~0.8M         |
| small  | 6       | 256    | 8      | 256 | ~4.7M         |
| medium | 8       | 384    | 8      | 512 | ~14M          |
| large  | 8       | 512    | 8      | 512 | ~25M          |

Treat these as starting points to adjust after the first run.

### ⚠️ The single most important design gotcha — read this
At these tiny sizes, **the embedding/LM-head table can dominate the parameter count**,
and embeddings are normally *not* ternarized. With a 50k GPT-2 vocab, a "tiny" model is
mostly embedding table, so ternarizing only the transformer blocks saves almost nothing in
bytes and your headline plot collapses. **Fix:** train a small custom BPE tokenizer on
TinyStories (target vocab ~4k–8k) so the transformer blocks dominate and the ternary byte
savings actually show up. Report byte accounting with the vocab choice stated. This single
decision makes or breaks the experiment.

## 6. Honest byte accounting (define this precisely in code)

`total_bytes = ternary_block_bytes + fp16_embedding_and_head_bytes + scale_factor_bytes`

- Quantized linear layers: `n_quantized_weights · 1.58 / 8` bytes (packed ternary)
- Embeddings + tied head: kept at fp16 → `2 · vocab · n_embd` bytes
- Per-layer/-channel scale factors: fp16, count them honestly (small but nonzero)
- For the fp16 baseline: every weight at 2 bytes

Write `count_bytes(model, precision)` as a tested function — the whole result depends on it
being correct.

## 7. The ternary layer (`BitLinear`)

- Keep a full-precision **latent** weight that the optimizer updates.
- Forward: `w_q = scale · round(clip(w_latent / scale, -1, 1))` with `scale = mean(|w_latent|)`
  (absmean), per-layer or per-output-channel. Per-layer adaptive scaling matches recent
  ternary-LM practice.
- Backward: **straight-through estimator** — gradient passes through the round/clip as
  identity (optionally clipped to [-1, 1]).
- Quantize only the linear layers inside transformer blocks. Leave embeddings, final norm,
  and LM head in fp16 for v1.
- For v1, leave **activations** in fp16 (isolates the weight-precision variable; add 8-bit
  activation quant only as a stretch goal).

## 8. Metrics & evaluation harness

- **Perplexity** on a held-out TinyStories split — cheap, run continuously.
- **Coherence judge**: feed model completions of held-out story prefixes to a judge model
  with a fixed rubric (grammar 0–5, consistency 0–5, completes-sensibly 0–5). Use a small
  open judge to stay free, or batch a few hundred completions through a paid API once at the
  end to control cost. Fix the rubric and the prefix set before looking at results.
- Report mean ± std across seeds. Note the train/val PPL ratio (overfitting signal — the
  ternary-as-regularizer claim lives here).

### Capability gate (frozen before any sweep run)

"Capable English" must be a single checkable predicate, fixed *before* runs, or "smallest
capable model" is unfalsifiable. A config counts as **capable** iff **BOTH** gates pass:

- **Coherence gate (primary):** mean judge score **≥ 4.0 / 5**, where the per-completion score
  is the mean of grammar(0–5), consistency(0–5), completes-sensibly(0–5), averaged over a
  **fixed set of N = 200 held-out TinyStories prefixes**.
- **Perplexity gate (secondary sanity guard):** val PPL **≤ T**, where **T = 1.5 × (best fp16
  val PPL achieved in the sweep)**. The *rule* (the 1.5× multiplier, anchored to the best fp16
  run) is frozen now; the numeric T resolves once the fp16 baseline runs complete. This is
  legitimate pre-registration — the policy is fixed, not the number — and is apples-to-apples
  because every run sees identical tokens / arch / tokenizer.

**Freeze list (commit to the repo before the first sweep run, never edit after):** the rubric
text, the 200 prefixes, the judge model identity (a fixed small open instruct model — the
free/local judge), and the judge prompt. Store them as committed files (e.g. `eval/rubric.md`,
`eval/prefixes.jsonl`) so "frozen" is enforced by version control. **Do not tune 4.0, N, the
1.5× multiplier, the rubric, or the prefixes after seeing sweep results.**

**Validity of the gate — pre-registration is not enough.** A frozen number in the wrong place,
or enforced by a noisy judge, measures nothing. Before freezing, run a one-time **sweep-blind
calibration + judge-reliability pass** and commit its results to `eval/calibration.md`:

- **Calibrate where 4.0 sits.** Score three reference sets through the *exact* frozen
  rubric+judge: (a) real TinyStories ground-truth continuations, (b) a published known-good
  small model's completions (e.g. TinyStories-33M), (c) deliberately degenerate text (shuffled /
  truncated / repetitive). Confirm good references clear 4.0, garbage falls well below it, and
  4.0 lands in the **discriminative band** for models of this scale. Pick the number from this
  calibration, *then* freeze it — done blind to sweep results, so it is calibration, not p-hacking.
- **Judge reliability (the gate is only as trustworthy as the judge):**
  - *Rank-ordering:* the judge must rank good > mediocre > bad across the reference sets above.
  - *Stability:* repeat-score the same completions and report **intra-judge std**. If the judge's
    noise band is wider than the good-vs-bad score gap, the gate is **invalid** — swap/upgrade the
    judge (a one-time paid-API pass for borderline configs is permitted) before proceeding.
- **Statistical power.** Report a **standard error / CI** on each config's mean judge score across
  the N = 200 prefixes (pooled over seeds), and state the expected CI width up front. A capability
  claim that separates adjacent tiers ("tier 2 capable, tier 1 not") counts **only if the CIs do
  not straddle 4.0**; if they do, report the tiers as **indistinguishable**, not a crossing.

## 9. Execution plan on free compute (Kaggle)

- ~30 GPU-hrs/week, ≤12-hr sessions, T4×2 (32GB) or P100.
- Each tiny/small run: well under an hour. Medium/large: a couple hours each.
- Full sweep = 4 sizes × 2 precisions × 2 seeds = 16 runs; comfortably one week's quota.
- **Checkpoint to Kaggle datasets / persistent output** every N steps so a session timeout
  never costs more than N steps. Resume logic is mandatory, not optional.
- Log everything (PPL curve, bytes, FLOPs, GPU-time) to a CSV per run; the plotting script
  reads the CSVs.

## 10. Definition of done

1. The coherence-vs-bytes frontier plot (fp16 vs ternary), with seeds, with the capability gate
   (§8) applied and the **smallest capable config in bytes** reported as the headline result
   (or the tiers reported as indistinguishable where CIs straddle the threshold).
2. A 1–2 page writeup: the finding (where ternary wins on bytes, where/why it crosses),
   the compute-axis caveat, and the embedding-dominance lesson.
3. Clean repo with reproducible run + plot scripts.

## 11. Stretch goals (only after v1 lands)
- Add a 4-bit curve (third line) to see the bit-width frontier.
- Vary the data constraint (vocab size / domain) and watch the emergence threshold move.
- Add 8-bit activation quantization and measure the additional byte/quality effect.
- Measure real inference latency/memory, not just theoretical bytes.

## 12. Risks / pitfalls
- **Embedding dominance** (see §5) — the #1 killer. Use a small custom vocab.
- Ternary training instability at tiny scale — watch for collapse; tune LR/scale.
- Judge-model cost and noise — fix rubric + prefixes up front; sample enough.
- Apples-to-apples discipline — identical tokens/arch/seeds, or the frontier is meaningless.
- Don't conflate byte savings (real, inference-side) with compute savings (negative, training-side).

---

## 13. What to hand a coding agent to scaffold this

Paste a brief like the following (adjust specifics), and require a plan before code:

> Build a research harness for a controlled ternary-vs-fp16 scaling study on TinyStories.
> Source of truth: the attached spec. **Before writing any code, restate the spec in your
> own words, list every ambiguity you see, and propose a file/module layout for my approval.
> Do not implement until I approve the plan.**
>
> Requirements the implementation must satisfy:
> - A decoder-only transformer with a config object exposing n_layer/n_embd/n_head/ctx/vocab.
> - A `BitLinear` module: full-precision latent weights, absmean per-layer scaling, ternary
>   forward, straight-through-estimator backward. A `precision` flag switches a model between
>   fp16-linear and BitLinear globally.
> - A small custom BPE tokenizer trained on TinyStories (configurable vocab, default ~8k).
> - A training loop with: seedable runs, checkpoint-and-resume (must survive a hard kill),
>   periodic held-out perplexity logging, and per-run CSV logging of step/PPL/tokens/FLOPs/wall-clock.
> - A tested `count_bytes(model, precision)` implementing the byte accounting in §6 exactly.
> - An eval script that generates completions from a fixed held-out prefix set and scores them
>   with a judge (pluggable: local model or API), emitting per-completion rubric scores.
> - A plotting script that reads all run CSVs and produces the coherence-vs-bytes frontier plot.
> - A `run_sweep` entrypoint that executes the §9 matrix and is safe to re-run (idempotent,
>   skips completed runs).
>
> Constraints: single-GPU (T4/P100), ≤12-hr sessions, PyTorch. No silent deviations from the
> spec — if something in the spec is wrong or infeasible, stop and flag it rather than
> "fixing" it yourself.

Keep this brief, the spec file, and a short `CLAUDE.md` (see the workflow notes) in the repo
so the agent reloads them every session.
