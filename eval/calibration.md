# Calibration & Judge Reliability (FROZEN)

- good (gold) mean: 4.568  (95% CI ±0.093)
- bad (degenerate) mean: 0.232
- good−bad gap: 4.337
- intra-judge std (mean over 3 re-scores): 0.0000

- rank-ordering good > bad: True
- judge reliable (intra_std < good−bad gap): True

If rank-ordering fails OR intra_std >= good−bad gap, STOP and upgrade the judge before freezing the gate (spec §8).
- mediocre (TinyStories-33M) mean: 4.378  (addendum)
- reference (TinyStories-33M, sampled temp1.0/topk40, n=50) mean: 3.687  (addendum, 2026-07-12)

## Decoding-policy correction (frozen 2026-07-12, before any greedy sweep scores existed)

**Defect found in a post-sweep audit:** evaluation decoding was never in the §8 freeze list.
The gate was calibrated only on decoding-free references (gold text, synthetic degenerate
text) and a *greedy* TinyStories-33M reference (4.378), while sweep models were judged on
temperature-1.0/top_k-40 *samples*. Scoring the 33M reference under that sampled policy
gives **3.687 < 4.0**: under sampled decoding the gate sits above what a known-good 33M
model achieves, so it has no discriminative power at the sweep's scales. The sampled-policy
gate was therefore never valid; this is an instrument correction, not gate tuning.

**Frozen rules (set blind — no greedy sweep-model scores had been computed):**
1. **Eval decoding = greedy** (temperature→0), uniformly, for every sweep model and every
   reference model. Greedy is the only policy the original calibration validated.
2. **Primary gate unchanged:** mean judge score ≥ 4.0 AND val PPL ≤ 1.5 × best fp16
   (spec §8, as pre-registered). Reported as the primary result.
3. **Reference-anchored secondary capability line:** a config is *reference-capable* iff its
   mean greedy judge score ≥ the mean greedy judge score of **roneneldan/TinyStories-1M**
   over the same 200 prefixes, same judge, same prompt. TinyStories-1M is the smallest
   checkpoint the TinyStories authors published and defend as producing grammatical,
   mostly-coherent stories, and is scale-matched to the sweep's tiny tier; it is chosen for
   those external reasons, before its score or any sweep model's greedy score is known.
   CI straddling the anchor ⇒ reported as indistinguishable, per the spec's power rule.
4. The original sampled-policy scores (five runs) are retained in the backup and reported
   alongside the greedy results in the writeup.
