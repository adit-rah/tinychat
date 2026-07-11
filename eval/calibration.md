# Calibration & Judge Reliability (FROZEN)

- good (gold) mean: 4.568  (95% CI ±0.093)
- bad (degenerate) mean: 0.232
- good−bad gap: 4.337
- intra-judge std (mean over 3 re-scores): 0.0000

- rank-ordering good > bad: True
- judge reliable (intra_std < good−bad gap): True

If rank-ordering fails OR intra_std >= good−bad gap, STOP and upgrade the judge before freezing the gate (spec §8).
