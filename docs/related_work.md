# Related work & positioning (probed 2026-07-13)

Where this project sits in the literature, from a web probe on 2026-07-13. For the writeup's
related-work section; re-verify citations before publishing.

## Provenance — what was known at design time vs found after

**Design-time inputs (pre-registration context):** the TinyStories paper, the BitNet b1.58
paper, and TernaryLM — the last indirectly, via the frozen spec (§2 cites its ~2× PPL penalty
as the motivating single-point comparison). Everything in the design (tiers, gate, byte
accounting) traces to these three.

**Found after the fact (2026-07-13 probe, mid-sweep, before any eval results):** ParetoQ, the
billion-scale quantization scaling-law literature, and the llama2.c/tinyllamas browser ports.
These inform the writeup's positioning and the post-v1 levers below, but influenced no frozen
design decision — the sweep was already running when they were found. State this in the
writeup; it is a feature of the pre-registration story, not a gap.

## The four adjacent clusters

1. **Emergence at tiny scale — TinyStories** (Eldan & Li, arXiv:2305.07759). Established
   coherent English at 1–10M params on a constrained distribution and pioneered LLM-judge
   grading. Axis: parameters, full precision. Bytes never enter. Released models 1M–33M
   (GPT-Neo, 50k vocab); our judge calibration scored TinyStories-33M at 4.378.
2. **Ternary at tiny scale — a single point, not a curve.** TernaryLM (arXiv:2602.07374)
   trains native ternary on TinyStories on a single T4 (same regime as ours): val PPL 58.4 vs
   FP32 baseline ~28. One size, PPL-only, no judge, no byte accounting, no capability gate.
   (Our tiny smoke pair echoes it: 52.4 vs 16.0.) BitNet b1.58 established ternary↔fp16
   parity only at ≥3B params — the opposite end of the scale.
3. **Low-bit scaling laws at large scale.** ParetoQ (arXiv:2502.02631) rigorously compares
   1/1.58/2/3/4-bit across sizes (600M–3B+, QAT on pretrained, benchmark accuracy): finds
   ternary/2-bit/3-bit roughly tied, often beating 4-bit. Related quantization scaling laws
   (arXiv:2411.17691) are also billion-scale. Nobody runs this program at the emergence
   threshold. ParetoQ's 2-bit result makes a 2-bit arm a sharper stretch goal than the
   spec §11's 4-bit suggestion.
4. **Tiny models in browsers — byte-careless.** Karpathy's llama2.c "tinyllamas"
   (stories260K/15M/42M/110M) have pure-JS and WASM browser ports — all fp32 (stories15M
   ≈ 60 MB; stories260K ≈ 1 MB but visibly below coherence). web-llm-class frameworks start
   at ~0.5B/4-bit (~300 MB).

## The unclaimed intersection (as of the probe)

Coherence-vs-**total-bytes** frontier, fp16 and ternary trained identically from scratch, at
1–30M params, with a pre-registered calibrated capability gate — and its corollary: a
gate-certified capable model at single-digit MB in a static page. The literature has single
points (TernaryLM), big-scale curves (ParetoQ), and byte-careless demos (tinyllamas).

**Positioning sentence:** ParetoQ maps the bit-width Pareto frontier at billion scale; we map
it at the emergence threshold, in bytes, with a judged capability gate.

**The foil:** stories260K proves a ~1 MB model can exist in a browser; it does not speak
coherent English. The question was never whether a 1 MB model can exist — it's where between
1 MB-incoherent and 60 MB-coherent the gate sits, under a judge calibrated to say so.

## Post-v1 levers for pushing the byte floor down (each faces the SAME frozen gate)

Ordered by expected MB-per-effort: (1) quantize embeddings — fp16 table is 86% of ternary
tiny's bytes; (2) a ~2.5–3M-param mini-tier in the tiny→small gap (~4 runs, frozen recipe,
declared as v2 extension); (3) a 2-bit arm (per ParetoQ); (4) constrain the distribution
further — changes the claim's fine print ("coherent English about X"), spec §11; (5) report
brotli transfer size alongside disk size, labeled as such. The frozen gate makes this a
ratchet: every new low is a record against the same bar, not a moved goalpost.
