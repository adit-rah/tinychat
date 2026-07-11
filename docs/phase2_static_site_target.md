# Phase 2 target — serverless in-browser inference from a static site

**Status:** declared goal, not yet in scope. Phase 1 (the spec's sweep + gate + frontier plot)
completes first and *selects the artifact*: the smallest capable checkpoint is the model that
ships. This doc records what "shippable" means so phase 1 decisions stay compatible with it.

## The goal

Bake the smallest capable model into the payload of a static site and run inference fully
client-side (no server, no API). The frontier plot's byte axis is therefore not just the
research metric — it is the deployment cost.

## Feasibility (from measured tier sizes, docs/emergence_floor_estimate.md)

- Ternary payloads: tiny 1.22 MB, small 3.05 MB, medium 5.94 MB (pre-brotli). All within
  ordinary web-page budgets even if the gate is only cleared at medium.
- Compute: ~3–4 MFLOPs/token (tiny) to ~12 (small) → hundreds of tokens/sec in WASM-SIMD on
  commodity hardware. No WebGPU required at these scales. KV cache at ctx 512 ≈ 2 MB.

## Deliverables (build after the sweep names the winner)

1. **Packer** — real file format for ternary weights: 5 trits/byte (3^5 = 243 ≤ 256, i.e.
   1.6 bits/weight, within 1.3% of the 1.58 accounting) + fp16 embeddings/head + per-layer
   fp16 scales + config header. The shipped file size must reconcile with `count_bytes`
   (report both; the small packing overhead is stated, not hidden).
2. **JS/WASM inference core** — int8 ternary matmuls, RMSNorm, RoPE, SwiGLU, KV cache,
   temperature sampling.
3. **Tokenizer port** — the frozen 4k BPE (`artifacts/tokenizer/tokenizer.json`) encoded/
   decoded in JS.
4. **The page** — single static HTML: type a story opening, the model continues it.
   Honest framing: a story completer, not a chatbot (a chat-tuned variant is phase 3,
   needs dialogue-format fine-tuning data).
5. **Model switcher (decided 2026-07-12):** the page offers *all* sweep models, not just
   the winner — an interactive frontier. Packed models ship as GitHub Release assets on a
   tagged release; the page discovers them via the `releases/latest` API (CORS-enabled, no
   backend, re-tag to update), lazy-fetches on selection with a progress bar (the byte size
   is deliberately visible UX), and caches via the Cache API. A `manifest.json` asset carries
   per-model bytes / val PPL / judge mean / gate verdict; the gate-passing smallest model is
   the preselected default. Consequence for the packer: the file format must be
   self-describing (config header), so one JS loader handles every tier/precision.

## Constraint phase 2 imposes on phase 1

None beyond what the spec already fixes — byte accounting stays the selection metric, and
checkpoints of gate-passing runs must be preserved (see the deliverables/ledger workflow in
docs/kaggle_runbook.md; final checkpoints of capable runs are download-and-keep artifacts).
