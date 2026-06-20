# Project: Ternary vs FP16 Emergence Frontier on TinyStories

Spec is the source of truth: @specs/idea2_ternary_emergence_frontier_spec.md

## Working contract
- Before writing code for a new component, restate what the spec asks for and list any
  ambiguities. Propose a module layout and wait for my approval. Do not implement past
  an unresolved ambiguity.
- Implement only against the spec. If something in the spec is wrong or infeasible, STOP
  and flag it — do not silently "fix" or reinterpret it.
- Verify each spec requirement before calling a task done.

## Non-negotiable design facts (these break the experiment if violated)
- Train a small custom BPE tokenizer (~8k vocab). Do NOT default to the 50k GPT-2 vocab —
  the embedding table would dominate byte count and negate the ternary compression we measure.
- Hold architecture, token budget, optimizer, schedule, and seed protocol FIXED across the
  whole sweep. The only things that vary are model size and precision (fp16 vs ternary).
- `count_bytes(model, precision)` must implement the spec's byte accounting exactly and have
  tests. The headline result depends on it being correct.
- Quantize only the linear layers in transformer blocks for v1. Embeddings, final norm, and
  LM head stay fp16. Activations stay fp16 for v1.
- Every run must checkpoint and be resumable — sessions hard-cap at 12 hours.

## Layout
- `src/` — implementation only.
- `docs/` — the spec, design notes, and the writeup. Keep intent/rationale here, not in code.
- `specs/` — the frozen spec.
- Runs log to per-run CSV; the plotting script reads CSVs. Never hand-edit results.
