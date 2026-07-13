# Demo site (`web/`) — the phase-2 static page, built on stand-in models

**What it is.** A static site (no server, no build step) that runs nanofable models fully
client-side: type a story opening, the model continues it token by token. Honest framing
per `docs/phase2_static_site_target.md`: a story completer, not a chatbot. Built 2026-07-12
against the stand-in `local/` checkpoints while the real sweep trains; when the sweep names
winners, re-pack and replace the models — nothing else changes.

## Pack format v1 (`.tpack`, magic `TCP1`)

Normative implementation: `src/nanofable/pack.py` (tested in `tests/test_pack.py`).
Little-endian: `TCP1 | uint32 header_len | JSON header | pad-to-8 | tensor payload`.
The JSON header is self-describing (config, precision, tensor manifest with offsets, tied
head), so the single JS loader (`web/js/pack.js`) handles every tier and precision.

- Ternary arm: in-block linears quantized at pack time with BitLinear's rule and stored at
  5 trits/byte; one absmean scale per tensor, **rounded to fp16 before writing** so Python
  and JS dequantize with bit-identical values. Embeddings/norms stay raw fp16.
- fp16 arm: same layout, everything raw fp16, no scales.
- `lm_head.weight` is never stored (tied); loaders reuse the embedding matrix.

### The size ledger — ternary's byte win lives in the `.tpack`

Ternary weights only shrink the artifact when the trits are actually packed (5/byte).
Torch-native formats (safetensors/.pt) have no sub-byte dtype, so the release
safetensors stores each quantized weight in a full fp16 slot — same size as the fp16
arm. Measured / exact-computed:

| tier | params | `model.safetensors` (either arm) | ternary `.tpack` |
|---|---|---|---|
| tiny | 1.38 M | 2,758,256 B | 1,225,952 B |
| small | 5.87 M | ~11.7 MB | 3,074,136 B |
| medium | 15.7 M | 31,477,256 B | 5,999,152 B |
| large | 27.8 M | ~55.6 MB | ~9.36 MB |

Both encodings hold bit-identical weights (tested); they differ only in storage. The
HF release ships both side by side: `model.safetensors` for standard tooling,
`model.tpack` as the honest small artifact.

### Getting a `.tpack`

`scripts/pack_model.py` accepts **either** source, and the two routes are guaranteed
byte-identical (pinned by `test_prequantized_pack_is_byte_identical`):

```sh
# from a training checkpoint (latents -> quantize at pack time)
python scripts/pack_model.py --ckpt local/tiny_ternary_0.pt --tier tiny \
    --precision ternary --out model.tpack

# from a release export (already-quantized; tier/precision read from config.json)
python scripts/pack_model.py --ckpt local/hf/tiny_ternary/model.safetensors \
    --out model.tpack
```

The safetensors route *recovers* the original scale (max|w|) rather than re-running
absmean on dequantized values — re-quantizing would silently compute a wrong scale.

### Inference on a `.tpack` — possible, but only via the project loaders

A `.tpack` is a complete inference artifact, but **no standard tooling reads it** — torch,
HF, and safetensors have no sub-byte dtype, so the packed trits are only usable through
the two loaders that implement the format:

- **Browser** (`web/js/pack.js` + `model.js`): the intended path. Unpacks trit5 → int8
  once at load and runs the ternary kernel directly on the trits (`scale` applied once
  per output row). This is true ternary execution.
- **Python** (`src/nanofable/pack.py`): `read_pack()` → `state_dict_from_pack()` →
  `load_state_dict` into a **`precision="fp16"`** `build_model` (not the ternary arm —
  weights come back already dequantized as `scale × {-1,0,+1}`, so they belong in plain
  `nn.Linear`; loading them into BitLinear would re-quantize a second time). Then
  `generate()` works normally. Output is bit-identical to the browser (verified).

There is no third path: handing a `.tpack` to torch/HF directly will not work. For
standard-tooling consumers (e.g. the HuggingFace release), export dequantized fp16
safetensors with `scripts/export_model.py` — same weights, verified bit-identical to the
`.tpack` dequant; the `.tpack` stays the honest small artifact (its trit values and the
safetensors values agree exactly, only the storage encoding differs).

### Reconciliation with `count_bytes` (spec §6) — shipped tiny models

| | file | count_bytes | delta |
|---|---|---|---|
| tiny ternary | 1,225,952 B | 1,216,895.7 B | +9,056 B (0.74%) |
| tiny fp16 | 2,758,680 B | 2,752,512.0 B | +6,168 B (0.22%) |

Delta decomposition (printed by the packer CLI, pinned by tests): 1.6-vs-1.58-bit ceil
padding + norm gains (deliberately uncounted by `count_bytes`) + header/scales/alignment.

## How to pack / serve / release

```sh
# pack a checkpoint (any tier, either precision)
python scripts/pack_model.py --ckpt local/tiny_ternary_0.pt --tier tiny \
    --precision ternary --out web/models/tiny_ternary.tpack

# serve locally (ES modules + Cache API need http on localhost, not file://)
python3 -m http.server 8000 -d web
```

`web/models/` commits **only** the baked default (`nanofable-16m-fp16.tpack`); everything
else there is gitignored. `web/manifest.json` lists all 8 sweep models under the HF release
naming (`NanoFable-<params>-<precision>`, tiers tiny/small/medium/large = 1M/6M/16M/28M):
the baked one by relative URL, the other seven by their HuggingFace repo URL
(`https://huggingface.co/adrahmana/NanoFable-<params>-<precision>/resolve/main/model.tpack`),
which the install-all flow streams into the Cache API. HF serves these cross-origin
(`Access-Control-Allow-Origin: *` on the LFS/Xet redirect target), so no proxy is needed.
A model that fails to download reports a graceful toast and stays greyed out in the picker.
**Release flow:** push a tier's `model.tpack` to its HF repo, then fill in that model's
`bytes` / `ppl` / `judge` / `gate` in the manifest. The page preselects the smallest
available `gate: true` model (falls back to the baked one).

**Deviation from the phase-2 doc, on purpose:** discovery is this committed static
manifest, not an API listing. Swapping later touches only `loadManifest()`.

## Verification

- `pytest tests/test_pack.py` — trit roundtrip, header integrity, BitLinear quantization
  fidelity, pack→unpack forward parity vs the original model, size reconciliation (exact
  delta, <2%), and byte-identity of `web/tokenizer.json` with the frozen artifact.
- **Cross-language greedy parity** (sampling RNGs can't match; greedy argmax must, since
  both sides use the header's fp16-rounded scales). Python reference:

  ```python
  # ids = tok.encode(prompt).ids; model = unpacked pack loaded into fp16-arm Transformer
  for _ in range(30):
      logits, _ = model(torch.tensor([ids])); ids.append(int(logits[0, -1].argmax()))
  ```

  Browser side: `window.__greedy(prompt, 30)` in the devtools console. Verified exact
  token-id match on 3 prompts × 30 tokens (2026-07-12), plus identical tokenizer ids on
  6 tricky strings (leading space, emoji, newlines, contractions, digits).
- Known, accepted: the config `seed` reproduces stories JS-vs-JS only (torch multinomial
  differs); the JS PRNG scrambles the seed splitmix-style before use because raw small
  seeds gave mulberry32 biased-tiny early outputs (which sampled EOS immediately).
- **Corpus mojibake, repaired at display time only.** The TinyStories dataset stores curly
  punctuation CP1252-double-encoded (`“` as the literal three characters `â€œ`), so models
  legitimately *generate* mojibake — the sequences are merged BPE tokens (e.g. `â` = id 535),
  while real curly quotes never earned a merge. `web/js/display.js` reverses the encoding
  *generically* (2026-07-13; a lookup table of known sequences missed valid-but-unlisted
  sequences like `â”€` → `─`, and couldn't handle orphaned fragments — continuation tokens
  emitted without their lead byte): each char maps back to its CP1252 byte, valid UTF-8 runs
  decode to the intended char, and anything that fails is residue. Residue is dropped from
  model output (the corpus's visible text is ASCII + mojibake only, so undecodable non-ASCII
  is never content; bare `â€` still renders `”`) but kept in user input (people type curly
  quotes/é/emoji). Streaming-safe across chunk splits. Token ids, the
  tokenizer, the dataset, and the parity hooks are untouched — cleaning the dataset itself
  would invalidate completed sweep runs, and the judge sees the same mojibake for every
  config, so gate comparisons stay apples-to-apples.

## Done criteria

- [x] Packer tests green; full suite green.
- [x] Reconciliation < 2%, stated (table above), CLI prints the decomposition.
- [x] Greedy parity: exact id match Python↔browser; tokenizer id parity.
- [x] UI: streaming into a storybook passage, stop button, seed-reproducible stories,
      model picker with locked states + tooltip, install-all with progress card
      (bottom-right) and per-model failure toasts (bottom-left), Cache API persistence
      across reloads, config popover (max tokens / temperature / top-k / seed — the full
      Python sampling surface), profanity + empty + too-long validation toasts,
      mobile-width sane, works from a bare `python3 -m http.server`.
- [ ] Replace stand-in packs with sweep winners; fill manifest metrics; tag a release.
