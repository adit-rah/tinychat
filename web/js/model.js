// Client-side forward pass for the nanofable decoder-only transformer, mirroring
// src/nanofable/model.py exactly: pre-norm RMSNorm (eps 1e-5), NeoX RoPE (base 10000),
// causal attention (1/sqrt(head_dim)), SwiGLU, tied embedding/LM head.
// Single-token steps with a KV cache; prefill is the same step run per prompt token.

const EPS = 1e-5;

// y = W x, W row-major [nOut, nIn]; ternary weights dispatch on .trits.
function linear(out, x, W, nOut, nIn) {
  const w = W.trits ?? W.f32;
  const scale = W.trits ? W.scale : 1;
  for (let o = 0; o < nOut; o++) {
    let acc = 0;
    const row = o * nIn;
    for (let i = 0; i < nIn; i++) acc += x[i] * w[row + i];
    out[o] = acc * scale;
  }
}

function rmsnorm(out, x, gain, n) {
  let ss = 0;
  for (let i = 0; i < n; i++) ss += x[i] * x[i];
  const inv = 1 / Math.sqrt(ss / n + EPS);
  for (let i = 0; i < n; i++) out[i] = x[i] * inv * gain[i];
}

function silu(v) {
  return v / (1 + Math.exp(-v));
}

export class Model {
  constructor({ header, tensors }) {
    const cfg = header.config;
    this.cfg = cfg;
    this.headDim = cfg.n_embd / cfg.n_head;

    const get = (name) => {
      const t = tensors.get(name);
      if (!t) throw new Error(`pack missing tensor ${name}`);
      return t;
    };
    this.tokEmb = get("tok_emb.weight").f32; // [vocab, n_embd], also the tied head
    this.finalNorm = get("final_norm.weight").f32;
    this.layers = [];
    for (let i = 0; i < cfg.n_layer; i++) {
      const p = `blocks.${i}.`;
      this.layers.push({
        attnNorm: get(p + "attn_norm.weight").f32,
        q: get(p + "attn.q.weight"), k: get(p + "attn.k.weight"),
        v: get(p + "attn.v.weight"), o: get(p + "attn.o.weight"),
        mlpNorm: get(p + "mlp_norm.weight").f32,
        gate: get(p + "mlp.gate.weight"), up: get(p + "mlp.up.weight"),
        down: get(p + "mlp.down.weight"),
      });
    }
    this.mlpHidden = get("blocks.0.mlp.gate.weight").trits
      ? tensors.get("blocks.0.mlp.gate.weight").trits.length / cfg.n_embd
      : tensors.get("blocks.0.mlp.gate.weight").f32.length / cfg.n_embd;

    // RoPE cache: cos/sin of pos * invFreq[j], j < headDim/2 (NeoX pairs j, j+hd/2).
    const half = this.headDim / 2;
    this.ropeCos = new Float32Array(cfg.ctx * half);
    this.ropeSin = new Float32Array(cfg.ctx * half);
    for (let p = 0; p < cfg.ctx; p++) {
      for (let j = 0; j < half; j++) {
        const angle = p / 10000 ** ((2 * j) / this.headDim);
        this.ropeCos[p * half + j] = Math.cos(angle);
        this.ropeSin[p * half + j] = Math.sin(angle);
      }
    }

    // KV cache + scratch buffers.
    const E = cfg.n_embd;
    this.kCache = this.layers.map(() => new Float32Array(cfg.ctx * E));
    this.vCache = this.layers.map(() => new Float32Array(cfg.ctx * E));
    this.pos = 0;
    this.x = new Float32Array(E);
    this.xn = new Float32Array(E);
    this.qB = new Float32Array(E);
    this.kB = new Float32Array(E);
    this.vB = new Float32Array(E);
    this.attnB = new Float32Array(E);
    this.projB = new Float32Array(E);
    this.gB = new Float32Array(this.mlpHidden);
    this.uB = new Float32Array(this.mlpHidden);
    this.scores = new Float32Array(cfg.ctx);
    this.logits = new Float32Array(cfg.vocab);
  }

  reset() {
    this.pos = 0;
  }

  applyRope(vec, pos) {
    const { n_head } = this.cfg;
    const hd = this.headDim, half = hd / 2;
    for (let h = 0; h < n_head; h++) {
      const base = h * hd;
      for (let j = 0; j < half; j++) {
        const c = this.ropeCos[pos * half + j];
        const s = this.ropeSin[pos * half + j];
        const a = vec[base + j], b = vec[base + j + half];
        vec[base + j] = a * c - b * s;
        vec[base + j + half] = b * c + a * s;
      }
    }
  }

  // Feed one token at the current position; returns logits (valid until next step).
  step(tokenId) {
    const cfg = this.cfg, E = cfg.n_embd, hd = this.headDim;
    const pos = this.pos;
    if (pos >= cfg.ctx) throw new Error("context window full");
    this.x.set(this.tokEmb.subarray(tokenId * E, (tokenId + 1) * E));

    for (let l = 0; l < this.layers.length; l++) {
      const L = this.layers[l];
      // attention
      rmsnorm(this.xn, this.x, L.attnNorm, E);
      linear(this.qB, this.xn, L.q, E, E);
      linear(this.kB, this.xn, L.k, E, E);
      linear(this.vB, this.xn, L.v, E, E);
      this.applyRope(this.qB, pos);
      this.applyRope(this.kB, pos);
      this.kCache[l].set(this.kB, pos * E);
      this.vCache[l].set(this.vB, pos * E);

      const K = this.kCache[l], V = this.vCache[l];
      const invSqrt = 1 / Math.sqrt(hd);
      for (let h = 0; h < cfg.n_head; h++) {
        const ho = h * hd;
        let max = -Infinity;
        for (let p = 0; p <= pos; p++) {
          let dot = 0;
          const ko = p * E + ho;
          for (let j = 0; j < hd; j++) dot += this.qB[ho + j] * K[ko + j];
          const sc = dot * invSqrt;
          this.scores[p] = sc;
          if (sc > max) max = sc;
        }
        let sum = 0;
        for (let p = 0; p <= pos; p++) {
          const e = Math.exp(this.scores[p] - max);
          this.scores[p] = e;
          sum += e;
        }
        for (let j = 0; j < hd; j++) {
          let acc = 0;
          for (let p = 0; p <= pos; p++) acc += this.scores[p] * V[p * E + ho + j];
          this.attnB[ho + j] = acc / sum;
        }
      }
      linear(this.projB, this.attnB, L.o, E, E);
      for (let i = 0; i < E; i++) this.x[i] += this.projB[i];

      // mlp
      rmsnorm(this.xn, this.x, L.mlpNorm, E);
      linear(this.gB, this.xn, L.gate, this.mlpHidden, E);
      linear(this.uB, this.xn, L.up, this.mlpHidden, E);
      for (let i = 0; i < this.mlpHidden; i++) this.gB[i] = silu(this.gB[i]) * this.uB[i];
      linear(this.projB, this.gB, L.down, E, this.mlpHidden);
      for (let i = 0; i < E; i++) this.x[i] += this.projB[i];
    }

    rmsnorm(this.xn, this.x, this.finalNorm, E);
    linear(this.logits, this.xn, { f32: this.tokEmb }, cfg.vocab, E);
    this.pos++;
    return this.logits;
  }

  // Run all prompt tokens; returns logits after the last one.
  prefill(ids) {
    let logits = null;
    for (const id of ids) logits = this.step(id);
    return logits;
  }
}

// Deterministic PRNG for reproducible sampling (JS-vs-JS; torch RNG differs).
// The seed is scrambled splitmix-style first: raw small seeds (0, 1, 2 …) give
// mulberry32 a near-empty state whose first few outputs are biased tiny, which
// visibly skews the first sampled tokens.
export function mulberry32(seed) {
  let a = (seed >>> 0) ^ 0x9e3779b9;
  a = Math.imul(a ^ (a >>> 16), 0x21f0aaad);
  a = Math.imul(a ^ (a >>> 15), 0x735a2d97);
  a = (a ^ (a >>> 15)) >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Mirrors src/nanofable/generate.py next_token: temperature floor 1e-6, top-k keeps
// ties at the k-th value, top_k=0 disables the mask, then multinomial via CDF.
export function sampleToken(logits, { temperature, topK }, rand) {
  const n = logits.length;
  const t = Math.max(temperature, 1e-6);
  const scaled = new Float32Array(n);
  for (let i = 0; i < n; i++) scaled[i] = logits[i] / t;

  if (topK) {
    const k = Math.min(topK, n);
    const sorted = Float32Array.from(scaled).sort().reverse();
    const threshold = sorted[k - 1];
    for (let i = 0; i < n; i++) if (scaled[i] < threshold) scaled[i] = -Infinity;
  }

  let max = -Infinity;
  for (let i = 0; i < n; i++) if (scaled[i] > max) max = scaled[i];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    scaled[i] = Math.exp(scaled[i] - max);
    sum += scaled[i];
  }
  const r = rand() * sum;
  let cum = 0;
  for (let i = 0; i < n; i++) {
    cum += scaled[i];
    if (r < cum) return i;
  }
  return n - 1;
}

// Dev/parity helper: greedy (argmax) continuation, comparable to a Python argmax loop.
export function greedy(model, ids, n) {
  model.reset();
  let logits = model.prefill(ids);
  const out = [];
  for (let i = 0; i < n && model.pos < model.cfg.ctx; i++) {
    let best = 0;
    for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
    out.push(best);
    logits = model.step(best);
  }
  return out;
}
