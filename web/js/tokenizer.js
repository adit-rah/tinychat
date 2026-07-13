// ByteLevel-BPE port of the frozen 4k tokenizer (web/tokenizer.json — byte-identity
// with artifacts/tokenizer/tokenizer.json is guarded by tests/test_pack.py).
// Mirrors src/nanofable/generate.py: no BOS prepended; special ids skipped on decode.

export const BOS_ID = 0, EOS_ID = 1, PAD_ID = 2;

// GPT-2 byte <-> unicode table: printable bytes map to themselves, the rest to 256+n.
function byteUnicodeTables() {
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const byteToChar = new Array(256), charToByte = new Map();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    const cp = bs.includes(b) ? b : 256 + n++;
    byteToChar[b] = String.fromCharCode(cp);
    charToByte.set(byteToChar[b], b);
  }
  return { byteToChar, charToByte };
}

// GPT-2 pre-tokenizer (ByteLevel use_regex:true, add_prefix_space:false).
const PRETOKEN_RE =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export class Tokenizer {
  constructor(tokenizerJson) {
    const model = tokenizerJson.model;
    this.vocab = new Map(Object.entries(model.vocab));
    this.idToToken = new Array(this.vocab.size);
    for (const [tok, id] of this.vocab) this.idToToken[id] = tok;
    this.ranks = new Map(model.merges.map(([a, b], i) => [a + " " + b, i]));
    const { byteToChar, charToByte } = byteUnicodeTables();
    this.byteToChar = byteToChar;
    this.charToByte = charToByte;
    this.cache = new Map(); // pretoken -> ids
    this.utf8 = new TextEncoder();
  }

  // Greedy BPE over one pre-token's byte-level chars.
  bpe(chars) {
    let parts = chars;
    for (;;) {
      let best = null, bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best === null) return parts;
      parts = parts
        .slice(0, best)
        .concat([parts[best] + parts[best + 1]], parts.slice(best + 2));
    }
  }

  encode(text) {
    const ids = [];
    for (const m of text.matchAll(PRETOKEN_RE)) {
      const pre = m[0];
      let cached = this.cache.get(pre);
      if (!cached) {
        const bytes = this.utf8.encode(pre);
        const chars = Array.from(bytes, (b) => this.byteToChar[b]);
        cached = this.bpe(chars)
          .map((t) => this.vocab.get(t))
          .filter((id) => id !== undefined);
        this.cache.set(pre, cached);
      }
      ids.push(...cached);
    }
    return ids;
  }

  decode(ids) {
    const d = this.streamDecoder();
    return ids.map((id) => d.push(id)).join("") + d.flush();
  }

  // Streaming decoder: emits text per token, holding back incomplete UTF-8 chars.
  streamDecoder() {
    const utf8 = new TextDecoder("utf-8", { fatal: false });
    const self = this;
    return {
      push(id) {
        if (id <= PAD_ID) return ""; // specials skipped, as in HF decode
        const tok = self.idToToken[id];
        if (tok === undefined) return "";
        const bytes = Uint8Array.from(tok, (c) => self.charToByte.get(c));
        return utf8.decode(bytes, { stream: true });
      },
      flush() {
        return utf8.decode();
      },
    };
  }
}

export async function loadTokenizer(url = "tokenizer.json") {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`tokenizer fetch failed: ${res.status}`);
  return new Tokenizer(await res.json());
}
