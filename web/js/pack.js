// .tpack (pack format v1 "TCP1") loader + model download/cache helpers.
// Format is defined normatively in src/nanofable/pack.py.

const MAGIC = 0x31504354; // "TCP1" little-endian
const ALIGN = 8;
const CACHE_NAME = "nanofable-models";

// 243 -> 5 trits lookup, built once.
const TRIT_LUT = (() => {
  const lut = new Int8Array(243 * 5);
  for (let b = 0; b < 243; b++) {
    let v = b;
    for (let j = 0; j < 5; j++) {
      lut[b * 5 + j] = (v % 3) - 1;
      v = (v / 3) | 0;
    }
  }
  return lut;
})();

function unpackTrits(bytes, numel) {
  const out = new Int8Array(numel);
  const full = Math.floor(numel / 5);
  let o = 0;
  for (let i = 0; i < full; i++) {
    const base = bytes[i] * 5;
    out[o++] = TRIT_LUT[base];
    out[o++] = TRIT_LUT[base + 1];
    out[o++] = TRIT_LUT[base + 2];
    out[o++] = TRIT_LUT[base + 3];
    out[o++] = TRIT_LUT[base + 4];
  }
  for (let j = 0; o < numel; j++) out[o++] = TRIT_LUT[bytes[full] * 5 + j];
  return out;
}

function halfToFloat32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h & 0x7c00) >> 10;
    const frac = h & 0x03ff;
    if (exp === 0) out[i] = sign * 2 ** -14 * (frac / 1024);
    else if (exp === 0x1f) out[i] = frac ? NaN : sign * Infinity;
    else out[i] = sign * 2 ** (exp - 15) * (1 + frac / 1024);
  }
  return out;
}

// Parse a pack into { header, tensors }. Tensor values are either
// { f32: Float32Array } or { trits: Int8Array, scale: number } — matmul-ready.
export function parsePack(buffer) {
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== MAGIC) throw new Error("not a TCP1 pack");
  const headerLen = view.getUint32(4, true);
  const header = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 8, headerLen))
  );
  const payload = Math.ceil((8 + headerLen) / ALIGN) * ALIGN;

  const tensors = new Map();
  for (const t of header.tensors) {
    const numel = t.shape.reduce((a, b) => a * b, 1);
    if (t.dtype === "trit5") {
      const bytes = new Uint8Array(buffer, payload + t.offset, t.bytes);
      tensors.set(t.name, { trits: unpackTrits(bytes, numel), scale: t.scale });
    } else {
      const u16 = new Uint16Array(buffer.slice(payload + t.offset, payload + t.offset + t.bytes));
      tensors.set(t.name, { f32: halfToFloat32(u16) });
    }
  }
  return { header, tensors };
}

export async function loadManifest(url = "manifest.json") {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`manifest fetch failed: ${res.status}`);
  return res.json();
}

// Installed packs are held in memory for this tab only. Nothing is written to the
// Cache API, and remote packs are fetched with `no-store` so they don't land in the
// HTTP disk cache either: closing the tab leaves nothing behind on the device.
const session = new Map(); // url -> ArrayBuffer

// Earlier builds persisted packs in the Cache API. Drop that store on load so anyone
// who installed back then gets those bytes reclaimed.
if (typeof caches !== "undefined") caches.delete(CACHE_NAME).catch(() => {});

export function isCached(url) {
  return session.has(url);
}

// Fetch a pack with download progress, keeping it in memory for the session.
// onProgress(loadedBytes, totalBytes|null) fires per chunk.
export async function fetchModel(url, onProgress) {
  const hit = session.get(url);
  if (hit) return hit;

  // The baked model is a same-origin static asset, so let the browser cache it the way
  // it caches any other asset. Downloaded models are the ones we refuse to persist.
  const remote = /^https?:\/\//.test(url);
  const res = await fetch(url, remote ? { cache: "no-store" } : undefined);
  if (!res.ok) throw new Error(`download failed (${res.status})`);
  const total = Number(res.headers.get("Content-Length")) || null;
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (onProgress) onProgress(loaded, total);
  }
  const data = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) {
    data.set(c, o);
    o += c.length;
  }
  session.set(url, data.buffer);
  return data.buffer;
}
