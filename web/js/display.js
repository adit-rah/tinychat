// Display-time repair of the TinyStories corpus's CP1252 mojibake.
// The dataset stores non-ASCII punctuation double-encoded (each UTF-8 byte re-read
// as CP1252), so the model legitimately *generates* mojibake — the sequences are
// merged BPE tokens — and it can also emit *fragments* of them (a continuation
// token without its lead byte), which no lookup table of known sequences can
// anticipate. So the encoding is reversed generically instead: map each char back
// to its CP1252 byte, decode valid UTF-8 runs to the intended character, and treat
// anything that fails as mojibake residue.
//
// Residue policy differs by source:
// - model output (mojibakeFilter): residue is DROPPED — the corpus's visible text
//   is ASCII + mojibake only, so an undecodable non-ASCII char from the model is a
//   broken fragment, never content. Exception: a bare "â€" whose third byte was
//   lost is the corpus's broken closing quote — rendered ”.
// - user input (fixMojibake): residue is KEPT — people legitimately type curly
//   quotes, é, emoji.
// Token ids, the tokenizer, the dataset, and the parity hooks are untouched.

// CP1252 codepoint -> byte for the 0x80–0x9F specials. Every other char below
// 0x100 maps to itself — including the bytes CP1252 leaves *undefined* (e.g.
// 0x9D), which the dataset pipeline carried through as literal C1 controls.
const CP1252_HIGH = {
  0x20ac: 0x80, 0x201a: 0x82, 0x0192: 0x83, 0x201e: 0x84, 0x2026: 0x85,
  0x2020: 0x86, 0x2021: 0x87, 0x02c6: 0x88, 0x2030: 0x89, 0x0160: 0x8a,
  0x2039: 0x8b, 0x0152: 0x8c, 0x017d: 0x8e, 0x2018: 0x91, 0x2019: 0x92,
  0x201c: 0x93, 0x201d: 0x94, 0x2022: 0x95, 0x2013: 0x96, 0x2014: 0x97,
  0x02dc: 0x98, 0x2122: 0x99, 0x0161: 0x9a, 0x203a: 0x9b, 0x0153: 0x9c,
  0x017e: 0x9e, 0x0178: 0x9f,
};

// undefined = not CP1252-expressible, so it cannot be part of mojibake.
const byteOf = (ch) => {
  const cp = ch.codePointAt(0);
  return cp < 0x100 ? cp : CP1252_HIGH[cp];
};

const seqLen = (b) =>
  b >= 0xc2 && b <= 0xdf ? 2 : b >= 0xe0 && b <= 0xef ? 3 : b >= 0xf0 && b <= 0xf4 ? 4 : 0;
const isCont = (b) => b >= 0x80 && b <= 0xbf;

const utf8 = new TextDecoder("utf-8", { fatal: true });

// One pass over `s`. When !atEnd, an unfinished candidate sequence at the tail is
// returned in `hold` for the next chunk to complete (streaming-safe splits).
function unmojibake(s, atEnd, keepResidue) {
  let out = "";
  let i = 0;
  while (i < s.length) {
    const ch = s[i];
    const b = byteOf(ch);
    if (b === undefined || b < 0x80) {
      out += ch;
      i++;
      continue;
    }
    const need = seqLen(b);
    const bytes = [b];
    let j = i + 1;
    while (j < s.length && bytes.length < need) {
      const cb = byteOf(s[j]);
      if (cb === undefined || !isCont(cb)) break;
      bytes.push(cb);
      j++;
    }
    if (need && bytes.length === need) {
      try {
        const decoded = utf8.decode(new Uint8Array(bytes));
        if (decoded.codePointAt(0) > 0x9f) {
          // (≤ 0x9f would be an invisible C1 control — treat as residue)
          out += decoded;
          i = j;
          continue;
        }
      } catch {
        /* not a real sequence — residue */
      }
    } else if (need && j === s.length && !atEnd) {
      return { out, hold: s.slice(i) };
    }
    if (b === 0xe2 && bytes.length >= 2 && bytes[1] === 0x80) {
      out += "”"; // corpus's broken closing quote (its 0x9d byte lost)
      i += 2;
      continue;
    }
    if (keepResidue) out += ch;
    i++;
  }
  return { out, hold: "" };
}

export function fixMojibake(s) {
  return unmojibake(s, true, true).out;
}

export function mojibakeFilter() {
  let buf = "";
  return {
    push(text) {
      const { out, hold } = unmojibake(buf + text, false, false);
      buf = hold;
      return out;
    },
    flush() {
      const { out } = unmojibake(buf, true, false);
      buf = "";
      return out;
    },
  };
}
