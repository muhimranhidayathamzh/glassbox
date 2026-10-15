// Byte-level BPE tokenizer: a line-by-line port of python/glassbox/tokenizer.py.
// Loads the tokenizer.json written by `python -m glassbox.export`.

export const EOT = "<|endoftext|>";

const utf8 = new TextEncoder();

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Replace every non-overlapping occurrence of (a, b), left to right, with newId. */
function mergePair(ids, a, b, newId) {
  const out = [];
  for (let i = 0; i < ids.length; ) {
    if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) {
      out.push(newId);
      i += 2;
    } else {
      out.push(ids[i]);
      i += 1;
    }
  }
  return out;
}

export class Tokenizer {
  constructor(json) {
    if (json.type !== "byte-level-bpe" || !json.js_pretoken_pattern) {
      throw new Error("not a glassbox web tokenizer (export it with python -m glassbox.export)");
    }
    this.ranks = new Map(); // left * 65536 + right -> merged id; lower id = higher priority
    this.vocab = [];
    for (let b = 0; b < 256; b++) this.vocab.push(Uint8Array.of(b));
    json.merges.forEach(([a, b], i) => {
      this.ranks.set(a * 65536 + b, 256 + i);
      const piece = new Uint8Array(this.vocab[a].length + this.vocab[b].length);
      piece.set(this.vocab[a]);
      piece.set(this.vocab[b], this.vocab[a].length);
      this.vocab.push(piece);
    });

    this.specialTokens = new Map();
    const specials = Object.entries(json.special_tokens).sort((x, y) => x[1] - y[1]);
    for (const [text, id] of specials) {
      if (id !== this.vocab.length) throw new Error(`special token ${text} has unexpected id ${id}`);
      this.specialTokens.set(text, id);
      this.vocab.push(utf8.encode(text));
    }
    if (this.vocab.length !== json.vocab_size) throw new Error("vocab size mismatch");

    this.pretok = new RegExp(json.js_pretoken_pattern, "gu");
    this.specialSplit = specials.length
      ? new RegExp("(" + specials.map(([t]) => escapeRegExp(t)).join("|") + ")", "u")
      : null;
    this.cache = new Map();
    this.decoder = new TextDecoder("utf-8");
  }

  get vocabSize() {
    return this.vocab.length;
  }

  get eotId() {
    return this.specialTokens.get(EOT);
  }

  encodeWord(word) {
    const cached = this.cache.get(word);
    if (cached) return cached;
    let ids = Array.from(utf8.encode(word));
    while (ids.length >= 2) {
      let bestRank = Infinity;
      let bestPos = -1;
      for (let i = 0; i < ids.length - 1; i++) {
        const r = this.ranks.get(ids[i] * 65536 + ids[i + 1]);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          bestPos = i;
        }
      }
      if (bestPos < 0) break;
      ids = mergePair(ids, ids[bestPos], ids[bestPos + 1], bestRank);
    }
    if (this.cache.size < 200_000) this.cache.set(word, ids);
    return ids;
  }

  encodeOrdinary(text) {
    const out = [];
    for (const m of text.matchAll(this.pretok)) {
      for (const id of this.encodeWord(m[0])) out.push(id);
    }
    return out;
  }

  /** With allowSpecial, literal special-token strings become their ids; otherwise plain text. */
  encode(text, { allowSpecial = true } = {}) {
    if (!allowSpecial || !this.specialSplit) return this.encodeOrdinary(text);
    const out = [];
    for (const part of text.split(this.specialSplit)) {
      if (this.specialTokens.has(part)) out.push(this.specialTokens.get(part));
      else if (part) for (const id of this.encodeOrdinary(part)) out.push(id);
    }
    return out;
  }

  decodeBytes(ids) {
    let n = 0;
    for (const id of ids) n += this.vocab[id].length;
    const out = new Uint8Array(n);
    let o = 0;
    for (const id of ids) {
      out.set(this.vocab[id], o);
      o += this.vocab[id].length;
    }
    return out;
  }

  decode(ids) {
    return this.decoder.decode(this.decodeBytes(ids));
  }

  /** Readable form of one token, for display. Invalid UTF-8 fragments show as hex bytes. */
  tokenStr(id) {
    try {
      return new TextDecoder("utf-8", { fatal: true }).decode(this.vocab[id]);
    } catch {
      return Array.from(this.vocab[id], (b) => `\\x${b.toString(16).padStart(2, "0")}`).join("");
    }
  }
}
