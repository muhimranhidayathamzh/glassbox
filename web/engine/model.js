// Forward pass of the glassbox transformer, one token at a time with a KV cache.
// Mirrors python/glassbox/model.py operation by operation:
//   x = wte[token] + wpe[pos]
//   for each block: x = x + attn(ln1(x)); x = x + mlp(ln2(x))
//   logits = ln_f(x) @ wte.T
// Weights stay float32 (as in PyTorch); dot products accumulate in float64.

import { decodeTensors } from "./weights.js";

function layerNorm(x, w, b, eps, out) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) {
    const d = x[i] - mean;
    variance += d * d;
  }
  variance /= n;
  const std = Math.sqrt(variance + eps);
  for (let i = 0; i < n; i++) out[i] = ((x[i] - mean) / std) * w[i] + b[i];
}

/** out = W x + b, with W stored as PyTorch nn.Linear does: [nOut, nIn], row-major. */
function linear(x, W, b, out, nIn, nOut) {
  for (let o = 0; o < nOut; o++) {
    let s = b[o];
    const row = o * nIn;
    for (let i = 0; i < nIn; i++) s += W[row + i] * x[i];
    out[o] = s;
  }
}

const GELU_C = Math.sqrt(2 / Math.PI);
function gelu(v) {
  return 0.5 * v * (1 + Math.tanh(GELU_C * (v + 0.044715 * v * v * v)));
}

export class GPT {
  /** @param config GPTConfig fields; @param tensors Map<string, Float32Array> (state_dict names) */
  constructor(config, tensors) {
    this.cfg = config;
    const get = (name) => {
      const t = tensors.get(name);
      if (!t) throw new Error(`missing tensor ${name}`);
      return t;
    };
    this.wte = get("wte.weight");
    this.wpe = get("wpe.weight");
    this.blocks = [];
    for (let l = 0; l < config.n_layer; l++) {
      const p = `blocks.${l}.`;
      this.blocks.push({
        ln1: [get(p + "ln1.weight"), get(p + "ln1.bias")],
        qkv: [get(p + "attn.qkv.weight"), get(p + "attn.qkv.bias")],
        attnProj: [get(p + "attn.proj.weight"), get(p + "attn.proj.bias")],
        ln2: [get(p + "ln2.weight"), get(p + "ln2.bias")],
        fc: [get(p + "mlp.fc.weight"), get(p + "mlp.fc.bias")],
        mlpProj: [get(p + "mlp.proj.weight"), get(p + "mlp.proj.bias")],
      });
    }
    this.lnF = [get("ln_f.weight"), get("ln_f.bias")];

    const d = config.d_model;
    this.dHead = d / config.n_head;
    this.x = new Float32Array(d);
    this.h = new Float32Array(d);
    this.qkv = new Float32Array(3 * d);
    this.attn = new Float32Array(d);
    this.tmp = new Float32Array(d);
    this.mlp = new Float32Array(config.d_mlp);
    this.scores = new Float64Array(config.n_ctx);
  }

  /** Build a model from manifest.json and the bytes of weights.bin. */
  static fromBuffers(manifest, buffer) {
    return new GPT(manifest.config, decodeTensors(manifest, buffer, "glassbox-weights"));
  }

  newCache() {
    const size = this.cfg.n_ctx * this.cfg.d_model;
    return {
      len: 0,
      k: this.blocks.map(() => new Float32Array(size)),
      v: this.blocks.map(() => new Float32Array(size)),
    };
  }

  /** Copy of the cache keeping only the first len positions (they do not depend on later ones). */
  cloneCache(cache, len = cache.len) {
    const d = this.cfg.d_model;
    const copy = this.newCache();
    for (let l = 0; l < this.blocks.length; l++) {
      copy.k[l].set(cache.k[l].subarray(0, len * d));
      copy.v[l].set(cache.v[l].subarray(0, len * d));
    }
    copy.len = len;
    return copy;
  }

  /**
   * Feed one token at position cache.len. Returns the next-token logits (a new Float32Array)
   * when wantLogits is true, otherwise null (saves the unembedding during prompt prefill).
   *
   * hooks: optional {layer: fn(x, pos)} called with the residual stream entering block `layer`
   * (layer == n_layer: entering ln_f), like resid_hooks in model.py. fn may modify x in place.
   */
  step(cache, token, wantLogits = true, hooks = null) {
    const { d_model: d, n_head: H, d_mlp: dMlp, n_ctx: T, vocab_size: V, ln_eps: eps } = this.cfg;
    const pos = cache.len;
    if (pos >= T) throw new Error(`KV cache is full (${T} tokens)`);
    if (!(token >= 0 && token < V)) throw new Error(`token id ${token} out of range`);
    const { x, h, qkv, attn, tmp, mlp, scores } = this;
    const dh = this.dHead;
    const scale = 1 / Math.sqrt(dh);

    for (let i = 0; i < d; i++) x[i] = this.wte[token * d + i] + this.wpe[pos * d + i];

    for (let l = 0; l < this.blocks.length; l++) {
      hooks?.[l]?.(x, pos);
      const blk = this.blocks[l];
      const K = cache.k[l];
      const Vc = cache.v[l];

      // attention
      layerNorm(x, blk.ln1[0], blk.ln1[1], eps, h);
      linear(h, blk.qkv[0], blk.qkv[1], qkv, d, 3 * d);
      K.set(qkv.subarray(d, 2 * d), pos * d);
      Vc.set(qkv.subarray(2 * d, 3 * d), pos * d);
      for (let hd = 0; hd < H; hd++) {
        const off = hd * dh;
        let max = -Infinity;
        for (let j = 0; j <= pos; j++) {
          let s = 0;
          const kj = j * d + off;
          for (let i = 0; i < dh; i++) s += qkv[off + i] * K[kj + i];
          s *= scale;
          scores[j] = s;
          if (s > max) max = s;
        }
        let sum = 0;
        for (let j = 0; j <= pos; j++) {
          scores[j] = Math.exp(scores[j] - max);
          sum += scores[j];
        }
        for (let i = 0; i < dh; i++) {
          let s = 0;
          for (let j = 0; j <= pos; j++) s += scores[j] * Vc[j * d + off + i];
          attn[off + i] = s / sum;
        }
      }
      linear(attn, blk.attnProj[0], blk.attnProj[1], tmp, d, d);
      for (let i = 0; i < d; i++) x[i] += tmp[i];

      // MLP
      layerNorm(x, blk.ln2[0], blk.ln2[1], eps, h);
      linear(h, blk.fc[0], blk.fc[1], mlp, d, dMlp);
      for (let i = 0; i < dMlp; i++) mlp[i] = gelu(mlp[i]);
      linear(mlp, blk.mlpProj[0], blk.mlpProj[1], tmp, dMlp, d);
      for (let i = 0; i < d; i++) x[i] += tmp[i];
    }
    hooks?.[this.blocks.length]?.(x, pos);
    cache.len = pos + 1;
    if (!wantLogits) return null;

    layerNorm(x, this.lnF[0], this.lnF[1], eps, h);
    const logits = new Float32Array(V);
    for (let v = 0; v < V; v++) {
      let s = 0;
      const row = v * d;
      for (let i = 0; i < d; i++) s += this.wte[row + i] * h[i];
      logits[v] = s;
    }
    return logits;
  }
}
