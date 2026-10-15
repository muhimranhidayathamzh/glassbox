// Top-k sparse autoencoder encoder, mirroring TopKSAE in python/glassbox/sae.py:
//   pre = (x * scale - b_dec) @ W_enc + b_enc;  codes = ReLU(top-k(pre))

import { decodeTensors } from "./weights.js";

export class SAE {
  constructor(config, tensors) {
    this.cfg = config;
    const get = (name) => {
      const t = tensors.get(name);
      if (!t) throw new Error(`missing tensor ${name}`);
      return t;
    };
    this.W_enc = get("W_enc"); // [d_in, n_features], row-major
    this.b_enc = get("b_enc");
    this.W_dec = get("W_dec"); // [n_features, d_in]
    this.b_dec = get("b_dec");
    this.acc = new Float64Array(config.n_features);
  }

  static fromBuffers(manifest, buffer) {
    return new SAE(manifest.config, decodeTensors(manifest, buffer, "glassbox-sae"));
  }

  /** Dense pre-activations for one residual vector x (Float32Array of d_in). */
  encodePre(x) {
    const { d_in: d, n_features: F, scale } = this.cfg;
    const acc = this.acc;
    for (let f = 0; f < F; f++) acc[f] = this.b_enc[f];
    for (let i = 0; i < d; i++) {
      const xi = x[i] * scale - this.b_dec[i];
      const row = i * F;
      for (let f = 0; f < F; f++) acc[f] += xi * this.W_enc[row + f];
    }
    return Float32Array.from(acc);
  }

  /**
   * Active features for x: the k largest pre-activations, kept only if positive.
   * Returns {ids, acts} sorted by activation, strongest first.
   */
  encode(x) {
    const pre = this.encodePre(x);
    const k = this.cfg.k;
    const ids = [];
    const vals = [];
    for (let f = 0; f < pre.length; f++) {
      const v = pre[f];
      if (ids.length === k && v <= vals[k - 1]) continue;
      let j = Math.min(ids.length, k - 1);
      if (ids.length < k) {
        ids.push(f);
        vals.push(v);
      }
      while (j > 0 && vals[j - 1] < v) {
        ids[j] = ids[j - 1];
        vals[j] = vals[j - 1];
        j--;
      }
      ids[j] = f;
      vals[j] = v;
    }
    const n = vals.findIndex((v) => !(v > 0));
    const keep = n === -1 ? vals.length : n;
    return { ids: Int32Array.from(ids.slice(0, keep)), acts: Float32Array.from(vals.slice(0, keep)) };
  }
}

/**
 * SAE features at every position of tokenIds (at most n_ctx tokens), read from the residual
 * stream entering block sae.cfg.layer. Position 0 is skipped (empty result), as in training:
 * it has no context. Returns [{ids, acts}] per position.
 */
export function featuresForTokens(model, sae, tokenIds) {
  if (tokenIds.length > model.cfg.n_ctx) throw new Error(`at most ${model.cfg.n_ctx} tokens`);
  const cache = model.newCache();
  const empty = { ids: new Int32Array(0), acts: new Float32Array(0) };
  return tokenIds.map((id, pos) => {
    let codes = empty;
    const hooks = pos === 0 ? null : { [sae.cfg.layer]: (x) => (codes = sae.encode(x)) };
    model.step(cache, id, false, hooks);
    return codes;
  });
}

/** What one unit of feature f adds to the residual stream: W_dec[f] / scale (as in attribution.py). */
export function featureVector(sae, f) {
  const d = sae.cfg.d_in;
  const out = new Float32Array(d);
  for (let i = 0; i < d; i++) out[i] = sae.W_dec[f * d + i] / sae.cfg.scale;
  return out;
}

/** coef times the unit vector along feature f's decoder direction (as in steer.py). */
export function steeringVector(sae, f, coef) {
  const v = featureVector(sae, f);
  const norm = Math.hypot(...v);
  return v.map((x) => (coef * x) / norm);
}

/** Hooks that add vec to the residual stream entering block `layer`, at positions >= 1. */
export function steeringHooks(layer, vec) {
  return {
    [layer]: (x, pos) => {
      if (pos === 0) return;
      for (let i = 0; i < x.length; i++) x[i] += vec[i];
    },
  };
}
