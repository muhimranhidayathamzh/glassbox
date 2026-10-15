// Which SAE features made the model predict a token? Exact answer by ablation, mirroring
// ablation_effects in python/glassbox/attribution.py: remove act * W_dec[f] / scale from the
// residual stream at the feature's position and measure the drop in the target token's logit.

import { featureVector } from "./sae.js";
import { softmax } from "./sampling.js";

/**
 * ids[0..t] are read and the next token y is predicted at position t. Candidates are the features
 * active at positions max(1, t - window + 1)..t. Returns {clean, prob, effects: [{pos, feature,
 * act, effect}]} where clean is y's logit, prob its probability, and effect = clean - ablated logit.
 */
export function ablationEffects(model, sae, ids, t, y, window = 8) {
  if (t < 1 || t >= ids.length || t >= model.cfg.n_ctx) throw new Error("bad target position");
  const layer = sae.cfg.layer;
  const first = Math.max(1, t - window + 1);
  const cands = [];
  const cache = model.newCache();
  let logits = null;
  for (let p = 0; p <= t; p++) {
    const hooks = p >= first
      ? { [layer]: (x) => {
          const c = sae.encode(x);
          c.ids.forEach((f, i) => cands.push({ pos: p, feature: f, act: c.acts[i] }));
        } }
      : null;
    logits = model.step(cache, ids[p], p === t, hooks);
  }
  const clean = logits[y];
  const prob = softmax(logits)[y];

  const effects = cands.map(({ pos, feature, act }) => {
    const d = featureVector(sae, feature);
    const c = model.cloneCache(cache, pos);
    const hooks = { [layer]: (x, q) => {
      if (q !== pos) return;
      for (let i = 0; i < x.length; i++) x[i] -= act * d[i];
    } };
    let out = null;
    for (let q = pos; q <= t; q++) out = model.step(c, ids[q], q === t, hooks);
    return { pos, feature, act, effect: clean - out[y] };
  });
  return { clean, prob, effects };
}
