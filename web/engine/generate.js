// Token generation with a KV cache. Follows GPT.generate in python/glassbox/model.py exactly,
// including the context rule: once the sequence would exceed n_ctx tokens, the window restarts
// from the last n_ctx / 2 tokens (the cache is rebuilt from those).

import { mulberry32, sampleToken } from "./sampling.js";

/**
 * Generate up to maxNewTokens after promptIds. Yields events:
 *   {type: "prefill", done, total}                    progress while reading the prompt
 *   {type: "token", id, prob, modelProbs, rebuilt}    one generated token
 * Stops early after emitting stopToken (if given). hooks ({layer: fn(x, pos)}) are applied at every
 * step, e.g. to steer; pos is the position in the current window, as in Python.
 */
export function* generateTokens(
  model,
  promptIds,
  { maxNewTokens = 200, temperature = 0.8, topK = 50, seed = 0, stopToken = undefined, hooks = null } = {},
) {
  if (!promptIds.length) throw new Error("prompt must contain at least one token");
  const nCtx = model.cfg.n_ctx;
  const keep = nCtx >> 1;
  const rng = mulberry32(seed);
  const tokens = [...promptIds];
  let start = 0;
  let cache;

  function* rebuild() {
    cache = model.newCache();
    let logits = null;
    const total = tokens.length - start;
    for (let i = start; i < tokens.length; i++) {
      logits = model.step(cache, tokens[i], i === tokens.length - 1, hooks);
      const done = i - start + 1;
      if (done % 32 === 0 && done < total) yield { type: "prefill", done, total };
    }
    return logits;
  }

  if (tokens.length - start > nCtx) start = tokens.length - keep;
  let logits = yield* rebuild();
  let rebuilt = false;
  for (let n = 0; n < maxNewTokens; n++) {
    const { id, prob, modelProbs } = sampleToken(logits, { temperature, topK, rng });
    tokens.push(id);
    yield { type: "token", id, prob, modelProbs, rebuilt };
    if (id === stopToken || n === maxNewTokens - 1) return;
    rebuilt = tokens.length - start > nCtx;
    if (rebuilt) {
      start = tokens.length - keep;
      logits = yield* rebuild();
    } else {
      logits = model.step(cache, id, true, hooks);
    }
  }
}

/** Convenience: run to completion and return the generated ids. */
export function generateIds(model, promptIds, options) {
  const out = [];
  for (const ev of generateTokens(model, promptIds, options)) if (ev.type === "token") out.push(ev.id);
  return out;
}
