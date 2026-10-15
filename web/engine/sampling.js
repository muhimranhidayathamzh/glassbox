// Seeded sampling. Semantics match GPT.generate in python/glassbox/model.py:
// temperature 0 = greedy (first maximum wins, like torch.argmax); top-k keeps every logit
// >= the k-th largest. The random numbers differ from torch, so only greedy output is
// expected to match Python token for token.

/** Small, fast, seedable PRNG (mulberry32). Returns floats in [0, 1). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

/** Softmax in float64 of logits / temperature. */
export function softmax(logits, temperature = 1) {
  const out = new Float64Array(logits.length);
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i] / temperature);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp(logits[i] / temperature - max);
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

/** The k most probable tokens as [{id, prob}], highest first. */
export function topTokens(probs, k) {
  const ids = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a] || a - b);
  return ids.slice(0, k).map((id) => ({ id, prob: probs[id] }));
}

/**
 * Pick the next token. Returns {id, prob} where prob is the model's own probability for the
 * chosen token (temperature 1, no top-k), which is what the UI displays.
 */
export function sampleToken(logits, { temperature = 1, topK = 0, rng = Math.random } = {}) {
  const modelProbs = softmax(logits);
  if (temperature === 0) {
    const id = argmax(logits);
    return { id, prob: modelProbs[id], modelProbs };
  }
  const scaled = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) scaled[i] = logits[i] / temperature;
  if (topK > 0 && topK < scaled.length) {
    const kth = Float64Array.from(scaled).sort()[scaled.length - topK];
    for (let i = 0; i < scaled.length; i++) if (scaled[i] < kth) scaled[i] = -Infinity;
  }
  const probs = softmax(scaled);
  const r = rng();
  let acc = 0;
  let id = probs.length - 1;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (r < acc) {
      id = i;
      break;
    }
  }
  while (probs[id] === 0) id--; // guard against rounding when r is close to 1
  return { id, prob: modelProbs[id], modelProbs };
}
