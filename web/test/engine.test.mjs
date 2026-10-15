// Unit tests for the JS engine that need no exported model:  node --test "web/test/*.test.mjs"
// Parity against PyTorch lives in tests/test_web_parity.py.

import assert from "node:assert/strict";
import { test } from "node:test";

import { halfToFloat } from "../engine/fp16.js";
import { generateIds } from "../engine/generate.js";
import { GPT } from "../engine/model.js";
import { ablationEffects } from "../engine/attribution.js";
import { SAE, featureVector, featuresForTokens, steeringHooks, steeringVector } from "../engine/sae.js";
import { argmax, mulberry32, sampleToken } from "../engine/sampling.js";
import { Tokenizer } from "../engine/tokenizer.js";

const CFG = { vocab_size: 20, n_ctx: 8, d_model: 8, n_layer: 2, n_head: 2, d_mlp: 16, ln_eps: 1e-5 };

function randomModel(cfg = CFG, seed = 0) {
  const rng = mulberry32(seed);
  const rand = (n) => Float32Array.from({ length: n }, () => (rng() - 0.5) * 2);
  const { vocab_size: V, n_ctx: T, d_model: d, d_mlp: m } = cfg;
  const t = new Map([
    ["wte.weight", rand(V * d)],
    ["wpe.weight", rand(T * d)],
    ["ln_f.weight", rand(d)],
    ["ln_f.bias", rand(d)],
  ]);
  for (let l = 0; l < cfg.n_layer; l++) {
    const p = `blocks.${l}.`;
    const shapes = {
      "ln1.weight": d, "ln1.bias": d, "attn.qkv.weight": 3 * d * d, "attn.qkv.bias": 3 * d,
      "attn.proj.weight": d * d, "attn.proj.bias": d, "ln2.weight": d, "ln2.bias": d,
      "mlp.fc.weight": m * d, "mlp.fc.bias": m, "mlp.proj.weight": d * m, "mlp.proj.bias": d,
    };
    for (const [name, n] of Object.entries(shapes)) t.set(p + name, rand(n));
  }
  return new GPT(cfg, t);
}

/** Reference: recompute every step from scratch, applying the same context rule. */
function naiveGreedy(model, prompt, n, hooks = null) {
  const tokens = [...prompt];
  const T = model.cfg.n_ctx;
  let start = 0;
  for (let k = 0; k < n; k++) {
    if (tokens.length - start > T) start = tokens.length - T / 2;
    const cache = model.newCache();
    let logits;
    for (let i = start; i < tokens.length; i++) logits = model.step(cache, tokens[i], true, hooks);
    tokens.push(argmax(logits));
  }
  return tokens.slice(prompt.length);
}

test("fp16 decoding of special values", () => {
  assert.equal(halfToFloat(0x0000), 0);
  assert.ok(Object.is(halfToFloat(0x8000), -0));
  assert.equal(halfToFloat(0x3c00), 1);
  assert.equal(halfToFloat(0xc000), -2);
  assert.equal(halfToFloat(0x3555), 0.333251953125);
  assert.equal(halfToFloat(0x7bff), 65504);
  assert.equal(halfToFloat(0x0001), 2 ** -24);
  assert.equal(halfToFloat(0x7c00), Infinity);
  assert.ok(Number.isNaN(halfToFloat(0x7e00)));
});

test("KV-cache generation equals recomputing from scratch, across context restarts", () => {
  const model = randomModel();
  const out = generateIds(model, [1, 2, 3], { maxNewTokens: 30, temperature: 0 });
  assert.equal(out.length, 30); // 3 + 30 tokens >> n_ctx = 8, so the window restarted
  assert.deepEqual(out, naiveGreedy(model, [1, 2, 3], 30));
});

test("long prompts start from the last n_ctx / 2 tokens, as in Python", () => {
  const model = randomModel();
  const prompt = Array.from({ length: 13 }, (_, i) => i % 20);
  const out = generateIds(model, prompt, { maxNewTokens: 5, temperature: 0 });
  assert.deepEqual(out, naiveGreedy(model, prompt, 5));
});

test("sampling is reproducible for a seed and stops at the stop token", () => {
  const model = randomModel();
  const opts = { maxNewTokens: 25, temperature: 1, topK: 0, seed: 7 };
  const a = generateIds(model, [4], opts);
  assert.deepEqual(a, generateIds(model, [4], opts));
  assert.notDeepEqual(a, generateIds(model, [4], { ...opts, seed: 8 }));
  const stopped = generateIds(model, [4], { ...opts, stopToken: a[2] });
  assert.ok(stopped.length <= 3 && stopped.at(-1) === a[2]);
});

test("sampleToken: greedy takes the first maximum, top-k 1 is greedy, frequencies follow softmax", () => {
  assert.equal(sampleToken(Float32Array.of(1, 3, 3, 2), { temperature: 0 }).id, 1);
  const rng = mulberry32(3);
  const logits = Float32Array.of(0.5, 2, -1, 1.9);
  for (let i = 0; i < 50; i++) assert.equal(sampleToken(logits, { topK: 1, rng }).id, 1);

  const counts = [0, 0, 0];
  const n = 30000;
  const l = Float32Array.of(0, Math.log(2), Math.log(3)); // probabilities 1/6, 2/6, 3/6
  for (let i = 0; i < n; i++) counts[sampleToken(l, { temperature: 1, rng }).id]++;
  [1 / 6, 2 / 6, 3 / 6].forEach((p, i) => assert.ok(Math.abs(counts[i] / n - p) < 0.012, `${i}: ${counts[i] / n}`));
});

test("model guards: full cache, bad token ids, bad weight buffers", () => {
  const model = randomModel();
  const cache = model.newCache();
  for (let i = 0; i < CFG.n_ctx; i++) model.step(cache, 0, false);
  assert.throws(() => model.step(cache, 0), /full/);
  assert.throws(() => model.step(model.newCache(), 20), /out of range/);
  const manifest = { format: "glassbox-weights", dtype: "float16", weights_bytes: 8, tensors: [], config: CFG };
  assert.throws(() => GPT.fromBuffers(manifest, new Uint8Array(8)), /ArrayBuffer/);
  assert.throws(() => GPT.fromBuffers(manifest, new ArrayBuffer(6)), /bytes/);
});

test("tokenizer: merges by priority, special tokens, byte fallback round trip", () => {
  const tok = new Tokenizer({
    type: "byte-level-bpe",
    vocab_size: 259,
    merges: [[104, 105], [32, 256]], // "hi" -> 256, " hi" -> 257
    special_tokens: { "<|endoftext|>": 258 },
    js_pretoken_pattern: " ?\\S+|\\s+",
  });
  assert.deepEqual(tok.encode("hi hi"), [256, 257]);
  assert.deepEqual(tok.encode("hi<|endoftext|>hi"), [256, 258, 256]);
  assert.ok(!tok.encode("hi<|endoftext|>", { allowSpecial: false }).includes(258));
  const text = "h\u00e9llo \u{1f642} hi";
  assert.equal(tok.decode(tok.encode(text)), text);
  assert.equal(tok.tokenStr(256), "hi");
  assert.equal(tok.tokenStr(0xc3), "\\xc3"); // lone UTF-8 lead byte
});

test("residual hooks see the stream entering each block and can change it", () => {
  const model = randomModel();
  const plain = model.step(model.newCache(), 3);
  const seen = {};
  const capture = Object.fromEntries(
    [0, 1, 2].map((l) => [l, (x, pos) => (seen[l] = { x: Float32Array.from(x), pos })]),
  );
  const hooked = model.step(model.newCache(), 3, true, capture);
  assert.deepEqual(hooked, plain);
  assert.equal(seen[0].pos, 0);
  // layer 0 is the embedding: wte[3] + wpe[0]
  const d = CFG.d_model;
  seen[0].x.forEach((v, i) => assert.equal(v, Math.fround(model.wte[3 * d + i] + model.wpe[i])));
  assert.notDeepEqual(seen[1].x, seen[0].x);
  const zeroed = model.step(model.newCache(), 3, true, { 1: (x) => x.fill(0) });
  assert.notDeepEqual(zeroed, plain);
});

test("SAE encode keeps the k largest positive pre-activations, strongest first", () => {
  const rng = mulberry32(5);
  const d = 6;
  const F = 40;
  const rand = (n) => Float32Array.from({ length: n }, () => rng() * 2 - 1);
  const tensors = new Map([["W_enc", rand(d * F)], ["b_enc", rand(F)], ["W_dec", rand(F * d)], ["b_dec", rand(d)]]);
  const sae = new SAE({ d_in: d, n_features: F, k: 5, layer: 1, scale: 0.7 }, tensors);
  for (let trial = 0; trial < 50; trial++) {
    const x = rand(d);
    const pre = sae.encodePre(x);
    // reference pre-activation and selection
    for (let f = 0; f < F; f++) {
      let s = tensors.get("b_enc")[f];
      for (let i = 0; i < d; i++) s += (x[i] * 0.7 - tensors.get("b_dec")[i]) * tensors.get("W_enc")[i * F + f];
      assert.ok(Math.abs(pre[f] - s) < 1e-5);
    }
    const order = Array.from(pre.keys()).sort((a, b) => pre[b] - pre[a] || a - b).slice(0, 5);
    const expected = order.filter((f) => pre[f] > 0);
    const { ids, acts } = sae.encode(x);
    assert.deepEqual(Array.from(ids), expected);
    assert.deepEqual(Array.from(acts), expected.map((f) => pre[f]));
  }
});

test("featuresForTokens skips position 0 and matches encoding the captured residual", () => {
  const model = randomModel();
  const rng = mulberry32(9);
  const d = CFG.d_model;
  const F = 30;
  const rand = (n) => Float32Array.from({ length: n }, () => rng() * 2 - 1);
  const tensors = new Map([["W_enc", rand(d * F)], ["b_enc", rand(F)], ["W_dec", rand(F * d)], ["b_dec", rand(d)]]);
  const sae = new SAE({ d_in: d, n_features: F, k: 4, layer: 1, scale: 1 }, tensors);
  const ids = [1, 5, 9, 2, 7];
  const out = featuresForTokens(model, sae, ids);
  assert.equal(out.length, ids.length);
  assert.equal(out[0].ids.length, 0);
  const cache = model.newCache();
  ids.forEach((id, pos) => {
    let resid = null;
    model.step(cache, id, false, { 1: (x) => (resid = Float32Array.from(x)) });
    if (pos > 0) assert.deepEqual(out[pos], sae.encode(resid));
  });
  assert.throws(() => featuresForTokens(model, sae, new Array(CFG.n_ctx + 1).fill(0)), /at most/);
});

function randomSAE(seed, F = 30, k = 4) {
  const rng = mulberry32(seed);
  const d = CFG.d_model;
  const rand = (n) => Float32Array.from({ length: n }, () => rng() * 2 - 1);
  const tensors = new Map([["W_enc", rand(d * F)], ["b_enc", rand(F)], ["W_dec", rand(F * d)], ["b_dec", rand(d)]]);
  return new SAE({ d_in: d, n_features: F, k, layer: 1, scale: 0.8 }, tensors);
}

test("cloneCache continues exactly like the original cache", () => {
  const model = randomModel();
  const cache = model.newCache();
  [1, 2, 3, 4].forEach((id) => model.step(cache, id, false));
  const copy = model.cloneCache(cache, 2);
  const fresh = model.newCache();
  [1, 2].forEach((id) => model.step(fresh, id, false));
  assert.deepEqual(model.step(copy, 9), model.step(fresh, 9));
  assert.equal(cache.len, 4); // the original is untouched
});

test("ablation effects equal a full rerun with the feature removed", () => {
  const model = randomModel();
  const sae = randomSAE(4);
  const ids = [3, 7, 1, 12, 5, 9, 2];
  const t = 6;
  const y = 11;
  const { clean, effects } = ablationEffects(model, sae, ids, t, y, 4);
  assert.ok(effects.length > 0 && effects.every((e) => e.pos >= 3 && e.pos <= t));
  for (const { pos, feature, act, effect } of effects) {
    const d = featureVector(sae, feature);
    const cache = model.newCache();
    let logits;
    ids.slice(0, t + 1).forEach((id, q) => {
      logits = model.step(cache, id, true, { 1: (x, p) => { if (p === pos) for (let i = 0; i < x.length; i++) x[i] -= act * d[i]; } });
    });
    assert.ok(Math.abs(clean - logits[y] - effect) < 1e-5);
  }
});

test("steering applies at positions >= 1 and survives context restarts", () => {
  const model = randomModel();
  const sae = randomSAE(5);
  const vec = steeringVector(sae, 3, 2.5);
  assert.ok(Math.abs(Math.hypot(...vec) - 2.5) < 1e-5);
  const hooks = steeringHooks(1, vec);
  const x0 = new Float32Array(CFG.d_model);
  hooks[1](x0, 0);
  assert.ok(x0.every((v) => v === 0));
  const steered = generateIds(model, [1, 2, 3], { maxNewTokens: 30, temperature: 0, hooks });
  assert.deepEqual(steered, naiveGreedy(model, [1, 2, 3], 30, hooks));
  assert.notDeepEqual(steered, generateIds(model, [1, 2, 3], { maxNewTokens: 30, temperature: 0 }));
});

test("name guard: logit, explanation and word check", async () => {
  const { NameGuard, sentenceStart } = await import("../engine/guard.js");
  const guard = new NameGuard({
    format: "glassbox-guard", layer: 1, bias: -1, threshold: 0, strict_threshold: -0.5,
    weights: [0.5, -2, 1.5], names: ["Lily"], exclude: ["I", "Mom"], labels: { 2: '" named"' },
  });
  const codes = { ids: Int32Array.of(0, 2, 1), acts: Float32Array.of(2, 1, 0.5) };
  assert.equal(guard.logit(codes), -1 + 1 + 1.5 - 1);
  assert.deepEqual(guard.explain(codes).map((p) => p.feature), [2, 0]);
  assert.equal(guard.explain(codes)[0].label, '" named"');
  assert.ok(guard.wordFlag("Lily", true));
  assert.ok(guard.wordFlag("Zed", false) && !guard.wordFlag("Zed", true));
  assert.ok(!guard.wordFlag("Mom", false) && !guard.wordFlag("cat", false) && !guard.wordFlag("", false));
  const t = 'Tom said, "Look at Sam!" Then';
  assert.deepEqual(["Tom", "Look", "Sam", "Then"].map((w) => sentenceStart(t, t.indexOf(w))), [true, true, false, true]);
});
