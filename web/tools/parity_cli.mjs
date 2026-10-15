// Bridge for the Python parity tests and reports:  node web/tools/parity_cli.mjs request.json
//
// Request fields (all optional except model_dir):
//   model_dir   directory with manifest.json, weights.bin, tokenizer.json
//   sequences   [[ids]]  next-token logits at every position, fed one token at a time through
//               the KV cache, written as float32 (sequence, position, vocab) to logits_out
//   logits_out  path for those logits
//   encode      [text]   token ids, special tokens allowed
//   decode      [[ids]]  text
//   greedy      [{prompt: [ids], n, steer?: {feature, coef}}]  greedy continuation, same context
//               rule as Python; with steer, coef * unit(W_dec[feature]) is added at the SAE's layer
//               at positions >= 1 (needs sae_dir)
//   bench       {prompt_tokens, new_tokens, repeats}  tokens/s for prefill and generation
//   sae_dir     directory with an exported SAE (needed by `features`)
//   features    [[ids]]  at every position: the residual stream entering the SAE's layer
//               (float32 to resid_out), the SAE pre-activations (float32 to pre_out), and the
//               active features as JSON [ids, acts]
//   attribution [{ids, t, y, window}]  ablation effect of each candidate feature on y's logit at t
//               (needs sae_dir)
// The response is JSON on stdout.

import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { generateIds } from "../engine/generate.js";
import { GPT } from "../engine/model.js";
import { ablationEffects } from "../engine/attribution.js";
import { SAE, steeringHooks, steeringVector } from "../engine/sae.js";
import { mulberry32, sampleToken } from "../engine/sampling.js";
import { Tokenizer } from "../engine/tokenizer.js";

function readExport(dir) {
  const manifest = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
  const bytes = readFileSync(join(dir, manifest.weights_file));
  if (createHash("sha256").update(bytes).digest("hex") !== manifest.weights_sha256) {
    throw new Error(`${dir}: weights.bin does not match the sha256 in manifest.json`);
  }
  return { manifest, buffer: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) };
}

function loadFromDir(dir) {
  const { manifest, buffer } = readExport(dir);
  const tokenizer = new Tokenizer(JSON.parse(readFileSync(join(dir, manifest.tokenizer_file), "utf8")));
  return { model: GPT.fromBuffers(manifest, buffer), tokenizer, manifest };
}

function writeFloat32(path, arrays) {
  const flat = new Float32Array(arrays.reduce((n, a) => n + a.length, 0));
  let o = 0;
  for (const a of arrays) {
    flat.set(a, o);
    o += a.length;
  }
  writeFileSync(path, new Uint8Array(flat.buffer));
}

function allLogits(model, ids) {
  const cache = model.newCache();
  return ids.map((id) => model.step(cache, id, true));
}

function bench(model, { prompt_tokens = 128, new_tokens = 128, repeats = 3 }) {
  const rng = mulberry32(1);
  const prompt = Array.from({ length: prompt_tokens }, () => Math.floor(rng() * model.cfg.vocab_size));
  const runs = [];
  for (let r = 0; r < repeats + 1; r++) {
    const cache = model.newCache();
    let t0 = performance.now();
    let logits = null;
    prompt.forEach((id, i) => (logits = model.step(cache, id, i === prompt.length - 1)));
    const prefillMs = performance.now() - t0;
    t0 = performance.now();
    for (let i = 0; i < new_tokens; i++) {
      const { id } = sampleToken(logits, { temperature: 0 });
      logits = model.step(cache, id, true);
    }
    const genMs = performance.now() - t0;
    if (r > 0) runs.push({ prefillMs, genMs }); // first run is warm-up
  }
  const median = (xs) => xs.sort((a, b) => a - b)[Math.floor(xs.length / 2)];
  return {
    prompt_tokens,
    new_tokens,
    repeats,
    prefill_tokens_per_s: prompt_tokens / (median(runs.map((r) => r.prefillMs)) / 1000),
    generate_tokens_per_s: new_tokens / (median(runs.map((r) => r.genMs)) / 1000),
    runtime: `node ${process.version}`,
  };
}

function features(model, sae, sequences) {
  const layer = sae.cfg.layer;
  const resids = [];
  const pres = [];
  const codes = sequences.map((ids) => {
    const cache = model.newCache();
    return ids.map((id) => {
      let resid = null;
      model.step(cache, id, false, { [layer]: (x) => (resid = Float32Array.from(x)) });
      resids.push(resid);
      pres.push(sae.encodePre(resid));
      const c = sae.encode(resid);
      return [Array.from(c.ids), Array.from(c.acts)];
    });
  });
  return { resids, pres, codes };
}

function main() {
  const req = JSON.parse(readFileSync(process.argv[2], "utf8"));
  const { model, tokenizer } = loadFromDir(req.model_dir);
  const res = {};
  if (req.sequences) {
    const rows = [];
    res.logits_shapes = req.sequences.map((ids) => {
      const seqRows = allLogits(model, ids);
      rows.push(...seqRows);
      return [seqRows.length, model.cfg.vocab_size];
    });
    writeFloat32(req.logits_out, rows);
  }
  if (req.encode) res.encoded = req.encode.map((t) => tokenizer.encode(t));
  if (req.decode) res.decoded = req.decode.map((ids) => tokenizer.decode(ids));
  let sae = null;
  if (req.sae_dir) {
    const { manifest, buffer } = readExport(req.sae_dir);
    sae = SAE.fromBuffers(manifest, buffer);
  }
  if (req.greedy) {
    res.greedy = req.greedy.map(({ prompt, n, steer }) => {
      const hooks = steer ? steeringHooks(sae.cfg.layer, steeringVector(sae, steer.feature, steer.coef)) : null;
      return generateIds(model, prompt, { maxNewTokens: n, temperature: 0, hooks });
    });
  }
  if (req.bench) res.bench = bench(model, req.bench);
  if (req.attribution) {
    res.attribution = req.attribution.map(({ ids, t, y, window }) => ablationEffects(model, sae, ids, t, y, window));
  }
  if (req.features) {
    const out = features(model, sae, req.features);
    writeFloat32(req.resid_out, out.resids);
    writeFloat32(req.pre_out, out.pres);
    res.codes = out.codes;
  }
  process.stdout.write(JSON.stringify(res));
}

main();
