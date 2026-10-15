// Browser loaders for the files written by python/glassbox/export.py and features.py.

import { GPT } from "./model.js";
import { SAE } from "./sae.js";
import { Tokenizer } from "./tokenizer.js";

async function fetchOk(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res;
}

async function fetchBytes(url, expectedBytes, onProgress) {
  const res = await fetchOk(url);
  if (!res.body) return res.arrayBuffer();
  const out = new Uint8Array(expectedBytes);
  const reader = res.body.getReader();
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (received + value.length > expectedBytes) throw new Error(`${url} is larger than expected`);
    out.set(value, received);
    received += value.length;
    onProgress?.(received, expectedBytes);
  }
  if (received !== expectedBytes) throw new Error(`${url}: got ${received} of ${expectedBytes} bytes`);
  return out.buffer;
}

async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

/** manifest.json + weights.bin from a directory URL, with size and (when available) sha256 checks. */
async function loadExport(baseUrl, onProgress) {
  const base = new URL(baseUrl, location.href);
  const manifest = await (await fetchOk(new URL("manifest.json", base))).json();
  const buffer = await fetchBytes(new URL(manifest.weights_file, base), manifest.weights_bytes, onProgress);
  if (globalThis.crypto?.subtle && (await sha256Hex(buffer)) !== manifest.weights_sha256) {
    throw new Error(`${base}weights.bin does not match the sha256 in manifest.json`);
  }
  return { base, manifest, buffer };
}

/** Load the exported model. onProgress(receivedBytes, totalBytes) reports the weights download. */
export async function loadModel(baseUrl, onProgress) {
  const { base, manifest, buffer } = await loadExport(baseUrl, onProgress);
  const tokJson = await (await fetchOk(new URL(manifest.tokenizer_file, base))).json();
  return { model: GPT.fromBuffers(manifest, buffer), tokenizer: new Tokenizer(tokJson), manifest };
}

/**
 * Load an exported SAE plus the requested companion files (features.json from features.py,
 * steering.json from steer.py). Refuses files made for a different model or SAE.
 */
export async function loadSAE(baseUrl, modelManifest, onProgress, { features = true, steering = false } = {}) {
  const { base, manifest, buffer } = await loadExport(baseUrl, onProgress);
  if (manifest.model_weights_sha256 !== modelManifest.weights_sha256) {
    throw new Error("this SAE was exported for a different model");
  }
  const out = { sae: SAE.fromBuffers(manifest, buffer), manifest };
  for (const [name, wanted] of [["features", features], ["steering", steering]]) {
    if (!wanted) continue;
    const json = await (await fetchOk(new URL(`${name}.json`, base))).json();
    if (json.sae_weights_sha256 !== manifest.weights_sha256) {
      throw new Error(`${name}.json was computed for a different SAE`);
    }
    out[name] = json;
  }
  return out;
}
