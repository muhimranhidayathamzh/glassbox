import { generateIds, generateTokens } from "../engine/generate.js";
import { loadModel, loadSAE } from "../engine/load.js";
import { steeringHooks, steeringVector } from "../engine/sae.js";
import { topTokens } from "../engine/sampling.js";

const $ = (id) => document.getElementById(id);

const EXAMPLES = [
  "Once upon a time",
  "Lily and Ben were playing in the park. Suddenly,",
  "Tom had a red ball. He",
  "The little bird was sad because",
  "",
];

let engine = null;
let steering = null; // {sae, steering} once loaded
let steeringLoad = null;
let running = false;
let stopRequested = false;

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}

function bindSlider(id, format) {
  const input = $(id);
  const update = () => ($(`${id}-out`).textContent = format(Number(input.value)));
  input.addEventListener("input", update);
  update();
}

function quote(s) {
  return JSON.stringify(s).slice(1, -1);
}

function tokenTitle(tokenizer, id, prob, modelProbs) {
  const lines = [`"${quote(tokenizer.tokenStr(id))}"  id ${id}  p = ${(prob * 100).toFixed(1)}%`, "Top alternatives:"];
  for (const alt of topTokens(modelProbs, 5)) {
    lines.push(`  "${quote(tokenizer.tokenStr(alt.id))}"  ${(alt.prob * 100).toFixed(1)}%`);
  }
  return lines.join("\n");
}

/** Load the SAE and the steering presets the first time they are needed. */
function ensureSteering() {
  steeringLoad ??= loadSAE(new URL("../model/sae/", import.meta.url), engine.manifest, null,
                           { features: false, steering: true })
    .then((s) => {
      steering = s;
      $("steer-layer").textContent = String(s.steering.layer);
      const select = $("steer-feature");
      for (const p of s.steering.presets) {
        select.append(new Option(`${p.word} (feature #${p.feature})`, String(p.feature)));
      }
      return s;
    });
  return steeringLoad;
}

/** Steering hooks for the current controls, or null when steering is off. */
async function currentSteering() {
  const feature = $("steer-feature").value;
  const alpha = Number($("steer-alpha").value);
  if (feature === "" || alpha === 0) return null;
  const { sae, steering: cfg } = await ensureSteering();
  const preset = cfg.presets.find((p) => String(p.feature) === feature);
  return {
    hooks: steeringHooks(cfg.layer, steeringVector(sae, Number(feature), alpha * cfg.resid_norm)),
    note: `steered toward "${preset.word}" (feature #${feature}) at strength ${alpha}`,
  };
}

const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));

async function generate() {
  if (running || !engine) return;
  running = true;
  stopRequested = false;
  $("generate").hidden = true;
  $("stop").hidden = false;

  const { model, tokenizer } = engine;
  const promptText = $("prompt").value;
  // A story in the training data always follows <|endoftext|>, so start from it.
  const ids = [tokenizer.eotId, ...tokenizer.encode(promptText, { allowSpecial: false })];
  const opts = {
    temperature: Number($("temperature").value),
    topK: Number($("topk").value),
    maxNewTokens: Number($("maxtok").value),
    seed: Number($("seed").value) >>> 0,
    stopToken: tokenizer.eotId,
  };
  let steer = null;
  try {
    steer = await currentSteering();
  } catch (err) {
    setStatus(`Steering unavailable: ${err.message}`, true);
  }
  if (steer) opts.hooks = steer.hooks;

  const out = $("output");
  out.replaceChildren();
  if (promptText) {
    const p = document.createElement("span");
    p.className = "prompt";
    p.textContent = promptText;
    out.append(p);
  }

  const decoder = new TextDecoder();
  const t0 = performance.now();
  let lastFrame = t0;
  let count = 0;
  let ended = false;
  try {
    for (const ev of generateTokens(model, ids, opts)) {
      if (stopRequested) break;
      if (ev.type === "prefill") {
        setStatus(`Reading prompt\u2026 ${ev.done}/${ev.total} tokens`);
      } else {
        count++;
        if (ev.id === tokenizer.eotId) {
          ended = true;
          const end = document.createElement("span");
          end.className = "end";
          end.textContent = "end of story";
          out.append(end);
        } else {
          const span = document.createElement("span");
          span.className = "tok";
          span.textContent = decoder.decode(tokenizer.vocab[ev.id], { stream: true });
          span.style.setProperty("--surprise", (1 - ev.prob).toFixed(3));
          span.title = tokenTitle(tokenizer, ev.id, ev.prob, ev.modelProbs);
          out.append(span);
        }
        setStatus("Generating\u2026");
      }
      if (performance.now() - lastFrame > 30) {
        await nextFrame(); // let the page render and react to Stop
        lastFrame = performance.now();
      }
    }
    const tail = decoder.decode();
    if (tail) out.append(tail);
    const seconds = (performance.now() - t0) / 1000;
    const why = (stopRequested ? "stopped" : ended ? "model ended the story" : "reached max tokens") +
      (steer ? ` \u00b7 ${steer.note}` : "");
    const rate = seconds > 0.01 ? ` (${(count / seconds).toFixed(0)} tokens/s)` : "";
    $("stats").textContent = `${count} tokens in ${seconds.toFixed(2)} s${rate} \u00b7 ${why}`;
    setStatus("Ready.");
  } catch (err) {
    setStatus(`Error: ${err.message}`, true);
  } finally {
    running = false;
    $("generate").hidden = false;
    $("stop").hidden = true;
  }
}

/** ?selftest: greedy-decode a fixed prompt and print the ids, for automated browser checks. */
function selftest() {
  const { model, tokenizer } = engine;
  const prompt = [tokenizer.eotId, ...tokenizer.encode("Once upon a time")];
  const ids = generateIds(model, prompt, { maxNewTokens: 40, temperature: 0 });
  $("selftest").textContent = JSON.stringify({ ok: true, prompt, ids, text: tokenizer.decode(ids) });
}

async function init() {
  bindSlider("temperature", (v) => (v === 0 ? "0 (greedy)" : v.toFixed(2)));
  bindSlider("topk", (v) => (v === 0 ? "off" : String(v)));
  bindSlider("maxtok", String);
  bindSlider("steer-alpha", (v) => (v === 0 ? "0 (off)" : v.toFixed(2)));
  $("reseed").addEventListener("click", () => ($("seed").value = Math.floor(Math.random() * 1e6)));
  $("show-tokens").addEventListener("change", (e) => $("output").classList.toggle("show-tokens", e.target.checked));
  $("generate").addEventListener("click", generate);
  $("stop").addEventListener("click", () => (stopRequested = true));
  $("prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) generate();
  });
  for (const text of EXAMPLES) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = text || "(empty: new story)";
    chip.addEventListener("click", () => {
      $("prompt").value = text;
      generate();
    });
    $("examples").append(chip);
  }

  // Shareable settings: ?prompt=...&temperature=0.8&topk=50&maxtok=208&seed=3&steer=819&alpha=0.5&tokens&run
  const params = new URLSearchParams(location.search);
  if (params.has("prompt")) $("prompt").value = params.get("prompt");
  if (params.has("alpha")) params.set("steer-alpha", params.get("alpha"));
  for (const id of ["temperature", "topk", "maxtok", "seed", "steer-alpha"]) {
    if (params.has(id)) {
      $(id).value = params.get(id);
      $(id).dispatchEvent(new Event("input"));
    }
  }
  if (params.has("tokens")) {
    $("show-tokens").checked = true;
    $("output").classList.add("show-tokens");
  }
  const selftestMode = params.has("selftest");
  try {
    const t0 = performance.now();
    engine = await loadModel(new URL("../model/", import.meta.url), (got, total) => {
      $("progress").value = got / total;
      setStatus(`Downloading weights\u2026 ${(got / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`);
    });
    const m = engine.manifest;
    $("param-count").textContent = `${(m.n_params / 1e6).toFixed(1)}M`;
    $("progress").hidden = true;
    const c = m.config;
    setStatus(
      `Ready: ${m.n_params.toLocaleString("en")} parameters (${c.n_layer} layers, d_model ${c.d_model}, ` +
        `${c.n_head} heads, context ${c.n_ctx}), ${(m.weights_bytes / 1e6).toFixed(1)} MB ${m.dtype}, ` +
        `loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s.`,
    );
    $("generate").disabled = false;
    $("generate").textContent = "Generate";
    if (selftestMode) selftest();
    $("steer-panel").addEventListener("toggle", () => {
      if ($("steer-panel").open) ensureSteering().catch((err) => setStatus(`Steering unavailable: ${err.message}`, true));
    });
    if (params.has("steer")) {
      $("steer-panel").open = true;
      await ensureSteering();
      $("steer-feature").value = params.get("steer");
    }
    if (params.has("run")) await generate();
  } catch (err) {
    setStatus(`Could not load the model: ${err.message}`, true);
    if (selftestMode) $("selftest").textContent = JSON.stringify({ ok: false, error: err.message });
  }
}

init();
