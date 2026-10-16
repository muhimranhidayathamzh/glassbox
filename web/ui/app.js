import { generateIds, generateTokens } from "../engine/generate.js";
import { NameGuard, sentenceStart } from "../engine/guard.js";
import { loadGuard, loadModel, loadSAE } from "../engine/load.js";
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
let saeLoad = null; // Promise of {sae, steering}: the default SAE, for steering
let guardLoad = null; // Promise of {sae, nameGuard}: the SAE named by guard.json
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

/** Load the default SAE and the steering presets the first time they are needed. */
function ensureSAE() {
  saeLoad ??= loadSAE(new URL("../model/sae/", import.meta.url), engine.manifest, null,
                      { features: false, steering: true })
    .then((s) => {
      $("steer-layer").textContent = String(s.steering.layer);
      const select = $("steer-feature");
      for (const p of s.steering.presets) {
        select.append(new Option(`${p.word} (feature #${p.feature})`, String(p.feature)));
      }
      return s;
    });
  return saeLoad;
}

/** Steering for the current controls, or null when steering is off. */
async function currentSteering() {
  const feature = $("steer-feature").value;
  const alpha = Number($("steer-alpha").value);
  if (feature === "" || alpha === 0) return null;
  const { sae, steering: cfg } = await ensureSAE();
  const preset = cfg.presets.find((p) => String(p.feature) === feature);
  return {
    layer: cfg.layer,
    hooks: steeringHooks(cfg.layer, steeringVector(sae, Number(feature), alpha * cfg.resid_norm)),
    note: `steered toward "${preset.word}" (feature #${feature}) at strength ${alpha}`,
  };
}

/** Load the name guard and the SAE it reads the first time it is needed. */
function ensureGuard() {
  guardLoad ??= loadGuard(new URL("../model/", import.meta.url), engine.manifest, null)
    .then(({ json, sae }) => ({ sae, nameGuard: new NameGuard(json) }));
  return guardLoad;
}

/** Name-guard state for the current controls, or null when the guard is off. */
async function currentGuard() {
  const mode = $("guard-mode").value;
  if (mode === "off") return null;
  const { sae, nameGuard } = await ensureGuard();
  return { sae, guard: nameGuard, mode, threshold: nameGuard.thresholds[mode], wordCheck: $("guard-word").checked,
           resid: null, pos: 0 };
}

/** Hooks that steer (if on) and record the residual the guard reads (if on), steering first. */
function combinedHooks(steer, guardState) {
  if (!steer && !guardState) return null;
  const layers = new Set([steer?.layer, guardState?.guard.layer].filter((l) => l !== undefined));
  const hooks = {};
  for (const layer of layers) {
    hooks[layer] = (x, pos) => {
      if (steer?.layer === layer) steer.hooks[layer](x, pos);
      if (guardState?.guard.layer === layer) {
        guardState.resid = x.slice();
        guardState.pos = pos;
      }
    };
  }
  return hooks;
}

function tokenSpan(text, ev, tokenizer) {
  const span = document.createElement("span");
  span.className = "tok";
  span.textContent = text;
  span.style.setProperty("--surprise", (1 - ev.prob).toFixed(3));
  span.title = tokenTitle(tokenizer, ev.id, ev.prob, ev.modelProbs);
  return span;
}

/**
 * Holds tokens until their word is complete, then shows the word or hides it. A word is hidden when
 * the guard predicted, before its first token was chosen, that a name was coming; or, with the word
 * check on, when the finished word is a known name or capitalized mid-sentence.
 */
class WordGate {
  constructor(out, tokenizer, g, context) {
    Object.assign(this, { out, tokenizer, g, raw: context, pending: [] });
    this.hidden = { before: 0, word: 0, both: 0 };
  }

  push(ev, decision) {
    const text = this.tokenizer.tokenStr(ev.id);
    const continues = this.pending.length && /^[A-Za-z]/.test(text) && /[A-Za-z]$/.test(this.pending.at(-1).text);
    if (!continues) this.flush();
    this.pending.push({ ev, text, ...decision });
  }

  flush() {
    if (!this.pending.length) return;
    const text = this.pending.map((p) => p.text).join("");
    const lead = text.match(/^[\s"']*/)[0];
    const word = (/^[A-Za-z]+/.exec(text.slice(lead.length)) ?? [""])[0];
    const atStart = sentenceStart(this.raw + text, this.raw.length + lead.length);
    const before = this.pending[0].flagged;
    const byWord = this.g.wordCheck && this.g.guard.wordFlag(word, atStart);
    if (before || byWord) {
      this.hidden[before && byWord ? "both" : before ? "before" : "word"]++;
      const span = document.createElement("span");
      span.className = "redacted";
      span.textContent = lead + "█".repeat(Math.max(1, text.length - lead.length));
      const reasons = [];
      if (before) {
        const why = this.pending[0].explain
          .map((e) => `#${e.feature}${e.label ? ` ${e.label}` : ""} (+${e.contribution.toFixed(2)})`).join(", ");
        reasons.push(`Predicted before it was written (score ${this.pending[0].logit.toFixed(2)} >= ` +
                     `${this.g.threshold.toFixed(2)})${why ? `; strongest features: ${why}` : ""}`);
      }
      if (byWord) {
        reasons.push(`Word check: "${word}" ${this.g.guard.names.has(word) ? "is in the name list" : "is capitalized mid-sentence"}`);
      }
      span.title = `Hidden: ${quote(text.slice(lead.length))}\n${reasons.join("\n")}`;
      this.out.append(span);
    } else {
      for (const p of this.pending) this.out.append(tokenSpan(p.text, p.ev, this.tokenizer));
    }
    this.raw += text;
    this.pending = [];
  }

  summary() {
    const { before, word, both } = this.hidden;
    const n = before + word + both;
    return `name guard (${this.g.mode}) hid ${n} word${n === 1 ? "" : "s"}: ${before + both} predicted before ` +
      `they were written, ${word + both} by the word check (${both} by both)`;
  }
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
  let g = null;
  try {
    steer = await currentSteering();
    g = await currentGuard();
  } catch (err) {
    setStatus(`Steering / name guard unavailable: ${err.message}`, true);
  }
  opts.hooks = combinedHooks(steer, g);

  const out = $("output");
  out.replaceChildren();
  if (promptText) {
    const p = document.createElement("span");
    p.className = "prompt";
    p.textContent = promptText;
    out.append(p);
  }
  const gate = g ? new WordGate(out, tokenizer, g, promptText) : null;

  const decoder = new TextDecoder();
  const t0 = performance.now();
  let lastFrame = t0;
  let count = 0;
  let ended = false;
  try {
    for (const ev of generateTokens(model, ids, opts)) {
      if (stopRequested) break;
      if (ev.type === "prefill") {
        setStatus(`Reading prompt… ${ev.done}/${ev.total} tokens`);
      } else {
        count++;
        if (ev.id === tokenizer.eotId) {
          ended = true;
          gate?.flush();
          const end = document.createElement("span");
          end.className = "end";
          end.textContent = "end of story";
          out.append(end);
        } else if (gate) {
          // The guard reads the position whose logits produced this token, i.e. before it was chosen.
          const codes = g.pos === 0 ? { ids: [], acts: [] } : g.sae.encode(g.resid);
          const logit = g.guard.logit(codes);
          gate.push(ev, { logit, flagged: logit >= g.threshold, explain: g.guard.explain(codes) });
        } else {
          out.append(tokenSpan(decoder.decode(tokenizer.vocab[ev.id], { stream: true }), ev, tokenizer));
        }
        setStatus("Generating…");
      }
      if (performance.now() - lastFrame > 30) {
        await nextFrame(); // let the page render and react to Stop
        lastFrame = performance.now();
      }
    }
    gate?.flush();
    const tail = decoder.decode();
    if (tail) out.append(tail);
    const seconds = (performance.now() - t0) / 1000;
    const why = (stopRequested ? "stopped" : ended ? "model ended the story" : "reached max tokens") +
      (steer ? ` · ${steer.note}` : "") + (gate ? ` · ${gate.summary()}` : "");
    const rate = seconds > 0.01 ? ` (${(count / seconds).toFixed(0)} tokens/s)` : "";
    $("stats").textContent = `${count} tokens in ${seconds.toFixed(2)} s${rate} · ${why}`;
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

  // Shareable settings: ?prompt=...&temperature=0.8&topk=50&maxtok=208&seed=3&steer=819&alpha=0.5
  //                     &guard=balanced&guardword=0&tokens&run
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
      if ($("steer-panel").open) ensureSAE().catch((err) => setStatus(`Steering unavailable: ${err.message}`, true));
    });
    $("guard-panel").addEventListener("toggle", () => {
      if ($("guard-panel").open) ensureGuard().catch((err) => setStatus(`Name guard unavailable: ${err.message}`, true));
    });
    if (params.has("steer")) {
      $("steer-panel").open = true;
      await ensureSAE();
      $("steer-feature").value = params.get("steer");
    }
    // ?guard=balanced|strict, &guardword=0 to switch the word check off
    if (["balanced", "strict"].includes(params.get("guard"))) {
      $("guard-panel").open = true;
      $("guard-mode").value = params.get("guard");
      $("guard-word").checked = params.get("guardword") !== "0";
    }
    if (params.has("run")) await generate();
  } catch (err) {
    setStatus(`Could not load the model: ${err.message}`, true);
    if (selftestMode) $("selftest").textContent = JSON.stringify({ ok: false, error: err.message });
  }
}

init();
