import { ablationEffects } from "../engine/attribution.js";
import { loadModel, loadSAE } from "../engine/load.js";
import { featuresForTokens } from "../engine/sae.js";

const $ = (id) => document.getElementById(id);

const DEFAULT_TEXT =
  "Once upon a time, there was a little girl named Lily. She had a red ball that she loved very much. " +
  'One day, Lily went to the park with her mom. "Can I play on the swings?" she asked. Her mom smiled and said, ' +
  '"Yes, you can!"';

let model = null;
let tokenizer = null;
let sae = null;
let features = null; // features.json
let analysis = { ids: [], codes: [] };
let selectedPos = null;
let selectedFeature = null;

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}

function shown(s) {
  return s.replace(/\n/g, "\u21b5");
}

function quoted(id) {
  return JSON.stringify(tokenizer.tokenStr(id));
}

function percent(x) {
  return `${(x * 100).toPrecision(2)}%`;
}

function actAt(pos, f) {
  const c = analysis.codes[pos];
  if (!c) return 0;
  const i = c.ids.indexOf(f);
  return i === -1 ? 0 : c.acts[i];
}

/** Activation relative to the feature's strongest validation example (dead features: 1). */
function relative(f, act) {
  const info = features.features[f];
  return info.dead ? 1 : act / info.max_act;
}

function heat(el, fraction) {
  el.style.setProperty("--heat-alpha", Math.max(0, Math.min(1, fraction)).toFixed(3));
}

/** One clickable feature. With act: a bar relative to the feature's strongest validation example, and
 * the activation (or, with asPercent, that relative value) as a number. */
function featureRow(f, act = null, asPercent = false) {
  const info = features.features[f];
  const li = document.createElement("li");
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "feature-row";
  btn.addEventListener("click", () => selectFeature(f));
  const id = document.createElement("span");
  id.className = "fid";
  id.textContent = `#${f}`;
  const label = document.createElement("span");
  label.className = "flabel";
  label.textContent = info.dead ? "never fired on validation text" : info.label;
  const meta = document.createElement("span");
  meta.className = "fmeta";
  if (act !== null) {
    const rel = relative(f, act);
    const bar = document.createElement("span");
    bar.className = "bar";
    bar.style.setProperty("--w", Math.min(1, rel).toFixed(3));
    meta.append(bar, document.createTextNode(asPercent ? `${Math.round(rel * 100)}%` : act.toFixed(2)));
    meta.title = `activation ${act.toFixed(2)}, ${Math.round(rel * 100)}% of this feature's strongest validation example`;
  } else if (!info.dead) {
    const few = info.n_examples < 10 ? ` of ${info.n_examples}` : "";
    meta.textContent = `${Math.round(info.label_share * 100)}%${few} \u00b7 ${percent(info.density)}`;
  }
  btn.append(id, label, meta);
  li.append(btn);
  return li;
}

function renderTokens() {
  const box = $("tokens");
  box.replaceChildren();
  const info = selectedFeature !== null ? features.features[selectedFeature] : null;
  analysis.ids.forEach((id, pos) => {
    if (pos === 0) return; // start-of-story token, skipped
    const span = document.createElement("span");
    span.className = "tok";
    span.textContent = tokenizer.tokenStr(id);
    span.dataset.pos = pos;
    if (info && !info.dead) heat(span, actAt(pos, selectedFeature) / info.max_act);
    if (pos === selectedPos) span.classList.add("selected");
    box.append(span);
  });
  box.classList.toggle("feature-view", info !== null);
  $("tokens-legend").textContent = info
    ? `Shaded by feature #${selectedFeature} (${info.dead ? "dead" : info.label}). Click a token to see its features.`
    : "Click a token to see the features active on it.";
}

function renderActive() {
  const list = $("active");
  list.replaceChildren();
  if (selectedPos !== null) {
    const c = analysis.codes[selectedPos];
    $("active-title").textContent = `Features on ${quoted(analysis.ids[selectedPos])} (position ${selectedPos})`;
    if (!c.ids.length) list.append(Object.assign(document.createElement("li"), { textContent: "No active features." }));
    c.ids.forEach((f, i) => list.append(featureRow(f, c.acts[i])));
    return;
  }
  $("active-title").textContent = "Features most active in this text";
  const best = new Map();
  analysis.codes.forEach((c) => c.ids.forEach((f, i) => {
    const rel = relative(f, c.acts[i]);
    if (!best.has(f) || rel > best.get(f).rel) best.set(f, { rel, act: c.acts[i] });
  }));
  [...best.entries()].sort((a, b) => b[1].rel - a[1].rel).slice(0, 12)
    .forEach(([f, v]) => list.append(featureRow(f, v.act, true)));
}

function renderSearch() {
  const q = $("search").value.trim().toLowerCase();
  const list = $("results");
  list.replaceChildren();
  if (!q) return;
  let hits;
  if (/^\d+$/.test(q)) {
    hits = features.features.filter((f) => String(f.id) === q);
  } else {
    const score = (f) => {
      const tokens = [...f.top_cur, ...f.top_prev, ...f.top_next].map(([t]) => tokenizer.tokenStr(t).trim().toLowerCase());
      const main = tokenizer.tokenStr(f.label_token).trim().toLowerCase();
      if (main === q) return 3;
      if (main.includes(q)) return 2;
      return tokens.some((t) => t.includes(q)) ? 1 : 0;
    };
    hits = features.features
      .filter((f) => !f.dead)
      .map((f) => ({ f, s: score(f) }))
      .filter((h) => h.s > 0)
      .sort((a, b) => b.s - a.s || b.f.label_share - a.f.label_share)
      .map((h) => h.f);
  }
  if (!hits.length) list.append(Object.assign(document.createElement("li"), { textContent: "No matching features." }));
  hits.slice(0, 40).forEach((f) => list.append(featureRow(f.id)));
}

function renderDist(el, pairs) {
  el.replaceChildren();
  for (const [t, share] of pairs) {
    const li = document.createElement("li");
    li.textContent = `${quoted(t)} ${Math.round(share * 100)}%`;
    el.append(li);
  }
}

function renderFeature() {
  const panel = $("feature");
  if (selectedFeature === null) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const f = features.features[selectedFeature];
  $("feature-title").textContent = f.dead ? `Feature #${f.id}: dead` : `Feature #${f.id}: ${f.label}`;
  if (f.dead) {
    $("feature-meta").textContent = `Never fired on ${features.eval_tokens.toLocaleString("en")} validation tokens.`;
    ["dist-cur", "dist-prev", "dist-next", "examples"].forEach((id) => $(id).replaceChildren());
    return;
  }
  let inText = 0;
  let maxInText = 0;
  analysis.codes.forEach((c) => {
    const i = c.ids.indexOf(f.id);
    if (i !== -1) {
      inText++;
      maxInText = Math.max(maxInText, c.acts[i]);
    }
  });
  $("feature-meta").textContent =
    `Label agrees with ${Math.round(f.label_share * 100)}% of its ${f.n_examples} strongest examples. ` +
    `Fires on ${percent(f.density)} of validation tokens; strongest activation ${f.max_act.toFixed(2)}. ` +
    `In your text: ${inText} token${inText === 1 ? "" : "s"}${inText ? ` (max ${maxInText.toFixed(2)})` : ""}.`;
  renderDist($("dist-cur"), f.top_cur);
  renderDist($("dist-prev"), f.top_prev);
  renderDist($("dist-next"), f.top_next);
  const list = $("examples");
  list.replaceChildren();
  for (const ex of f.examples) {
    const li = document.createElement("li");
    ex.tokens.forEach((t, i) => {
      const span = document.createElement("span");
      span.className = "tok";
      if (t === tokenizer.eotId) {
        span.classList.add("eot");
        span.textContent = "\u00b6 ";
        span.title = "<|endoftext|>: boundary between stories";
      } else {
        span.textContent = shown(tokenizer.tokenStr(t));
      }
      heat(span, ex.acts[i] / 100);
      if (i === ex.at) span.classList.add("peak");
      li.append(span);
    });
    list.append(li);
  }
}

const SVG = "http://www.w3.org/2000/svg";
const CIRCUIT_WINDOW = 8;
const CIRCUIT_FEATURES = 8;

function svg(tag, attrs, text) {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text !== undefined) el.textContent = text;
  return el;
}

function clip(s, n) {
  return s.length > n ? `${s.slice(0, n - 1)}\u2026` : s;
}

function drawCircuit(q, result) {
  const { effects, prob } = result;
  const top = [...effects].sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect)).slice(0, CIRCUIT_FEATURES);
  const positions = [...new Set(top.map((e) => e.pos))].sort((a, b) => a - b);
  const rowH = 34;
  const rows = Math.max(positions.length, top.length, 1);
  const H = rows * rowH + 16;
  const W = 760;
  const yOf = (i, n) => 8 + (H - 16) * (i + 0.5) / n;
  const graph = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "circuit", role: "img",
                             "aria-label": "Features that led to the selected token" });
  const maxEffect = Math.max(...top.map((e) => Math.abs(e.effect)), 1e-9);
  const tokenY = new Map(positions.map((p, i) => [p, yOf(i, positions.length)]));
  const targetY = H / 2;
  top.forEach((e, i) => {
    const fy = yOf(i, top.length);
    graph.append(svg("line", { x1: 170, y1: tokenY.get(e.pos), x2: 232, y2: fy, class: "edge-active" }));
    graph.append(svg("line", { x1: 560, y1: fy, x2: 628, y2: targetY,
                               class: e.effect >= 0 ? "edge-plus" : "edge-minus",
                               "stroke-width": (1 + 7 * Math.abs(e.effect) / maxEffect).toFixed(2) }));
  });
  for (const p of positions) {
    graph.append(svg("text", { x: 164, y: tokenY.get(p) + 5, "text-anchor": "end", class: "node-token" },
                     `${clip(JSON.stringify(tokenizer.tokenStr(analysis.ids[p])), 14)} @${p}`));
  }
  top.forEach((e, i) => {
    const fy = yOf(i, top.length);
    const info = features.features[e.feature];
    const g = svg("g", { class: "node-feature", tabindex: 0, role: "button" });
    g.append(svg("rect", { x: 232, y: fy - 13, width: 328, height: 26, rx: 6 }));
    const name = clip(`#${e.feature} ${info.dead ? "(no label)" : info.label}`, 29);
    g.append(svg("text", { x: 242, y: fy + 5 }, `${name} @${e.pos}`));
    g.append(svg("text", { x: 552, y: fy + 5, "text-anchor": "end", class: "node-effect" },
                 `${e.effect >= 0 ? "+" : ""}${e.effect.toFixed(2)}`));
    g.addEventListener("click", () => selectFeature(e.feature));
    g.addEventListener("keydown", (ev) => ev.key === "Enter" && selectFeature(e.feature));
    graph.append(g);
  });
  const target = svg("g", { class: "node-target" });
  target.append(svg("rect", { x: 628, y: targetY - 24, width: 128, height: 48, rx: 8 }));
  target.append(svg("text", { x: 692, y: targetY - 3, "text-anchor": "middle" },
                    clip(JSON.stringify(tokenizer.tokenStr(analysis.ids[q])), 14)));
  target.append(svg("text", { x: 692, y: targetY + 15, "text-anchor": "middle", class: "node-effect" },
                    `p = ${(prob * 100).toFixed(1)}%`));
  graph.append(target);
  $("circuit-graph").replaceChildren(graph);
  return top.length;
}

async function renderCircuit() {
  const panel = $("circuit");
  const q = selectedPos;
  if (q === null) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  $("circuit-title").textContent = `Why the model wrote ${quoted(analysis.ids[q])} (position ${q})`;
  if (q < 2) {
    $("circuit-meta").textContent =
      "The first token of a story is predicted from the start-of-story token alone, which the SAE skips, " +
      "so there are no features to show.";
    $("circuit-graph").replaceChildren();
    return;
  }
  $("circuit-meta").textContent = "Computing\u2026";
  $("circuit-graph").replaceChildren();
  await new Promise((r) => setTimeout(r, 0)); // let the page paint first
  if (selectedPos !== q) return;
  const t0 = performance.now();
  const result = ablationEffects(model, sae, analysis.ids, q - 1, analysis.ids[q], CIRCUIT_WINDOW);
  if (selectedPos !== q) return;
  const shown = drawCircuit(q, result);
  $("circuit-meta").textContent =
    `Predicted at position ${q - 1} with probability ${(result.prob * 100).toFixed(1)}%. ` +
    `Showing the ${shown} largest of ${result.effects.length} features active in the last ` +
    `${Math.min(CIRCUIT_WINDOW, q - 1)} positions (change in the token's logit when removed); ` +
    `computed in ${((performance.now() - t0) / 1000).toFixed(2)} s.`;
}

function selectFeature(f) {
  selectedFeature = f;
  renderTokens();
  renderFeature();
  $("feature").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function analyze() {
  const all = [tokenizer.eotId, ...tokenizer.encode($("text").value, { allowSpecial: false })];
  const ids = all.slice(0, model.cfg.n_ctx);
  $("analyze-note").textContent =
    all.length > ids.length ? `Only the first ${ids.length - 1} tokens fit in the model's context.` : "";
  const t0 = performance.now();
  analysis = { ids, codes: featuresForTokens(model, sae, ids) };
  selectedPos = null;
  renderCircuit();
  setStatus(`Analyzed ${ids.length - 1} tokens in ${((performance.now() - t0) / 1000).toFixed(2)} s.`);
  renderTokens();
  renderActive();
  renderFeature();
}

async function init() {
  $("text").value = DEFAULT_TEXT;
  $("analyze").addEventListener("click", analyze);
  $("text").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyze();
  });
  $("search").addEventListener("input", renderSearch);
  $("tokens").addEventListener("click", (e) => {
    const pos = e.target.dataset?.pos;
    if (pos === undefined) return;
    selectedPos = Number(pos) === selectedPos ? null : Number(pos);
    renderTokens();
    renderActive();
    renderCircuit();
  });
  try {
    const progress = (got, total) => {
      $("progress").value = got / total;
      setStatus(`Downloading\u2026 ${(got / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`);
    };
    const m = await loadModel(new URL("../model/", import.meta.url), progress);
    ({ model, tokenizer } = m);
    const s = await loadSAE(new URL("../model/sae/", import.meta.url), m.manifest, progress);
    ({ sae, features } = s);
    $("progress").hidden = true;
    $("n-features").textContent = sae.cfg.n_features.toLocaleString("en");
    $("sae-layer").textContent = String(sae.cfg.layer);
    const dead = features.features.filter((f) => f.dead).length;
    setStatus(`Ready: SAE with ${sae.cfg.n_features} features (k = ${sae.cfg.k}, ${dead} dead) on layer ${sae.cfg.layer}.`);
    $("analyze").disabled = false;
    $("analyze").textContent = "Analyze";
    const params = new URLSearchParams(location.search);
    if (params.has("text")) $("text").value = params.get("text");
    analyze();
    if (params.has("feature")) selectFeature(Number(params.get("feature")));
    if (params.has("q")) {
      $("search").value = params.get("q");
      renderSearch();
    }
  } catch (err) {
    setStatus(`Could not load: ${err.message}`, true);
  }
}

init();
