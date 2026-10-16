// Training replay: everything shown comes from web/model/replay.json (python -m glassbox.induction).

const $ = (id) => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";

let data = null;
let idx = 0;
let metric = "prefix";
let head = [0, 0];
let playTimer = null;
let sharedPrompt = ""; // the fixed prompt: the text every checkpoint's first sample starts with

function el(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

const fmt = (v, d = 3) => v.toFixed(d);
const fmtStep = (s) => (s >= 1000 ? `${+(s / 1000).toFixed(s % 1000 ? 1 : 0)}K` : String(s));
const fmtTokens = (t) => (t >= 1e6 ? `${(t / 1e6).toFixed(1)}M` : t >= 1e3 ? `${Math.round(t / 1e3)}K` : String(t));
const maxOf = (m) => Math.max(...m.flat());
const argmaxOf = (m) => {
  let best = [0, 0];
  m.forEach((row, l) => row.forEach((v, h) => { if (v > m[best[0]][best[1]]) best = [l, h]; }));
  return best;
};

// ---------------------------------------------------------------- tooltip

function showTip(evt, rows) {
  const tip = $("tooltip");
  tip.replaceChildren();
  for (const { value, label, key } of rows) {
    const row = document.createElement("div");
    row.className = "tip-row";
    if (key) {
      const k = document.createElement("span");
      k.className = `key ${key}`;
      row.append(k);
    }
    const v = document.createElement("strong");
    v.textContent = value;
    const l = document.createElement("span");
    l.textContent = ` ${label}`;
    row.append(v, l);
    tip.append(row);
  }
  tip.hidden = false;
  const pad = 14;
  const x = Math.min(evt.clientX + pad, window.innerWidth - tip.offsetWidth - 8);
  const y = Math.max(8, evt.clientY - tip.offsetHeight - pad);
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}

function hideTip() {
  $("tooltip").hidden = true;
}

// ---------------------------------------------------------------- shared x axis

function logX(x0, x1) {
  const maxStep = data.checkpoints.at(-1).step;
  const lmax = Math.log10(maxStep + 1);
  return (s) => x0 + ((x1 - x0) * Math.log10(s + 1)) / lmax;
}

function xAxis(svg, x, y0, ticks) {
  for (const s of ticks) {
    svg.append(el("line", { x1: x(s), x2: x(s), y1: y0, y2: y0 + 4, class: "axis" }));
    svg.append(el("text", { x: x(s), y: y0 + 17, "text-anchor": "middle", class: "tick" }, fmtStep(s)));
  }
  svg.append(el("text", { x: x(ticks.at(-1)), y: y0 + 30, "text-anchor": "end", class: "tick" },
                "training step (log scale)"));
}

function nearestCheckpoint(step) {
  let best = 0;
  data.checkpoints.forEach((c, i) => {
    if (Math.abs(Math.log10(c.step + 1) - Math.log10(step + 1)) <
        Math.abs(Math.log10(data.checkpoints[best].step + 1) - Math.log10(step + 1))) best = i;
  });
  return best;
}

function stepAt(svg, evt, x0, x1) {
  const pt = svg.createSVGPoint();
  pt.x = evt.clientX;
  pt.y = evt.clientY;
  const p = pt.matrixTransform(svg.getScreenCTM().inverse());
  const maxStep = data.checkpoints.at(-1).step;
  const frac = Math.min(1, Math.max(0, (p.x - x0) / (x1 - x0)));
  return 10 ** (frac * Math.log10(maxStep + 1)) - 1;
}

const ticksFor = () => [0, 10, 100, 1000, data.checkpoints.at(-1).step];

// ---------------------------------------------------------------- loss chart

/** Chart width in SVG units = the container's width in pixels, so text stays at its CSS size. */
const chartWidth = (id) => Math.max(300, Math.min(760, $(id).clientWidth || 760));

function drawLoss() {
  const W = chartWidth("loss-chart");
  const H = 280;
  const m = { l: 44, r: 16, t: 12, b: 40 };
  const x = logX(m.l, W - m.r);
  const train = data.train_curve;
  const val = [...new Map(data.val_curve.map(([s, v]) => [s, v])).entries()];
  const all = [...train.map((p) => p[1]), ...val.map((p) => p[1])];
  const yMin = Math.floor(Math.min(...all));
  const yMax = Math.ceil(Math.max(...all));
  const y = (v) => m.t + ((H - m.t - m.b) * (yMax - v)) / (yMax - yMin);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Training and validation loss by step" });
  const f = data.formation;
  svg.append(el("rect", { x: x(f.from_step), y: m.t, width: x(f.to_step) - x(f.from_step), height: H - m.t - m.b,
                          class: "band" }));
  for (let v = yMin; v <= yMax; v++) {
    svg.append(el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: "grid" }));
    svg.append(el("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end", class: "tick" }, String(v)));
  }
  xAxis(svg, x, H - m.b, ticksFor());
  const path = (pts) => pts.map(([s, v], i) => `${i ? "L" : "M"}${x(s).toFixed(1)},${y(v).toFixed(1)}`).join("");
  svg.append(el("path", { d: path(train), class: "line s1" }));
  svg.append(el("path", { d: path(val), class: "line s2" }));
  const c = data.checkpoints[idx];
  svg.append(el("line", { x1: x(c.step), x2: x(c.step), y1: m.t, y2: H - m.b, class: "selected" }));
  svg.append(el("circle", { cx: x(c.step), cy: y(c.val_loss), r: 5, class: "dot s2" }));
  const cross = el("line", { y1: m.t, y2: H - m.b, class: "crosshair", visibility: "hidden" });
  svg.append(cross);
  const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, class: "hit" });
  hit.addEventListener("pointermove", (evt) => {
    const s = stepAt(svg, evt, m.l, W - m.r);
    const near = (pts) => pts.reduce((a, b) => (Math.abs(b[0] - s) < Math.abs(a[0] - s) ? b : a));
    const t = near(train);
    const v = near(val);
    cross.setAttribute("x1", x(t[0]));
    cross.setAttribute("x2", x(t[0]));
    cross.setAttribute("visibility", "visible");
    showTip(evt, [
      { value: fmt(t[1]), label: `training loss, step ${t[0].toLocaleString("en")}`, key: "key-1" },
      { value: fmt(v[1]), label: `validation loss, step ${v[0].toLocaleString("en")}`, key: "key-2" },
      { value: "", label: "click to jump to the nearest checkpoint" },
    ]);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  hit.addEventListener("click", (evt) => select(nearestCheckpoint(stepAt(svg, evt, m.l, W - m.r))));
  svg.append(hit);
  $("loss-chart").replaceChildren(svg);
}

// ---------------------------------------------------------------- score chart

function drawScores() {
  const W = chartWidth("score-chart");
  const H = 240;
  const m = { l: 44, r: 16, t: 12, b: 40 };
  const x = logX(m.l, W - m.r);
  const y = (v) => m.t + (H - m.t - m.b) * (1 - v);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Best induction and previous-token scores" });
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    svg.append(el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: "grid" }));
    svg.append(el("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end", class: "tick" }, v.toFixed(2)));
  }
  const refs = [
    [data.induction_threshold, "induction-head threshold"],
    [data.positive_control.fixed_length, "control model trained to copy"],
  ];
  for (const [v, label] of refs) {
    svg.append(el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: "ref" }));
    svg.append(el("text", { x: W - m.r - 4, y: y(v) - 6, "text-anchor": "end", class: "ref-label" },
                  `${label} (${v.toFixed(2)})`));
  }
  xAxis(svg, x, H - m.b, ticksFor());
  const series = [["prefix", "s1"], ["prev", "s2"]];
  for (const [key, cls] of series) {
    const pts = data.checkpoints.map((c) => [c.step, maxOf(c[key])]);
    svg.append(el("path", { d: pts.map(([s, v], i) => `${i ? "L" : "M"}${x(s).toFixed(1)},${y(v).toFixed(1)}`).join(""),
                            class: `line ${cls}` }));
    pts.forEach(([s, v], i) => svg.append(el("circle", { cx: x(s), cy: y(v), r: i === idx ? 6 : 4, class: `dot ${cls}` })));
  }
  const c = data.checkpoints[idx];
  svg.append(el("line", { x1: x(c.step), x2: x(c.step), y1: m.t, y2: H - m.b, class: "selected" }));
  const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, class: "hit" });
  hit.addEventListener("pointermove", (evt) => {
    const k = data.checkpoints[nearestCheckpoint(stepAt(svg, evt, m.l, W - m.r))];
    showTip(evt, [
      { value: fmt(maxOf(k.prefix)), label: `best induction score, step ${k.step.toLocaleString("en")}`, key: "key-1" },
      { value: fmt(maxOf(k.prev)), label: "best previous-token score", key: "key-2" },
      { value: "", label: "click to select this checkpoint" },
    ]);
  });
  hit.addEventListener("pointerleave", hideTip);
  hit.addEventListener("click", (evt) => select(nearestCheckpoint(stepAt(svg, evt, m.l, W - m.r))));
  svg.append(hit);
  $("score-chart").replaceChildren(svg);
}

// ---------------------------------------------------------------- heads grid and pattern

function drawHeads() {
  const c = data.checkpoints[idx];
  const grid = $("heads");
  grid.replaceChildren();
  grid.style.setProperty("--cols", data.n_head + 1);
  grid.append(document.createElement("span"));
  for (let h = 0; h < data.n_head; h++) grid.append(Object.assign(document.createElement("span"), { className: "head-axis", textContent: `H${h}` }));
  c[metric].forEach((row, l) => {
    grid.append(Object.assign(document.createElement("span"), { className: "head-axis", textContent: `L${l}` }));
    row.forEach((v, h) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "head-cell";
      if (l === head[0] && h === head[1]) b.classList.add("on");
      b.style.setProperty("--a", Math.min(1, v).toFixed(3));
      b.textContent = v.toFixed(2);
      b.title = `layer ${l}, head ${h}: ${metric === "prefix" ? "induction" : "previous-token"} score ${v.toFixed(3)}`;
      b.addEventListener("click", () => { head = [l, h]; drawHeads(); drawPattern(); });
      grid.append(b);
    });
  });
}

function drawPattern() {
  const c = data.checkpoints[idx];
  const toks = data.ui_sequence;
  const T = toks.length;
  const L = (T - 1) / 2;
  const [l, h] = head;
  const pat = c.patterns[l][h];
  $("pattern-title").textContent = `Attention of head L${l}.H${h} on a repeated random sequence`;
  const labelW = 84;
  const cell = 15;
  const W = labelW + T * cell + 4;
  const svg = el("svg", { viewBox: `0 0 ${W} ${T * cell + 4}`, role: "img",
                          "aria-label": `Attention pattern of layer ${l} head ${h}` });
  toks.forEach((t, q) => {
    svg.append(el("text", { x: labelW - 6, y: q * cell + cell - 3, "text-anchor": "end", class: "tick" },
                  `${JSON.stringify(t).slice(0, 10)} ${q}`));
    for (let k = 0; k <= q; k++) {
      const v = pat[q * T + k] / 100;
      const r = el("rect", { x: labelW + k * cell, y: q * cell, width: cell - 1, height: cell - 1, class: "cell",
                             style: `--a:${v.toFixed(3)}` });
      r.addEventListener("pointermove", (evt) => showTip(evt, [
        { value: `${Math.round(v * 100)}%`, label: `of position ${q} ${JSON.stringify(t)} looks at position ${k} ${JSON.stringify(toks[k])}` },
      ]));
      r.addEventListener("pointerleave", hideTip);
      svg.append(r);
    }
  });
  for (let j = 0; j < L; j++) {
    const q = 1 + L + j;
    const k = 2 + j;
    svg.append(el("rect", { x: labelW + k * cell - 0.5, y: q * cell - 0.5, width: cell, height: cell, class: "target" }));
  }
  $("pattern").replaceChildren(svg);
}

// ---------------------------------------------------------------- text panels

function drawTiles() {
  const c = data.checkpoints[idx];
  $("ckpt-label").textContent =
    `Checkpoint ${idx + 1} of ${data.checkpoints.length}: step ${c.step.toLocaleString("en")}, ` +
    `${fmtTokens(c.tokens)} training tokens seen.`;
  $("t-val").textContent = fmt(c.val_loss);
  const bi = argmaxOf(c.prefix);
  const bp = argmaxOf(c.prev);
  $("t-ind").textContent = fmt(maxOf(c.prefix));
  $("t-ind-head").textContent = `head L${bi[0]}.H${bi[1]}`;
  $("t-prev").textContent = fmt(maxOf(c.prev));
  $("t-prev-head").textContent = `head L${bp[0]}.H${bp[1]}`;
  $("t-copy").textContent = `${c.repeat_loss_first.toFixed(2)} → ${c.repeat_loss_second.toFixed(2)}`;
  const story = c.samples[0];
  const prompt = sharedPrompt;
  $("ckpt-story").replaceChildren();
  if (prompt && story.startsWith(prompt)) {
    $("ckpt-story").append(Object.assign(document.createElement("span"), { className: "prompt", textContent: prompt }),
                           story.slice(prompt.length));
  } else {
    $("ckpt-story").textContent = story;
  }
  const list = $("samples");
  list.replaceChildren();
  for (const s of c.samples) list.append(Object.assign(document.createElement("li"), { textContent: s }));
}

function drawTable() {
  const t = $("table");
  t.replaceChildren();
  const head = t.createTHead().insertRow();
  for (const h of ["step", "tokens", "val loss", "best induction", "best previous-token", "copy loss 1st", "copy loss 2nd"]) {
    head.append(Object.assign(document.createElement("th"), { textContent: h }));
  }
  const body = t.createTBody();
  for (const c of data.checkpoints) {
    const r = body.insertRow();
    for (const v of [c.step.toLocaleString("en"), fmtTokens(c.tokens), fmt(c.val_loss), fmt(maxOf(c.prefix)),
                     fmt(maxOf(c.prev)), c.repeat_loss_first.toFixed(2), c.repeat_loss_second.toFixed(2)]) {
      r.insertCell().textContent = v;
    }
  }
}

function verdict() {
  const f = data.formation;
  const last = data.checkpoints.at(-1);
  const pc = data.positive_control;
  $("verdict").textContent = f.induction_step === null
    ? `Result: no head reaches the threshold (${data.induction_threshold}) at any checkpoint. The best final score is ` +
      `${fmt(maxOf(last.prefix))}, and the loss on the second copy (${last.repeat_loss_second.toFixed(2)}) is barely ` +
      `below the first (${last.repeat_loss_first.toFixed(2)}): this model did not learn to copy from context. ` +
      `The probe itself works: a tiny model trained to copy by position reaches ${fmt(pc.fixed_length)}. ` +
      `Details: results/induction.md.`
    : `Result: an induction head first passes the threshold at step ${f.induction_step.toLocaleString("en")}.`;
}

function select(i) {
  idx = i;
  $("ckpt").value = String(i);
  drawTiles();
  drawLoss();
  drawScores();
  drawHeads();
  drawPattern();
}

/** Step through the checkpoints from the first (or the current one) to the last. */
function togglePlay() {
  if (playTimer !== null) {
    stopPlay();
    return;
  }
  if (idx >= data.checkpoints.length - 1) select(0);
  $("play").textContent = "❚❚ Pause";
  playTimer = setInterval(() => {
    if (idx >= data.checkpoints.length - 1) stopPlay();
    else select(idx + 1);
  }, 900);
}

function stopPlay() {
  clearInterval(playTimer);
  playTimer = null;
  $("play").textContent = "▶ Play training";
}

async function init() {
  try {
    const res = await fetch(new URL("../model/replay.json", import.meta.url));
    if (!res.ok) throw new Error(`replay.json: HTTP ${res.status}`);
    data = await res.json();
  } catch (err) {
    $("status").textContent = `Could not load the replay data: ${err.message}`;
    $("status").classList.add("error");
    return;
  }
  $("n-ckpt").textContent = String(data.checkpoints.length);
  sharedPrompt = data.checkpoints.map((c) => c.samples[0]).reduce((a, b) => {
    let i = 0;
    while (i < a.length && a[i] === b[i]) i++;
    return a.slice(0, i);
  });
  $("ckpt").max = String(data.checkpoints.length - 1);
  head = argmaxOf(data.checkpoints.at(-1).prefix);
  $("ckpt").addEventListener("input", (e) => {
    stopPlay();
    select(Number(e.target.value));
  });
  $("play").addEventListener("click", togglePlay);
  for (const [id, m] of [["m-prefix", "prefix"], ["m-prev", "prev"]]) {
    $(id).addEventListener("click", () => {
      metric = m;
      $("m-prefix").classList.toggle("seg-on", m === "prefix");
      $("m-prev").classList.toggle("seg-on", m === "prev");
      drawHeads();
    });
  }
  const params = new URLSearchParams(location.search);
  verdict();
  drawTable();
  // first visit: start untrained, so Play shows the whole story; ?ckpt=N opens a given checkpoint
  select(params.has("ckpt") ? Math.min(Number(params.get("ckpt")), data.checkpoints.length - 1) : 0);
  let pending = 0;
  window.addEventListener("resize", () => {
    cancelAnimationFrame(pending);
    pending = requestAnimationFrame(() => { drawLoss(); drawScores(); });
  });
}

init();
