// Records the frames of docs/demo.gif with headless Chrome over the DevTools protocol.
//
//   python -m http.server 8765 --bind 127.0.0.1 -d web      (in another terminal)
//   node docs/demo/frames.mjs http://127.0.0.1:8765/ docs/demo/frames
//   python docs/demo/make_gif.py docs/demo/frames docs/demo.gif
//
// Set CHROME to the browser executable if it is not in the default Windows location.

import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [base, outDir] = process.argv.slice(2);
const CHROME = process.env.CHROME ?? "C:/Program Files/Google/Chrome/Application/chrome.exe";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
mkdirSync(outDir, { recursive: true });

const port = 9400 + Math.floor(Math.random() * 400);
const chrome = spawn(CHROME, ["--headless=new", "--disable-gpu", "--no-first-run", "--hide-scrollbars",
  `--remote-debugging-port=${port}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), "demo-"))}`, "about:blank"],
  { stdio: "ignore" });
let targets;
for (let i = 0; i < 50 && !targets; i++) {
  try { targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json(); } catch { await sleep(200); }
}
const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let nextId = 1;
const pending = new Map();
ws.addEventListener("message", (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
});
const send = (method, params = {}) => new Promise((resolve) => {
  const id = nextId++;
  pending.set(id, resolve);
  ws.send(JSON.stringify({ id, method, params }));
});
const evaluate = async (expression) =>
  (await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true })).result?.result?.value;
const waitFor = async (expr, timeoutMs = 120000) => {
  for (let t = 0; t < timeoutMs; t += 250) {
    if (await evaluate(expr)) return;
    await sleep(250);
  }
  throw new Error(`timed out waiting for ${expr}`);
};

const W = 900;
await send("Emulation.setDeviceMetricsOverride", { width: W, height: 900, deviceScaleFactor: 1, mobile: false });
await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "light" }] });
await send("Page.enable");

let frame = 0;
async function shot(scene, fromSel, toSel, holdMs) {
  const box = await evaluate(`(() => {
    const a = document.querySelector(${JSON.stringify(fromSel)}).getBoundingClientRect();
    const b = document.querySelector(${JSON.stringify(toSel)}).getBoundingClientRect();
    return { y: a.top + scrollY - 12, h: b.bottom - a.top + 24 };
  })()`);
  const img = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true,
    clip: { x: 0, y: Math.max(0, box.y), width: W, height: box.h, scale: 1 } });
  const name = `${String(frame++).padStart(3, "0")}_${scene}_${holdMs}.png`;
  writeFileSync(join(outDir, name), Buffer.from(img.result.data, "base64"));
}

// Scene 1: a story streams in with the name guard on. The CPU is throttled so the streaming is
// visible, so the stats line (which would show the slowed-down speed) is left out of the frame.
await send("Page.navigate", { url: `${base}index.html?guard=balanced&seed=3` });
await waitFor("!document.getElementById('generate').disabled && document.getElementById('guard-panel').open");
await send("Emulation.setCPUThrottlingRate", { rate: 10 });
await evaluate("document.getElementById('generate').click()");
for (let i = 0; i < 60; i++) {
  await shot("generate", "#guard-panel", "#output", 120);
  if (await evaluate("document.getElementById('stats').textContent.includes('tokens in')")) break;
  await sleep(150);
}
await send("Emulation.setCPUThrottlingRate", { rate: 1 });
await shot("generate", "#guard-panel", "#output", 2500);

// Scene 2: the microscope explains why the model wrote " Lily".
await send("Page.navigate", { url: `${base}microscope.html` });
await waitFor("document.getElementById('status').textContent.startsWith('Analyzed')");
await evaluate("[...document.querySelectorAll('#tokens .tok')].find((s) => s.textContent === ' Lily').click()");
await waitFor("document.getElementById('circuit-meta').textContent.startsWith('Predicted')");
await evaluate("document.getElementById('circuit').scrollIntoView()");
await shot("microscope", "#circuit", "#circuit", 3500);

// Scene 3: replaying training, checkpoint by checkpoint.
await send("Page.navigate", { url: `${base}replay.html?ckpt=6` });
await waitFor("document.getElementById('t-val').textContent.length > 0");
for (const k of [6, 9, 11, 13, 15, 17, 19, 22]) {
  await evaluate(`(() => { const s = document.getElementById('ckpt'); s.value = '${k}';
    s.dispatchEvent(new Event('input')); })()`);
  await sleep(200);
  await shot("replay", "#ckpt", "#loss-chart", k === 22 ? 3000 : 700);
}

ws.close();
chrome.kill();
console.log(`${frame} frames in ${outDir}`);
process.exit(0);
