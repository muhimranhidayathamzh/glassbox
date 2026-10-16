// Makes web/img/circuit.png (a screenshot of the microscope's circuit graph, used on the case study
// page) and web/og.png (the 1200x630 link preview) with headless Chrome over the DevTools protocol.
// The numbers on the preview are read from results/*.json, never typed.
//
//   python -m http.server 8765 --bind 127.0.0.1 -d web      (in another terminal)
//   node docs/demo/og.mjs http://127.0.0.1:8765/
//
// Set CHROME to the browser executable if it is not in the default Windows location.

import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [base] = process.argv.slice(2);
const CHROME = process.env.CHROME ?? "C:/Program Files/Google/Chrome/Application/chrome.exe";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = (name) => JSON.parse(readFileSync(`results/${name}.json`, "utf-8"));

const port = 9400 + Math.floor(Math.random() * 400);
const chrome = spawn(CHROME, ["--headless=new", "--disable-gpu", "--no-first-run", "--hide-scrollbars",
  `--remote-debugging-port=${port}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), "og-"))}`, "about:blank"],
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
await send("Page.enable");
await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "light" }] });

// 1. The microscope explains why the model wrote " Lily" (the same scene as in docs/demo.gif).
await send("Emulation.setDeviceMetricsOverride", { width: 820, height: 1000, deviceScaleFactor: 2, mobile: false });
await send("Page.navigate", { url: `${base}microscope.html` });
await waitFor("document.getElementById('status').textContent.startsWith('Analyzed')");
await evaluate("[...document.querySelectorAll('#tokens .tok')].find((s) => s.textContent === ' Lily').click()");
await waitFor("document.getElementById('circuit-meta').textContent.startsWith('Predicted')");
const box = await evaluate(`(() => {
  const panel = document.getElementById('circuit').getBoundingClientRect();
  const graph = document.getElementById('circuit-graph').getBoundingClientRect();
  return { x: panel.left + scrollX, y: panel.top + scrollY, w: panel.width, h: graph.bottom - panel.top + 16 };
})()`);
const circuit = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true,
  clip: { x: box.x, y: box.y, width: box.w, height: box.h, scale: 1 } });
mkdirSync("web/img", { recursive: true });
writeFileSync("web/img/circuit.png", Buffer.from(circuit.result.data, "base64"));
console.log(`web/img/circuit.png: ${Math.round(box.w * 2)}x${Math.round(box.h * 2)}`);

// 2. The link preview: a fixed layout around that screenshot, numbers from results/.
const lm = results("lm_eval");
const sae = results("sae_eval");
const steer = results("steering").summary;
const best = steer.by_alpha.reduce((a, b) => (b.feature_mention > a.feature_mention ? b : a));
const pct = (x) => `${Math.round(x * 100)}%`;
const img = `data:image/png;base64,${circuit.result.data}`;
const html = `<!doctype html><html><head><meta charset="utf-8"><style>
  * { box-sizing: border-box; margin: 0; }
  body { width: 1200px; height: 630px; overflow: hidden; background: #f7f5f0; color: #1f1d19;
         font-family: system-ui, "Segoe UI", Roboto, sans-serif; display: flex; }
  .text { width: 560px; flex: none; padding: 64px 0 56px 64px; display: flex; flex-direction: column; }
  .brand { font-size: 22px; font-weight: 700; letter-spacing: 0.08em; color: #3d5bd9; text-transform: uppercase; }
  h1 { margin-top: 22px; font-size: 48px; line-height: 1.08; letter-spacing: -0.02em; }
  .sub { margin-top: 20px; font-size: 23px; line-height: 1.4; color: #6d675c; }
  .stats { margin-top: auto; display: flex; gap: 36px; }
  .stat b { display: block; font-size: 36px; letter-spacing: -0.01em; }
  .stat span { display: block; margin-top: 4px; font-size: 17px; line-height: 1.35; color: #6d675c; max-width: 230px; }
  .shot { flex: 1; position: relative; }
  .card { position: absolute; left: 28px; top: 50%; transform: translateY(-50%); width: 584px; background: #fff; border: 1px solid #e2ddd2;
          border-radius: 16px; overflow: hidden; box-shadow: 0 18px 50px rgb(31 29 25 / 0.14); }
  .card img { display: block; width: 100%; }
</style></head><body>
  <div class="text">
    <div class="brand">Glassbox</div>
    <h1>An LLM built from scratch, and a microscope to look inside it</h1>
    <p class="sub">${(lm.n_params / 1e6).toFixed(1)}M parameters, trained on a laptop CPU, running in your browser.</p>
    <div class="stats">
      <div class="stat"><b>${pct(best.feature_mention)} vs ${pct(best.random_mention)}</b>
        <span>stories steered to a concept: feature vs random direction</span></div>
      <div class="stat"><b>${sae.n_features.toLocaleString("en-US")}</b>
        <span>features found by a sparse autoencoder</span></div>
    </div>
  </div>
  <div class="shot"><div class="card"><img src="${img}"></div></div>
</body></html>`;
await send("Emulation.setDeviceMetricsOverride", { width: 1200, height: 630, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url: "about:blank" });
await sleep(300);
const frameId = (await send("Page.getFrameTree")).result.frameTree.frame.id;
await send("Page.setDocumentContent", { frameId, html });
await waitFor("document.querySelector('.card img')?.complete");
await sleep(300);
const og = await send("Page.captureScreenshot", { format: "png", clip: { x: 0, y: 0, width: 1200, height: 630, scale: 1 } });
writeFileSync("web/og.png", Buffer.from(og.result.data, "base64"));
console.log("web/og.png: 1200x630");

ws.close();
chrome.kill();
process.exit(0);
