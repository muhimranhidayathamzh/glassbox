"""Milestones 2-3 report: does the JavaScript engine reproduce PyTorch?

    python -m glassbox.export                 # writes web/model
    python -m glassbox.eval.web_parity        # writes results/web_parity.{json,md}

Compares, on real validation data:
  * float16 export vs the float32 checkpoint (cost of shipping half-precision weights)
  * JS engine vs PyTorch running the same exported weights (next-token logits at every position)
  * JS tokenizer vs Python tokenizer on validation stories
  * greedy generation, JS vs PyTorch, long enough to cross the context window
  * JS speed in Node.js (V8, the engine behind Chrome)
  * if web/model/sae exists: residual stream, SAE pre-activations and active features, JS vs PyTorch
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from ..data import VALID_FILE, split_stories
from ..export import load_exported, load_exported_sae
from ..sae import story_windows
from ..train import load_checkpoint, load_tokens

REPO = Path(__file__).resolve().parents[3]
NODE_CLI = REPO / "web" / "tools" / "parity_cli.mjs"

# Max |logit_js - logit_torch| allowed for the same weights. Set before measuring: float32
# PyTorch vs float64-accumulating JS should differ around 1e-5; any real bug is >> 1e-2.
PARITY_MAX_ABS_LOGIT_DIFF = 1e-3

# Active-feature sets may differ when the k-th and (k+1)-th pre-activations are this close.
SAE_NEAR_TIE = 1e-3

GREEDY_PROMPTS = ["Once upon a time", "Tom and Sue went to the beach.", "The cat"]


def node_available() -> bool:
    return shutil.which("node") is not None


def _split_rows(path: str, lengths: list[int], width: int) -> list[np.ndarray]:
    flat = np.fromfile(path, dtype="<f4")
    if flat.size != sum(lengths) * width:
        raise RuntimeError(f"{path}: expected {sum(lengths) * width} floats, got {flat.size}")
    out, o = [], 0
    for n in lengths:
        out.append(flat[o : o + n * width].reshape(n, width))
        o += n * width
    return out


def run_node(request: dict, widths: dict | None = None) -> tuple[dict, dict]:
    """Run web/tools/parity_cli.mjs.

    Returns the JSON response and the binary outputs as lists of (positions, width) float32
    arrays, one per input sequence: "logits" if `sequences` was given; "resid" and "pre" if
    `features` was given (widths must then give {"resid": d_model, "pre": n_features})."""
    with tempfile.TemporaryDirectory() as tmp:
        req = dict(request)
        req["model_dir"] = str(Path(req["model_dir"]).resolve())
        if "sae_dir" in req:
            req["sae_dir"] = str(Path(req["sae_dir"]).resolve())
        if "sequences" in req:
            req["logits_out"] = str(Path(tmp) / "logits.bin")
        if "features" in req:
            req["resid_out"] = str(Path(tmp) / "resid.bin")
            req["pre_out"] = str(Path(tmp) / "pre.bin")
        req_path = Path(tmp) / "request.json"
        req_path.write_text(json.dumps(req), encoding="utf-8")
        proc = subprocess.run(
            ["node", str(NODE_CLI), str(req_path)], capture_output=True, encoding="utf-8"
        )
        if proc.returncode != 0:
            raise RuntimeError(f"node failed:\n{proc.stderr}")
        res = json.loads(proc.stdout)
        arrays = {}
        if "sequences" in req:
            lengths = [rows for rows, _ in res["logits_shapes"]]
            arrays["logits"] = _split_rows(req["logits_out"], lengths, res["logits_shapes"][0][1])
        if "features" in req:
            lengths = [len(ids) for ids in req["features"]]
            arrays["resid"] = _split_rows(req["resid_out"], lengths, widths["resid"])
            arrays["pre"] = _split_rows(req["pre_out"], lengths, widths["pre"])
    return res, arrays


@torch.no_grad()
def torch_logits(model, ids: list[int]) -> np.ndarray:
    logits, _ = model(torch.tensor([ids]))
    return logits[0].numpy()


class DiffStats:
    """Streaming comparison of two logit arrays of shape (positions, vocab)."""

    def __init__(self):
        self.max = 0.0
        self.total = 0.0
        self.count = 0
        self.agree = 0
        self.rows = 0

    def add(self, a: np.ndarray, b: np.ndarray) -> None:
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        self.max = max(self.max, float(diff.max()))
        self.total += float(diff.sum())
        self.count += diff.size
        self.agree += int((a.argmax(-1) == b.argmax(-1)).sum())
        self.rows += a.shape[0]

    def result(self) -> dict:
        return {"max_abs_diff": self.max, "mean_abs_diff": self.total / self.count,
                "top1_agreement": self.agree / self.rows}


def mean_nll(logits: np.ndarray, targets: np.ndarray) -> float:
    lp = torch.log_softmax(torch.from_numpy(logits).double(), -1)
    return float(-lp.gather(-1, torch.from_numpy(targets)[..., None]).mean())


def write_markdown(r: dict, path: Path) -> None:
    f, j, t, g, b = r["fp16_vs_fp32"], r["js_vs_torch"], r["tokenizer"], r["greedy"], r["speed"]
    lines = [
        "# Milestones 2-3: the JavaScript engine vs PyTorch",
        "",
        "Generated by `python -m glassbox.eval.web_parity`. Do not edit by hand.",
        "",
        "## Shipped files",
        "",
        f"- `web/model/weights.bin`: {r['files']['weights_bytes'] / 1e6:.2f} MB, {r['files']['dtype']}, "
        f"{r['files']['n_params']:,} parameters (from `{r['files']['source']['checkpoint']}`, "
        f"step {r['files']['source']['step']})",
        f"- `web/model/tokenizer.json`: {r['files']['tokenizer_bytes'] / 1e3:.0f} kB; "
        f"`manifest.json`: {r['files']['manifest_bytes'] / 1e3:.0f} kB",
        "",
        "## Cost of float16 weights (PyTorch fp32 checkpoint vs PyTorch fp16 export)",
        "",
        f"{f['windows']} validation windows x {f['positions_per_window']} positions.",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| val loss, float32 weights | {f['loss_fp32']:.5f} |",
        f"| val loss, float16 weights | {f['loss_fp16']:.5f} |",
        f"| max abs logit difference | {f['max_abs_diff']:.4f} |",
        f"| mean abs logit difference | {f['mean_abs_diff']:.6f} |",
        f"| top-1 token agreement | {f['top1_agreement']:.4%} |",
        "",
        "## Parity: JavaScript engine vs PyTorch (same exported weights)",
        "",
        f"Next-token logits at every position of {j['windows']} validation windows "
        f"({j['positions']:,} positions x {j['vocab']:,} logits). JS feeds one token at a time "
        "through its KV cache; PyTorch runs the full causal forward pass.",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| max abs logit difference | {j['max_abs_diff']:.2e} |",
        f"| mean abs logit difference | {j['mean_abs_diff']:.2e} |",
        f"| top-1 token agreement | {j['top1_agreement']:.4%} |",
        f"| threshold (max abs diff) | {j['threshold']:.0e} |",
        f"| **result** | **{'PASS' if j['pass'] else 'FAIL'}** |",
        "",
        "## Tokenizer parity",
        "",
        f"- {t['identical']:,} / {t['stories']:,} validation stories encode to identical token ids "
        f"({t['tokens']:,} tokens)",
        f"- {t['decode_identical']:,} / {t['stories']:,} decode back to the original text",
        "",
        "## Greedy generation parity",
        "",
        f"Greedy decoding of {g['new_tokens']} tokens (crosses the {g['n_ctx']}-token context, so the "
        "context-restart rule is exercised).",
        "",
        "| prompt | identical tokens |",
        "|---|---:|",
    ]
    lines += [f"| `{p['prompt']}` | {p['identical_prefix']} / {p['new_tokens']} |" for p in g["prompts"]]
    sae = r.get("sae")
    if sae:
        lines += [
            "",
            "## SAE in the browser",
            "",
            f"Top-k SAE (k = {sae['k']}, {sae['n_features']} features) on the residual stream entering block "
            f"{sae['layer']}, over {sae['windows']} validation windows that begin at a story boundary "
            f"({sae['positions']:,} positions >= 1). JS reads the residual through its KV-cache forward pass.",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| max abs residual difference (JS vs PyTorch) | {sae['max_abs_resid_diff']:.2e} |",
            f"| max abs SAE pre-activation difference | {sae['max_abs_pre_diff']:.2e} |",
            f"| positions with identical active features | {sae['identical']:,} / {sae['positions']:,} |",
            f"| differing positions at a near-tie (k-th and (k+1)-th within {SAE_NEAR_TIE:g}) | "
            f"{sae['near_tie_mismatches']} / {sae['positions'] - sae['identical']} |",
        ]
    lines += [
        "",
        "## Speed (JS engine in Node.js, single thread)",
        "",
        f"{b['runtime']}, {b['cpu']}. Median of {b['repeats']} runs after a warm-up. "
        "Browsers run the same JavaScript, but speed there depends on the browser and device.",
        "",
        f"- Prompt processing: {b['prefill_tokens_per_s']:.0f} tokens/s ({b['prompt_tokens']} tokens)",
        f"- Generation: {b['generate_tokens_per_s']:.0f} tokens/s ({b['new_tokens']} tokens, "
        "including softmax and sampling)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    from .lm_eval import cpu_name

    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=Path("web/model"))
    p.add_argument("--ckpt", type=Path, default=Path("runs/main/final.pt"))
    p.add_argument("--data-dir", type=Path, default=Path("data/tinystories"))
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--fp16-windows", type=int, default=64)
    p.add_argument("--js-windows", type=int, default=8)
    p.add_argument("--tokenizer-stories", type=int, default=2000)
    p.add_argument("--greedy-tokens", type=int, default=300)
    a = p.parse_args()
    if not node_available():
        raise SystemExit("node is not installed")

    t0 = time.time()
    exported, tok, manifest = load_exported(a.model_dir)
    original, _ = load_checkpoint(a.ckpt)
    T, V = exported.cfg.n_ctx, exported.cfg.vocab_size
    val = load_tokens(a.data_dir / "val.bin")
    starts = np.linspace(0, len(val) - T - 2, a.fp16_windows).astype(np.int64)
    windows = np.stack([np.asarray(val[s : s + T + 1], dtype=np.int64) for s in starts])

    # 1. float16 vs float32
    fp16_stats, nll32, nll16, ref16 = DiffStats(), [], [], []
    for i, w in enumerate(windows):
        lo32 = torch_logits(original, w[:-1].tolist())
        lo16 = torch_logits(exported, w[:-1].tolist())
        fp16_stats.add(lo16, lo32)
        nll32.append(mean_nll(lo32, w[1:]))
        nll16.append(mean_nll(lo16, w[1:]))
        if i < a.js_windows:
            ref16.append(lo16)
    fp16 = {"windows": len(windows), "positions_per_window": T, **fp16_stats.result(),
            "loss_fp32": float(np.mean(nll32)), "loss_fp16": float(np.mean(nll16))}

    # 2. JS vs PyTorch on the exported weights
    js_seqs = [w[:-1].tolist() for w in windows[: a.js_windows]]
    js_logits = run_node({"model_dir": a.model_dir, "sequences": js_seqs})[1]["logits"]
    js_stats = DiffStats()
    for got, ref in zip(js_logits, ref16):
        js_stats.add(got, ref)
    js = js_stats.result()
    js.update({"windows": len(js_seqs), "positions": len(js_seqs) * T, "vocab": V,
               "threshold": PARITY_MAX_ABS_LOGIT_DIFF})
    js["pass"] = js["max_abs_diff"] < PARITY_MAX_ABS_LOGIT_DIFF

    # 3. tokenizer on raw validation stories
    stories, _ = split_stories((a.raw_dir / VALID_FILE).read_text(encoding="utf-8"))
    stories = stories[: a.tokenizer_stories]
    py_ids = [tok.encode(s) for s in stories]
    res, _ = run_node({"model_dir": a.model_dir, "encode": stories, "decode": py_ids})
    tok_report = {
        "stories": len(stories),
        "tokens": sum(map(len, py_ids)),
        "identical": sum(x == y for x, y in zip(res["encoded"], py_ids)),
        "decode_identical": sum(x == y for x, y in zip(res["decoded"], stories)),
    }

    # 4. greedy generation across the context window
    prompts = [[tok.eot_id] + tok.encode(pr) for pr in GREEDY_PROMPTS]
    res, _ = run_node({"model_dir": a.model_dir,
                       "greedy": [{"prompt": pr, "n": a.greedy_tokens} for pr in prompts]})
    greedy = {"new_tokens": a.greedy_tokens, "n_ctx": T, "prompts": []}
    for text, pr, js_ids in zip(GREEDY_PROMPTS, prompts, res["greedy"]):
        py = exported.generate(torch.tensor([pr]), a.greedy_tokens, temperature=0)[0, len(pr):].tolist()
        same = next((i for i, (x, y) in enumerate(zip(py, js_ids)) if x != y), min(len(py), len(js_ids)))
        greedy["prompts"].append({"prompt": text, "new_tokens": len(py), "identical_prefix": same})

    # 5. speed
    res, _ = run_node({"model_dir": a.model_dir,
                       "bench": {"prompt_tokens": 128, "new_tokens": 128, "repeats": 3}})
    speed = {**res["bench"], "cpu": cpu_name()}

    # 6. SAE in the browser, on the story-aligned windows the SAE is used on
    sae_report = None
    if (a.model_dir / "sae" / "manifest.json").exists():
        sae, _ = load_exported_sae(a.model_dir / "sae")
        k = sae.cfg.k
        x_sae, _ = story_windows(val, tok.eot_id, T, a.js_windows)
        seqs = x_sae.tolist()
        res, arr = run_node({"model_dir": a.model_dir, "sae_dir": a.model_dir / "sae", "features": seqs},
                            widths={"resid": exported.cfg.d_model, "pre": sae.cfg.n_features})
        sae_report = {"k": k, "n_features": sae.cfg.n_features, "layer": sae.cfg.layer, "windows": len(seqs),
                      "positions": 0, "identical": 0, "near_tie_mismatches": 0,
                      "max_abs_resid_diff": 0.0, "max_abs_pre_diff": 0.0}
        for ids, resid_js, pre_js, codes in zip(seqs, arr["resid"], arr["pre"], res["codes"]):
            with torch.no_grad():
                resid = exported.resid_at(torch.tensor([ids]), sae.cfg.layer)[0]
                pre = sae.encode_pre(resid)
            sae_report["max_abs_resid_diff"] = max(sae_report["max_abs_resid_diff"],
                                                   float(np.abs(resid_js[1:] - resid[1:].numpy()).max()))
            sae_report["max_abs_pre_diff"] = max(sae_report["max_abs_pre_diff"],
                                                 float(np.abs(pre_js[1:] - pre[1:].numpy()).max()))
            for pos in range(1, len(ids)):
                vals, idx = pre[pos].topk(k + 1)
                expected = {int(i) for i, v in zip(idx[:k], vals[:k]) if v > 0}
                sae_report["positions"] += 1
                if set(codes[pos][0]) == expected:
                    sae_report["identical"] += 1
                elif vals[k - 1] - vals[k] < SAE_NEAR_TIE or (vals[:k].abs() < SAE_NEAR_TIE).any():
                    sae_report["near_tie_mismatches"] += 1

    report = {
        "files": {
            "dtype": manifest["dtype"], "n_params": manifest["n_params"],
            "weights_bytes": manifest["weights_bytes"], "source": manifest["source"],
            "tokenizer_bytes": (a.model_dir / manifest["tokenizer_file"]).stat().st_size,
            "manifest_bytes": (a.model_dir / "manifest.json").stat().st_size,
        },
        "fp16_vs_fp32": fp16,
        "js_vs_torch": js,
        "tokenizer": tok_report,
        "greedy": greedy,
        "speed": speed,
        "sae": sae_report,
        "eval_seconds": round(time.time() - t0, 1),
    }
    a.results_dir.mkdir(parents=True, exist_ok=True)
    (a.results_dir / "web_parity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, a.results_dir / "web_parity.md")
    print(f"JS vs PyTorch max |logit diff| {js['max_abs_diff']:.2e} "
          f"({'PASS' if js['pass'] else 'FAIL'} < {PARITY_MAX_ABS_LOGIT_DIFF:.0e})")
    print(f"tokenizer identical {tok_report['identical']}/{tok_report['stories']}; "
          f"greedy identical {[g['identical_prefix'] for g in greedy['prompts']]}")
    if sae_report:
        print(f"SAE: identical active features at {sae_report['identical']}/{sae_report['positions']} positions, "
              f"max |pre diff| {sae_report['max_abs_pre_diff']:.2e}")
    print(f"wrote {a.results_dir / 'web_parity.json'} and {a.results_dir / 'web_parity.md'}")


if __name__ == "__main__":
    main()
