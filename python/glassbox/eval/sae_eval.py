"""Milestone 3 report: SAE reconstruction quality and the cost of splicing it into the model.

    python -m glassbox.eval.sae_eval --runs runs/sae_k8 runs/sae_k16 runs/sae_k32

For each SAE (float32, as trained), on validation windows that begin at a story boundary
(<|endoftext|>), with position 0 skipped:
  * variance explained, L0 (active features per token), dead and dense features
  * next-token loss when the residual stream at the SAE's layer is replaced by the SAE's
    reconstruction (positions >= 1; loss is measured on those positions)
Baselines: zero ablation, mean ablation, and PCA keeping as many components as the SAE's k
(mean and components fitted on activations from story-aligned training windows).

The SAE used by the UI is selected by a rule fixed before looking at the results: the smallest
k whose fraction of loss recovered (vs mean ablation) is at least SELECT_MIN_RECOVERED;
if none qualifies, the largest k.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..sae import load_sae, story_starts, story_windows
from ..tokenizer import BPETokenizer
from ..train import load_checkpoint, load_tokens

SELECT_MIN_RECOVERED = 0.85
DENSE_THRESHOLD = 0.1


def fit_mean_and_pca(model, tokens: np.ndarray, eot_id: int, layer: int, n_windows: int, seed: int):
    rng = np.random.default_rng(seed)
    T = model.cfg.n_ctx
    candidates = story_starts(tokens, eot_id, T)
    starts = candidates[rng.integers(0, len(candidates), size=n_windows)]
    xs = []
    with torch.no_grad():
        for i in range(0, n_windows, 16):
            idx = torch.from_numpy(np.stack([tokens[s : s + T] for s in starts[i : i + 16]]).astype(np.int64))
            xs.append(model.resid_at(idx, layer)[:, 1:].reshape(-1, model.cfg.d_model))
    acts = torch.cat(xs).double()
    mean = acts.mean(0)
    cov = (acts - mean).T @ (acts - mean) / len(acts)
    evals, evecs = torch.linalg.eigh(cov)
    order = evals.argsort(descending=True)
    return mean.float(), evecs[:, order].float(), int(len(acts))


def pca_reconstruct(x: torch.Tensor, mean: torch.Tensor, comps: torch.Tensor, k: int) -> torch.Tensor:
    P = comps[:, :k]
    return mean + ((x - mean) @ P) @ P.T


def splice(fn):
    """Replace the residual at positions >= 1 with fn(residual); keep position 0."""
    return lambda x: torch.cat([x[:, :1], fn(x[:, 1:])], dim=1)


@torch.no_grad()
def spliced_losses(model, layer: int, x: np.ndarray, y: np.ndarray, replacements: dict, batch: int = 16) -> dict:
    """Mean next-token loss over positions >= 1 for each replacement (None = clean model)."""
    totals = {name: 0.0 for name in replacements}
    count = 0
    for i in range(0, len(x), batch):
        xb, yb = torch.from_numpy(x[i : i + batch]), torch.from_numpy(y[i : i + batch])
        for name, fn in replacements.items():
            hooks = {layer: splice(fn)} if fn is not None else None
            logits, _ = model(xb, resid_hooks=hooks)
            totals[name] += F.cross_entropy(logits[:, 1:].reshape(-1, logits.size(-1)), yb[:, 1:].reshape(-1),
                                            reduction="sum").item()
        count += yb[:, 1:].numel()
    return {name: v / count for name, v in totals.items()}


@torch.no_grad()
def reconstruction_stats(model, layer: int, x: np.ndarray, saes: dict, mean, comps, batch: int = 16) -> dict:
    """Variance explained, L0, firing counts (SAEs) and PCA variance explained, in float64 sums."""
    d = model.cfg.d_model
    sum_x = torch.zeros(d, dtype=torch.float64)
    sum_sq = 0.0
    n = 0
    sse = {name: 0.0 for name in saes}
    pca_sse = {name: 0.0 for name in saes}
    l0 = {name: 0.0 for name in saes}
    fired = {name: torch.zeros(s.cfg.n_features, dtype=torch.long) for name, s in saes.items()}
    max_pre = {name: torch.full((s.cfg.n_features,), float("-inf")) for name, s in saes.items()}
    for i in range(0, len(x), batch):
        r = model.resid_at(torch.from_numpy(x[i : i + batch]), layer)[:, 1:].reshape(-1, d)
        r64 = r.double()
        sum_x += r64.sum(0)
        sum_sq += r64.pow(2).sum().item()
        n += r.shape[0]
        for name, sae in saes.items():
            pre = sae.encode_pre(r)
            z = sae.topk_codes(pre)
            rec = sae.decode(z)
            max_pre[name] = torch.maximum(max_pre[name], pre.max(0).values)
            sse[name] += (r64 - rec.double()).pow(2).sum().item()
            l0[name] += (z > 0).sum().item()
            fired[name] += (z > 0).sum(0)
            pca = pca_reconstruct(r, mean, comps, sae.cfg.k)
            pca_sse[name] += (r64 - pca.double()).pow(2).sum().item()
    sst = sum_sq - (sum_x.pow(2).sum().item() / n)
    out = {}
    for name, sae in saes.items():
        density = fired[name].double() / n
        out[name] = {
            "variance_explained": 1 - sse[name] / sst,
            "pca_variance_explained": 1 - pca_sse[name] / sst,
            "l0": l0[name] / n,
            "dead": int((fired[name] == 0).sum()),
            # dead features whose pre-activation never went above zero: ReLU blocks AuxK's gradient
            "dead_never_positive": int(((fired[name] == 0) & (max_pre[name] <= 0)).sum()),
            "dense": int((density > DENSE_THRESHOLD).sum()),
            "tokens": n,
        }
    return out


def write_markdown(r: dict, path: Path) -> None:
    b = r["baselines"]
    lines = [
        "# Milestone 3: sparse autoencoders",
        "",
        "Generated by `python -m glassbox.eval.sae_eval`. Do not edit by hand.",
        "",
        f"Top-k SAEs with {r['n_features']} features ({r['n_features'] / r['d_model']:.1f}x d_model = "
        f"{r['d_model']}) on the residual stream entering block {r['layer']} of {r['n_layer']}. Trained on "
        "activations from training windows that begin at a story boundary (<|endoftext|>), like text in the "
        "web UI; evaluated on non-overlapping validation windows of the same kind. Position 0 (always "
        "<|endoftext|>) is skipped everywhere.",
        "",
        f"- Reconstruction metrics: {r['recon_tokens']:,} validation tokens.",
        f"- Loss metrics: {r['loss_tokens']:,} validation tokens; the residual at positions >= 1 is replaced "
        "and the next-token loss is measured on those positions.",
        f"- Mean and PCA components for the baselines: {r['fit_tokens']:,} training-split tokens.",
        "",
        "## Baselines (next-token loss, nats)",
        "",
        "| residual stream at this layer | loss |",
        "|---|---:|",
        f"| unchanged (clean model) | {b['clean']:.4f} |",
        f"| replaced by its mean (mean ablation) | {b['mean_ablation']:.4f} |",
        f"| replaced by zeros (zero ablation) | {b['zero_ablation']:.4f} |",
        "",
        "## SAE vs PCA with the same number of active components",
        "",
        "Loss recovered = (mean-ablation loss - spliced loss) / (mean-ablation loss - clean loss).",
        "",
        "| k | variance explained | L0 | dead | dense (>10%) | loss spliced | loss increase | loss recovered "
        "| PCA-k variance explained | PCA-k loss increase | PCA-k loss recovered | SAE train time |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in r["saes"]:
        lines.append(
            f"| {s['k']} | {s['variance_explained']:.3f} | {s['l0']:.2f} | {s['dead']} ({s['dead'] / r['n_features']:.1%}) "
            f"| {s['dense']} | {s['loss_spliced']:.4f} | +{s['loss_increase']:.4f} (+{s['relative_loss_increase']:.1%}) "
            f"| {s['loss_recovered']:.1%} "
            f"| {s['pca_variance_explained']:.3f} | +{s['pca_loss_increase']:.4f} | {s['pca_loss_recovered']:.1%} "
            f"| {s['train_minutes']:.1f} min |"
        )
    lines += [
        "",
        "## Dead features",
        "",
        "Dead = never active on the reconstruction tokens. The second column counts dead features whose "
        "pre-activation never went above zero (the ReLU then passes them no gradient at all); the rest had "
        "positive pre-activations but were always outranked by k other features. The AuxK loss is the fraction "
        "of the remaining reconstruction error that the dead features fail to explain (averaged over the last "
        "logging interval): close to 1 means AuxK was active but did not bring them back.",
        "",
        "| k | dead on validation | of which pre-activation never > 0 | dead at end of training | final AuxK loss |",
        "|---:|---:|---:|---:|---:|",
    ]
    lines += [f"| {s['k']} | {s['dead']} | {s['dead_never_positive']} | {s['final_train_dead']} | "
              f"{s['final_train_aux']:.4f} |" for s in r["saes"]]
    sel = r["selected"]
    chosen = next(s for s in r["saes"] if s["k"] == sel["k"])
    lines += [
        "",
        "## SAE used by the web microscope",
        "",
        f"Rule (fixed in advance): {sel['rule']}",
        "",
        f"Selected: **k = {sel['k']}** (`{sel['run']}`). {sel['reason']}",
        "",
        f"Read the percentage with care: mean ablation raises the loss from {b['clean']:.2f} to "
        f"{b['mean_ablation']:.2f}, so even {chosen['loss_recovered']:.1%} recovered leaves the selected SAE "
        f"{chosen['relative_loss_increase']:.1%} above the clean loss. In hindsight the threshold is lenient for "
        "this model; it is kept because it was fixed before the results were seen. All three SAEs are available "
        "to later milestones.",
        "",
        f"SAE training: Adam, lr {r['train']['lr']:g} (best of a short pilot, `results/sae_lr_pilot.json`), "
        f"{r['train']['total_tokens']:,} tokens, batch {r['train']['batch_size']}, AuxK (k_aux {r['train']['k_aux']}, "
        f"coefficient {r['train']['aux_coef']:.4g}), seed {r['train']['seed']}.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--ckpt", type=Path, default=Path("runs/main/final.pt"))
    p.add_argument("--data-dir", type=Path, default=Path("data/tinystories"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--recon-tokens", type=int, default=1 << 20)
    p.add_argument("--loss-tokens", type=int, default=1 << 17)
    p.add_argument("--fit-windows", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()

    model, _ = load_checkpoint(a.ckpt)
    loaded = {run: load_sae(run / "sae.pt") for run in a.runs}
    saes = {run: s for run, (s, _) in loaded.items()}
    layers = {s.cfg.layer for s in saes.values()}
    n_features = {s.cfg.n_features for s in saes.values()}
    if len(layers) != 1 or len(n_features) != 1:
        raise SystemExit("all SAEs must share the layer and the number of features")
    layer = layers.pop()
    T = model.cfg.n_ctx

    eot_id = BPETokenizer.load(a.data_dir / "tokenizer.json").eot_id
    mean, comps, fit_tokens = fit_mean_and_pca(model, load_tokens(a.data_dir / "train.bin"), eot_id, layer,
                                               a.fit_windows, a.seed)
    val = load_tokens(a.data_dir / "val.bin")
    x_rec, _ = story_windows(val, eot_id, T, a.recon_tokens // T)
    x_loss, y_loss = story_windows(val, eot_id, T, a.loss_tokens // T)

    recon = reconstruction_stats(model, layer, x_rec, saes, mean, comps)
    replacements = {"clean": None, "mean_ablation": lambda r: mean.expand_as(r), "zero_ablation": torch.zeros_like}
    for run, sae in saes.items():
        replacements[f"sae:{run}"] = lambda r, s=sae: s(r)[0]
        replacements[f"pca:{run}"] = lambda r, k=sae.cfg.k: pca_reconstruct(r, mean, comps, k)
    losses = spliced_losses(model, layer, x_loss, y_loss, replacements)
    clean, mean_abl = losses["clean"], losses["mean_ablation"]

    def recovered(loss):
        return (mean_abl - loss) / (mean_abl - clean)

    rows = []
    for run, sae in saes.items():
        train_log = json.loads((run / "train_log.json").read_text(encoding="utf-8"))
        rows.append({
            "run": run.as_posix(), "k": sae.cfg.k, **recon[run],
            "loss_spliced": losses[f"sae:{run}"], "loss_increase": losses[f"sae:{run}"] - clean,
            "relative_loss_increase": (losses[f"sae:{run}"] - clean) / clean,
            "loss_recovered": recovered(losses[f"sae:{run}"]),
            "pca_loss_increase": losses[f"pca:{run}"] - clean, "pca_loss_recovered": recovered(losses[f"pca:{run}"]),
            "final_train_fvu": train_log["history"][-1]["fvu"], "train_minutes": train_log["elapsed_s"] / 60,
            "final_train_aux": train_log["history"][-1]["aux"], "final_train_dead": train_log["history"][-1]["dead"],
        })
    rows.sort(key=lambda s: s["k"])

    rule = (f"the smallest k whose loss recovered is >= {SELECT_MIN_RECOVERED:.0%}; "
            "if none qualifies, the largest k.")
    ok = [s for s in rows if s["loss_recovered"] >= SELECT_MIN_RECOVERED]
    chosen = ok[0] if ok else rows[-1]
    reason = (f"Loss recovered {chosen['loss_recovered']:.1%}." if ok else
              f"No SAE reached {SELECT_MIN_RECOVERED:.0%}; the largest k recovers {chosen['loss_recovered']:.1%}.")
    first_cfg = loaded[a.runs[0]][1]
    train_cfg = json.loads((a.runs[0] / "train_log.json").read_text(encoding="utf-8"))["config"]
    report = {
        "layer": layer, "n_layer": model.cfg.n_layer, "d_model": model.cfg.d_model,
        "n_features": n_features.pop(), "recon_tokens": recon[a.runs[0]]["tokens"],
        "loss_tokens": int(y_loss[:, 1:].size), "fit_tokens": fit_tokens,
        "baselines": {"clean": clean, "mean_ablation": mean_abl, "zero_ablation": losses["zero_ablation"]},
        "saes": rows,
        "selected": {"k": chosen["k"], "run": chosen["run"], "rule": rule, "reason": reason},
        "train": {key: train_cfg[key] for key in ("lr", "total_tokens", "batch_size", "k_aux", "aux_coef", "seed")},
        "source_checkpoint": first_cfg.get("source_checkpoint"),
        "seconds": round(time.time() - t0, 1),
    }
    a.results_dir.mkdir(parents=True, exist_ok=True)
    (a.results_dir / "sae_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, a.results_dir / "sae_eval.md")
    for s in rows:
        print(f"k={s['k']:3d}: VE {s['variance_explained']:.3f} | L0 {s['l0']:.1f} | dead {s['dead']} | "
              f"loss +{s['loss_increase']:.4f} | recovered {s['loss_recovered']:.1%} | PCA-k recovered "
              f"{s['pca_loss_recovered']:.1%}")
    print(f"selected k={chosen['k']} ({chosen['run']}); wrote {a.results_dir / 'sae_eval.md'}")


if __name__ == "__main__":
    main()
