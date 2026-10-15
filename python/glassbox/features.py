"""Feature analysis for a trained SAE: densities, top-activating examples, automatic labels,
and a token-specificity comparison against non-SAE bases.

    python -m glassbox.features --model-dir web/model --sae-dir web/model/sae

Uses the exported (float16) model and SAE, i.e. exactly what the browser runs, on validation
windows that begin at a story boundary. Position 0 (<|endoftext|>) is skipped, as in training.

Automatic label: over a feature's strongest distinct examples (one per window and one per
4-token context, so "Once upon a time" counts once), look at the current
token, the previous token and the next token. Whichever is most concentrated becomes the
label: '"ball"' (current token), 'after "named"' (previous token) or 'before "said"' (next
token), together with its share of the examples.

Token specificity: for each direction (an SAE feature, a residual-stream dimension, a random
direction, or a feature of an untrained SAE), the share of its top examples that sit on its
single most common token. It measures one narrow kind of interpretability: features that
track a context rather than a token score low even when they are meaningful.

Writes web/model/sae/features.json (for the UI) and results/sae_features.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .export import load_exported, load_exported_sae
from .sae import TopKSAE, story_windows
from .tokenizer import BPETokenizer
from .train import load_tokens

POSITION_BUCKETS = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 64), (64, 256)]
N_EXAMPLES_UI = 8
N_EXAMPLES_STATS = 20
MIN_EXAMPLES_FOR_METRIC = 10
CONTEXT_BEFORE, CONTEXT_AFTER = 16, 4


class TopTracker:
    """For each of n_dirs directions, the n_cand largest activations seen and where."""

    def __init__(self, n_dirs: int, n_cand: int):
        self.n_cand = n_cand
        self.acts = torch.full((n_dirs, n_cand), float("-inf"))
        self.where = torch.full((n_dirs, n_cand), -1, dtype=torch.long)

    def update(self, acts: torch.Tensor, where: torch.Tensor) -> None:
        """acts: (N, n_dirs); where: (N,) global positions."""
        cand_a = torch.cat([self.acts, acts.T], dim=1)
        cand_w = torch.cat([self.where, where.expand(acts.shape[1], -1)], dim=1)
        top = cand_a.topk(self.n_cand, dim=1)
        self.acts, self.where = top.values, cand_w.gather(1, top.indices)


CONTEXT_KEY = 4  # examples are distinct if their last 4 tokens (up to and including the token) differ


def distinct_examples(
    acts: list[float], where: list[int], x: np.ndarray, n: int, min_act: float
) -> list[tuple[float, int, int]]:
    """Strongest activations (already sorted descending), at most one per window and at most
    one per local context, so a formula such as "Once upon a time" counts once."""
    T = x.shape[1]
    seen_windows, seen_contexts, out = set(), set(), []
    for a, w in zip(acts, where):
        if w < 0 or not a > min_act:
            break
        win, pos = divmod(w, T)
        context = tuple(x[win, max(0, pos - CONTEXT_KEY + 1) : pos + 1].tolist())
        if win in seen_windows or context in seen_contexts:
            continue
        seen_windows.add(win)
        seen_contexts.add(context)
        out.append((a, win, pos))
        if len(out) == n:
            break
    return out


def top_share(ids: list[int]) -> tuple[int, float]:
    tok, count = Counter(ids).most_common(1)[0]
    return tok, count / len(ids)


def auto_label(tok: BPETokenizer, cur: list[int], prev: list[int], nxt: list[int]) -> dict:
    kinds = [("token", cur), ("after", prev), ("before", nxt)]  # order breaks ties
    best = max(((kind, *top_share(ids)) for kind, ids in kinds), key=lambda t: t[2])
    kind, token, share = best
    name = json.dumps(tok.token_str(token))
    text = name if kind == "token" else f"{kind} {name}"
    return {"label": text, "label_kind": kind, "label_token": token, "label_share": share}


def distribution(ids: list[int], k: int = 3) -> list[list]:
    return [[t, c / len(ids)] for t, c in Counter(ids).most_common(k)]


@torch.no_grad()
def scan(model, sae: TopKSAE, bases: dict, x: np.ndarray, layer: int, mean: torch.Tensor, n_cand: int,
         batch_windows: int = 16):
    """One pass over the windows: SAE firing counts and top activations for every basis.
    bases maps a name to either a TopKSAE or a (d, n_dirs) matrix of directions."""
    n, T = x.shape
    F = sae.cfg.n_features
    trackers = {"sae": TopTracker(F, n_cand)}
    for name, b in bases.items():
        trackers[name] = TopTracker(b.cfg.n_features if isinstance(b, TopKSAE) else b.shape[1], n_cand)
    fired = torch.zeros(F, dtype=torch.long)
    total = 0
    pos = torch.arange(1, T)
    for i in range(0, n, batch_windows):
        xb = torch.from_numpy(x[i : i + batch_windows])
        r = model.resid_at(xb, layer)[:, 1:]
        flat = r.reshape(-1, r.shape[-1])
        where = ((torch.arange(i, i + xb.shape[0])[:, None] * T) + pos[None, :]).reshape(-1)
        codes = sae.encode(flat)
        fired += (codes > 0).sum(0)
        total += flat.shape[0]
        trackers["sae"].update(codes, where)
        for name, b in bases.items():
            acts = b.encode(flat) if isinstance(b, TopKSAE) else (flat - mean) @ b
            trackers[name].update(acts, where)
    return trackers, fired, total


@torch.no_grad()
def snippets(model, sae: TopKSAE, x: np.ndarray, layer: int, requests: dict[int, list[tuple[int, int, dict]]],
             batch_windows: int = 16) -> None:
    """Fill each example dict with context tokens and the feature's activation on each of them.
    requests maps window -> [(feature, position, example dict)]."""
    T = x.shape[1]
    wins = sorted(requests)
    for i in range(0, len(wins), batch_windows):
        chunk = wins[i : i + batch_windows]
        xb = torch.from_numpy(x[chunk])
        codes = sae.encode(model.resid_at(xb, layer))
        codes[:, 0] = 0  # position 0 is excluded everywhere
        for j, win in enumerate(chunk):
            for f, p, ex in requests[win]:
                lo, hi = max(0, p - CONTEXT_BEFORE), min(T, p + CONTEXT_AFTER + 1)
                ex["tokens"] = x[win, lo:hi].tolist()
                ex["acts_raw"] = codes[j, lo:hi, f].tolist()
                ex["at"] = p - lo


def specificity(tracker: TopTracker, x: np.ndarray, min_act: float, alive: torch.Tensor | None = None) -> dict:
    shares, skipped = [], 0
    for d in range(tracker.acts.shape[0]):
        if alive is not None and not alive[d]:
            continue
        ex = distinct_examples(tracker.acts[d].tolist(), tracker.where[d].tolist(), x, N_EXAMPLES_STATS, min_act)
        if len(ex) < MIN_EXAMPLES_FOR_METRIC:
            skipped += 1
            continue
        shares.append(top_share([int(x[w, p]) for _, w, p in ex])[1])
    s = np.array(shares)
    return {"directions_scored": len(s), "directions_skipped": skipped,
            "mean_top_token_share": float(s.mean()) if len(s) else None,
            "share_ge_0.5": float((s >= 0.5).mean()) if len(s) else None,
            "share_ge_0.8": float((s >= 0.8).mean()) if len(s) else None}


def highlight(tok: BPETokenizer, ex: dict) -> str:
    """Example as text, with the strongest token wrapped in [[ ]]."""
    t = ex["tokens"]
    s = tok.decode(t[: ex["at"]]) + "[[" + tok.decode([t[ex["at"]]]) + "]]" + tok.decode(t[ex["at"] + 1 :])
    return s.replace("\n", " ").replace("|", "\\|")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=Path("web/model"))
    p.add_argument("--sae-dir", type=Path, default=Path("web/model/sae"))
    p.add_argument("--data-dir", type=Path, default=Path("data/tinystories"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--eval-tokens", type=int, default=2 << 20)
    p.add_argument("--n-cand", type=int, default=1000,
                   help="strongest activations kept per direction before removing repeated contexts")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--markdown-only", action="store_true", help="rewrite the .md from the existing .json")
    a = p.parse_args()
    if a.markdown_only:
        report = json.loads((a.results_dir / "sae_features.json").read_text(encoding="utf-8"))
        write_markdown(report, a.results_dir / "sae_features.md")
        return
    torch.set_num_threads(a.threads)
    t0 = time.time()

    model, tok, model_manifest = load_exported(a.model_dir)
    sae, sae_manifest = load_exported_sae(a.sae_dir)
    if sae_manifest["model_weights_sha256"] != model_manifest["weights_sha256"]:
        raise SystemExit("this SAE export was made for a different model export")
    layer, d, T = sae.cfg.layer, model.cfg.d_model, model.cfg.n_ctx
    x, y = story_windows(load_tokens(a.data_dir / "val.bin"), tok.eot_id, T, a.eval_tokens // T)

    # Bases to compare against, all deterministic.
    g = torch.Generator().manual_seed(a.seed)
    with torch.no_grad():
        mean = model.resid_at(torch.from_numpy(x[:64]), layer)[:, 1:].reshape(-1, d).mean(0)
    rand = torch.randn(d, 2 * d, generator=g)
    untrained = TopKSAE(sae.cfg)
    untrained.init_weights(g, mean)
    bases = {
        "residual dimensions (+/-)": torch.cat([torch.eye(d), -torch.eye(d)], dim=1),
        "random directions": rand / rand.norm(dim=0, keepdim=True),
        "untrained SAE (same k)": untrained,
    }
    trackers, fired, total = scan(model, sae, bases, x, layer, mean, a.n_cand)
    density = (fired.double() / total).tolist()
    alive = fired > 0

    # Per-feature examples and labels.
    features, requests = [], defaultdict(list)
    example_positions = []
    tr = trackers["sae"]
    for f in range(sae.cfg.n_features):
        if not alive[f]:
            features.append({"id": f, "dead": True, "density": 0.0})
            continue
        ex = distinct_examples(tr.acts[f].tolist(), tr.where[f].tolist(), x, N_EXAMPLES_STATS, 0.0)
        example_positions += [p for _, _, p in ex]
        cur = [int(x[w, p]) for _, w, p in ex]
        prev = [int(x[w, p - 1]) for _, w, p in ex]
        nxt = [int(y[w, p]) for _, w, p in ex]
        feat = {"id": f, "density": density[f], "max_act": ex[0][0], "n_examples": len(ex),
                **auto_label(tok, cur, prev, nxt),
                "top_cur": distribution(cur), "top_prev": distribution(prev), "top_next": distribution(nxt),
                "examples": []}
        for act, w, pos in ex[:N_EXAMPLES_UI]:
            e = {"act": act}
            feat["examples"].append(e)
            requests[w].append((f, pos, e))
        features.append(feat)
    snippets(model, sae, x, layer, requests)
    for feat in features:
        for e in feat.get("examples", []):
            e["acts"] = [int(round(100 * max(v, 0.0) / feat["max_act"])) for v in e.pop("acts_raw")]
            e["act"] = round(e["act"], 3)

    # Token specificity vs the other bases.
    spec = {"SAE (trained)": specificity(trackers["sae"], x, 0.0, alive)}
    for name, b in bases.items():
        min_act = 0.0 if isinstance(b, TopKSAE) else float("-inf")
        spec[name] = specificity(trackers[name], x, min_act)

    # Where the strongest examples sit, and how large the residual stream is there.
    with torch.no_grad():
        norms = model.resid_at(torch.from_numpy(x[:64]), layer).norm(dim=-1).mean(0)
    pos_arr = np.array(example_positions)
    positions = [
        {"positions": f"{lo}-{hi - 1}", "share_of_examples": float(((pos_arr >= lo) & (pos_arr < hi)).mean()),
         "share_if_uniform": (min(hi, T) - lo) / (T - 1), "mean_residual_norm": float(norms[lo:hi].mean())}
        for lo, hi in POSITION_BUCKETS
    ]

    alive_feats = [f for f in features if not f.get("dead")]
    labelled = [f for f in alive_feats if f["n_examples"] >= MIN_EXAMPLES_FOR_METRIC]
    dens = np.array([f["density"] for f in alive_feats])
    shares = np.array([f["label_share"] for f in labelled])
    rng = np.random.default_rng(a.seed)
    sample_ids = sorted(rng.choice([f["id"] for f in alive_feats], size=min(12, len(alive_feats)), replace=False))
    report = {
        "sae": sae_manifest["config"],
        "eval_tokens": total,
        "eval_windows": int(x.shape[0]),
        "n_features": sae.cfg.n_features,
        "dead": int((~alive).sum()),
        "density_buckets": {
            "<1e-4": int((dens < 1e-4).sum()), "1e-4..1e-3": int(((dens >= 1e-4) & (dens < 1e-3)).sum()),
            "1e-3..1e-2": int(((dens >= 1e-3) & (dens < 1e-2)).sum()),
            "1e-2..1e-1": int(((dens >= 1e-2) & (dens < 1e-1)).sum()), ">=0.1": int((dens >= 0.1).sum()),
        },
        "token_specificity": spec,
        "example_positions": positions,
        "labels": {
            "features_with_enough_examples": len(labelled),
            "too_rare": len(alive_feats) - len(labelled),
            "kind": dict(Counter(f["label_kind"] for f in labelled)),
            "share_ge_0.8": int((shares >= 0.8).sum()), "share_0.5_to_0.8": int(((shares >= 0.5) & (shares < 0.8)).sum()),
            "share_lt_0.5": int((shares < 0.5).sum()),
        },
        "random_sample": [
            {"id": f["id"], "label": f["label"], "label_share": f["label_share"], "density": f["density"],
             "n_examples": f["n_examples"], "top_example": highlight(tok, f["examples"][0])}
            for f in alive_feats if f["id"] in set(sample_ids)
        ],
        "settings": {"examples_for_stats": N_EXAMPLES_STATS, "examples_for_ui": N_EXAMPLES_UI,
                     "distinct_context_tokens": CONTEXT_KEY, "candidates": a.n_cand,
                     "min_examples_for_metric": MIN_EXAMPLES_FOR_METRIC, "seed": a.seed},
        "seconds": round(time.time() - t0, 1),
    }

    ui = {"format": "glassbox-features", "version": 1, "sae": sae_manifest["config"],
          "sae_weights_sha256": sae_manifest["weights_sha256"], "eval_tokens": total,
          "context_before": CONTEXT_BEFORE, "context_after": CONTEXT_AFTER, "features": features}
    for f in features:
        if "label_share" in f:
            f["label_share"] = round(f["label_share"], 3)
            f["density"] = float(f"{f['density']:.4g}")
            f["max_act"] = round(f["max_act"], 3)
            for key in ("top_cur", "top_prev", "top_next"):
                f[key] = [[t, round(s, 3)] for t, s in f[key]]
    (a.sae_dir / "features.json").write_text(json.dumps(ui, separators=(",", ":")), encoding="utf-8")
    a.results_dir.mkdir(parents=True, exist_ok=True)
    (a.results_dir / "sae_features.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, a.results_dir / "sae_features.md")
    print(f"{len(alive_feats)} alive / {sae.cfg.n_features} features; wrote {a.sae_dir / 'features.json'} "
          f"and {a.results_dir / 'sae_features.md'} in {report['seconds']} s")


def position_summary(buckets: list[dict]) -> str:
    """One sentence comparing where the strongest examples sit with a uniform spread."""
    over = [f"{b['positions']} ({b['share_of_examples'] / b['share_if_uniform']:.2f}x)" for b in buckets
            if b["share_of_examples"] > 1.25 * b["share_if_uniform"]]
    under = [f"{b['positions']} ({b['share_of_examples'] / b['share_if_uniform']:.2f}x)" for b in buckets
             if b["share_of_examples"] < 0.8 * b["share_if_uniform"]]
    parts = []
    if over:
        parts.append("over-represented at positions " + ", ".join(over))
    if under:
        parts.append("under-represented at " + ", ".join(under))
    if not parts:
        return "The strongest examples are spread roughly uniformly over positions."
    return ("Compared with a uniform spread, the strongest examples are " + "; ".join(parts) +
            " (ratio to uniform). Residual norms are larger early in the window (last column), which may "
            "contribute, since larger inputs give larger SAE activations.")


def write_markdown(r: dict, path: Path) -> None:
    s = r["sae"]
    lines = [
        "# Milestone 3: SAE features",
        "",
        "Generated by `python -m glassbox.features`. Do not edit by hand.",
        "",
        f"SAE: {r['n_features']} features, top-k with k = {s['k']}, residual stream entering block {s['layer']}. "
        f"Statistics over {r['eval_tokens']:,} validation tokens ({r['eval_windows']:,} windows, position 0 "
        "skipped), using the exported float16 weights that the browser runs.",
        "",
        "## Feature density",
        "",
        f"- Dead (never fired): {r['dead']} / {r['n_features']} ({r['dead'] / r['n_features']:.1%})",
        "",
        "| fraction of tokens a feature fires on | features |",
        "|---|---:|",
    ]
    lines += [f"| {k} | {v} |" for k, v in r["density_buckets"].items()]
    st = r["settings"]
    lines += [
        "",
        "## Token specificity vs other bases",
        "",
        f"For each direction: the share of its {st['examples_for_stats']} strongest examples that "
        "sit on its most common token. Examples are distinct: at most one per window and one per "
        f"{st['distinct_context_tokens']}-token context, so a stock phrase counts once. Directions with fewer than "
        f"{st['min_examples_for_metric']} examples are skipped. This measures token-level features only; "
        "features that track context score low even when they are meaningful.",
        "",
        "| basis | directions scored | mean top-token share | share >= 0.5 | share >= 0.8 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, v in r["token_specificity"].items():
        if v["directions_scored"]:
            lines.append(f"| {name} | {v['directions_scored']:,} | {v['mean_top_token_share']:.3f} | "
                         f"{v['share_ge_0.5']:.1%} | {v['share_ge_0.8']:.1%} |")
        else:
            lines.append(f"| {name} | 0 | - | - | - |")
    lines += [
        "",
        "## Where the strongest examples sit",
        "",
        "Position within the window (0 is <|endoftext|>, 1 is the first token of a story). "
        + position_summary(r["example_positions"]),
        "",
        "| positions | share of strongest examples | share if uniform | mean residual norm |",
        "|---|---:|---:|---:|",
    ]
    lines += [f"| {b['positions']} | {b['share_of_examples']:.1%} | {b['share_if_uniform']:.1%} | "
              f"{b['mean_residual_norm']:.1f} |" for b in r["example_positions"]]
    lb = r["labels"]
    lines += [
        "",
        "## Automatic labels",
        "",
        f"Over the {lb['features_with_enough_examples']:,} live features with at least "
        f"{st['min_examples_for_metric']} distinct examples. The other {lb['too_rare']:,} live features either "
        f"fire rarely or keep firing in the same few contexts (their {st['candidates']:,} strongest activations "
        "repeat the same 4-token context), so their labels rest on few examples; the UI shows how many.",
        "",
        "Label kind (which of current / previous / next token is most concentrated in the top examples): "
        + ", ".join(f"{k}: {v}" for k, v in sorted(lb["kind"].items())) + ".",
        "",
        f"- Label share >= 0.8: {lb['share_ge_0.8']} features",
        f"- 0.5 to 0.8: {lb['share_0.5_to_0.8']} features",
        f"- below 0.5 (no single token explains the feature): {lb['share_lt_0.5']} features",
        "",
        f"## {len(r['random_sample'])} features chosen at random (seed {st['seed']}), not cherry-picked",
        "",
        "Strongest example shown with the top-activating token in [[ ]].",
        "",
        "| feature | label | share | examples | density | strongest example |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for f in r["random_sample"]:
        lines.append(f"| {f['id']} | `{f['label']}` | {f['label_share']:.2f} | {f['n_examples']} | "
                     f"{f['density']:.2e} | {f['top_example']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
