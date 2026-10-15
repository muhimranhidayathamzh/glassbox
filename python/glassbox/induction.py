"""Training replay: induction heads and loss across the log-spaced checkpoints saved by train.py.

    python -m glassbox.induction        # -> results/induction.{json,md}, web/model/replay.json

Test input: random token sequences repeated twice, [<|endoftext|>, r_1..r_L, r_1..r_L]. Nothing in
the first copy is predictable; the second copy is, if the model copies from earlier context.

Per checkpoint and per attention head:
  * prefix-matching score (induction): attention from the second copy of r_j to r_{j+1} in the
    first copy, i.e. to the token that followed the earlier occurrence of the current token
  * previous-token score: attention from each position to the one just before it (induction heads
    rely on such a head in an earlier layer)
and for the whole model the loss on the first and on the second copy.

Follow-up probes, added after the primary probe came out negative (reported as such): the same
measurements on repeats of the 500 most frequent training tokens, and on repeated stretches of
natural validation text. Positive control: a tiny 2-layer model trained on repeated sequences, to
show that the probe reads high when a model does attend to the earlier copy.

Formation and loss rule (fixed before looking at the data):
  * the formation interval is the pair of consecutive checkpoints with the largest increase in the
    best head's prefix-matching score
  * it "coincides with a sharp loss drop" if the validation loss falls at least SHARP_RATIO times
    faster there (per unit of ln(step + 1)) than the mean rate of the neighbouring intervals
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import GPT, GPTConfig
from .sae import story_windows
from .tokenizer import BPETokenizer
from .train import load_checkpoint, load_tokens

REPEAT_LEN = 50
N_SEQUENCES = 64
UI_REPEAT_LEN = 10
SAMPLE_PROMPTS = ["Once upon a time", "Lily and Ben went to the", "The little dog was"]
SAMPLE_TOKENS = 48
SHARP_RATIO = 2.0
FREQUENT_TOKENS = 500
INDUCTION_THRESHOLD = 0.5


def repeated_sequences(n: int, L: int, low: int, high: int, eot: int, seed: int) -> torch.Tensor:
    """(n, 2L + 1): <|endoftext|>, then L random tokens from [low, high), then the same L again."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randint(low, high, (n, L), generator=g)
    return torch.cat([torch.full((n, 1), eot), r, r], dim=1)


@torch.no_grad()
def attention_patterns(model, idx: torch.Tensor) -> torch.Tensor:
    """(n_layer, B, n_head, T, T) attention weights."""
    x = model.embed(idx)
    out = []
    for block in model.blocks:
        out.append(block.attn.attention_pattern(block.ln1(x)))
        x = block(x)
    return torch.stack(out)


def head_scores(patterns: torch.Tensor, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Prefix-matching and previous-token scores, each (n_layer, n_head)."""
    j = torch.arange(L)
    queries, keys = 1 + L + j, 2 + j  # second copy of r_j -> first copy of r_{j+1}
    prefix = patterns[..., queries, keys].mean(dim=(1, 3))
    pos = torch.arange(1, patterns.shape[-1])
    prev = patterns[..., pos, pos - 1].mean(dim=(1, 3))
    return prefix.numpy(), prev.numpy()


def uniform_prefix_baseline(L: int) -> float:
    """Prefix-matching score of a head that spreads attention evenly over all earlier positions."""
    return float(np.mean([1 / (q + 1) for q in range(1 + L, 1 + 2 * L)]))


@torch.no_grad()
def repeat_losses(model, idx: torch.Tensor, L: int) -> tuple[float, float]:
    """Mean loss predicting r_2..r_L in the first copy and in the second copy."""
    logits, _ = model(idx[:, :-1])
    nll = F.cross_entropy(logits.transpose(1, 2), idx[:, 1:], reduction="none")  # nll[:, i] predicts idx[i+1]
    first = nll[:, 1:L].mean()  # targets at positions 2..L
    second = nll[:, L + 1 : 2 * L].mean()  # targets at positions L+2..2L
    return float(first), float(second)


def formation_interval(scores: list[float]) -> int:
    """Index i of the consecutive pair (i, i+1) with the largest increase in score."""
    return int(np.argmax(np.diff(scores)))


def sharp_drop(steps: list[int], losses: list[float], i: int) -> dict:
    """Loss drop per unit ln(step + 1) in interval i vs the mean of its neighbouring intervals."""
    def rate(k):
        return (losses[k] - losses[k + 1]) / (math.log(steps[k + 1] + 1) - math.log(steps[k] + 1))

    neighbours = [k for k in (i - 1, i + 1) if 0 <= k < len(steps) - 1]
    here = rate(i)
    around = float(np.mean([rate(k) for k in neighbours]))
    ratio = here / around if around > 0 else math.inf
    return {"rate": here, "neighbour_rate": around, "ratio": ratio, "sharp": bool(ratio >= SHARP_RATIO)}


def first_step_above(steps: list[int], values: list[float], threshold: float) -> int | None:
    return next((s for s, v in zip(steps, values) if v >= threshold), None)


def probe(model, seqs: torch.Tensor, L: int) -> dict:
    prefix, prev = head_scores(attention_patterns(model, seqs), L)
    first, second = repeat_losses(model, seqs, L)
    return {"best_prefix": float(prefix.max()), "best_prev": float(prev.max()),
            "repeat_loss_first": first, "repeat_loss_second": second}


def positive_control(variable_length: bool, steps: int, seed: int = 0) -> list[dict]:
    """Train a tiny 2-layer model on sequences containing a repeated random segment and probe it.

    With a fixed repeat length the earlier copy can be found by position alone; with a variable
    length and offset only content matching works. Probed on the standard test (fixed length)."""
    torch.manual_seed(seed)
    V, T, L = 64, 64, 31
    model = GPT(GPTConfig(vocab_size=V, n_ctx=T, d_model=64, n_layer=2, n_head=2, d_mlp=256))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    test = repeated_sequences(64, L, 0, V - 1, V - 1, seed=seed + 123)
    history = []
    for step in range(1, steps + 1):
        g = torch.Generator().manual_seed(seed * 100_000 + step)
        if variable_length:
            seq = torch.randint(0, V - 1, (32, T), generator=g)
            seq[:, 0] = V - 1
            for b in range(32):
                n = int(torch.randint(8, 30, (1,), generator=g))
                start = int(torch.randint(1, T - 2 * n + 1, (1,), generator=g))
                seq[b, start + n : start + 2 * n] = seq[b, start : start + n]
        else:
            seq = repeated_sequences(32, L, 0, V - 1, V - 1, seed=seed * 100_000 + step)
        _, loss = model(seq[:, :-1], seq[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == steps:
            model.eval()
            history.append({"step": step, **probe(model, test, L)})
            model.train()
    return history


def write_markdown(r: dict, path: Path) -> None:
    f, c = r["formation"], r["checkpoints"]
    lines = [
        "# Milestone 5: training replay and induction heads",
        "",
        "Generated by `python -m glassbox.induction`. Do not edit by hand.",
        "",
        f"Each of the {len(c)} checkpoints saved during training (log-spaced in steps) is given "
        f"{r['n_sequences']} sequences of {r['repeat_len']} random tokens repeated twice. A head's **prefix-matching "
        "score** is its attention from the second copy of a token to the token that followed the first copy "
        "(1 = a perfect induction head). Its **previous-token score** is its attention to the position just "
        f"before. A head spreading attention evenly would score {r['uniform_prefix_baseline']:.3f} on prefix "
        "matching. Repeat loss: the model's loss on the first copy (unpredictable) and on the second copy "
        "(predictable by copying).",
        "",
        "| step | tokens seen | val loss | best induction head | prefix score | best previous-token head | "
        "prev score | loss, 1st copy | loss, 2nd copy |",
        "|---:|---:|---:|---|---:|---|---:|---:|---:|",
    ]
    for k in c:
        lines.append(
            f"| {k['step']:,} | {k['tokens']:,} | {k['val_loss']:.3f} | {k['best_induction_head']} | "
            f"{k['best_prefix']:.3f} | {k['best_prev_head']} | {k['best_prev']:.3f} | "
            f"{k['repeat_loss_first']:.2f} | {k['repeat_loss_second']:.2f} |"
        )
    formed = f["induction_step"] is not None
    lines += [
        "",
        "## Did induction heads form?",
        "",
        (f"Yes: the first checkpoint with a head at prefix-matching score >= {r['induction_threshold']:g} is step "
         f"{f['induction_step']:,}." if formed else
         f"**No.** No head reaches a prefix-matching score of {r['induction_threshold']:g} at any checkpoint: the "
         f"best at the end of training is {f['final_score']:.3f} ({f['final_head']}), against "
         f"{r['uniform_prefix_baseline']:.3f} for evenly spread attention, and the loss on the second copy is "
         f"hardly lower than on the first ({c[-1]['repeat_loss_second']:.2f} vs {c[-1]['repeat_loss_first']:.2f}). "
         "The model has not learned to copy from earlier context."),
        "",
        f"- Largest jump in the best prefix-matching score: between step {f['from_step']:,} and step "
        f"{f['to_step']:,} ({f['score_from']:.3f} -> {f['score_to']:.3f}).",
        f"- Best induction head at the end of training: {f['final_head']} (score {f['final_score']:.3f}); "
        f"previous-token heads at the end: {', '.join(f['final_prev_heads']) or 'none above threshold'}.",
        f"- First checkpoint with a previous-token score >= {r['induction_threshold']:g}: "
        f"{'step ' + format(f['prev_head_step'], ',') if f['prev_head_step'] is not None else 'none'}; the best "
        f"previous-token score grows from {c[0]['best_prev']:.3f} to {c[-1]['best_prev']:.3f} (previous-token "
        "heads are the first half of an induction circuit).",
        f"- Repeat loss on the second copy goes from {f['second_copy_from']:.2f} to {f['second_copy_to']:.2f} "
        "across the formation interval (first copy stays near "
        f"{f['first_copy_mean']:.2f}, the cost of guessing a random token).",
        f"- Validation loss falls {f['drop']['rate']:.3f} per unit of ln(step + 1) in the formation interval, vs "
        f"{f['drop']['neighbour_rate']:.3f} in the neighbouring intervals (ratio {f['drop']['ratio']:.2f}).",
        "",
        f"Rule fixed in advance: a ratio of at least {r['sharp_ratio']:g} counts as a sharp drop coinciding with "
        f"induction-head formation. **Result: {'yes' if f['drop']['sharp'] else 'no'}.**"
        + ("" if formed else " Since no induction head formed, the interval only tracks a small change and the "
           "test has little meaning here; it is reported because it was planned."),
        "",
        f"Resolution is limited by the checkpoints: {len(c)} log-spaced steps, so a formation interval can span "
        "a factor of about 1.5 in steps.",
        "",
        "## Follow-up probes",
        "",
        "Added after the primary probe came out negative, to check that rare random tokens were not the problem. "
        f"Frequent: repeats of random tokens drawn from the {r['frequent_tokens']} most frequent training tokens. "
        "Natural: stretches of validation stories repeated twice.",
        "",
        "| step | prefix score, frequent | loss 1st -> 2nd copy, frequent | prefix score, natural | "
        "loss 1st -> 2nd copy, natural |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in c:
        fr, na = k["frequent"], k["natural"]
        lines.append(f"| {k['step']:,} | {fr['best_prefix']:.3f} | {fr['repeat_loss_first']:.2f} -> "
                     f"{fr['repeat_loss_second']:.2f} | {na['best_prefix']:.3f} | {na['repeat_loss_first']:.2f} -> "
                     f"{na['repeat_loss_second']:.2f} |")
    lines += [
        "",
        "## Positive control: does the probe detect copying when it exists?",
        "",
        "A tiny 2-layer model (d_model 64, vocabulary 64) trained on sequences with a repeated random segment, "
        "probed with the same code on repeats of length 31.",
        "",
        "| training data | step | prefix score | loss 1st -> 2nd copy |",
        "|---|---:|---:|---:|",
    ]
    for name, hist in (("fixed repeat length (copy can be found by position)", r["control_fixed"]),
                       ("variable length and offset (only content matching works)", r["control_variable"])):
        for h in hist:
            lines.append(f"| {name} | {h['step']:,} | {h['best_prefix']:.3f} | {h['repeat_loss_first']:.2f} -> "
                         f"{h['repeat_loss_second']:.2f} |")
    lines += [
        "",
        "When a model does attend to the token after the earlier copy, the probe reads close to 1 and the copy "
        "loss collapses, so the measurement works. Learning to find the copy by its content is slower: in "
        "this small control it had not happened within the steps shown.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=Path("runs/main"))
    p.add_argument("--tokenizer", type=Path, default=Path("data/tinystories/tokenizer.json"))
    p.add_argument("--data-dir", type=Path, default=Path("data/tinystories"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--ui-out", type=Path, default=Path("web/model/replay.json"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--markdown-only", action="store_true", help="rewrite the .md from the existing .json")
    a = p.parse_args()
    if a.markdown_only:
        report = json.loads((a.results_dir / "induction.json").read_text(encoding="utf-8"))
        write_markdown(report, a.results_dir / "induction.md")
        return
    torch.set_num_threads(a.threads)
    t0 = time.time()

    tok = BPETokenizer.load(a.tokenizer)
    log = json.loads((a.run_dir / "train_log.json").read_text(encoding="utf-8"))
    paths = sorted(a.run_dir.glob("ckpt_step*.pt"), key=lambda q: int(re.search(r"(\d+)", q.stem).group(1)))
    eot, low, high = tok.eot_id, 256, tok.eot_id  # merged tokens only: raw bytes are mostly unseen
    seqs = repeated_sequences(N_SEQUENCES, REPEAT_LEN, low, high, eot, a.seed)
    ui_seq = repeated_sequences(1, UI_REPEAT_LEN, low, high, eot, a.seed + 1)
    prompts = [[eot] + tok.encode(pr) for pr in SAMPLE_PROMPTS]
    counts = np.bincount(np.asarray(load_tokens(a.data_dir / "train.bin")), minlength=tok.vocab_size)
    counts[eot] = 0
    frequent = torch.from_numpy(np.argsort(-counts, kind="stable")[:FREQUENT_TOKENS].copy())
    g = torch.Generator().manual_seed(a.seed + 2)
    r_freq = frequent[torch.randint(0, FREQUENT_TOKENS, (N_SEQUENCES, REPEAT_LEN), generator=g)]
    freq_seqs = torch.cat([torch.full((N_SEQUENCES, 1), eot), r_freq, r_freq], dim=1)
    xs, _ = story_windows(load_tokens(a.data_dir / "val.bin"), eot, REPEAT_LEN + 1, N_SEQUENCES)
    r_nat = torch.from_numpy(xs[:, 1:])
    natural_seqs = torch.cat([torch.full((N_SEQUENCES, 1), eot), r_nat, r_nat], dim=1)
    tokens_per_step = log["config"]["batch_size"] * log["config"]["model"]["n_ctx"]

    ckpts = []
    for path in paths:
        model, ckpt = load_checkpoint(path)
        prefix, prev = head_scores(attention_patterns(model, seqs), REPEAT_LEN)
        first, second = repeat_losses(model, seqs, REPEAT_LEN)
        pats = attention_patterns(model, ui_seq)[:, 0]  # (n_layer, n_head, T, T)
        samples = []
        for pr, text in zip(prompts, SAMPLE_PROMPTS):
            out = model.generate(torch.tensor([pr]), SAMPLE_TOKENS, temperature=0, stop_token=eot)[0, len(pr):]
            samples.append(text + tok.decode([t for t in out.tolist() if t != eot]))
        bi, bp = np.unravel_index(prefix.argmax(), prefix.shape), np.unravel_index(prev.argmax(), prev.shape)
        ckpts.append({
            "step": ckpt["step"], "tokens": ckpt["step"] * tokens_per_step, "val_loss": ckpt["val_loss"],
            "prefix": prefix.round(4).tolist(), "prev": prev.round(4).tolist(),
            "best_induction_head": f"L{bi[0]}.H{bi[1]}", "best_prefix": float(prefix.max()),
            "best_prev_head": f"L{bp[0]}.H{bp[1]}", "best_prev": float(prev.max()),
            "repeat_loss_first": first, "repeat_loss_second": second, "samples": samples,
            "frequent": probe(model, freq_seqs, REPEAT_LEN), "natural": probe(model, natural_seqs, REPEAT_LEN),
            "patterns": (pats * 100).round().int().flatten(start_dim=2).tolist(),
        })
        print(f"step {ckpt['step']:5d} | val {ckpt['val_loss']:.3f} | induction {prefix.max():.3f} "
              f"({ckpts[-1]['best_induction_head']}) | prev {prev.max():.3f} | repeat {first:.2f} -> {second:.2f}",
              flush=True)

    steps = [k["step"] for k in ckpts]
    best = [k["best_prefix"] for k in ckpts]
    i = formation_interval(best)
    final = np.array(ckpts[-1]["prefix"])
    final_prev = np.array(ckpts[-1]["prev"])
    formation = {
        "from_step": steps[i], "to_step": steps[i + 1], "score_from": best[i], "score_to": best[i + 1],
        "final_head": ckpts[-1]["best_induction_head"], "final_score": best[-1],
        "final_prev_heads": [f"L{l}.H{h}" for l, h in zip(*np.nonzero(final_prev >= INDUCTION_THRESHOLD))],
        "prev_head_step": first_step_above(steps, [k["best_prev"] for k in ckpts], INDUCTION_THRESHOLD),
        "induction_step": first_step_above(steps, best, INDUCTION_THRESHOLD),
        "second_copy_from": ckpts[i]["repeat_loss_second"], "second_copy_to": ckpts[i + 1]["repeat_loss_second"],
        "first_copy_mean": float(np.mean([k["repeat_loss_first"] for k in ckpts[1:]])),
        "drop": sharp_drop(steps, [k["val_loss"] for k in ckpts], i),
        "interval_index": i,
    }
    print("positive controls...", flush=True)
    report = {
        "repeat_len": REPEAT_LEN, "n_sequences": N_SEQUENCES, "seed": a.seed, "sharp_ratio": SHARP_RATIO,
        "induction_threshold": INDUCTION_THRESHOLD, "frequent_tokens": FREQUENT_TOKENS,
        "control_fixed": positive_control(False, 1500, a.seed),
        "control_variable": positive_control(True, 3000, a.seed),
        "uniform_prefix_baseline": uniform_prefix_baseline(REPEAT_LEN), "formation": formation,
        "checkpoints": [{k: v for k, v in c.items() if k not in ("patterns", "samples")} for c in ckpts],
        "seconds": round(time.time() - t0, 1),
    }
    a.results_dir.mkdir(parents=True, exist_ok=True)
    (a.results_dir / "induction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, a.results_dir / "induction.md")

    ui = {
        "format": "glassbox-replay", "version": 1, "n_layer": final.shape[0], "n_head": final.shape[1],
        "tokens_per_step": tokens_per_step, "uniform_prefix_baseline": report["uniform_prefix_baseline"],
        "induction_threshold": INDUCTION_THRESHOLD,
        "positive_control": {"fixed_length": report["control_fixed"][-1]["best_prefix"],
                             "variable_length": report["control_variable"][-1]["best_prefix"]},
        "train_curve": [[e["step"], round(e["loss"], 4)] for e in log["train"]],
        "val_curve": [[e["step"], round(e["loss"], 4)] for e in log["val"]],
        "ui_sequence": [tok.token_str(t) for t in ui_seq[0].tolist()],
        "formation": formation,
        "checkpoints": [{k: c[k] for k in ("step", "tokens", "val_loss", "prefix", "prev", "repeat_loss_first",
                                           "repeat_loss_second", "samples", "patterns")} for c in ckpts],
    }
    a.ui_out.write_text(json.dumps(ui, separators=(",", ":")), encoding="utf-8")
    print(f"formation {steps[i]} -> {steps[i + 1]}; sharp drop: {formation['drop']['sharp']} "
          f"(ratio {formation['drop']['ratio']:.2f}); wrote {a.results_dir / 'induction.md'} and {a.ui_out}")


if __name__ == "__main__":
    main()
