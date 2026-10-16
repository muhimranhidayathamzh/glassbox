"""Name guard: stop the model from showing a name, by reading its internal state before the token
is chosen, compared with a regex + name list and with logistic-regression classifiers.

    python -m glassbox.guard        # -> results/guard.{json,md}, web/model/guard.json

Labelled data. Validation stories (never seen in training), where names are found with a lexicon
built automatically from the training stories: words that are capitalized in the middle of a
sentence and almost never lowercase, minus a hand-written list of capitalized non-names (I, Mr,
Wow, TV, days, ...). Stories with other capitalized mid-sentence words (possible unlabelled names)
are dropped and counted. Two test conditions on the same stories:
  * known names: the names as written;
  * new names: every name replaced (consistently within a story) by a name outside the lexicon.

Task. At each position t the model has read tokens 0..t; is token t+1 the first token of a name?
Positions whose next token continues a name are skipped (a guard that fires on the first token
also hides the rest of the word).

Methods (classifiers are fitted, and every threshold chosen for best F1, on separate calibration
stories with known names only):
  1. regex + name list: looks at the next word itself, so it acts after the token is chosen
  2. logistic regression on the two previous tokens (text only, before the token is chosen)
  3. logistic regression on the raw residual stream at the SAE's layer (internal, before)
  4. logistic regression on SAE features at that layer (internal, before; explainable by labels)
  5. a single SAE feature with a threshold (internal, before; the simplest explanation)

Added after the first results (and labelled as such in the report): a strict threshold for the
SAE classifier (catching at least STRICT_RECALL of the calibration names), the combination
"regex or SAE guard", and a probe on the last layer's residual stream as a reference.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import TRAIN_HEAD_NAME, VALID_FILE, split_stories
from .export import load_exported, load_exported_sae

# Capitalized words that are not names (decided by hand after inspecting the automatic lexicon).
EXCLUDE = {
    "I", "Mr", "Mrs", "Ms", "Miss", "Dr", "Sir", "Wow", "Hey", "Oh", "Yes", "No", "OK", "Ok", "TV", "CD", "God",
    "Christmas", "Halloween", "Easter", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday", "January", "February", "March", "July", "September", "October", "November", "December",
    "Mom", "Dad", "Mommy", "Daddy", "Mama", "Papa", "Grandma", "Grandpa",
}
# Replacement names for the "new names" condition; any that the training data uses as a name is removed.
NEW_NAMES = [
    "Aiko", "Bayu", "Chidi", "Dewi", "Emeka", "Farida", "Gita", "Hiro", "Imani", "Jabari", "Kemal", "Laila",
    "Mateo", "Nia", "Oluwa", "Priya", "Rania", "Sakura", "Tariq", "Uma", "Vikram", "Wulan", "Yara", "Zainab",
    "Budi", "Citra", "Dimas", "Fajar", "Gilang", "Indah", "Joko", "Kirana", "Lestari", "Nanda", "Putri",
    "Rizki", "Sari", "Teguh", "Yusuf", "Anjali", "Bilal", "Dalia", "Esra", "Hamza", "Ines", "Kofi", "Leila",
    "Malik", "Noor", "Omar", "Quan", "Ravi", "Selin", "Thabo", "Umar", "Valentina", "Xiomara", "Zuri",
]
MIN_COUNT, MAX_LOWER_SHARE = 20, 0.02
STRICT_RECALL = 0.9
L2 = 1e-4
WORD = re.compile(r"[A-Za-z]+")


def sentence_start(text: str, i: int) -> bool:
    """Is the word at text[i] the first word of a sentence? A word right after an opening quote
    starts a sentence (dialogue); otherwise look back past spaces and quotes for . ! ? or a newline."""
    if i > 0 and text[i - 1] in "\"'":
        return True
    j = i - 1
    while j >= 0 and text[j] in " \t\"'":
        j -= 1
    return j < 0 or text[j] in ".!?\n"


def build_lexicon(stories: list[str]) -> tuple[set[str], Counter]:
    """Names: capitalized mid-sentence >= MIN_COUNT times, lowercase < MAX_LOWER_SHARE of the time."""
    cap_mid, cap_any, lower = Counter(), Counter(), Counter()
    for s in stories:
        for m in WORD.finditer(s):
            w = m.group(0)
            if w[0].isupper():
                cap_any[w] += 1
                if not sentence_start(s, m.start()):
                    cap_mid[w] += 1
            else:
                lower[w.lower()] += 1
    names = {w for w, n in cap_mid.items()
             if n >= MIN_COUNT and lower[w.lower()] / (cap_any[w] + lower[w.lower()]) < MAX_LOWER_SHARE
             and w not in EXCLUDE}
    return names, cap_any


def has_unknown_capitals(story: str, names: set[str]) -> bool:
    """A capitalized mid-sentence word that is neither a known name nor excluded: maybe an unlabelled name."""
    return any(m.group(0)[0].isupper() and not sentence_start(story, m.start())
               and m.group(0) not in names and m.group(0) not in EXCLUDE for m in WORD.finditer(story))


def name_spans(story: str, names: set[str]) -> list[tuple[int, int]]:
    return [m.span() for m in WORD.finditer(story) if m.group(0) in names]


def substitute(story: str, names: set[str], pool: list[str], rng: np.random.Generator) -> tuple[str, set[str]]:
    """Replace every name by a pool name, the same replacement for the same name within the story."""
    mapping: dict[str, str] = {}
    used: set[str] = set()

    def repl(m):
        w = m.group(0)
        if w not in names:
            return w
        if w not in mapping:
            choices = [p for p in pool if p not in used]
            mapping[w] = choices[int(rng.integers(len(choices)))]
            used.add(mapping[w])
        return mapping[w]

    return WORD.sub(repl, story), set(mapping.values())


@dataclass
class Example:
    ids: list[int]
    labels: list[int]  # per position t: 1 = next token starts a name, 0 = it does not, -1 = skip
    next_word: list[str]  # the word that starts at token t+1 ("" if it is not a word)
    next_sentence_start: list[bool]


def label_story(tok, story: str, spans: list[tuple[int, int]], n_ctx: int) -> Example:
    ids = [tok.eot_id] + tok.encode(story, allow_special=False)[: n_ctx - 1]
    starts, pos = [None], 0  # character offset of every token (story is ASCII, so bytes = chars)
    for i in ids[1:]:
        starts.append(pos)
        pos += len(tok.vocab[i])
    kind = [0] * len(ids)  # per token: 1 name start, -1 name continuation, 0 other
    for s, e in spans:
        for i in range(1, len(ids)):
            a, b = starts[i], starts[i] + len(tok.vocab[ids[i]])
            if a <= s < b:
                kind[i] = 1
            elif s < a < e:
                kind[i] = -1
    labels, words, sent = [], [], []
    for t in range(len(ids)):
        if t + 1 >= len(ids):
            labels.append(-1), words.append(""), sent.append(False)
            continue
        labels.append(kind[t + 1])
        a = starts[t + 1]
        while a < len(story) and story[a] in " \t\n\"'":
            a += 1
        m = WORD.match(story, a)
        words.append(m.group(0) if m and starts[t + 1] <= a < starts[t + 1] + len(tok.vocab[ids[t + 1]]) else "")
        sent.append(sentence_start(story, a))
    return Example(ids, labels, words, sent)


# ----------------------------------------------------------------------------- features


@dataclass
class Features:
    resid: np.ndarray  # (N, d) at the SAE's layer
    resid_last: np.ndarray  # (N, d) entering ln_f (reference only)
    code_idx: np.ndarray  # (N, k) SAE feature ids
    code_val: np.ndarray  # (N, k) activations (0 where fewer than k are active)
    prev2: np.ndarray  # (N, 2) previous token and the one before (vocab_size if none)
    y: np.ndarray  # (N,)
    words: list[str]
    sentence_start: np.ndarray


@torch.no_grad()
def extract(model, sae, examples: list[Example], batch: int = 16) -> Features:
    layer, V = sae.cfg.layer, model.cfg.vocab_size
    resid, last, cidx, cval, prev2, y, words, sent = [], [], [], [], [], [], [], []
    for i in range(0, len(examples), batch):
        chunk = examples[i : i + batch]
        T = max(len(e.ids) for e in chunk)
        idx = torch.tensor([e.ids + [0] * (T - len(e.ids)) for e in chunk])
        r = model.resid_at(idx, layer)
        r_last = model.resid_at(idx, model.cfg.n_layer)
        codes = sae.encode(r)
        codes[:, 0] = 0  # position 0 is skipped by the SAE, as everywhere else
        top = codes.topk(sae.cfg.k, dim=-1)
        for b, e in enumerate(chunk):
            keep = [t for t, lab in enumerate(e.labels) if lab != -1]
            resid.append(r[b, keep].numpy())
            last.append(r_last[b, keep].numpy())
            cidx.append(top.indices[b, keep].numpy())
            cval.append(top.values[b, keep].numpy())
            prev2.append(np.array([[e.ids[t], e.ids[t - 1] if t > 0 else V] for t in keep]))
            y += [e.labels[t] for t in keep]
            words += [e.next_word[t] for t in keep]
            sent += [e.next_sentence_start[t] for t in keep]
    return Features(np.concatenate(resid), np.concatenate(last), np.concatenate(cidx), np.concatenate(cval),
                    np.concatenate(prev2), np.array(y), words, np.array(sent))


# ----------------------------------------------------------------------------- classifiers


def fit_logreg(score_fn, params: list[torch.Tensor], y: np.ndarray, iters: int = 150) -> None:
    """Full-batch L-BFGS on mean BCE + L2 * ||weights||^2 (params[0] are the weights). Deterministic."""
    target = torch.from_numpy(y.astype(np.float32))
    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(score_fn(), target) + L2 * params[0].pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)


class ResidualProbe:
    def __init__(self, f: Features, field: str = "resid"):
        self.field = field
        x = getattr(f, field)
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        self.w = torch.zeros(x.shape[1], requires_grad=True)
        self.b = torch.zeros((), requires_grad=True)
        xt = torch.from_numpy((x - self.mean) / self.std)
        fit_logreg(lambda: xt @ self.w + self.b, [self.w, self.b], f.y)

    @torch.no_grad()
    def scores(self, f: Features) -> np.ndarray:
        return (torch.from_numpy((getattr(f, self.field) - self.mean) / self.std) @ self.w + self.b).numpy()


class BagLR:
    """Logistic regression over sparse (index, value) inputs: SAE features, or previous tokens."""

    def __init__(self, n: int, idx: np.ndarray, val: np.ndarray, y: np.ndarray):
        self.w = torch.zeros(n, requires_grad=True)
        self.b = torch.zeros((), requires_grad=True)
        i, v = torch.from_numpy(idx.astype(np.int64)), torch.from_numpy(val.astype(np.float32))
        fit_logreg(lambda: self._logits(i, v), [self.w, self.b], y)

    def _logits(self, i, v):
        return F.embedding_bag(i, self.w[:, None], per_sample_weights=v, mode="sum")[:, 0] + self.b

    @torch.no_grad()
    def scores(self, idx: np.ndarray, val: np.ndarray) -> np.ndarray:
        return self._logits(torch.from_numpy(idx.astype(np.int64)), torch.from_numpy(val.astype(np.float32))).numpy()


def text_inputs(f: Features, V: int) -> tuple[np.ndarray, np.ndarray]:
    """Previous token (ids 0..V) and the one before (ids V+1..2V+1), as a bag of two."""
    return np.stack([f.prev2[:, 0], f.prev2[:, 1] + V + 1], axis=1), np.ones((len(f.y), 2), dtype=np.float32)


def single_feature_scores(f: Features, feature: int) -> np.ndarray:
    return np.where(f.code_idx == feature, f.code_val, 0).sum(1)


def regex_flags(f: Features, names: set[str]) -> np.ndarray:
    """Regex + name list on the next word: flag known names, and capitalized words mid-sentence."""
    return np.array([bool(w) and (w in names or (w[0].isupper() and not s and w not in EXCLUDE))
                     for w, s in zip(f.words, f.sentence_start)])


def best_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Threshold with the best F1 (predict positive when score >= threshold)."""
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(y[order] == 1)
    fp = np.cumsum(y[order] == 0)
    f1 = 2 * tp / (tp + fp + (y == 1).sum())
    k = int(np.argmax(f1))
    return float(scores[order][k])


def recall_threshold(scores: np.ndarray, y: np.ndarray, target: float) -> float:
    """Highest threshold that still catches at least `target` of the positives."""
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(y[order] == 1)
    k = int(np.searchsorted(tp, target * (y == 1).sum()))
    return float(scores[order][min(k, len(order) - 1)])


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn,
            "false_blocks_per_1000_tokens": 1000 * fp / len(y)}


def best_single_feature(f: Features, min_hits: int = 20) -> tuple[int, float, float]:
    """The SAE feature whose activation alone gives the best calibration F1. Only positions where a
    feature is active can be flagged by it, so each feature is scored on those positions."""
    active = f.code_val > 0
    pos = np.repeat(np.arange(len(f.y)), f.code_idx.shape[1]).reshape(f.code_idx.shape)[active]
    feat, val = f.code_idx[active], f.code_val[active]
    order = np.lexsort((-val, feat))
    pos, feat, val = pos[order], feat[order], val[order]
    n_pos = int((f.y == 1).sum())
    best = (-1, 0.0, -1.0)
    groups, starts = np.unique(feat, return_index=True)
    for g, a, b in zip(groups, starts, list(starts[1:]) + [len(feat)]):
        if b - a < min_hits:
            continue
        yv = f.y[pos[a:b]]
        tp, fp = np.cumsum(yv == 1), np.cumsum(yv == 0)
        f1 = 2 * tp / (tp + fp + n_pos)
        k = int(np.argmax(f1))
        if f1[k] > best[2]:
            best = (int(g), float(val[a + k]), float(f1[k]))
    return best


# ----------------------------------------------------------------------------- report


def write_markdown(r: dict, path: Path) -> None:
    d = r["data"]
    lines = [
        "# Milestone 6: name guard",
        "",
        "Generated by `python -m glassbox.guard`. Do not edit by hand.",
        "",
        "Task: at each position the model has read the story so far; is the next token the first token of a "
        "name? A guard that says yes hides that token (and the rest of the word) before it is shown.",
        "",
        "## Data",
        "",
        f"- Name lexicon: {d['lexicon_size']} names found automatically in {d['lexicon_stories']:,} training stories "
        f"(capitalized mid-sentence at least {MIN_COUNT} times, lowercase less than {MAX_LOWER_SHARE:.0%} of the "
        f"time), minus {len(EXCLUDE)} hand-listed capitalized non-names. Most frequent: {', '.join(d['top_names'])}.",
        f"- Stories: {d['calibration_stories']:,} for calibration and {d['test_stories']:,} for testing, all from the "
        f"validation split; {d['dropped_stories']:,} stories with other capitalized mid-sentence words (possible "
        "unlabelled names) were dropped.",
        f"- Test positions: {d['test_positions']:,}, of which {d['test_names']:,} are followed by a name "
        f"({d['test_names'] / d['test_positions']:.1%}).",
        f"- New-names condition: every name replaced by one of {d['new_names_used']} names that are not in the "
        f"lexicon (e.g. {', '.join(d['new_names_example'])}), the same replacement within a story.",
        "",
        "Classifiers are fitted, and all thresholds chosen for the best F1, on calibration stories with known "
        "names only. The new-names test therefore measures generalization to names never used for fitting.",
        "",
        "Rows marked * were added after the first results: a strict threshold for the SAE classifier (the "
        f"highest one that still catches {r['sae_guard']['strict_recall']:.0%} of calibration names), the "
        "combination that hides a word when either the regex or the SAE guard fires, and a probe on the last "
        "layer as a reference (the SAE, and so the guard, reads layer "
        f"{r['sae_layer']}).",
        "",
    ]
    for cond, title in (("known", "Test: known names"), ("new", "Test: new names")):
        lines += [f"## {title}", "",
                  "| method | decides | precision | recall | F1 | false blocks / 1000 tokens |",
                  "|---|---|---:|---:|---:|---:|"]
        for m in r["methods"]:
            s = m[cond]
            star = " *" if m.get("added_later") else ""
            lines.append(f"| {m['name']}{star} | {m['decides']} | {s['precision']:.3f} | {s['recall']:.3f} | "
                         f"**{s['f1']:.3f}** | {s['false_blocks_per_1000_tokens']:.1f} |")
        lines.append("")
        if cond == "known":
            lines += ["The regex uses the same lexicon that defines the labels, and stories with other capitalized "
                      "mid-sentence words were dropped, so its near-perfect score on known names holds by "
                      "construction. The new-names test is the fair comparison.", ""]
    lat = r.get("latency")
    if lat:
        lines += [
            "## Added latency per generated token (JavaScript engine, Node.js)",
            "",
            f"{lat['runtime']}; a model step without any guard takes {lat['base_ms']:.2f} ms (median of "
            f"{lat['tokens']} tokens).",
            "",
            "| guard | added time per token | relative |",
            "|---|---:|---:|",
        ]
        for name, ms in lat["added_ms"].items():
            lines.append(f"| {name} | {ms:.3f} ms | {ms / lat['base_ms']:+.1%} |")
        lines.append("")
    g = r["sae_guard"]
    lines += [
        "## What the SAE guard looks at",
        "",
        f"Features with the largest positive weights in the SAE-feature classifier (threshold "
        f"{g['threshold']:.3f} on the logit). Labels come from `python -m glassbox.features`.",
        "",
        "| feature | weight | label |",
        "|---:|---:|---|",
    ]
    lines += [f"| #{t['id']} | {t['weight']:+.2f} | `{t['label']}` |" for t in g["top_features"][:12]]
    sf = r["single_feature"]
    lines += [
        "",
        f"Best single feature: #{sf['id']} `{sf['label']}`, blocking when its activation is at least "
        f"{sf['threshold']:.2f}.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=Path("web/model"))
    p.add_argument("--sae-dir", type=Path, default=Path("web/model/sae"))
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--calibration-stories", type=int, default=1500)
    p.add_argument("--test-stories", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    t0 = time.time()

    model, tok, _ = load_exported(a.model_dir)
    sae, sae_manifest = load_exported_sae(a.sae_dir)
    labels = {f["id"]: f.get("label", "(dead)") for f in
              json.loads((a.sae_dir / "features.json").read_text(encoding="utf-8"))["features"]}
    train_stories, _ = split_stories((a.raw_dir / TRAIN_HEAD_NAME).read_text(encoding="utf-8"))
    names, cap_counts = build_lexicon(train_stories)
    pool = [n for n in NEW_NAMES if n not in names and cap_counts[n] < 5]
    print(f"lexicon: {len(names)} names; {len(pool)} new names available", flush=True)

    val_stories, _ = split_stories((a.raw_dir / VALID_FILE).read_text(encoding="utf-8"))
    kept = [s for s in val_stories if not has_unknown_capitals(s, names)]
    dropped = len(val_stories) - len(kept)
    cal = kept[: a.calibration_stories]
    test = kept[a.calibration_stories : a.calibration_stories + a.test_stories]
    rng = np.random.default_rng(a.seed)
    T = model.cfg.n_ctx
    cal_ex = [label_story(tok, s, name_spans(s, names), T) for s in cal]
    known_ex = [label_story(tok, s, name_spans(s, names), T) for s in test]
    new_ex, used_new = [], set()
    for s in test:
        s2, used = substitute(s, names, pool, rng)
        used_new |= used
        new_ex.append(label_story(tok, s2, name_spans(s2, used), T))
    fc, fk, fn = (extract(model, sae, ex) for ex in (cal_ex, known_ex, new_ex))
    print(f"positions: calibration {len(fc.y):,} ({int((fc.y == 1).sum())} names), test {len(fk.y):,}", flush=True)

    V = model.cfg.vocab_size
    probe = ResidualProbe(fc)
    probe_last = ResidualProbe(fc, "resid_last")
    sae_lr = BagLR(sae.cfg.n_features, fc.code_idx, fc.code_val, fc.y)
    text_lr = BagLR(2 * V + 2, *text_inputs(fc, V), fc.y)
    single, single_th, _ = best_single_feature(fc)

    scorers = {
        "text classifier (previous 2 tokens)": lambda f: text_lr.scores(*text_inputs(f, V)),
        "residual-stream probe": probe.scores,
        "SAE-feature classifier": lambda f: sae_lr.scores(f.code_idx, f.code_val),
        f"single SAE feature #{single}": lambda f: single_feature_scores(f, single),
    }
    decides = {"regex + name list": "after the token is chosen"}
    methods = [{"name": "regex + name list", "decides": decides["regex + name list"],
                "known": metrics(regex_flags(fk, names), fk.y), "new": metrics(regex_flags(fn, names), fn.y),
                "calibration": metrics(regex_flags(fc, names), fc.y)}]
    thresholds = {}
    for name, score in scorers.items():
        th = best_threshold(score(fc), fc.y) if not name.startswith("single") else single_th
        thresholds[name] = th
        methods.append({"name": name, "decides": "before the token is chosen", "threshold": th,
                        "calibration": metrics(score(fc) >= th, fc.y),
                        "known": metrics(score(fk) >= th, fk.y), "new": metrics(score(fn) >= th, fn.y)})
        print(f"{name:38s} F1 known {methods[-1]['known']['f1']:.3f} | new {methods[-1]['new']['f1']:.3f}", flush=True)
    print(f"{'regex + name list':38s} F1 known {methods[0]['known']['f1']:.3f} | new {methods[0]['new']['f1']:.3f}")

    # Added after the first results: strict threshold, regex-or-guard, and a last-layer reference.
    sae_score = scorers["SAE-feature classifier"]
    strict_th = recall_threshold(sae_score(fc), fc.y, STRICT_RECALL)
    added = [
        ("SAE-feature classifier, strict", "before the token is chosen", lambda f: sae_score(f) >= strict_th),
        ("regex or SAE guard", "both",
         lambda f: regex_flags(f, names) | (sae_score(f) >= thresholds["SAE-feature classifier"])),
        ("regex or SAE guard, strict", "both", lambda f: regex_flags(f, names) | (sae_score(f) >= strict_th)),
    ]
    last_th = best_threshold(probe_last.scores(fc), fc.y)
    added.append(("residual-stream probe, last layer (reference)", "before the token is chosen",
                  lambda f: probe_last.scores(f) >= last_th))
    for name, dec, pred in added:
        methods.append({"name": name, "decides": dec, "added_later": True, "calibration": metrics(pred(fc), fc.y),
                        "known": metrics(pred(fk), fk.y), "new": metrics(pred(fn), fn.y)})
        print(f"{name:38s} F1 known {methods[-1]['known']['f1']:.3f} | new {methods[-1]['new']['f1']:.3f} "
              f"(recall new {methods[-1]['new']['recall']:.3f})", flush=True)

    w = sae_lr.w.detach().numpy()
    top = [{"id": int(i), "weight": float(w[i]), "label": labels.get(int(i), "?")} for i in np.argsort(-w)[:40]]
    sae_th = thresholds["SAE-feature classifier"]
    guard_ui = {
        "format": "glassbox-guard", "version": 1, "layer": sae.cfg.layer, "sae_dir": a.sae_dir.name,
        "sae_weights_sha256": sae_manifest["weights_sha256"], "bias": float(sae_lr.b), "threshold": sae_th,
        "strict_threshold": strict_th, "strict_recall": STRICT_RECALL,
        "names": sorted(names), "exclude": sorted(EXCLUDE),
        "labels": {str(int(i)): labels.get(int(i), "?") for i in np.nonzero(w > 0.05)[0]},
        "weights": [round(float(v), 5) for v in w], "top_features": top,
    }
    guard_file = a.model_dir / "guard.json"  # the page's guard; guard_layers.py may later replace it
    guard_file.write_text(json.dumps(guard_ui, separators=(",", ":")), encoding="utf-8")

    used_new = sorted(used_new)
    report = {
        "data": {"lexicon_size": len(names), "lexicon_stories": len(train_stories),
                 "top_names": [n for n, _ in Counter({n: cap_counts[n] for n in names}).most_common(10)],
                 "calibration_stories": len(cal), "test_stories": len(test), "dropped_stories": dropped,
                 "test_positions": int(len(fk.y)), "test_names": int((fk.y == 1).sum()),
                 "new_names_used": len(used_new), "new_names_example": used_new[:6]},
        "methods": methods,
        "sae_guard": {"threshold": sae_th, "strict_threshold": strict_th, "strict_recall": STRICT_RECALL,
                      "bias": float(sae_lr.b), "top_features": top[:20]},
        "single_feature": {"id": single, "threshold": single_th, "label": labels.get(single, "?")},
        "sae_layer": sae.cfg.layer, "seed": a.seed, "seconds": round(time.time() - t0, 1),
    }
    from .eval.web_parity import node_available, run_node

    if node_available():  # added latency of the guard in the browser engine
        bench = run_node({"model_dir": a.model_dir, "sae_dir": a.sae_dir,
                          "guard_bench": {"file": str(guard_file.resolve()), "tokens": 200,
                                          "repeats": 5}})[0]["guard_bench"]
        report["latency"] = {"runtime": bench["runtime"], "tokens": bench["tokens"], "base_ms": bench["base_ms"],
                             "added_ms": {"SAE guard + word check": bench["guarded_ms"] - bench["base_ms"]}}
    a.results_dir.mkdir(parents=True, exist_ok=True)
    (a.results_dir / "guard.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, a.results_dir / "guard.md")
    print(f"wrote {a.results_dir / 'guard.md'} and {guard_file} in {report['seconds']} s")


if __name__ == "__main__":
    main()
