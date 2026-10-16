"""Write the case study page and the LinkedIn drafts from results/*.json, so every number in them is
generated, never typed.

    python -m glassbox.eval.writeup            # -> web/case-study.html, docs/linkedin_post_{en,id}.md
    python -m glassbox.eval.writeup --check    # exit 1 if any of them is out of date (used in CI)
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

SITE = "https://muhimranhidayathamzh.github.io/glassbox/"
REPO = "https://github.com/muhimranhidayathamzh/glassbox"
AUTHOR = "Muhammad Imran"


def load(results: Path, name: str) -> dict:
    return json.loads((results / f"{name}.json").read_text(encoding="utf-8"))


def pct(x: float, digits: int = 0) -> str:
    return f"{x * 100:.{digits}f}%"


def num_id(text: str) -> str:
    """English number punctuation to Indonesian: 2,048 -> 2.048 and 2.6 -> 2,6."""
    return text.translate(str.maketrans({",": ".", ".": ","}))


def e(text: str) -> str:
    return html.escape(text, quote=True)


def excerpt(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0]


class Numbers:
    """Every figure the texts use, read from the result files."""

    def __init__(self, results: Path):
        lm, web, sae = load(results, "lm_eval"), load(results, "web_parity"), load(results, "sae_eval")
        feats = {8: load(results, "sae_features"), 16: load(results, "sae_features_k16"),
                 32: load(results, "sae_features_k32")}
        steer, attr, ind = load(results, "steering"), load(results, "attribution"), load(results, "induction")
        guard, layers = load(results, "guard"), load(results, "guard_layers")

        t = lm["training"]
        self.params_m = lm["n_params"] / 1e6
        self.hours = t["time"]["active_s"] / 3600
        self.tokens_m = t["tokens_seen"] / 1e6
        self.cpu = t["cpu"].replace("(R)", "").replace("(TM)", "").split(" @ ")[0].removeprefix("11th Gen ")
        self.loss, self.bigram, self.unigram = (lm["val_loss"][k] for k in ("model", "bigram", "unigram"))
        # the first sample with a written prompt (sample 0 is unconditional)
        i, s = next((i, s) for i, s in enumerate(lm["samples"]) if s["prompt"] != "<|endoftext|>")
        self.sample_prompt, self.sample_text = s["prompt"], s["completion"]
        st = lm["sample_settings"]
        self.sample_settings = f"temperature {st['temperature']:g}, top-k {st['top_k']}, seed {st['seed'] + i}"

        self.parity = web["js_vs_torch"]["max_abs_diff"]
        self.tok_same, self.tok_total = web["tokenizer"]["identical"], web["tokenizer"]["stories"]
        g = web["greedy"]
        same = sum(q["identical_prefix"] == q["new_tokens"] for q in g["prompts"])
        self.greedy = (f"greedy generation matches PyTorch token for token over {g['new_tokens']} tokens on "
                       + (f"all {len(g['prompts'])} test prompts" if same == len(g["prompts"])
                          else f"{same} of {len(g['prompts'])} test prompts"))
        self.speed = web["speed"]["generate_tokens_per_s"]
        self.weights_mb = web["files"]["weights_bytes"] / 1e6

        self.sae_rows = {r["k"]: r for r in sae["saes"]}
        self.sae_k = sae["selected"]["k"]
        self.sae_layer = sae["layer"]
        self.n_features = sae["n_features"]
        self.spec = {k: f["token_specificity"]["SAE (trained)"]["share_ge_0.5"] for k, f in feats.items()}
        self.spec_base = max(v["share_ge_0.5"] for name, v in feats[8]["token_specificity"].items()
                             if name != "SAE (trained)")
        self.dead_feats = {k: f["dead"] for k, f in feats.items()}

        s = steer["summary"]
        self.steer_rows = [(0.0, s["alpha_0_mention"], None)] + [
            (r["alpha"], r["feature_mention"], r["random_mention"]) for r in s["by_alpha"]]
        self.steer_best = max(s["by_alpha"], key=lambda r: r["feature_mention"])
        self.steer_none_nll = s["alpha_0_nll"]
        self.steer_words = [c["word"] for c in steer["concepts"] if c.get("feature") is not None]
        self.steer_missing = [c["word"] for c in steer["concepts"] if c.get("feature") is None]
        self.steer_n = steer["samples_per_condition"]
        self.steer_example = steer["example"]

        est = attr["estimates"]
        self.attr_grad, self.attr_grad_top1 = (est["gradient x activation"][k] for k in ("spearman", "top1"))
        self.attr_dla, self.attr_dla_top1 = (est["direct logit attribution"][k] for k in ("spearman", "top1"))
        self.attr_earlier = 1 - attr["effects"]["share_at_t"]

        self.ind_score, self.ind_threshold = ind["formation"]["final_score"], ind["induction_threshold"]
        self.ind_control = ind["control_fixed"][-1]["best_prefix"]
        self.ind_ckpts = len(ind["checkpoints"])
        self.prev_score = ind["checkpoints"][-1]["best_prev"]

        gm = {m["name"]: m for m in guard["methods"]}
        self.g_regex, self.g_sae = gm["regex + name list"], gm["SAE-feature classifier"]
        self.g_text = gm["text classifier (previous 2 tokens)"]
        self.g_both, self.g_strict = gm["regex or SAE guard"], gm["regex or SAE guard, strict"]
        self.g_latency = guard["latency"]["added_ms"]["SAE guard + word check"]
        self.g_base_ms = guard["latency"]["base_ms"]
        self.g_new_names = guard["data"]["new_names_used"]
        self.g_test_stories = guard["data"]["test_stories"]
        cands = {c["name"]: c for c in layers["candidates"]}
        self.gl_selected = cands[layers["selected"]]
        self.gl_k8 = next(c for c in layers["candidates"] if c.get("dir") == "sae")
        self.gl_best_resid = cands[layers["best_residual"]]


# ----------------------------------------------------------------------------- charts (HTML)


def hbar(value: float, cls: str, label: str) -> str:
    return (f'<div class="htrack"><span class="hbar {cls}" style="--v:{value:.4f}"></span>'
            f'<span class="hval">{e(label)}</span></div>')


def steering_chart(n: Numbers) -> str:
    rows = []
    for alpha, feat, rand in n.steer_rows:
        if rand is None:
            name, bars = "no steering", hbar(feat, "s0", pct(feat))
            tip = f"no steering: {pct(feat, 1)} mention the concept"
        else:
            name, bars = f"strength {alpha:g}", hbar(feat, "s1", pct(feat)) + hbar(rand, "s2", pct(rand))
            tip = f"strength {alpha:g}: feature {pct(feat, 1)}, random {pct(rand, 1)}"
        rows.append(f'<div class="hgroup" title="{e(tip)}"><span class="hlabel">{name}</span>'
                    f'<div class="htracks">{bars}</div></div>')
    table = "".join(
        f"<tr><td>{a:g}</td><td>{pct(f, 1)}</td><td>{'-' if r is None else pct(r, 1)}</td></tr>"
        for a, f, r in n.steer_rows)
    return f"""<figure class="fig">
      <figcaption><strong>Stories that mention the steered concept.</strong> Mean over {len(n.steer_words)} concepts,
        {n.steer_n} stories per concept and setting, the same prompts and seeds in every setting.</figcaption>
      <div class="legend-row"><span class="legend-item"><span class="key key-bar key-1"></span>feature direction</span>
        <span class="legend-item"><span class="key key-bar key-2"></span>random direction, same size</span>
        <span class="legend-item"><span class="key key-bar key-0"></span>no steering</span></div>
      <div class="hbars">{''.join(rows)}</div>
      <details><summary>Data table</summary><div class="table-wrap"><table class="data"><tr><th>strength</th>
        <th>feature</th><th>random</th></tr>{table}</table></div></details>
    </figure>"""


def guard_chart(n: Numbers) -> str:
    rows = [
        ("regex + name list", n.g_regex),
        ("SAE-feature guard", n.g_sae),
        ("regex or SAE guard", n.g_both),
        ("regex or SAE guard, strict", n.g_strict),
    ]
    body = "".join(
        f'<div class="hgroup" title="{e(name)}: recall {pct(m["new"]["recall"], 1)}, precision '
        f'{pct(m["new"]["precision"], 1)}"><span class="hlabel">{e(name)}</span><div class="htracks">'
        f'{hbar(m["new"]["recall"], "s1", pct(m["new"]["recall"]))}'
        f'<span class="hnote">{m["new"]["false_blocks_per_1000_tokens"]:.0f} false blocks per 1,000 tokens</span>'
        "</div></div>" for name, m in rows)
    table = "".join(
        f"<tr><td>{e(name)}</td><td>{m['new']['precision']:.3f}</td><td>{m['new']['recall']:.3f}</td>"
        f"<td>{m['new']['f1']:.3f}</td><td>{m['new']['false_blocks_per_1000_tokens']:.1f}</td></tr>"
        for name, m in rows)
    return f"""<figure class="fig">
      <figcaption><strong>Names caught, among names never used to fit or tune anything.</strong>
        {n.g_test_stories:,} test stories with every name replaced by one of {n.g_new_names} names outside the
        training lexicon.</figcaption>
      <div class="hbars">{body}</div>
      <details><summary>Data table</summary><div class="table-wrap"><table class="data"><tr><th>method</th>
        <th>precision</th><th>recall</th><th>F1</th><th>false blocks / 1,000 tokens</th></tr>{table}</table></div>
      </details>
    </figure>"""


# ----------------------------------------------------------------------------- case study


def case_study(n: Numbers) -> str:
    k8, k32 = n.sae_rows[8], n.sae_rows[32]
    sel = n.sae_rows[n.sae_k]
    best = n.steer_best
    ex = n.steer_example
    words = ", ".join(f"<em>{e(w)}</em>" for w in n.steer_words)
    missing = ", ".join(f"<em>{e(w)}</em>" for w in n.steer_missing)
    steered = excerpt(ex["steered"], 140)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Glassbox · case study</title>
  <meta name="description" content="Building a small language model from scratch, the tools to look inside it, and a name guard that reads its internal features. Every number on the page is generated from the experiments.">
  <meta property="og:type" content="article">
  <meta property="og:site_name" content="Glassbox">
  <meta property="og:title" content="Glassbox: an LLM built from scratch, and a microscope to look inside it">
  <meta property="og:description" content="A {n.params_m:.1f}M-parameter transformer trained on a laptop CPU, running in your browser, taken apart with sparse autoencoders, steering and attribution.">
  <meta property="og:image" content="{SITE}og.png">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="630">
  <meta property="og:url" content="{SITE}case-study.html">
  <meta name="twitter:card" content="summary_large_image">
  <link rel="stylesheet" href="ui/style.css">
</head>
<body>
  <!-- Generated by `python -m glassbox.eval.writeup` from results/*.json. Do not edit by hand. -->
  <header>
    <nav class="nav"><a href="index.html">Generate</a><a href="microscope.html">Microscope</a><a href="replay.html">Replay</a><a href="case-study.html" aria-current="page">Case study</a></nav>
    <p class="kicker">Case study &middot; {e(AUTHOR)}</p>
    <h1>I built a language model from scratch, then the tools to look inside it</h1>
    <p class="lede">A {n.params_m:.1f}M-parameter transformer trained on a laptop CPU, a JavaScript engine that runs it
      in your browser, a sparse autoencoder that splits its internal state into readable features, and a guard that
      reads those features to stop the model writing names before it writes them.</p>
    <p class="cta"><a class="button primary" href="index.html">Open the demo</a>
      <a class="button" href="{REPO}">Code on GitHub</a></p>
  </header>

  <main class="article viz-root">
    <div class="tiles">
      <div class="tile"><span class="tile-value">{n.params_m:.1f}M</span><span class="tile-label">parameters, every
        layer written by hand</span></div>
      <div class="tile"><span class="tile-value">{n.hours:.1f} h</span><span class="tile-label">of training on a laptop
        CPU, no GPU</span></div>
      <div class="tile"><span class="tile-value">{n.n_features:,}</span><span class="tile-label">features found by the
        sparse autoencoder</span></div>
      <div class="tile"><span class="tile-value">{n.parity:.1e}</span><span class="tile-label">largest logit gap,
        browser vs PyTorch</span></div>
    </div>

    <figure class="fig shot">
      <img src="img/circuit.png" width="1576" height="826" alt="The microscope explaining why the model wrote Lily after 'a little girl named': the features active on the previous tokens, each linked to the predicted word by an edge whose width is that feature's effect.">
      <figcaption>Screenshot of the microscope, explaining why the model wrote <q>Lily</q>. Each active feature is
        removed in turn and the model rerun; a blue edge means the word became less likely without that feature, and
        its width is how much.</figcaption>
    </figure>

    <section class="tldr">
      <h2>In short</h2>
      <ul>
        <li>Adding one learned feature direction to the model's internal state makes
          <strong>{pct(best['feature_mention'])}</strong> of stories mention its concept, against
          {pct(best['random_mention'])} for a random direction of the same size.</li>
        <li>A guard that reads features <em>before</em> a token is chosen catches names a regex cannot see. Together
          they catch <strong>{pct(n.g_both['new']['recall'])}</strong> of names never used in fitting, where the regex
          alone catches {pct(n.g_regex['new']['recall'])}. The price: {n.g_both['new']['false_blocks_per_1000_tokens']:.0f}
          other tokens per 1,000 held back by mistake, and {n.g_latency:.2f} ms per token.</li>
        <li>The browser engine matches PyTorch to {n.parity:.1e} in the logits, checked on every push.</li>
        <li>Negative results stay in: no induction heads formed, and the feature guard on its own does not beat a
          regex.</li>
      </ul>
    </section>

    <h2>Why I built it</h2>
    <p>Interpretability research usually happens on large models with large budgets. I wanted to find out how much of
      it one person can do end to end on a laptop: build the model myself so no part of it is a black box, look inside
      it with current techniques, and turn one finding into a feature whose value can be measured. It all had to run
      without a GPU, a server or an API key, so anyone can open it.</p>

    <h2>What I built</h2>
    <ul>
      <li><strong>The model.</strong> A byte-level BPE tokenizer, a GPT-style transformer and a training loop, written
        with plain PyTorch primitives. Trained for {n.hours:.1f} hours on an {e(n.cpu)} laptop CPU
        ({n.tokens_m:.1f}M tokens of TinyStories), it reaches a validation loss of {n.loss:.2f} nats per token,
        against {n.bigram:.2f} for a bigram model and {n.unigram:.2f} for unigram frequencies.</li>
      <li><strong>The browser engine.</strong> The same forward pass in dependency-free JavaScript, with a KV cache.
        The weights take {n.weights_mb:.1f} MB, and it generates about {n.speed:.0f} tokens per second in Node.js on
        the same laptop. {n.tok_same:,} of {n.tok_total:,} validation stories tokenize exactly as in Python, and
        {n.greedy}.</li>
      <li><strong>The microscope.</strong> A top-k sparse autoencoder (SAE) on the residual stream entering block
        {n.sae_layer}, with {n.n_features:,} features, automatic feature labels, steering, attribution checked by
        ablation, and a replay of training across {n.ind_ckpts} saved checkpoints.</li>
      <li><strong>The guard.</strong> A classifier on SAE features that predicts, before the model picks the next
        token, whether that token starts a name, plus a word check once the word is written.</li>
    </ul>
    <figure class="fig">
      <figcaption><strong>Real model output.</strong> The first prompted sample in the evaluation report, not
        cherry-picked ({e(n.sample_settings)}); the prompt is in grey.</figcaption>
      <p class="story-sample"><span class="prompt">{e(n.sample_prompt)}</span>{e(n.sample_text)}</p>
    </figure>

    <h2>How I kept myself honest</h2>
    <ul>
      <li><strong>No typed numbers.</strong> Every number in the README, on this page and in the reports is written by
        a script from the experiment results, and CI fails if a page no longer matches them.</li>
      <li><strong>A baseline for every claim.</strong> Random directions for steering, PCA for the SAE, an untrained
        SAE for feature quality, a regex and simpler classifiers for the guard.</li>
      <li><strong>Rules before results.</strong> Which SAE the UI uses, which concepts to steer, what counts as a sharp
        drop in the loss and which guard ships were each decided by a rule written down before the data came in.
        Anything added after a first look is marked as such in the reports.</li>
      <li><strong>Tests across two languages.</strong> JavaScript and PyTorch are compared on the same weights for
        logits, tokens, SAE features, steering, attribution and guard scores, on every push.</li>
    </ul>

    <h2>Finding 1: the features are real, and they steer</h2>
    <p>The SAE the interface uses (k&nbsp;=&nbsp;{n.sae_k} active features per token) explains
      {pct(sel['variance_explained'], 1)} of the variance of the model's internal state. Swapping that state for the
      SAE's reconstruction raises the model's loss by {pct(sel['relative_loss_increase'], 1)}.
      {pct(n.spec[n.sae_k])} of the features put most of their strongest examples on a single token, against at most
      {pct(n.spec_base)} for the model's own dimensions, random directions or an untrained SAE.</p>
    <p>To steer, I add a feature's direction to the internal state while the model writes. I fixed the concepts in
      advance ({words}; {missing} had no matching feature and was dropped) and compared each with random directions of
      the same size. Strength is the size of the added vector relative to the average size of the internal state.</p>
    {steering_chart(n)}
    <p>The price is fluency. At strength {best['alpha']:g} the steered stories are harder for the unsteered model to
      predict (loss {best['feature_nll']:.2f} nats per token, against {n.steer_none_nll:.2f} without steering and
      {best['random_nll']:.2f} with a random direction). Pushed harder, the text falls apart. The example in the report
      (not cherry-picked) is the <em>{e(ex['word'])}</em> feature at strength {ex['alpha']:g}:
      <q>{e(steered)}&hellip;</q></p>

    <h2>Finding 2: reading features beats reading text, but not a regex</h2>
    <p>The guard has to decide before the next token exists, and that is genuinely uncertain: after
      <q>Lily and</q> the next word may be a name or <q>her</q>. Reading the model's internal features clearly helps:
      F1 {n.g_sae['new']['f1']:.2f}, against {n.g_text['new']['f1']:.2f} for a classifier that sees only the previous
      two tokens. But a regex that checks the finished word scores {n.g_regex['new']['f1']:.2f}. The regex's blind
      spot is a name at the start of a sentence, which looks like any other capitalized word, and that is where the
      internal signal adds the most. The combination and the strict mode were added after the first results; their
      thresholds still come only from the calibration stories.</p>
    {guard_chart(n)}
    <p>The guard adds {n.g_latency:.2f} ms to a {n.g_base_ms:.2f} ms generation step in the JavaScript engine. A
      follow-up compared SAEs of different sizes and the raw internal state at every layer, choosing on a separate split
      with yet another set of names. The k&nbsp;=&nbsp;{n.gl_selected['k']} SAE won (test F1 {n.gl_selected['new']['f1']:.2f}, against
      {n.gl_k8['new']['f1']:.2f} for k&nbsp;=&nbsp;8), slightly ahead of the best raw layer
      ({n.gl_best_resid['new']['f1']:.2f}), so the demo now uses it.</p>

    <h2>Finding 3: what did not work</h2>
    <ul>
      <li><strong>No induction heads.</strong> I scored every attention head at all {n.ind_ckpts} checkpoints on
        repeated random tokens. The best prefix-matching score is {n.ind_score:.3f}, far below the
        {n.ind_threshold:g} threshold, while a small control model trained to copy reaches {n.ind_control:.2f}, so the
        probe works. Previous-token heads, the first half of that circuit, reached only {n.prev_score:.2f}. This model
        and budget did not get there.</li>
      <li><strong>The cheap attribution method is weak.</strong> Direct logit attribution, the method the project
        brief named, correlates with the true effect of removing a feature at only {n.attr_dla:.2f} (Spearman), because
        {pct(n.attr_earlier)} of that effect travels through attention from earlier positions. Gradient times activation
        ranks features far better ({n.attr_grad:.2f}), though it picks the single most important feature less often
        ({pct(n.attr_grad_top1)} vs {pct(n.attr_dla_top1)}). The circuit graph in the demo uses exact ablation
        instead.</li>
      <li><strong>A lenient rule.</strong> The rule that picked the interface's SAE chose k&nbsp;=&nbsp;{n.sae_k},
        which raises the loss by {pct(k8['relative_loss_increase'], 1)}. The k&nbsp;=&nbsp;32 SAE raises it by only
        {pct(k32['relative_loss_increase'], 1)}, is just as token-specific ({pct(n.spec[32])}) and has
        {n.dead_feats[32]} dead feature against {n.dead_feats[8]}. I kept the rule's choice as the default and made
        the microscope switchable.</li>
    </ul>

    <h2>Limits</h2>
    <ul>
      <li>The model is tiny and trained on simple children's stories; what holds here may not hold for a large model.</li>
      <li>The new names in the guard test were swapped into real stories by a script, so the test measures names in
        familiar places, not every way a name can appear.</li>
      <li>Speed and latency were measured in Node.js on one laptop; browsers and devices differ.</li>
    </ul>

    <h2>What I would do next</h2>
    <ul>
      <li>Train longer, on a GPU, to see whether induction heads appear and whether the loss drops when they do.</li>
      <li>Repeat the study on Indonesian text, asking whether SAE features capture affixes such as <em>me-</em>,
        <em>di-</em> and <em>-kan</em>.</li>
    </ul>

    <h2>How it was built</h2>
    <p>I built Glassbox with Claude Code, an AI coding assistant, as a pair programmer, working from a written
      project brief with explicit rules: no invented numbers, a baseline for every claim, negative results reported. I
      set the goals and constraints, made the decisions along the way and reviewed the results at every milestone;
      the assistant wrote most of the code and ran the experiments.</p>
    <p class="stack">Python &middot; PyTorch (primitives only) &middot; NumPy &middot; pytest &middot; JavaScript (no
      dependencies) &middot; GitHub Actions &middot; GitHub Pages</p>
    <p class="links"><a href="index.html">Demo</a> &middot; <a href="{REPO}">Code</a> &middot;
      <a href="{REPO}/tree/main/results">All reports</a> &middot;
      <a href="{REPO}#reproduce-everything">Reproduce it</a></p>
  </main>
</body>
</html>
"""


# ----------------------------------------------------------------------------- LinkedIn drafts


def post_en(n: Numbers) -> str:
    words = ", ".join(f'"{w}"' for w in n.steer_words[:3])
    best = n.steer_best
    false_per_100 = n.g_both["new"]["false_blocks_per_1000_tokens"] / 10
    return f"""I built a language model from scratch, then built the tools to look inside it.

Glassbox is a {n.params_m:.1f}M-parameter transformer that I trained on a laptop CPU in {n.hours:.0f} hours. It runs entirely in your browser: no server, no API key.

Then I opened it up:
→ A sparse autoencoder splits its internal state into {n.n_features:,} readable features.
→ Steering with a single feature ({words}, ...) makes {pct(best['feature_mention'])} of stories mention that concept, against {pct(best['random_mention'])} for a random direction of the same size.
→ A name guard reads those features before each token is chosen. Together with a regex it catches {pct(n.g_both['new']['recall'])} of names it was never fitted on; the regex alone catches {pct(n.g_regex['new']['recall'])}. The price: about {false_per_100:.0f} in 100 other tokens are held back by mistake.

The part I care about most is the discipline:
• every number is generated by a script from the experiments, and CI checks the pages still match
• every claim is measured against a baseline
• selection rules were written down before the results came in
• negative results stay in: no induction heads formed, and the feature guard on its own does not beat a regex

Built with Claude Code as an AI pair programmer, from a written project brief. I set the goals, made the decisions and reviewed the results.

Try it: {SITE}
Case study: {SITE}case-study.html
Code: {REPO}

#MachineLearning #Interpretability #LLM #AIEngineering
"""


def post_id(n: Numbers) -> str:
    words = ", ".join(f'"{w}"' for w in n.steer_words[:3])
    best = n.steer_best
    false_per_100 = n.g_both["new"]["false_blocks_per_1000_tokens"] / 10
    return f"""Saya membangun language model dari nol, lalu membuat alat untuk melihat isi kepalanya.

Glassbox adalah transformer {num_id(f"{n.params_m:.1f}")} juta parameter yang saya latih di CPU laptop dalam {n.hours:.0f} jam. Semuanya berjalan di browser: tanpa server, tanpa API key.

Lalu saya bedah:
→ Sparse autoencoder memecah state internalnya menjadi {num_id(f"{n.n_features:,}")} fitur yang bisa dibaca.
→ Steering dengan satu fitur ({words}, ...) membuat {pct(best['feature_mention'])} cerita menyebut konsep itu, dibanding {pct(best['random_mention'])} dengan arah acak berukuran sama.
→ Pengaman nama membaca fitur-fitur itu sebelum tiap token dipilih. Digabung dengan regex, ia menangkap {pct(n.g_both['new']['recall'])} nama yang tidak pernah dipakai untuk melatihnya; regex saja hanya {pct(n.g_regex['new']['recall'])}. Harganya: sekitar {false_per_100:.0f} dari 100 token lain ikut tertahan karena salah deteksi.

Yang paling saya jaga adalah disiplinnya:
• setiap angka dibuat oleh skrip dari hasil eksperimen, dan CI mengecek halaman masih cocok
• setiap klaim diukur terhadap pembanding
• aturan pemilihan ditulis sebelum hasilnya keluar
• hasil negatif tetap dilaporkan: induction head tidak terbentuk, dan pengaman fitur sendirian tidak mengalahkan regex

Dibangun bersama Claude Code sebagai AI pair programmer, berdasarkan brief proyek tertulis. Tujuan, keputusan, dan review hasil ada di tangan saya.

Coba demonya: {SITE}
Studi kasus (EN): {SITE}case-study.html
Kode: {REPO}

#MachineLearning #Interpretability #LLM #AIEngineering
"""


def outputs(results: Path, web: Path, docs: Path) -> dict[Path, str]:
    n = Numbers(results)
    return {web / "case-study.html": case_study(n), docs / "linkedin_post_en.md": post_en(n),
            docs / "linkedin_post_id.md": post_id(n)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--web-dir", type=Path, default=Path("web"))
    p.add_argument("--docs-dir", type=Path, default=Path("docs"))
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    files = outputs(a.results_dir, a.web_dir, a.docs_dir)
    stale = [path for path, text in files.items()
             if not path.exists() or path.read_text(encoding="utf-8") != text]
    if a.check:
        if stale:
            sys.exit("out of date: " + ", ".join(str(s) for s in stale) + " (run python -m glassbox.eval.writeup)")
        print("case study and posts are up to date")
        return
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    print("wrote " + ", ".join(str(p) for p in files))


if __name__ == "__main__":
    main()
