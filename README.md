# Glassbox

A small language model built from scratch, plus tools to look inside it (work in progress).

Status:
- **Milestone 1** (tokenizer, model, training) done: [results/lm_eval.md](results/lm_eval.md)
- **Milestone 2** (inference in the browser) done: [results/web_parity.md](results/web_parity.md)
- **Milestone 3** (sparse autoencoder + feature microscope) done:
  [results/sae_eval.md](results/sae_eval.md), [results/sae_features.md](results/sae_features.md)
- **Milestone 4** (steering + attribution) done:
  [results/steering.md](results/steering.md), [results/attribution.md](results/attribution.md)

Every number in `results/` is written by a script; none are typed by hand.

## Try the web demo locally

The demo is a static page: the model (5 MB) is downloaded once and runs in the browser, with
no server-side computation and no API key. ES modules need to be served over HTTP:

```bash
python -m http.server 8000 -d web
# open http://localhost:8000 (generate) or http://localhost:8000/microscope.html (SAE features)
```

- **Generate**: write a story; open *Steering* to push it toward a concept with an SAE feature.
- **Microscope**: type text, click a token to see the features active on it and a circuit graph of the
  features that led the model to write it (computed by ablation in the browser).

URL parameters make a story shareable, e.g.
`?prompt=Tom%20had%20a%20red%20ball.&seed=3&temperature=0.8&topk=50&tokens&run`.

## Setup

Python 3.10+, CPU only. Node.js 20+ for the JavaScript tests (no npm packages needed).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch numpy pytest   # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pip install --no-deps -e .
```

## Reproduce milestone 1: train the model

```bash
python -m glassbox.data download     # TinyStories V2 (CDLA-Sharing-1.0): val + first 150 MiB of train
python -m glassbox.data prepare      # clean, train BPE tokenizer (vocab 4000), encode to data/tinystories/*.bin
python -m glassbox.train --out-dir runs/main --max-steps 7000 --threads 4 --log-interval 50
python -m glassbox.eval.lm_eval --run-dir runs/main   # writes results/lm_eval.{json,md}
python -m glassbox.sample --prompt "Once upon a time" --n 3
```

Training is deterministic (fixed seed) and resumable: stop with `--stop-at N` (a multiple of
`--eval-interval`) and continue with `--resume`.

## Reproduce milestone 2: export for the web and check parity

```bash
python -m glassbox.export              # runs/main/final.pt -> web/model/ (float16 weights, manifest, tokenizer)
python -m glassbox.eval.web_parity     # JS vs PyTorch on val data -> results/web_parity.{json,md}
```

## Reproduce milestone 3: sparse autoencoder and microscope

```bash
python -m glassbox.sae pilot                              # short lr sweep -> results/sae_lr_pilot.json
python -m glassbox.sae train --k 8  --out runs/sae_k8     # lr defaults to the pilot's best
python -m glassbox.sae train --k 16 --out runs/sae_k16
python -m glassbox.sae train --k 32 --out runs/sae_k32
python -m glassbox.eval.sae_eval --runs runs/sae_k8 runs/sae_k16 runs/sae_k32   # metrics, picks the UI's SAE
python -m glassbox.export sae          # the selected SAE -> web/model/sae
python -m glassbox.features            # examples + labels -> web/model/sae/features.json, results/sae_features.*
```

## Reproduce milestone 4: steering and attribution

```bash
python -m glassbox.steer           # feature vs random directions -> results/steering.*, web/model/sae/steering.json
python -m glassbox.attribution     # DLA and gradient estimates vs ablation -> results/attribution.*
```

## Tests

```bash
pytest                                  # Python tests, JS unit tests, and JS-vs-PyTorch parity
cd web && node --test "test/*.test.mjs" # JS unit tests only
```

The parity tests run the JavaScript engine (through Node.js) and PyTorch on the same exported
weights and require the max absolute logit difference to stay below 1e-3. They also require
identical tokenization and identical greedy generation, including past the 256-token context,
and, for the SAE, matching residual activations, pre-activations and active features, identical
steered generation, and matching ablation effects.
They run on a small random model every time and on the shipped `web/model` when present.

## Layout

```
python/glassbox/
  tokenizer.py     byte-level BPE from scratch (train, encode, decode, save/load)
  data.py          download, cleaning rules, tokenizer training, encoding to uint16 .bin
  model.py         decoder-only transformer; attention, LayerNorm, GELU written out by hand
  train.py         AdamW + warmup/cosine, log-spaced checkpoints for training replay, resume
  sample.py        text generation CLI
  export.py        float16 weights + manifest + tokenizer for the web engine (model and SAE)
  sae.py           top-k sparse autoencoder, activation stream, training
  features.py      feature densities, top examples, automatic labels, comparison with other bases
  steer.py         steering with feature directions, benchmark against random directions
  attribution.py   feature attribution (DLA, gradient x activation) validated by ablation
  eval/lm_eval.py      val loss vs uniform/unigram/bigram baselines, loss curve, samples
  eval/web_parity.py   JS engine vs PyTorch: logits, tokenizer, greedy generation, speed
  eval/sae_eval.py     SAE variance explained, L0, dead features, loss when spliced in; PCA baselines
web/
  engine/          JS inference: tokenizer, forward pass with KV cache and residual hooks, SAE,
                   steering, ablation-based attribution, sampling
  ui/              page logic and styles (index.html: generate, microscope.html: SAE features)
  model/           exported model served to the browser
  test/            JS unit tests (node --test)
  tools/           Node bridge used by the Python parity tests
tests/             pytest suite
results/           script-generated reports
```

## Data

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (Eldan & Li, 2023),
license CDLA-Sharing-1.0. Cleaning rules, applied in `glassbox/data.py` and counted in the report:
stories containing non-ASCII characters (mostly mojibake) are dropped, and a story cut off at the
end of a file is dropped.
