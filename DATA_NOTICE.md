# Data notice

The model was trained on **TinyStories** by Ronen Eldan and Yuanzhi Li (2023), *TinyStories: How Small
Can Language Models Be and Still Speak Coherent English?*, published at
https://huggingface.co/datasets/roneneldan/TinyStories under the
[Community Data License Agreement – Sharing, Version 1.0](https://spdx.org/licenses/CDLA-Sharing-1.0.html).

The dataset is **not** redistributed here (`data/` is excluded from the repository). This repository
does contain material derived from it:

| file | what it contains |
|---|---|
| `web/model/sae/features.json` | short excerpts (up to 21 tokens each) of TinyStories validation stories, used as feature examples |
| `results/sae_features.md`, `results/sae_features.json` | twelve of those excerpts |
| `web/model/sae/guard.json` | a list of character names found in the TinyStories training stories |

These excerpts and the name list are made available under CDLA-Sharing-1.0, with the attribution above.
Model weights, scores and other numbers in `results/` are results of computational use of the data.
All code in this repository is under the MIT license (`LICENSE`).
