"""Sample text from a trained checkpoint.

    python -m glassbox.sample --prompt "Once upon a time" --n 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .tokenizer import EOT, BPETokenizer
from .train import load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("runs/main/final.pt"))
    p.add_argument("--tokenizer", type=Path, default=Path("data/tinystories/tokenizer.json"))
    p.add_argument("--prompt", default=EOT, help="default: start a new story")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    tok = BPETokenizer.load(a.tokenizer)
    model, _ = load_checkpoint(a.ckpt)
    idx = torch.tensor([tok.encode(a.prompt)])
    for i in range(a.n):
        g = torch.Generator().manual_seed(a.seed + i)
        out = model.generate(idx, a.max_new_tokens, a.temperature, a.top_k, tok.eot_id, g)
        text = tok.decode(t for t in out[0].tolist() if t != tok.eot_id)
        print(f"--- sample {i} (seed {a.seed + i}) ---\n{text}\n")


if __name__ == "__main__":
    main()
