"""Dataset preparation: download TinyStories, clean, train the tokenizer, encode to .bin files.

    python -m glassbox.data download
    python -m glassbox.data prepare

TinyStories (Eldan & Li, 2023) is released under CDLA-Sharing-1.0:
https://huggingface.co/datasets/roneneldan/TinyStories
We use the GPT-4-only "V2" split. To keep the download and the CPU budget small, only the
first TRAIN_HEAD_BYTES of the train file are fetched (HTTP Range request).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from .tokenizer import EOT, BPETokenizer

HF_BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
VALID_FILE = "TinyStoriesV2-GPT4-valid.txt"
TRAIN_FILE = "TinyStoriesV2-GPT4-train.txt"
TRAIN_HEAD_BYTES = 150 * 1024 * 1024
TRAIN_HEAD_NAME = "TinyStoriesV2-GPT4-train-head150MB.txt"

DEFAULT_RAW = Path("data/raw")
DEFAULT_OUT = Path("data/tinystories")


def download(raw_dir: Path = DEFAULT_RAW) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (VALID_FILE, VALID_FILE, None),
        (TRAIN_FILE, TRAIN_HEAD_NAME, f"bytes=0-{TRAIN_HEAD_BYTES - 1}"),
    ]
    for remote, local, byte_range in jobs:
        dest = raw_dir / local
        if dest.exists():
            print(f"exists: {dest}")
            continue
        req = urllib.request.Request(HF_BASE + remote)
        if byte_range:
            req.add_header("Range", byte_range)
        print(f"downloading {remote} -> {dest}")
        with urllib.request.urlopen(req) as resp, open(dest.with_suffix(".part"), "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
        dest.with_suffix(".part").rename(dest)


def split_stories(text: str) -> tuple[list[str], dict]:
    """Split raw TinyStories text into clean stories.

    Rules (counts are reported so nothing is dropped silently):
      * stories are separated by the literal <|endoftext|> marker
      * if the file does not end with the marker, the last story may be truncated: drop it
      * stories containing non-ASCII characters (mostly mojibake such as 'â€™') are dropped
    """
    segments = [s for s in (seg.strip() for seg in text.split(EOT)) if s]
    n_segments = len(segments)
    truncated_tail = bool(segments) and not text.rstrip().endswith(EOT)
    if truncated_tail:
        segments = segments[:-1]
    stories = [s for s in segments if s.isascii()]
    stats = {
        "segments": n_segments,
        "dropped_truncated_tail": int(truncated_tail),
        "dropped_non_ascii": len(segments) - len(stories),
        "kept": len(stories),
    }
    return stories, stats


def encode_stories(tok: BPETokenizer, stories: list[str]) -> np.ndarray:
    """Concatenate stories as `story <|endoftext|> story <|endoftext|> ...` token ids."""
    assert tok.vocab_size <= 65536
    eot = tok.eot_id
    ids: list[int] = []
    for s in stories:
        ids.extend(tok.encode(s, allow_special=False))
        ids.append(eot)
    return np.asarray(ids, dtype=np.uint16)


def prepare(
    raw_dir: Path = DEFAULT_RAW,
    out_dir: Path = DEFAULT_OUT,
    vocab_size: int = 4000,
    tokenizer_train_stories: int = 40_000,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_text = (raw_dir / TRAIN_HEAD_NAME).read_text(encoding="utf-8")
    valid_text = (raw_dir / VALID_FILE).read_text(encoding="utf-8")
    train_stories, train_stats = split_stories(train_text)
    valid_stories, valid_stats = split_stories(valid_text)

    t0 = time.time()
    tok_corpus = f"\n{EOT}\n".join(train_stories[:tokenizer_train_stories])
    tok = BPETokenizer.train(tok_corpus, vocab_size=vocab_size)
    tok_time = time.time() - t0
    tok.save(out_dir / "tokenizer.json")
    print(f"tokenizer: {tok.vocab_size} tokens, trained on {len(tok_corpus):,} chars in {tok_time:.1f}s")

    stats = {
        "source": "roneneldan/TinyStories " + TRAIN_FILE + " (first 150 MiB) / " + VALID_FILE,
        "license": "CDLA-Sharing-1.0",
        "vocab_size": tok.vocab_size,
        "tokenizer_train_stories": min(tokenizer_train_stories, len(train_stories)),
        "tokenizer_train_chars": len(tok_corpus),
        "tokenizer_train_seconds": round(tok_time, 2),
    }
    for name, stories, split_stats in [
        ("val", valid_stories, valid_stats),
        ("train", train_stories, train_stats),
    ]:
        t0 = time.time()
        ids = encode_stories(tok, stories)
        ids.tofile(out_dir / f"{name}.bin")
        n_chars = sum(len(s) for s in stories)
        stats[name] = {
            **split_stats,
            "tokens": int(ids.size),
            "chars": n_chars,
            "chars_per_token": round(n_chars / ids.size, 3),
            "encode_seconds": round(time.time() - t0, 1),
        }
        print(f"{name}: {split_stats['kept']:,} stories -> {ids.size:,} tokens")

    (out_dir / "data_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["download", "prepare"])
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--vocab-size", type=int, default=4000)
    p.add_argument("--tokenizer-train-stories", type=int, default=40_000)
    a = p.parse_args()
    if a.command == "download":
        download(a.raw_dir)
    else:
        prepare(a.raw_dir, a.out_dir, a.vocab_size, a.tokenizer_train_stories)


if __name__ == "__main__":
    main()
