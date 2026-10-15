"""Byte-level BPE tokenizer, built from scratch.

Vocabulary layout (the JS engine in milestone 2 relies on this):
    0..255                 single bytes
    256..256+M-1           merges, in the order they were learned (lower id = higher priority)
    256+M..                special tokens (e.g. <|endoftext|>)

Text is first split into "pre-tokens" with a regex (GPT-2 style: a word keeps its leading
space), and merges never cross a pre-token boundary.
"""

from __future__ import annotations

import functools
import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

EOT = "<|endoftext|>"

# Contractions, words with an optional leading space, single digits, punctuation runs,
# and whitespace (a run of spaces leaves its last space to attach to the next word).
PRETOKEN_PATTERN = r"""'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""


@functools.cache
def js_pretoken_pattern() -> str:
    r"""PRETOKEN_PATTERN with `\s` spelled out as the exact code points Python's `re` treats as
    whitespace. JavaScript's `\s` differs (it lacks U+001C-U+001F and U+0085 and adds U+FEFF),
    so the web tokenizer uses this form (with the `u` flag). It is also a valid Python pattern."""
    chars = "".join(chr(c) for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF)
    cps = sorted({ord(c) for c in re.findall(r"\s", chars)})
    ranges: list[list[int]] = []
    for c in cps:
        if ranges and c == ranges[-1][1] + 1:
            ranges[-1][1] = c
        else:
            ranges.append([c, c])
    ws = "".join(
        f"\\u{lo:04x}" if lo == hi else f"\\u{lo:04x}-\\u{hi:04x}" for lo, hi in ranges
    )
    return (
        f"'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]| ?[^{ws}A-Za-z0-9]+"
        f"|[{ws}]+(?![^{ws}])|[{ws}]+"
    )


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` (left to right) with `new_id`."""
    out = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]], special_tokens: Iterable[str] = (EOT,)):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: 256 + i for i, pair in enumerate(self.merges)}

        self.vocab: list[bytes] = [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])

        self.special_tokens: dict[str, int] = {}
        for tok in special_tokens:
            self.special_tokens[tok] = len(self.vocab)
            self.vocab.append(tok.encode("utf-8"))
        self.special_ids = set(self.special_tokens.values())

        self._pretok = re.compile(PRETOKEN_PATTERN)
        self._special_split = (
            re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ training

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        special_tokens: Iterable[str] = (EOT,),
    ) -> "BPETokenizer":
        """Learn merges until the vocabulary (bytes + merges + specials) reaches `vocab_size`.

        Ties between equally frequent pairs are broken by the smaller (left, right) id pair,
        so training is fully deterministic.
        """
        special_tokens = list(special_tokens)
        n_merges = vocab_size - 256 - len(special_tokens)
        if n_merges < 0:
            raise ValueError(f"vocab_size {vocab_size} is smaller than 256 bytes + specials")

        # Count pre-tokens; special tokens are separators, never merged into words.
        pretok = re.compile(PRETOKEN_PATTERN)
        chunks = re.split("|".join(re.escape(t) for t in special_tokens), text) if special_tokens else [text]
        word_counts: Counter[str] = Counter()
        for chunk in chunks:
            word_counts.update(pretok.findall(chunk))

        words = [list(w.encode("utf-8")) for w in word_counts]
        freqs = list(word_counts.values())

        pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        pair_where: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, ids in enumerate(words):
            for pair in zip(ids, ids[1:]):
                pair_counts[pair] += freqs[wi]
                pair_where[pair].add(wi)

        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        while len(merges) < n_merges and heap:
            neg_count, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) != -neg_count or neg_count == 0:
                continue  # stale heap entry
            new_id = 256 + len(merges)
            merges.append(pair)

            changed: set[tuple[int, int]] = set()
            for wi in pair_where.pop(pair):
                old = words[wi]
                new = _merge(old, pair, new_id)
                if len(new) == len(old):
                    continue  # word no longer contains the pair (lazy index)
                f = freqs[wi]
                for p in zip(old, old[1:]):
                    pair_counts[p] -= f
                    changed.add(p)
                for p in zip(new, new[1:]):
                    pair_counts[p] += f
                    pair_where[p].add(wi)
                    changed.add(p)
                words[wi] = new
            pair_counts.pop(pair, None)
            changed.discard(pair)
            for p in changed:
                c = pair_counts[p]
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                else:
                    pair_counts.pop(p, None)

        if len(merges) < n_merges:
            raise ValueError(
                f"corpus only supports {len(merges)} merges, {n_merges} requested; use more text"
            )
        return cls(merges, special_tokens)

    # ------------------------------------------------------------------ encode / decode

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def eot_id(self) -> int:
        return self.special_tokens[EOT]

    def _encode_word(self, word: str) -> list[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = list(word.encode("utf-8"))
        while len(ids) >= 2:
            best = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, 1 << 30))
            rank = self.ranks.get(best)
            if rank is None:
                break
            ids = _merge(ids, best, rank)
        if len(self._cache) < 200_000:
            self._cache[word] = ids
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for word in self._pretok.findall(text):
            out.extend(self._encode_word(word))
        return out

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        """Encode text. With allow_special, literal special-token strings become their ids;
        otherwise they are encoded as ordinary bytes."""
        if not allow_special or self._special_split is None:
            return self._encode_ordinary(text)
        out: list[int] = []
        for part in self._special_split.split(text):
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
            elif part:
                out.extend(self._encode_ordinary(part))
        return out

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        return b"".join(self.vocab[i] for i in ids)

    def decode(self, ids: Iterable[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def token_str(self, i: int) -> str:
        """Readable form of a single token (for inspection / UI)."""
        return self.vocab[i].decode("utf-8", errors="backslashreplace")

    # ------------------------------------------------------------------ persistence

    def to_dict(self) -> dict:
        return {
            "type": "byte-level-bpe",
            "version": 1,
            "pretoken_pattern": PRETOKEN_PATTERN,
            "vocab_size": self.vocab_size,
            "merges": [list(m) for m in self.merges],
            "special_tokens": self.special_tokens,
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("type") != "byte-level-bpe" or d.get("pretoken_pattern") != PRETOKEN_PATTERN:
            raise ValueError(f"{path} is not a compatible glassbox tokenizer")
        specials = sorted(d["special_tokens"], key=d["special_tokens"].get)
        tok = cls([tuple(m) for m in d["merges"]], specials)
        if tok.special_tokens != d["special_tokens"] or tok.vocab_size != d["vocab_size"]:
            raise ValueError(f"{path} has inconsistent special tokens or vocab size")
        return tok
