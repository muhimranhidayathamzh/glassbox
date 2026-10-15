import re
from collections import Counter

import pytest

from glassbox.tokenizer import EOT, PRETOKEN_PATTERN, BPETokenizer, _merge

CORPUS = (
    "Once upon a time, there was a little cat named Tom. Tom liked to play in the sun.\n"
    "One day, Tom saw a big red ball. \"Can I play with it?\" he asked his mom.\n"
    "His mom said, \"Yes, you can!\" Tom was very happy. They played all day.\n"
    "Lily and Ben went to the park. They didn't want to go home, but it's late.\n"
) * 20


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(CORPUS, vocab_size=320)


def naive_bpe_merges(text: str, n_merges: int) -> list[tuple[int, int]]:
    """Reference implementation: recount every pair after every merge."""
    words = Counter(re.findall(PRETOKEN_PATTERN, text))
    seqs = {w: list(w.encode("utf-8")) for w in words}
    merges = []
    for i in range(n_merges):
        counts = Counter()
        for w, ids in seqs.items():
            for p in zip(ids, ids[1:]):
                counts[p] += words[w]
        if not counts:
            break
        best = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
        merges.append(best)
        seqs = {w: _merge(ids, best, 256 + i) for w, ids in seqs.items()}
    return merges


def test_vocab_layout(tok):
    assert tok.vocab_size == 320
    assert tok.eot_id == 319
    assert len(tok.merges) == 320 - 256 - 1
    assert tok.vocab[65] == b"A"


def test_incremental_training_matches_naive_reference():
    fast = BPETokenizer.train(CORPUS, vocab_size=256 + 80 + 1)
    assert fast.merges == naive_bpe_merges(CORPUS, 80)


def test_most_frequent_pair_merged_first_with_deterministic_tie_break():
    # (a, b) occurs 3x and wins outright.
    assert BPETokenizer.train("ab ab ab cd", 258).merges[0] == (ord("a"), ord("b"))
    # All pairs occur once: the smallest pair id wins, i.e. (' ', 'c') = (32, 99).
    assert BPETokenizer.train("ab cd", 258).merges[0] == (32, ord("c"))


def test_training_is_deterministic():
    a = BPETokenizer.train(CORPUS, vocab_size=300)
    b = BPETokenizer.train(CORPUS, vocab_size=300)
    assert a.merges == b.merges


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello",
        CORPUS[:300],
        "  multiple   spaces\n\n\ttabs and trailing space ",
        "don't it's we're I'll you've I'm he'd",
        "numbers 123 4567 and 3.14",
        "Ünïcödé ✓ 日本語 🙂 bytes never seen in training",
    ],
)
def test_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_special_token(tok):
    text = f"The end.{EOT}Once upon"
    ids = tok.encode(text)
    assert ids.count(tok.eot_id) == 1
    assert tok.decode(ids) == text
    # Without allow_special the marker is just text and is never mapped to the special id.
    plain = tok.encode(text, allow_special=False)
    assert tok.eot_id not in plain
    assert tok.decode(plain) == text


def test_encode_equals_applying_merges_in_training_order(tok):
    for word in set(re.findall(PRETOKEN_PATTERN, CORPUS)):
        ids = list(word.encode("utf-8"))
        for i, pair in enumerate(tok.merges):
            ids = _merge(ids, pair, 256 + i)
        assert tok.encode(word) == ids, word


def test_merges_never_cross_word_boundaries(tok):
    for i in range(256, 256 + len(tok.merges)):
        piece = tok.vocab[i]
        # Whitespace is only allowed as a leading space or as a pure-whitespace token.
        assert piece.isspace() or not any(c in b" \n\t" for c in piece[1:]), piece


def test_compresses_training_text(tok):
    assert len(tok.encode(CORPUS)) < 0.55 * len(CORPUS.encode("utf-8"))


def test_save_load_roundtrip(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.merges == tok.merges
    assert loaded.special_tokens == tok.special_tokens
    sample = "Tom and Lily played. 🙂" + EOT
    assert loaded.encode(sample) == tok.encode(sample)


def test_vocab_too_small_raises():
    with pytest.raises(ValueError):
        BPETokenizer.train("abc", vocab_size=100)
    with pytest.raises(ValueError):
        BPETokenizer.train("ab", vocab_size=400)  # not enough distinct pairs
