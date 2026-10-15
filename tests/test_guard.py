import numpy as np
import pytest

from glassbox.guard import (
    recall_threshold,
    BagLR,
    Features,
    best_single_feature,
    best_threshold,
    build_lexicon,
    has_unknown_capitals,
    label_story,
    metrics,
    name_spans,
    regex_flags,
    sentence_start,
    single_feature_scores,
    substitute,
)
from glassbox.tokenizer import BPETokenizer


def test_sentence_start():
    t = 'Lily ran. Tom said, "Look at Sam!" Then Ben came.'
    assert sentence_start(t, t.index("Lily"))
    assert sentence_start(t, t.index("Tom"))
    assert sentence_start(t, t.index("Look"))  # dialogue opens a sentence
    assert not sentence_start(t, t.index("Sam"))
    assert sentence_start(t, t.index("Then"))
    assert not sentence_start(t, t.index("Ben"))


def test_lexicon_keeps_mid_sentence_capitalized_words_only():
    stories = ["Then Lily and Tom played. The cat saw Lily. Big dogs like Tom."] * 25 + ["the big cat."] * 5
    names, counts = build_lexicon(stories)
    assert names == {"Lily", "Tom"}  # "Then", "The", "Big" only start sentences
    assert counts["Lily"] == 50


def test_unknown_capitals_and_substitution():
    names = {"Lily", "Tom"}
    assert not has_unknown_capitals("Lily saw Tom. I said, \"Wow!\" Mom smiled.", names)
    assert has_unknown_capitals("Lily saw Zed.", names)
    rng = np.random.default_rng(0)
    story, used = substitute("Lily met Tom. Lily's dog liked Tom.", names, ["Aiko", "Budi", "Citra"], rng)
    words = story.replace(".", "").replace("'s", "").split()
    assert words[0] == words[3] and words[2] == words[6] and words[0] != words[2]  # consistent, distinct
    assert used == {words[0], words[2]} and not ({"Lily", "Tom"} & set(words))


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train("Lily saw Tom. Tom ran to the park with Lily and Mimi. " * 30, vocab_size=280)


def test_label_story_marks_the_token_before_each_name(tok):
    story = 'Lily saw Mimi. "Tom!" said Lily.'
    ex = label_story(tok, story, name_spans(story, {"Lily", "Tom", "Mimi"}), n_ctx=64)
    pieces = [tok.token_str(i) for i in ex.ids]
    for t, lab in enumerate(ex.labels):
        nxt = pieces[t + 1] if t + 1 < len(pieces) else None
        if lab == 1:  # the next token is the first piece of a name
            assert any(name.startswith(nxt.lstrip(' "')) for name in ("Lily", "Tom", "Mimi")), nxt
    starts = [ex.next_word[t] for t, lab in enumerate(ex.labels) if lab == 1]
    assert starts == ["Lily", "Mimi", "Tom", "Lily"]
    assert ex.labels[-1] == -1
    first = ex.labels.index(1)
    assert first == 0 and ex.next_sentence_start[0]  # position 0 (<|endoftext|>) predicts "Lily"


def test_regex_flags():
    f = Features(resid=None, resid_last=None, code_idx=None, code_val=None, prev2=None, y=np.zeros(5),
                 words=["Lily", "Zed", "Zed", "The", "cat"],
                 sentence_start=np.array([True, False, True, True, False]))
    assert regex_flags(f, {"Lily"}).tolist() == [True, True, False, False, False]


def test_threshold_and_metrics():
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    y = np.array([1, 1, 0, 1, 0])
    th = best_threshold(scores, y)
    assert th == 0.2  # top four: precision 3/4, recall 1, F1 6/7 (better than the 0.8 cut's 0.8)
    m = metrics(scores >= th, y)
    assert (m["tp"], m["fp"], m["fn"]) == (3, 1, 0) and m["f1"] == pytest.approx(6 / 7)


def test_best_single_feature_matches_brute_force():
    rng = np.random.default_rng(0)
    N, k = 400, 3
    idx = np.stack([rng.choice(30, k, replace=False) for _ in range(N)])
    val = rng.random((N, k)).astype(np.float32)
    val[rng.random((N, k)) < 0.3] = 0
    y = (rng.random(N) < 0.2).astype(int)
    y[(idx == 7).any(1) & (rng.random(N) < 0.8)] = 1  # feature 7 predicts the label
    f = Features(resid=None, resid_last=None, code_idx=idx, code_val=val, prev2=None, y=y, words=[],
                 sentence_start=np.zeros(N, bool))
    feat, th, f1 = best_single_feature(f, min_hits=5)
    brute = []  # every activation value of every feature as a threshold
    for g in range(30):
        s = single_feature_scores(f, g)
        if (s > 0).sum() < 5:
            continue
        brute += [(metrics(s >= t, y)["f1"], g) for t in np.unique(s[s > 0])]
    assert feat == max(brute)[1] == 7
    assert f1 == pytest.approx(max(brute)[0])


def test_bag_lr_learns_a_separable_problem():
    rng = np.random.default_rng(1)
    idx = rng.integers(0, 10, (300, 2))
    y = (idx == 3).any(1).astype(int)
    lr = BagLR(10, idx, np.ones((300, 2), np.float32), y)
    s = lr.scores(idx, np.ones((300, 2), np.float32))
    assert metrics(s >= best_threshold(s, y), y)["f1"] == pytest.approx(1.0)
    assert int(np.argmax(lr.w.detach().numpy())) == 3


def test_recall_threshold_catches_the_target_share():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    y = np.array([1, 0, 1, 1, 0, 1])
    assert recall_threshold(scores, y, 0.5) == 0.7  # 2 of 4 caught at 0.7
    assert recall_threshold(scores, y, 0.75) == 0.6
    assert recall_threshold(scores, y, 1.0) == 0.4
