import numpy as np
import pytest

from glassbox.guard import Example, metrics
from glassbox.guard_layers import DenseProbe, evaluate, kept_positions, text_view


def test_text_view_keeps_labelled_positions_only():
    ex = Example(ids=[0, 1, 2, 3], labels=[1, -1, 0, -1], next_word=["Lily", "", "cat", ""],
                 next_sentence_start=[True, False, False, False])
    assert kept_positions(ex) == [0, 2]
    v = text_view([ex, ex])
    assert v.y.tolist() == [1, 0, 1, 0] and v.words == ["Lily", "cat", "Lily", "cat"]


def test_dense_probe_learns_a_linear_rule():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 5)).astype(np.float32)
    y = (x[:, 2] > 0.3).astype(int)
    s = DenseProbe(x, y).scores(x)
    assert metrics(s >= 0, y)["f1"] > 0.95


def test_evaluate_uses_the_selection_threshold_for_every_split():
    def view(y, words):
        return text_view([Example(ids=list(range(len(y) + 1)), labels=list(y) + [-1], next_word=words + [""],
                                  next_sentence_start=[False] * (len(y) + 1))])

    views = {"selection": view([1, 0, 1, 0], ["", "", "", ""]), "known": view([1, 0], ["Lily", ""]),
             "new": view([1, 0], ["", ""])}
    row = evaluate("c", "test", np.array([0.9, 0.1, 0.8, 0.2]), {"known": np.array([0.85, 0.5]),
                                                                  "new": np.array([0.5, 0.95])}, views, {"Lily"})
    assert row["threshold"] == pytest.approx(0.8)
    assert row["selection"]["f1"] == 1.0
    assert row["known"]["f1"] == 1.0 and row["new"]["f1"] == 0.0  # 0.5 < 0.8 misses, 0.95 is a false block
    assert row["regex_or_known"]["recall"] == 1.0
