import math

import numpy as np
import pytest

from glassbox.eval.lm_eval import eval_pairs, ngram_baselines, training_time


def test_eval_pairs_are_contiguous_non_overlapping_windows():
    val = np.arange(1000, dtype=np.uint16)
    x, y = eval_pairs(val, n_tokens=100, T=8)
    assert x.shape == y.shape == (12, 8)
    np.testing.assert_array_equal(y, x + 1)
    np.testing.assert_array_equal(x[1, 0], x[0, -1] + 1)


def test_ngram_baselines_rank_as_expected_on_a_deterministic_cycle():
    V = 16
    rng = np.random.default_rng(0)
    perm = rng.permutation(V)
    train = perm[np.arange(1_200_000) % V]  # next token fully determined by the previous one
    val = perm[np.arange(4097) % V]
    x, y = eval_pairs(val, 4096, 64)

    summary, per_token = ngram_baselines(train, x, y, V)

    assert summary["uniform"] == pytest.approx(math.log(V))
    assert summary["unigram"] == pytest.approx(math.log(V), abs=1e-3)  # all tokens equally frequent
    assert summary["bigram"] < 0.01
    assert per_token.shape == y.shape
    assert summary["bigram_mu"] == 0.1  # least smoothing wins on noise-free data


def test_training_time_separates_pauses_from_active_time():
    # Ten 60 s log intervals, one of which also contains a 1 h suspend.
    elapsed, t = [], 0.0
    for i in range(10):
        t += 60 + (3600 if i == 6 else 0)
        elapsed.append(t)
    log = {"train": [{"elapsed_s": e} for e in elapsed], "total_elapsed_s": elapsed[-1]}

    times = training_time(log)

    assert times["n_pauses"] == 1
    assert times["paused_s"] == pytest.approx(3600)
    assert times["active_s"] == pytest.approx(600)
