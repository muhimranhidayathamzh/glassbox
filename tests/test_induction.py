import math

import numpy as np
import pytest
import torch

from glassbox.induction import (
    attention_patterns,
    first_step_above,
    formation_interval,
    head_scores,
    repeat_losses,
    repeated_sequences,
    sharp_drop,
    uniform_prefix_baseline,
)
from glassbox.model import GPT, GPTConfig

L = 6
T = 2 * L + 1


def test_repeated_sequences_layout():
    s = repeated_sequences(3, L, 10, 20, eot=99, seed=0)
    assert s.shape == (3, T)
    assert (s[:, 0] == 99).all()
    assert torch.equal(s[:, 1 : L + 1], s[:, L + 1 :])
    assert ((s[:, 1:] >= 10) & (s[:, 1:] < 20)).all()
    assert torch.equal(s, repeated_sequences(3, L, 10, 20, eot=99, seed=0))


def test_perfect_induction_and_previous_token_heads_score_one():
    pat = torch.zeros(2, 1, 2, T, T)  # (layer, batch, head, query, key)
    pat[..., 0, 0] = 1
    for q in range(1, T):
        pat[0, 0, 0, q, q - 1] = 1  # layer 0 head 0: previous-token head
        pat[1, 0, 1, q, q] = 1  # layer 1 head 1: attends to itself
    for j in range(L):
        pat[1, 0, 1, 1 + L + j] = 0
        pat[1, 0, 1, 1 + L + j, 2 + j] = 1  # induction: copy what followed the earlier occurrence
    prefix, prev = head_scores(pat, L)
    assert prev[0, 0] == pytest.approx(1.0) and prefix[1, 1] == pytest.approx(1.0)
    assert prefix[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_uniform_attention_matches_the_baseline():
    pat = torch.tril(torch.ones(T, T))
    pat = (pat / pat.sum(-1, keepdim=True)).expand(1, 1, 1, T, T)
    prefix, _ = head_scores(pat, L)
    assert prefix[0, 0] == pytest.approx(uniform_prefix_baseline(L))


class Oracle:
    """Fake model: knows the next token exactly in the second copy, guesses uniformly elsewhere."""

    def __init__(self, V):
        self.V = V

    def __call__(self, idx):
        B, n = idx.shape
        logits = torch.zeros(B, n, self.V)
        for i in range(L + 1, n):  # second copy: predict what followed the first occurrence
            logits[torch.arange(B), i, idx[:, i - L + 1]] = 50.0
        return logits, None


def test_repeat_losses_split_the_copies():
    V = 30
    seqs = repeated_sequences(4, L, 0, V - 1, eot=V - 1, seed=1)
    first, second = repeat_losses(Oracle(V), seqs, L)
    assert first == pytest.approx(math.log(V), rel=1e-5)
    assert second < 1e-6


def test_attention_patterns_are_causal_distributions_that_reproduce_attention():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=40, n_ctx=16, d_model=16, n_layer=2, n_head=2, d_mlp=32)).eval()
    idx = torch.randint(0, 40, (3, 9))
    pats = attention_patterns(model, idx)
    assert pats.shape == (2, 3, 2, 9, 9)
    torch.testing.assert_close(pats.sum(-1), torch.ones(2, 3, 2, 9))
    assert (pats.triu(diagonal=1) == 0).all()
    x = model.embed(idx)
    attn = model.blocks[0].attn
    with torch.no_grad():
        _, _, v = attn._heads(model.blocks[0].ln1(x))
        y = (pats[0] @ v).transpose(1, 2).reshape(3, 9, 16)
        torch.testing.assert_close(attn.proj(y), attn(model.blocks[0].ln1(x)))


def test_formation_interval_and_sharp_drop_rule():
    steps = [0, 10, 100, 1000, 10000]
    assert formation_interval([0.0, 0.05, 0.1, 0.8, 0.85]) == 2  # 100 -> 1000
    # loss falls 0.1 per ln-unit before and after, 1.0 in the interval -> sharp
    lns = [math.log(s + 1) for s in steps]
    losses = [5.0]
    for k, r in enumerate([0.1, 0.1, 1.0, 0.1]):
        losses.append(losses[-1] - r * (lns[k + 1] - lns[k]))
    d = sharp_drop(steps, losses, 2)
    assert d["rate"] == pytest.approx(1.0) and d["neighbour_rate"] == pytest.approx(0.1)
    assert d["sharp"] and d["ratio"] == pytest.approx(10.0)
    assert not sharp_drop(steps, losses, 0)["sharp"]  # edge interval: one neighbour only
    assert first_step_above(steps, [0.1, 0.2, 0.6, 0.9, 0.9], 0.5) == 100
    assert first_step_above(steps, [0.1] * 5, 0.5) is None


REPLAY = __import__("pathlib").Path(__file__).resolve().parents[1] / "web" / "model" / "replay.json"


@pytest.mark.skipif(not REPLAY.exists(), reason="run python -m glassbox.induction")
def test_shipped_replay_data_is_consistent():
    import json

    d = json.loads(REPLAY.read_text(encoding="utf-8"))
    steps = [c["step"] for c in d["checkpoints"]]
    assert steps == sorted(set(steps)) and steps[0] == 0
    T = len(d["ui_sequence"])
    for c in d["checkpoints"]:
        assert np.array(c["prefix"]).shape == np.array(c["prev"]).shape == (d["n_layer"], d["n_head"])
        assert len(c["samples"]) == 3
        for layer in c["patterns"]:
            for pat in layer:
                rows = np.array(pat).reshape(T, T)
                assert (np.triu(rows, 1) == 0).all()  # causal
                assert (np.abs(rows.sum(1) - 100) <= T).all()  # rows are rounded percentages
    f = d["formation"]
    assert f["from_step"] in steps and f["to_step"] in steps and f["from_step"] < f["to_step"]
    assert set(d["positive_control"]) == {"fixed_length", "variable_length"}
