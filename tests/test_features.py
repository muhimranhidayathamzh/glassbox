import numpy as np
import pytest
import torch

from glassbox.eval.sae_eval import pca_reconstruct, reconstruction_stats, splice, spliced_losses
from glassbox.features import TopTracker, auto_label, distinct_examples
from glassbox.model import GPT, GPTConfig
from glassbox.sae import SAEConfig, TopKSAE
from glassbox.tokenizer import BPETokenizer


def test_top_tracker_matches_brute_force():
    g = torch.Generator().manual_seed(0)
    acts = torch.randn(1000, 7, generator=g)
    where = torch.arange(1000) * 3 + 5
    tr = TopTracker(7, 12)
    for i in range(0, 1000, 64):  # streamed in uneven batches
        tr.update(acts[i : i + 64], where[i : i + 64])
    for d in range(7):
        top = acts[:, d].topk(12)
        torch.testing.assert_close(tr.acts[d], top.values)
        assert tr.where[d].tolist() == where[top.indices].tolist()


def test_distinct_examples_keeps_one_per_window_and_stops_at_min_act():
    T = 10
    x = np.arange(100).reshape(10, T)  # every context is different
    acts = [9.0, 8.0, 7.0, 6.0, 0.0, -1.0]
    where = [3 * T + 4, 3 * T + 5, 7 * T + 1, 2 * T + 9, 5 * T + 2, 6 * T + 2]
    assert distinct_examples(acts, where, x, n=10, min_act=0.0) == [(9.0, 3, 4), (7.0, 7, 1), (6.0, 2, 9)]
    assert distinct_examples(acts, where, x, n=2, min_act=0.0) == [(9.0, 3, 4), (7.0, 7, 1)]
    assert len(distinct_examples(acts, where, x, n=10, min_act=float("-inf"))) == 5


def test_distinct_examples_drops_repeats_of_the_same_local_context():
    T = 8
    x = np.zeros((4, T), dtype=np.int64)
    x[0] = [9, 1, 2, 3, 4, 5, 6, 7]
    x[1] = [9, 1, 2, 3, 4, 0, 0, 0]  # same 4 tokens up to position 4 as window 0
    x[2] = [9, 8, 2, 3, 4, 0, 0, 0]  # last 4 tokens (8, 2, 3, 4) at position 4: different
    x[3] = [9, 7, 1, 2, 3, 4, 0, 0]  # (1, 2, 3, 4) again, at position 5
    where = [0 * T + 4, 1 * T + 4, 2 * T + 4, 3 * T + 5]
    out = distinct_examples([4.0, 3.0, 2.0, 1.0], where, x, n=10, min_act=0.0)
    assert [w for _, w, _ in out] == [0, 2]


def test_auto_label_picks_the_most_concentrated_position():
    tok = BPETokenizer.train("the cat sat on the mat and the dog ran " * 5, vocab_size=270)
    a, b = ord("a"), ord("b")  # single-byte tokens: token_str is "a" / "b"
    lab = auto_label(tok, cur=[b, b, b, a], prev=[a, b, a, b], nxt=[1, 2, 3, 4])
    assert lab["label_kind"] == "token" and lab["label_token"] == b and lab["label_share"] == 0.75
    assert lab["label"] == '"b"'
    lab = auto_label(tok, cur=[1, 2, 3, 4], prev=[a, a, a, a], nxt=[b, b, 3, 4])
    assert lab["label"] == 'after "a"' and lab["label_share"] == 1.0
    lab = auto_label(tok, cur=[1, 2, 3, 4], prev=[5, 6, 7, 8], nxt=[b, b, b, 4])
    assert lab["label"] == 'before "b"'
    lab = auto_label(tok, cur=[a, a, 3, 4], prev=[b, b, 7, 8], nxt=[1, 2, 3, 4])
    assert lab["label_kind"] == "token"  # tie between token and after: token wins


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=40, n_ctx=12, d_model=16, n_layer=2, n_head=2, d_mlp=32)).eval()
    x = np.random.default_rng(0).integers(0, 40, (8, 12))
    return model, x


def test_identity_splice_leaves_the_loss_unchanged(tiny):
    model, x = tiny
    y = np.roll(x, -1, axis=1)
    out = spliced_losses(model, 1, x, y, {"clean": None, "identity": lambda r: r, "zero": torch.zeros_like})
    assert out["identity"] == pytest.approx(out["clean"], abs=1e-6)
    assert out["zero"] != pytest.approx(out["clean"], abs=1e-3)


def test_splice_keeps_position_zero():
    x = torch.randn(2, 5, 3)
    out = splice(torch.zeros_like)(x)
    torch.testing.assert_close(out[:, 0], x[:, 0])
    assert (out[:, 1:] == 0).all()


def test_full_rank_pca_explains_everything(tiny):
    model, x = tiny
    with torch.no_grad():
        acts = model.resid_at(torch.from_numpy(x), 1)[:, 1:].reshape(-1, 16).double()
    mean = acts.mean(0)
    _, comps = torch.linalg.eigh((acts - mean).T @ (acts - mean))
    sae = TopKSAE(SAEConfig(d_in=16, n_features=16, k=16, layer=1))  # k = d -> PCA keeps all 16
    sae.init_weights(torch.Generator().manual_seed(0))
    stats = reconstruction_stats(model, 1, x, {"s": sae}, mean.float(), comps.float().flip(1))
    assert stats["s"]["pca_variance_explained"] == pytest.approx(1.0, abs=1e-5)
    assert stats["s"]["tokens"] == 8 * 11
    assert stats["s"]["l0"] <= 16 and stats["s"]["dead"] + stats["s"]["dense"] <= 16
    r = acts[:5].float()
    torch.testing.assert_close(pca_reconstruct(r, mean.float(), comps.float(), 16), r, atol=1e-4, rtol=0)
