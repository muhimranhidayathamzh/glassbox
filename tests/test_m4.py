import numpy as np
import pytest
import torch

from glassbox.attribution import (
    ablation_effects,
    candidates,
    direct_logit_attribution,
    feature_vectors,
    gradient_x_activation,
    spearman,
)
from glassbox.model import GPT, GPTConfig
from glassbox.sae import SAEConfig, TopKSAE
from glassbox.steer import generate_batch, mentions, pick_feature, steering_hook
from glassbox.tokenizer import BPETokenizer


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=50, n_ctx=64, d_model=16, n_layer=2, n_head=2, d_mlp=32)).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.3)
    sae = TopKSAE(SAEConfig(d_in=16, n_features=40, k=4, layer=1, scale=0.5))
    sae.init_weights(torch.Generator().manual_seed(1), torch.zeros(16))
    with torch.no_grad():
        sae.b_enc.normal_(0, 0.5)
    ids = torch.randint(0, 50, (20,), generator=torch.Generator().manual_seed(2)).tolist()
    return model, sae, ids


def test_candidates_are_the_active_features_in_the_window(tiny):
    model, sae, ids = tiny
    cands = candidates(model, sae, ids, t=12, window=5)
    assert {p for p, _, _ in cands} <= set(range(8, 13))
    with torch.no_grad():
        codes = sae.encode(model.resid_at(torch.tensor([ids[:13]]), 1)[0])
    for p, f, act in cands:
        assert act == pytest.approx(float(codes[p, f])) and act > 0
    assert len(cands) == int((codes[8:13] > 0).sum())


def test_ablation_matches_a_manual_rerun(tiny):
    model, sae, ids = tiny
    t, y = 12, 7
    cands = candidates(model, sae, ids, t)[:5]
    effects = ablation_effects(model, sae, ids, t, y, cands)
    d = feature_vectors(sae)
    with torch.no_grad():
        clean = model(torch.tensor([ids[: t + 1]]))[0][0, t, y]
        for (p, f, act), e in zip(cands, effects):
            def hook(x, p=p, f=f, act=act):
                x = x.clone()
                x[0, p] -= act * d[f]
                return x
            ablated = model(torch.tensor([ids[: t + 1]]), resid_hooks={1: hook})[0][0, t, y]
            assert e == pytest.approx(float(clean - ablated), abs=1e-5)


def test_gradient_estimate_equals_ablation_when_the_change_is_tiny(tiny):
    model, sae, ids = tiny
    t, y = 15, 3
    cands = [(p, f, act * 1e-4) for p, f, act in candidates(model, sae, ids, t)]  # first-order regime
    effects = ablation_effects(model, sae, ids, t, y, cands)
    estimates = gradient_x_activation(model, sae, ids, t, y, cands)
    np.testing.assert_allclose(estimates, effects, rtol=2e-2, atol=1e-6)


def test_dla_is_zero_off_the_predicting_position_and_follows_the_formula(tiny):
    model, sae, ids = tiny
    t, y = 15, 3
    cands = candidates(model, sae, ids, t)
    dla = direct_logit_attribution(model, sae, ids, t, y, cands)
    with torch.no_grad():
        final = model.resid_at(torch.tensor([ids[: t + 1]]), 2)[0, t]
        sigma = torch.sqrt(final.var(unbiased=False) + 1e-5)
        d = feature_vectors(sae)
    for (p, f, act), v in zip(cands, dla):
        if p != t:
            assert v == 0
        else:
            dc = d[f] - d[f].mean()
            expected = act * float((dc / sigma * model.ln_f.weight) @ model.wte.weight[y])
            assert v == pytest.approx(expected, rel=1e-5)


def test_spearman():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(a, a * 10) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)


def test_steering_hook_leaves_position_zero_alone():
    x = torch.zeros(2, 5, 3)
    out = steering_hook(torch.ones(3))(x)
    assert (out[:, 0] == 0).all() and (out[:, 1:] == 1).all()


def test_mentions_matches_whole_words_and_plurals():
    assert mentions("She threw the Ball.", "ball") and mentions("two balls", "ball")
    assert not mentions("a balloon", "ball") and not mentions("football", "ball")


def test_pick_feature_follows_the_fixed_rule():
    tok = BPETokenizer.train("the ball and the dog played with the ball " * 20, vocab_size=270)
    ball = tok.encode(" ball")
    assert len(ball) == 1
    feats = [
        {"id": 0, "dead": True},
        {"id": 1, "label_kind": "token", "label_token": ball[0], "n_examples": 20, "label_share": 0.9, "density": 0.1},
        {"id": 2, "label_kind": "token", "label_token": ball[0], "n_examples": 20, "label_share": 1.0, "density": 0.01},
        {"id": 3, "label_kind": "token", "label_token": ball[0], "n_examples": 5, "label_share": 1.0, "density": 0.5},
        {"id": 4, "label_kind": "after", "label_token": ball[0], "n_examples": 20, "label_share": 1.0, "density": 0.5},
    ]
    assert pick_feature(feats, tok, "ball")["id"] == 2
    assert pick_feature(feats, tok, "zebra") is None  # not a single token here


def test_generate_batch_is_reproducible_and_steering_changes_it(tiny):
    model, _, _ = tiny
    a = generate_batch(model, [1, 2, 3], n=4, seed=5)
    assert a == generate_batch(model, [1, 2, 3], n=4, seed=5)
    assert all(len(r) == 60 for r in a)
    steered = generate_batch(model, [1, 2, 3], n=4, seed=5, hooks={1: steering_hook(3.0 * torch.randn(16, generator=torch.Generator().manual_seed(0)))})
    assert steered != a
