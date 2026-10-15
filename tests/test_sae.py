import numpy as np
import pytest
import torch

from glassbox.model import GPT, GPTConfig
from glassbox.sae import (
    ActivationStream,
    SAEConfig,
    SAETrainConfig,
    TopKSAE,
    fit_sae,
    load_sae,
    lr_at,
    sae_losses,
    save_sae,
    train_sae,
)


def make_sae(d=16, n=64, k=4, scale=1.0, seed=0):
    sae = TopKSAE(SAEConfig(d_in=d, n_features=n, k=k, layer=1, scale=scale))
    sae.init_weights(torch.Generator().manual_seed(seed))
    return sae


def test_codes_are_topk_sparse_and_nonnegative():
    sae = make_sae()
    with torch.no_grad():
        sae.b_enc.normal_()
    z = sae.encode(torch.randn(100, 16))
    assert ((z > 0).sum(-1) <= 4).all()
    assert (z >= 0).all()
    pre = sae.encode_pre(torch.randn(5, 16))
    z = sae.topk_codes(pre)
    for row_pre, row_z in zip(pre, z):
        top = set(row_pre.topk(4).indices.tolist())
        assert set(row_z.nonzero().flatten().tolist()) <= top


def test_scale_is_applied_on_the_way_in_and_undone_on_the_way_out():
    sae = make_sae(scale=0.1)
    x = torch.randn(3, 16) * 10
    torch.testing.assert_close(sae.encode_pre(x), make_sae(scale=1.0).encode_pre(x * 0.1))
    torch.testing.assert_close(sae.decode(torch.zeros(1, 64)), (sae.b_dec / 0.1)[None])


def test_decoder_rows_stay_unit_norm_and_parallel_grad_is_removed():
    sae = make_sae()
    _, z = sae(torch.randn(32, 16))
    sae.decode(z).pow(2).sum().backward()
    before = sae.W_dec.grad.norm(dim=1).clone()
    sae.remove_parallel_grad()
    along = (sae.W_dec.grad * sae.W_dec).sum(1)
    assert (along.abs() <= 1e-5 * before.clamp_min(1e-12)).all()  # zero up to float32 rounding
    assert sae.W_dec.grad.abs().sum() > 0
    with torch.no_grad():
        sae.W_dec.mul_(3.0)
    sae.normalize_decoder()
    torch.testing.assert_close(sae.W_dec.norm(dim=1), torch.ones(64))


def test_auxk_loss_only_trains_dead_features():
    sae = make_sae()
    x = torch.randn(64, 16)
    dead = torch.zeros(64, dtype=torch.bool)
    dead[[3, 10, 40]] = True
    _, aux, _ = sae_losses(sae, x, dead, k_aux=2)
    assert aux > 0
    aux.backward()
    grad_cols = sae.W_enc.grad.abs().sum(0)
    assert (grad_cols[~dead] == 0).all()
    assert (grad_cols[dead] > 0).sum() >= 2
    _, aux_none, _ = sae_losses(sae, x, torch.zeros(64, dtype=torch.bool), k_aux=2)
    assert aux_none == 0


def true_dictionary(seed=0, n_true=32, d=16):
    D = torch.randn(n_true, d, generator=torch.Generator().manual_seed(seed))
    return D / D.norm(dim=1, keepdim=True)


def synthetic_batches(D, k_true=3, batch=512, seed=1):
    """x = sum of k_true distinct true features with positive coefficients, plus noise and an
    offset (so b_dec has something to learn)."""
    g = torch.Generator().manual_seed(seed)
    n_true, d = D.shape
    while True:
        idx = torch.stack([torch.randperm(n_true, generator=g)[:k_true] for _ in range(batch)])
        coef = torch.rand(batch, k_true, generator=g) * 2 + 0.5
        yield (coef[..., None] * D[idx]).sum(1) + 0.01 * torch.randn(batch, d, generator=g) + 1.0


def test_fit_recovers_a_synthetic_sparse_dictionary():
    D = true_dictionary()
    cfg = SAETrainConfig(total_tokens=512 * 1500, batch_size=512, lr=3e-3, warmup_steps=20,
                         dead_tokens=512 * 50, k_aux=8, sae=SAEConfig(d_in=16, n_features=64, k=3, layer=0))
    sae, log = fit_sae(synthetic_batches(D), cfg, verbose=False)
    # FVU plateaus near 0.075 even with 2.7x more steps: 32 features share 16 dimensions, so a
    # linear encoder cannot read their coefficients exactly (interference). Recovery of the
    # dictionary itself, checked below, is the real criterion.
    assert log["history"][-1]["fvu"] < 0.1
    # every true feature should have a learned decoder row pointing the same way
    best_cos = (D @ sae.W_dec.detach().T).max(dim=1).values
    assert (best_cos > 0.9).float().mean() > 0.9, best_cos


def test_lr_schedule_warms_up_holds_and_decays():
    cfg = SAETrainConfig(lr=1e-3, warmup_steps=10, decay_frac=0.2)
    lrs = [lr_at(s, 100, cfg) for s in range(100)]
    assert lrs[0] == pytest.approx(1e-4) and lrs[9] == pytest.approx(1e-3)
    assert lrs[50] == pytest.approx(1e-3)
    assert lrs[99] < lrs[90] < lrs[80] and lrs[99] > 0


@pytest.fixture(scope="module")
def tiny_lm():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=50, n_ctx=16, d_model=24, n_layer=2, n_head=2, d_mlp=48)).eval()
    tokens = np.random.default_rng(0).integers(0, 50, 5000).astype(np.uint16)
    return model, tokens


def test_activation_stream_is_deterministic_and_skips_position_zero(tiny_lm):
    model, tokens = tiny_lm
    a = ActivationStream(model, tokens, layer=1, batch_size=64, buffer_tokens=512, windows_per_forward=4, seed=3)
    b = ActivationStream(model, tokens, layer=1, batch_size=64, buffer_tokens=512, windows_per_forward=4, seed=3)
    fa, fb = a._fresh(100), b._fresh(100)
    torch.testing.assert_close(fa, fb, rtol=0, atol=0)

    starts = np.random.default_rng(3).integers(0, len(tokens) - 16, size=4)
    idx = torch.from_numpy(np.stack([tokens[s : s + 16] for s in starts]).astype(np.int64))
    with torch.no_grad():
        expected = model.resid_at(idx, 1)[:, 1:].reshape(-1, 24)
    torch.testing.assert_close(fa[:60], expected, rtol=0, atol=0)  # 4 windows x 15 positions

    batches_a, batches_b = iter(a), iter(b)
    for _ in range(10):  # crosses a buffer refill (256 tokens served per round)
        torch.testing.assert_close(next(batches_a), next(batches_b), rtol=0, atol=0)


def test_train_sae_end_to_end_and_save_load(tiny_lm, tmp_path):
    model, tokens = tiny_lm
    cfg = SAETrainConfig(total_tokens=64 * 60, batch_size=64, lr=1e-3, warmup_steps=5, buffer_tokens=1024,
                         log_every=10, sae=SAEConfig(d_in=24, n_features=96, k=4, layer=1))
    sae, log = train_sae(model, tokens, cfg, verbose=False)
    assert log["history"][-1]["fvu"] < log["history"][0]["fvu"]
    assert sae.cfg.scale != 1.0

    save_sae(sae, tmp_path / "sae.pt", note="x")
    loaded, ckpt = load_sae(tmp_path / "sae.pt")
    x = torch.randn(5, 24)
    torch.testing.assert_close(loaded.encode(x), sae.encode(x), rtol=0, atol=0)
    assert ckpt["note"] == "x" and loaded.cfg == sae.cfg


def test_story_windows_begin_at_eot_and_do_not_overlap():
    from glassbox.sae import story_starts, story_windows

    eot = 9
    tokens = np.array([1, 9, 2, 3, 9, 4, 5, 6, 9, 7, 9, 8, 8, 8], dtype=np.uint16)
    assert story_starts(tokens, eot, 4).tolist() == [1, 4, 8, 10]
    assert story_starts(tokens, eot, 5).tolist() == [1, 4, 8]
    x, y = story_windows(tokens, eot, T=4, n_windows=10)
    assert x[:, 0].tolist() == [eot, eot]  # starts 1 and 8; 4 would overlap the first window
    assert x.tolist() == [[9, 2, 3, 9], [9, 7, 9, 8]]
    assert y.tolist() == [[2, 3, 9, 4], [7, 9, 8, 8]]
    assert len(story_windows(tokens, eot, T=4, n_windows=1)[0]) == 1


def test_activation_stream_uses_the_given_window_starts(tiny_lm):
    model, tokens = tiny_lm
    starts = np.array([0, 100, 2000])
    stream = ActivationStream(model, tokens, layer=1, batch_size=8, buffer_tokens=64, windows_per_forward=3,
                              seed=0, starts=starts)
    acts = stream._fresh(45)
    chosen = starts[np.random.default_rng(0).integers(0, 3, size=3)]
    idx = torch.from_numpy(np.stack([tokens[s : s + 16] for s in chosen]).astype(np.int64))
    with torch.no_grad():
        expected = model.resid_at(idx, 1)[:, 1:].reshape(-1, 24)
    torch.testing.assert_close(acts, expected, rtol=0, atol=0)
