import math

import pytest
import torch
import torch.nn.functional as F

from glassbox.model import GPT, CausalSelfAttention, GPTConfig, LayerNorm, gelu

TINY = GPTConfig(vocab_size=64, n_ctx=16, d_model=32, n_layer=2, n_head=4, d_mlp=128)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(TINY)


def test_default_config_is_within_parameter_budget():
    n = GPT(GPTConfig()).num_params()
    assert 1_000_000 <= n <= 5_000_000, n


def test_output_shapes(model):
    idx = torch.randint(0, TINY.vocab_size, (3, 10))
    logits, loss = model(idx)
    assert logits.shape == (3, 10, TINY.vocab_size)
    assert loss is None
    _, loss = model(idx, idx)
    assert loss.ndim == 0


def test_rejects_sequence_longer_than_context(model):
    with pytest.raises(ValueError):
        model(torch.zeros(1, TINY.n_ctx + 1, dtype=torch.long))


def test_initial_loss_is_close_to_uniform(model):
    idx = torch.randint(0, TINY.vocab_size, (8, 16))
    _, loss = model(idx, torch.randint(0, TINY.vocab_size, (8, 16)))
    assert abs(loss.item() - math.log(TINY.vocab_size)) < 0.1


def test_causal_future_tokens_do_not_change_past_logits(model):
    idx = torch.randint(0, TINY.vocab_size, (1, 12))
    changed = idx.clone()
    changed[0, 7:] = (changed[0, 7:] + 1) % TINY.vocab_size
    a, _ = model(idx)
    b, _ = model(changed)
    torch.testing.assert_close(a[0, :7], b[0, :7])
    assert not torch.allclose(a[0, 7:], b[0, 7:])


def test_handwritten_ops_match_torch_reference():
    torch.manual_seed(1)
    x = torch.randn(2, 5, 32)

    ln = LayerNorm(32, 1e-5)
    with torch.no_grad():
        ln.weight.normal_()
        ln.bias.normal_()
    torch.testing.assert_close(ln(x), F.layer_norm(x, (32,), ln.weight, ln.bias, 1e-5))

    torch.testing.assert_close(gelu(x), F.gelu(x, approximate="tanh"))

    attn = CausalSelfAttention(TINY)
    q, k, v = attn.qkv(x).split(32, dim=-1)
    heads = lambda t: t.view(2, 5, 4, 8).transpose(1, 2)  # noqa: E731
    ref = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), is_causal=True)
    ref = attn.proj(ref.transpose(1, 2).reshape(2, 5, 32))
    torch.testing.assert_close(attn(x), ref)


def test_output_layer_is_tied_to_input_embedding(model):
    # Token 5 never appears in the input, so its embedding row can only get gradient
    # through the (tied) unembedding.
    idx = torch.tensor([[1, 2, 3, 4]])
    logits, _ = model(idx)
    logits[..., 5].sum().backward()
    assert model.wte.weight.grad[5].abs().sum() > 0
    assert model.wte.weight.grad[6].abs().sum() == 0


def test_can_overfit_a_single_batch(model):
    torch.manual_seed(0)
    idx = torch.randint(0, TINY.vocab_size, (4, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(150):
        _, loss = model(idx[:, :-1], idx[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_generate_is_reproducible_and_stops(model):
    prompt = torch.tensor([[1, 2, 3]])
    g1, g2 = torch.Generator().manual_seed(42), torch.Generator().manual_seed(42)
    a = model.generate(prompt, 20, temperature=1.0, top_k=10, generator=g1)
    b = model.generate(prompt, 20, temperature=1.0, top_k=10, generator=g2)
    assert torch.equal(a, b) and a.shape == (1, 23)
    assert torch.equal(a[:, :3], prompt)

    greedy = model.generate(prompt, 5, temperature=0)
    stop = greedy[0, 3].item()
    stopped = model.generate(prompt, 5, temperature=0, stop_token=stop)
    assert stopped.shape == (1, 4)


def test_generate_slides_window_beyond_context(model):
    out = model.generate(torch.zeros(1, 1, dtype=torch.long), TINY.n_ctx + 5, temperature=0)
    assert out.shape == (1, TINY.n_ctx + 6)


def test_resid_hooks_read_and_replace_the_residual_stream(model):
    idx = torch.randint(0, TINY.vocab_size, (2, 9))
    plain, _ = model(idx)
    seen = {}

    def capture(layer):
        def hook(x):
            seen[layer] = x.clone()
            return x
        return hook

    hooked, _ = model(idx, resid_hooks={l: capture(l) for l in range(TINY.n_layer + 1)})
    torch.testing.assert_close(hooked, plain, rtol=0, atol=0)
    for layer in range(TINY.n_layer + 1):
        torch.testing.assert_close(model.resid_at(idx, layer), seen[layer], rtol=0, atol=0)

    zeroed, _ = model(idx, resid_hooks={1: torch.zeros_like})
    assert not torch.allclose(zeroed, plain)
    with pytest.raises(ValueError):
        model.resid_at(idx, TINY.n_layer + 1)
