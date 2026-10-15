"""Parity: the JavaScript engine (web/engine) must reproduce PyTorch on the same weights.

Needs Node.js; skipped otherwise. Runs on a small random model (and SAE) every time, and on the
shipped model in web/model (and web/model/sae) when they exist.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from glassbox.eval.web_parity import PARITY_MAX_ABS_LOGIT_DIFF, REPO, node_available, run_node, torch_logits
from glassbox.export import export, export_sae, load_exported, load_exported_sae
from glassbox.model import GPT, GPTConfig
from glassbox.sae import SAEConfig, TopKSAE
from glassbox.tokenizer import EOT, BPETokenizer

pytestmark = pytest.mark.skipif(not node_available(), reason="node is not installed")

TEXTS = [
    "",
    "Once upon a time, there was a little girl named Lily.",
    f"The end.{EOT}Once upon a time",
    "  odd   spacing\n\n\tand tabs  ",
    "Unicode: café — 日本語 \U0001f642 \x1c\x1d\x85 　﻿ done",
    "don't stop, it's 3:45pm! (really?)",
]
SHIPPED = REPO / "web" / "model"
SHIPPED_SAE = SHIPPED / "sae"
# Same budget as the logits: residuals and SAE pre-activations are O(1-30) in magnitude.
SAE_PARITY_MAX_ABS_DIFF = 1e-3
NEAR_TIE = 1e-3


def test_js_unit_tests_pass():
    proc = subprocess.run(
        ["node", "--test", "test/engine.test.mjs"], cwd=REPO / "web", capture_output=True, encoding="utf-8"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.fixture(scope="module", params=["float16", "float32"])
def tiny_export(request, tmp_path_factory):
    tok = BPETokenizer.train(
        "Lily and Tom played with a big red ball in the park. They laughed! " * 30, vocab_size=300
    )
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=300, n_ctx=32, d_model=48, n_layer=2, n_head=3, d_mlp=96))
    with torch.no_grad():  # larger weights than the init, so logits are far from uniform
        for p in model.parameters():
            p.normal_(0, 0.3)
    out = tmp_path_factory.mktemp(f"export_{request.param}")
    m = export(model, tok, out, dtype=request.param)

    sae = TopKSAE(SAEConfig(d_in=48, n_features=96, k=6, layer=1, scale=0.5))
    sae.init_weights(torch.Generator().manual_seed(1), torch.randn(48))
    with torch.no_grad():
        sae.b_enc.normal_(0, 0.5)
    export_sae(sae, out / "sae", dtype=request.param, model_weights_sha256=m["weights_sha256"])
    return out


def check_parity(model_dir: Path, sequences: list[list[int]], greedy_tokens: int) -> None:
    model, tok, _ = load_exported(model_dir)
    prompts = [[tok.eot_id] + tok.encode("Once upon a time"), tok.encode("Lily")]
    res, arrays = run_node({
        "model_dir": model_dir,
        "sequences": sequences,
        "encode": TEXTS,
        "decode": [tok.encode(t) for t in TEXTS],
        "greedy": [{"prompt": p, "n": greedy_tokens} for p in prompts],
    })

    for ids, got in zip(sequences, arrays["logits"]):
        ref = torch_logits(model, ids)
        assert got.shape == ref.shape
        assert np.abs(got - ref).max() < PARITY_MAX_ABS_LOGIT_DIFF
        assert ref.std() > 0.1  # make sure the comparison is not trivially between ~0 values

    assert res["encoded"] == [tok.encode(t) for t in TEXTS]
    assert res["decoded"] == TEXTS

    for p, js_ids in zip(prompts, res["greedy"]):
        py_ids = model.generate(torch.tensor([p]), greedy_tokens, temperature=0)[0, len(p):].tolist()
        assert js_ids == py_ids


def check_sae_parity(model_dir: Path, sae_dir: Path, sequences: list[list[int]]) -> None:
    model, _, _ = load_exported(model_dir)
    sae, _ = load_exported_sae(sae_dir)
    k, layer = sae.cfg.k, sae.cfg.layer
    res, arrays = run_node(
        {"model_dir": model_dir, "sae_dir": sae_dir, "features": sequences},
        widths={"resid": model.cfg.d_model, "pre": sae.cfg.n_features},
    )
    checked = skipped = 0
    for ids, resid_js, pre_js, codes_js in zip(sequences, arrays["resid"], arrays["pre"], res["codes"]):
        with torch.no_grad():
            resid = model.resid_at(torch.tensor([ids]), layer)[0]
            pre = sae.encode_pre(resid)
        assert np.abs(resid_js - resid.numpy()).max() < SAE_PARITY_MAX_ABS_DIFF
        assert np.abs(pre_js - pre.numpy()).max() < SAE_PARITY_MAX_ABS_DIFF
        assert pre.std() > 0.1

        for p, (js_ids, js_acts) in enumerate(codes_js):
            vals, idx = pre[p].topk(k + 1)
            if vals[k - 1] - vals[k] < NEAR_TIE or (vals[:k].abs() < NEAR_TIE).any():
                skipped += 1  # selection is decided by float noise here
                continue
            expected = {int(i): float(v) for i, v in zip(idx[:k], vals[:k]) if v > 0}
            got = dict(zip(js_ids, js_acts))
            assert got.keys() == expected.keys(), (p, got, expected)
            for i in got:
                assert abs(got[i] - expected[i]) < SAE_PARITY_MAX_ABS_DIFF
            assert list(js_acts) == sorted(js_acts, reverse=True)
            checked += 1
    assert checked > 0 and skipped <= 0.05 * (checked + skipped)


def test_tiny_model_parity(tiny_export):
    rng = np.random.default_rng(0)
    seqs = [[5], rng.integers(0, 300, 7).tolist(), rng.integers(0, 300, 32).tolist()]
    check_parity(tiny_export, seqs, greedy_tokens=80)  # 80 > n_ctx: context restarts


def test_tiny_sae_parity(tiny_export):
    rng = np.random.default_rng(1)
    check_sae_parity(tiny_export, tiny_export / "sae", [rng.integers(0, 300, 32).tolist(), [7, 8, 9]])


@pytest.mark.skipif(not (SHIPPED / "manifest.json").exists(), reason="run python -m glassbox.export")
def test_shipped_model_parity():
    _, tok, manifest = load_exported(SHIPPED)
    story = tok.encode("Once upon a time, there was a little dog named Max. Max liked to run. " * 20)
    n_ctx = manifest["config"]["n_ctx"]
    seqs = [[tok.eot_id], tok.encode("Tom had a red ball. He"), story[:n_ctx]]
    check_parity(SHIPPED, seqs, greedy_tokens=n_ctx + 50)


@pytest.mark.skipif(not (SHIPPED_SAE / "manifest.json").exists(), reason="run python -m glassbox.export sae")
def test_shipped_sae_parity():
    _, tok, manifest = load_exported(SHIPPED)
    story = [tok.eot_id] + tok.encode(
        "Lily found a red ball in the park. She said, \"Can I play with it?\" Her mom smiled. " * 8
    )
    check_sae_parity(SHIPPED, SHIPPED_SAE, [story[: manifest["config"]["n_ctx"]]])
