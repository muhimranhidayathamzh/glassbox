import json
import re

import numpy as np
import pytest
import torch

from glassbox.export import export, load_exported
from glassbox.model import GPT, GPTConfig
from glassbox.tokenizer import EOT, PRETOKEN_PATTERN, BPETokenizer, js_pretoken_pattern

TRICKY = (
    "Tom's dog didn't bark.\n\n  Then   \t he ran!!! 42 cats, 3.5 m\u00e8tres \u2014 \u65e5\u672c "
    "\U0001f642\x1c\x1d\x1e\x1f\x85\u00a0\u2003\u3000\ufeff\u200b end \r\n"
)


def test_js_pattern_has_exactly_pythons_whitespace():
    chars = "".join(chr(c) for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF)
    ws_class = re.search(r"\|\[([^\]]+)\]\+\(\?!", js_pretoken_pattern()).group(1)
    assert re.findall(f"[{ws_class}]", chars) == re.findall(r"\s", chars)


def test_js_pattern_splits_like_the_python_pattern():
    for text in [TRICKY, TRICKY * 3, "a  b", "x\u2003\u2003y", "end with spaces   "]:
        assert re.findall(js_pretoken_pattern(), text) == re.findall(PRETOKEN_PATTERN, text)


@pytest.fixture(scope="module")
def tiny():
    tok = BPETokenizer.train("the cat sat on the mat. the dog ran to the cat! " * 20, vocab_size=270)
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=270, n_ctx=16, d_model=24, n_layer=2, n_head=3, d_mlp=48))
    return model, tok


def test_float32_export_round_trips_exactly(tiny, tmp_path):
    model, tok = tiny
    manifest = export(model, tok, tmp_path, dtype="float32", source={"step": 1})
    loaded, tok2, m2 = load_exported(tmp_path)
    for k, v in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v), k
    assert tok2.merges == tok.merges and m2 == manifest
    assert manifest["weights_bytes"] == 4 * model.num_params()


def test_float16_export_equals_rounding_the_weights(tiny, tmp_path):
    model, tok = tiny
    export(model, tok, tmp_path, dtype="float16")
    loaded, _, manifest = load_exported(tmp_path)
    assert manifest["weights_bytes"] == 2 * model.num_params()
    for k, v in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v.half().float()), k


def test_manifest_layout_is_contiguous_and_complete(tiny, tmp_path):
    model, tok = tiny
    m = export(model, tok, tmp_path)
    offset = 0
    for t in m["tensors"]:
        assert t["offset"] == offset and t["length"] == int(np.prod(t["shape"]))
        offset += t["length"]
    assert offset == model.num_params()
    assert {t["name"] for t in m["tensors"]} == set(model.state_dict())
    web_tok = json.loads((tmp_path / "tokenizer.json").read_text())
    assert web_tok["js_pretoken_pattern"] == js_pretoken_pattern()
    assert web_tok["special_tokens"] == {EOT: 269}


def test_corrupted_weights_are_rejected(tiny, tmp_path):
    model, tok = tiny
    export(model, tok, tmp_path)
    blob = bytearray((tmp_path / "weights.bin").read_bytes())
    blob[10] ^= 0xFF
    (tmp_path / "weights.bin").write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="sha256"):
        load_exported(tmp_path)


def test_weights_too_large_for_float16_are_rejected(tiny, tmp_path):
    model, tok = tiny
    big = GPT(model.cfg)
    with torch.no_grad():
        big.wte.weight[0, 0] = 1e6
    with pytest.raises(ValueError, match="float16"):
        export(big, tok, tmp_path)


def test_sae_export_round_trips_through_float16(tmp_path):
    from glassbox.export import export_sae, load_exported_sae
    from glassbox.sae import SAEConfig, TopKSAE

    sae = TopKSAE(SAEConfig(d_in=24, n_features=96, k=4, layer=1, scale=0.37))
    sae.init_weights(torch.Generator().manual_seed(0), torch.randn(24))
    m = export_sae(sae, tmp_path, source={"sae": "x"}, model_weights_sha256="abc")
    loaded, m2 = load_exported_sae(tmp_path)
    assert m2 == m and m["model_weights_sha256"] == "abc"
    assert loaded.cfg == sae.cfg
    for k, v in sae.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v.half().float()), k
    with pytest.raises(ValueError, match="glassbox-weights"):
        load_exported(tmp_path)  # an SAE export is not a model export
