"""End-to-end runs of the milestone 3 scripts on a tiny model and dataset."""

import json
import sys

import numpy as np
import pytest
import torch

from glassbox import features
from glassbox.eval import sae_eval
from glassbox.export import export, export_sae
from glassbox.model import GPT, GPTConfig
from glassbox.sae import SAEConfig, TopKSAE, save_sae
from glassbox.tokenizer import BPETokenizer
from glassbox.train import save_checkpoint

CORPUS = "Once upon a time, Lily saw a red ball. She was happy. Tom ran to the park and played. "


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("m3")
    tok = BPETokenizer.train(CORPUS * 20, vocab_size=300)
    data = root / "data"
    data.mkdir()
    tok.save(data / "tokenizer.json")
    rng = np.random.default_rng(0)
    stories = [tok.encode(" ".join(rng.permutation(CORPUS.split())[:12].tolist())) for _ in range(400)]
    stream = np.concatenate([np.array(s + [tok.eot_id]) for s in stories]).astype(np.uint16)
    stream.tofile(data / "train.bin")
    stream[: len(stream) // 2].tofile(data / "val.bin")

    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=300, n_ctx=32, d_model=24, n_layer=2, n_head=2, d_mlp=48)).eval()
    save_checkpoint(root / "ckpt.pt", model, step=0)
    m = export(model, tok, root / "web")

    runs = []
    for k in (2, 4):
        sae = TopKSAE(SAEConfig(d_in=24, n_features=64, k=k, layer=1, scale=0.5))
        sae.init_weights(torch.Generator().manual_seed(k))
        run = root / f"sae_k{k}"
        save_sae(sae, run / "sae.pt", source_checkpoint="ckpt.pt")
        log = {"config": {"lr": 1e-3, "total_tokens": 100, "batch_size": 10, "k_aux": 8, "aux_coef": 0.03,
                          "seed": 0}, "history": [{"fvu": 0.5, "aux": 1.0, "dead": 3}], "elapsed_s": 6.0}
        (run / "train_log.json").write_text(json.dumps(log))
        runs.append(run)
    export_sae(sae, root / "web" / "sae", model_weights_sha256=m["weights_sha256"])
    return root, runs, tok


def run_main(module, monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["prog", *map(str, args)])
    module.main()


def test_features_script_writes_consistent_ui_data(setup, monkeypatch):
    root, _, tok = setup
    run_main(features, monkeypatch, [
        "--model-dir", root / "web", "--sae-dir", root / "web" / "sae", "--data-dir", root / "data",
        "--results-dir", root / "results", "--eval-tokens", 32 * 40, "--n-cand", 30, "--threads", 1,
    ])
    ui = json.loads((root / "web" / "sae" / "features.json").read_text())
    assert len(ui["features"]) == 64 and [f["id"] for f in ui["features"]] == list(range(64))
    alive = [f for f in ui["features"] if not f.get("dead")]
    assert alive
    for f in alive:
        assert 0 < f["label_share"] <= 1 and f["max_act"] > 0 and 0 < f["density"] <= 1
        assert 1 <= len(f["examples"]) <= features.N_EXAMPLES_UI
        for e in f["examples"]:
            assert len(e["tokens"]) == len(e["acts"]) and 0 <= e["at"] < len(e["tokens"])
            assert all(0 <= v <= 100 for v in e["acts"])
            assert e["acts"][e["at"]] > 0  # the example's own token is active
        assert f["examples"][0]["acts"][f["examples"][0]["at"]] == 100  # strongest example = max_act
    report = json.loads((root / "results" / "sae_features.json").read_text())
    assert set(report["token_specificity"]) == {"SAE (trained)", "residual dimensions (+/-)", "random directions",
                                                "untrained SAE (same k)"}
    assert abs(sum(b["share_of_examples"] for b in report["example_positions"]) - 1) < 1e-9
    assert (root / "results" / "sae_features.md").read_text().startswith("# Milestone 3")


def test_sae_eval_script_reports_every_sae_and_applies_the_selection_rule(setup, monkeypatch):
    root, runs, _ = setup
    run_main(sae_eval, monkeypatch, [
        "--runs", *runs, "--ckpt", root / "ckpt.pt", "--data-dir", root / "data", "--results-dir", root / "results",
        "--recon-tokens", 32 * 20, "--loss-tokens", 32 * 8, "--fit-windows", 16, "--threads", 1,
    ])
    r = json.loads((root / "results" / "sae_eval.json").read_text())
    assert [s["k"] for s in r["saes"]] == [2, 4]
    for s in r["saes"]:
        assert s["l0"] <= s["k"] and s["variance_explained"] <= 1
        assert s["loss_increase"] == pytest.approx(s["loss_spliced"] - r["baselines"]["clean"])
        assert 0 <= s["dead_never_positive"] <= s["dead"]
    ok = [s for s in r["saes"] if s["loss_recovered"] >= sae_eval.SELECT_MIN_RECOVERED]
    assert r["selected"]["k"] == (ok[0]["k"] if ok else 4)
    assert "loss recovered" in (root / "results" / "sae_eval.md").read_text()
