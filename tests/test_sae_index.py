import json
from pathlib import Path

import pytest

from glassbox.eval.sae_index import build_index

MODEL = Path(__file__).resolve().parents[1] / "web" / "model"


def make_sae_dir(root: Path, name: str, k: int, with_features: bool = True) -> None:
    d = root / name
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"config": {"k": k, "n_features": 100, "layer": 2},
                                                 "weights_sha256": f"sha{k}"}))
    if with_features:
        (d / "features.json").write_text("{}")


def test_build_index_lists_complete_exports_with_their_metrics(tmp_path):
    make_sae_dir(tmp_path, "sae", 8)
    make_sae_dir(tmp_path, "sae_k16", 16)
    make_sae_dir(tmp_path, "sae_k32", 32, with_features=False)  # incomplete: skipped
    rows = [{"k": k, "variance_explained": v, "relative_loss_increase": r, "dead": d}
            for k, v, r, d in [(8, 0.8, 0.1, 10), (16, 0.9, 0.05, 2), (32, 0.95, 0.02, 0)]]
    index = build_index(tmp_path, {"saes": rows})
    assert [e["dir"] for e in index["saes"]] == ["sae", "sae_k16"]
    assert index["saes"][0]["default"] and not index["saes"][1]["default"]
    assert index["saes"][1]["variance_explained"] == 0.9 and index["saes"][0]["dead_share"] == 0.1


@pytest.mark.skipif(not (MODEL / "saes.json").exists(), reason="run python -m glassbox.eval.sae_index")
def test_shipped_index_matches_the_exports():
    index = json.loads((MODEL / "saes.json").read_text(encoding="utf-8"))
    for e in index["saes"]:
        manifest = json.loads((MODEL / e["dir"] / "manifest.json").read_text(encoding="utf-8"))
        features = json.loads((MODEL / e["dir"] / "features.json").read_text(encoding="utf-8"))
        assert manifest["weights_sha256"] == e["weights_sha256"] == features["sae_weights_sha256"]
        assert manifest["config"]["k"] == e["k"]
    assert sum(e["default"] for e in index["saes"]) == 1
