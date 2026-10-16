"""Index of the sparse autoencoders shipped to the browser, for the microscope's selector.

    python -m glassbox.eval.sae_index        # -> web/model/saes.json

Lists every web/model/sae*/ directory that has an SAE export and features.json, with its metrics
from results/sae_eval.json. web/model/sae is the default: it is the SAE selected by sae_eval's
rule, and the one the Generate page uses for steering and the name guard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_DIR = "sae"


def build_index(model_dir: Path, sae_eval: dict) -> dict:
    rows = {r["k"]: r for r in sae_eval["saes"]}
    entries = []
    for d in sorted(model_dir.glob("sae*")):
        manifest_path, features_path = d / "manifest.json", d / "features.json"
        if not (d.is_dir() and manifest_path.exists() and features_path.exists()):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cfg = manifest["config"]
        row = rows[cfg["k"]]
        entries.append({
            "dir": d.name, "k": cfg["k"], "n_features": cfg["n_features"], "layer": cfg["layer"],
            "default": d.name == DEFAULT_DIR, "weights_sha256": manifest["weights_sha256"],
            "variance_explained": row["variance_explained"],
            "relative_loss_increase": row["relative_loss_increase"],
            "dead_share": row["dead"] / cfg["n_features"],
        })
    if sum(e["default"] for e in entries) != 1:
        raise SystemExit(f"expected exactly one default SAE in {model_dir / DEFAULT_DIR}")
    return {"format": "glassbox-sae-index", "version": 1, "saes": sorted(entries, key=lambda e: e["k"])}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=Path("web/model"))
    p.add_argument("--sae-eval", type=Path, default=Path("results/sae_eval.json"))
    a = p.parse_args()
    index = build_index(a.model_dir, json.loads(a.sae_eval.read_text(encoding="utf-8")))
    (a.model_dir / "saes.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"wrote {a.model_dir / 'saes.json'}: " + ", ".join(f"{e['dir']} (k={e['k']})" for e in index["saes"]))


if __name__ == "__main__":
    main()
