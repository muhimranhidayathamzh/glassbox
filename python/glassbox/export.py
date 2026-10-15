"""Export a checkpoint (and optionally a sparse autoencoder) for the web engine.

    python -m glassbox.export          # model -> web/model
    python -m glassbox.export sae      # SAE selected in results/sae_eval.json -> web/model/sae

Writes to out_dir:
    weights.bin      all tensors, little-endian float16 (or float32), concatenated
    manifest.json    config, tensor names/shapes/offsets (in elements), sha256, provenance
    tokenizer.json   (model only) the tokenizer plus a JavaScript-compatible pre-token regex

`load_exported` / `load_exported_sae` rebuild PyTorch modules from these exact files, so parity
tests compare the JS engine against PyTorch running the very same (rounded) weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .model import GPT, GPTConfig
from .sae import SAEConfig, TopKSAE
from .tokenizer import BPETokenizer, js_pretoken_pattern

FORMAT = "glassbox-weights"
SAE_FORMAT = "glassbox-sae"
DTYPES = {"float16": np.float16, "float32": np.float32}


def _write_weights(state: dict[str, torch.Tensor], out: Path, dtype: str) -> dict:
    little_endian = np.dtype(DTYPES[dtype]).newbyteorder("<")
    tensors, chunks, offset = [], [], 0
    for name, t in state.items():
        with np.errstate(over="ignore"):  # overflow is detected just below
            arr = t.detach().cpu().numpy().astype(little_endian)
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} has values that do not fit in {dtype}")
        tensors.append({"name": name, "shape": list(arr.shape), "offset": offset, "length": arr.size})
        chunks.append(arr.tobytes())
        offset += arr.size
    blob = b"".join(chunks)
    (out / "weights.bin").write_bytes(blob)
    return {
        "dtype": dtype,
        "weights_file": "weights.bin",
        "weights_bytes": len(blob),
        "weights_sha256": hashlib.sha256(blob).hexdigest(),
        "tensors": tensors,
    }


def _read_weights(manifest: dict, out: Path) -> dict[str, torch.Tensor]:
    blob = (out / manifest["weights_file"]).read_bytes()
    if hashlib.sha256(blob).hexdigest() != manifest["weights_sha256"]:
        raise ValueError(f"{out / manifest['weights_file']} does not match the sha256 in manifest.json")
    flat = np.frombuffer(blob, dtype=np.dtype(DTYPES[manifest["dtype"]]).newbyteorder("<"))
    return {
        t["name"]: torch.from_numpy(
            flat[t["offset"] : t["offset"] + t["length"]].astype(np.float32).reshape(t["shape"])
        )
        for t in manifest["tensors"]
    }


def _read_manifest(out: Path, fmt: str) -> dict:
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != fmt:
        raise ValueError(f"{out} does not contain a {fmt} export")
    return manifest


def export(
    model: GPT,
    tok: BPETokenizer,
    out_dir: str | Path,
    dtype: str = "float16",
    source: dict | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights = _write_weights(model.state_dict(), out, dtype)

    tok_dict = tok.to_dict()
    tok_dict["js_pretoken_pattern"] = js_pretoken_pattern()
    (out / "tokenizer.json").write_text(json.dumps(tok_dict), encoding="utf-8")

    manifest = {
        "format": FORMAT,
        "version": 1,
        "dtype": dtype,
        "config": model.cfg.to_dict(),
        "n_params": model.num_params(),
        "weights_file": weights["weights_file"],
        "weights_bytes": weights["weights_bytes"],
        "weights_sha256": weights["weights_sha256"],
        "tokenizer_file": "tokenizer.json",
        "tensors": weights["tensors"],
        "source": source or {},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def load_exported(out_dir: str | Path) -> tuple[GPT, BPETokenizer, dict]:
    out = Path(out_dir)
    manifest = _read_manifest(out, FORMAT)
    model = GPT(GPTConfig(**manifest["config"]))
    model.load_state_dict(_read_weights(manifest, out))
    model.eval()
    tok = BPETokenizer.load(out / manifest["tokenizer_file"])
    return model, tok, manifest


def export_sae(
    sae: TopKSAE,
    out_dir: str | Path,
    dtype: str = "float16",
    source: dict | None = None,
    model_weights_sha256: str | None = None,
) -> dict:
    """model_weights_sha256 ties the SAE to the exported model whose activations it reads."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights = _write_weights(sae.state_dict(), out, dtype)
    manifest = {
        "format": SAE_FORMAT,
        "version": 1,
        "config": sae.cfg.to_dict(),
        **weights,
        "model_weights_sha256": model_weights_sha256,
        "source": source or {},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def load_exported_sae(out_dir: str | Path) -> tuple[TopKSAE, dict]:
    out = Path(out_dir)
    manifest = _read_manifest(out, SAE_FORMAT)
    sae = TopKSAE(SAEConfig(**manifest["config"]))
    sae.load_state_dict(_read_weights(manifest, out))
    sae.eval()
    return sae, manifest


def main() -> None:
    from .sae import load_sae
    from .train import load_checkpoint

    p = argparse.ArgumentParser()
    p.add_argument("what", nargs="?", choices=["model", "sae"], default="model")
    p.add_argument("--ckpt", type=Path, default=Path("runs/main/final.pt"))
    p.add_argument("--tokenizer", type=Path, default=Path("data/tinystories/tokenizer.json"))
    p.add_argument("--sae", type=Path, help="sae: path to sae.pt (default: the one selected by sae_eval)")
    p.add_argument("--model-dir", type=Path, default=Path("web/model"), help="sae: the exported model")
    p.add_argument("--out", type=Path, default=None, help="default: web/model, or web/model/sae")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="float16")
    a = p.parse_args()

    if a.what == "model":
        out = a.out or Path("web/model")
        model, ckpt = load_checkpoint(a.ckpt)
        tok = BPETokenizer.load(a.tokenizer)
        source = {"checkpoint": a.ckpt.as_posix(), "step": ckpt["step"], "val_loss": ckpt.get("val_loss")}
        m = export(model, tok, out, a.dtype, source)
        print(f"wrote {out}: {m['n_params']:,} params, {m['weights_bytes'] / 1e6:.2f} MB ({a.dtype})")
        return

    if a.sae is None:  # default: the SAE selected by python -m glassbox.eval.sae_eval
        selected = json.loads(Path("results/sae_eval.json").read_text(encoding="utf-8"))["selected"]
        a.sae = Path(selected["run"]) / "sae.pt"
    out = a.out or a.model_dir / "sae"
    sae, ckpt = load_sae(a.sae)
    model_manifest = _read_manifest(a.model_dir, FORMAT)
    source = {"sae": a.sae.as_posix(), "source_checkpoint": ckpt.get("source_checkpoint")}
    m = export_sae(sae, out, a.dtype, source, model_manifest["weights_sha256"])
    print(f"wrote {out}: {m['config']['n_features']} features, k={m['config']['k']}, "
          f"{m['weights_bytes'] / 1e6:.2f} MB ({a.dtype})")


if __name__ == "__main__":
    main()
