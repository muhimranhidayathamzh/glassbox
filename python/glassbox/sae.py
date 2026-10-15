"""Top-k sparse autoencoder on the residual stream (Gao et al., 2024, "Scaling and evaluating
sparse autoencoders"), built from torch primitives.

    python -m glassbox.sae pilot                              # choose the learning rate
    python -m glassbox.sae train --k 16 --out runs/sae_k16    # train one SAE

    x_s   = x * scale                                  (scale makes E||x_s||^2 = d_model)
    pre   = (x_s - b_dec) @ W_enc + b_enc
    z     = ReLU(top-k(pre))                           (all other features are exactly 0)
    x_hat = (z @ W_dec + b_dec) / scale                (decoder rows have unit norm)

Loss: normalized MSE + aux_coef * AuxK, where AuxK reconstructs the remaining error using the
top k_aux pre-activations among features that have not fired recently ("dead"), which gives
them gradient and brings them back.

Activations come from the residual stream entering block `layer`, on windows of the training
split that begin at a story boundary (<|endoftext|>), like text in the web UI. (Windows at
random offsets would start mid-story with no context, a state the UI never produces.)
Position 0 of every window is excluded: it is always <|endoftext|> and carries no context.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .model import GPT
from .tokenizer import BPETokenizer
from .train import load_checkpoint, load_tokens


@dataclass
class SAEConfig:
    d_in: int = 192
    n_features: int = 2048
    k: int = 16
    layer: int = 2
    scale: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


class TopKSAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        self.W_enc = nn.Parameter(torch.empty(cfg.d_in, cfg.n_features))
        self.b_enc = nn.Parameter(torch.zeros(cfg.n_features))
        self.W_dec = nn.Parameter(torch.empty(cfg.n_features, cfg.d_in))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in))

    @torch.no_grad()
    def init_weights(self, generator: torch.Generator | None = None, data_mean: torch.Tensor | None = None):
        """Unit-norm random decoder rows, encoder = decoder transpose, b_dec = data mean."""
        w = torch.randn(self.cfg.n_features, self.cfg.d_in, generator=generator)
        self.W_dec.copy_(w / w.norm(dim=1, keepdim=True))
        self.W_enc.copy_(self.W_dec.T)
        self.b_enc.zero_()
        self.b_dec.copy_(data_mean * self.cfg.scale if data_mean is not None else torch.zeros(self.cfg.d_in))

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.cfg.scale - self.b_dec) @ self.W_enc + self.b_enc

    def topk_codes(self, pre: torch.Tensor, k: int | None = None) -> torch.Tensor:
        vals, idx = pre.topk(k or self.cfg.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, idx, torch.relu(vals))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.topk_codes(self.encode_pre(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return (z @ self.W_dec + self.b_dec) / self.cfg.scale

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True))

    @torch.no_grad()
    def remove_parallel_grad(self) -> None:
        """Drop the gradient component along each decoder row (keeps the unit-norm constraint
        from fighting the optimizer)."""
        if self.W_dec.grad is not None:
            g = self.W_dec.grad
            g.sub_((g * self.W_dec).sum(dim=1, keepdim=True) * self.W_dec)


# --------------------------------------------------------------------------- activations


def story_starts(tokens: np.ndarray, eot_id: int, length: int) -> np.ndarray:
    """Positions of <|endoftext|> from which `length` tokens fit: windows that begin exactly at
    a story boundary."""
    eots = np.flatnonzero(np.asarray(tokens) == eot_id)
    return eots[eots + length <= len(tokens)]


def story_windows(tokens: np.ndarray, eot_id: int, T: int, n_windows: int) -> tuple[np.ndarray, np.ndarray]:
    """The first n_windows non-overlapping windows that begin at <|endoftext|>, in order.
    Returns x (n, T) and the next tokens y (n, T)."""
    chosen, next_free = [], -1
    for s in story_starts(tokens, eot_id, T + 1):
        if s >= next_free:
            chosen.append(int(s))
            next_free = s + T
            if len(chosen) == n_windows:
                break
    x = np.stack([tokens[s : s + T] for s in chosen]).astype(np.int64)
    y = np.stack([tokens[s + 1 : s + T + 1] for s in chosen]).astype(np.int64)
    return x, y


class ActivationStream:
    """Shuffled residual-stream activations from training windows starting at random entries of
    `starts` (default: any offset).

    Keeps a buffer of `buffer_tokens` activations; each round it is shuffled, the first half
    is served in batches, and that half is refilled with fresh activations. Deterministic
    for a given seed."""

    def __init__(
        self,
        model: GPT,
        tokens: np.ndarray,
        layer: int,
        batch_size: int = 4096,
        buffer_tokens: int = 1 << 18,
        windows_per_forward: int = 32,
        seed: int = 0,
        starts: np.ndarray | None = None,
    ):
        self.model, self.tokens, self.layer = model, tokens, layer
        self.starts = starts
        self.batch_size, self.buffer_tokens = batch_size, buffer_tokens
        self.windows_per_forward = windows_per_forward
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.T = model.cfg.n_ctx

    @torch.no_grad()
    def _fresh(self, n_tokens: int) -> torch.Tensor:
        out, have = [], 0
        while have < n_tokens:
            if self.starts is None:
                starts = self.rng.integers(0, len(self.tokens) - self.T, size=self.windows_per_forward)
            else:
                starts = self.starts[self.rng.integers(0, len(self.starts), size=self.windows_per_forward)]
            idx = torch.from_numpy(np.stack([self.tokens[s : s + self.T] for s in starts]).astype(np.int64))
            acts = self.model.resid_at(idx, self.layer)[:, 1:].reshape(-1, self.model.cfg.d_model)
            out.append(acts)
            have += acts.shape[0]
        return torch.cat(out)[:n_tokens]

    def __iter__(self):
        buf = self._fresh(self.buffer_tokens)
        half = self.buffer_tokens // 2
        while True:
            buf = buf[torch.randperm(len(buf), generator=self.gen)]
            for i in range(0, half - self.batch_size + 1, self.batch_size):
                yield buf[i : i + self.batch_size]
            buf[:half] = self._fresh(half)


def estimate_scale_and_mean(stream_batches, d_in: int) -> tuple[float, torch.Tensor]:
    """scale such that E||x * scale||^2 = d_in, and the per-dimension mean, from a few batches."""
    xs = torch.cat(list(stream_batches))
    return float(math.sqrt(d_in / xs.pow(2).sum(-1).mean().item())), xs.mean(0)


# --------------------------------------------------------------------------- training


@dataclass
class SAETrainConfig:
    total_tokens: int = 8_000_000
    batch_size: int = 4096
    lr: float = 5e-4
    warmup_steps: int = 100
    decay_frac: float = 0.2  # linear decay to 0 over the last 20% of steps
    k_aux: int = 64
    aux_coef: float = 1 / 32
    dead_tokens: int = 1_000_000  # a feature not fired for this many tokens counts as dead
    buffer_tokens: int = 1 << 18
    seed: int = 0
    threads: int = 0
    log_every: int = 50
    sae: SAEConfig = field(default_factory=SAEConfig)


def lr_at(step: int, n_steps: int, cfg: SAETrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    decay_start = int(n_steps * (1 - cfg.decay_frac))
    if step < decay_start:
        return cfg.lr
    return cfg.lr * max(0.0, (n_steps - step) / max(1, n_steps - decay_start))


def sae_losses(
    sae: TopKSAE, x: torch.Tensor, dead_mask: torch.Tensor, k_aux: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (normalized MSE, AuxK loss, codes). Both losses are in the scaled space and
    normalized by the batch variance, so they are fractions of unexplained variance."""
    xs = x * sae.cfg.scale
    pre = sae.encode_pre(x)
    z = sae.topk_codes(pre)
    xs_hat = z @ sae.W_dec + sae.b_dec
    err = xs - xs_hat
    var = (xs - xs.mean(0)).pow(2).sum(-1).mean()
    mse = err.pow(2).sum(-1).mean() / var

    n_dead = int(dead_mask.sum())
    if n_dead == 0 or k_aux == 0:
        return mse, torch.zeros(()), z
    pre_dead = pre.masked_fill(~dead_mask, float("-inf"))
    z_aux = sae.topk_codes(pre_dead, k=min(k_aux, n_dead))
    err_hat = z_aux @ sae.W_dec
    aux = (err.detach() - err_hat).pow(2).sum(-1).mean() / err.detach().pow(2).sum(-1).mean().clamp_min(1e-12)
    return mse, aux, z


def train_sae(
    model: GPT, tokens: np.ndarray, cfg: SAETrainConfig, verbose: bool = True, starts: np.ndarray | None = None
) -> tuple[TopKSAE, dict]:
    """Train on residual-stream activations of `model` over windows of `tokens` beginning at
    entries of `starts` (default: random offsets)."""
    if cfg.threads > 0:
        torch.set_num_threads(cfg.threads)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    stream = ActivationStream(model, tokens, cfg.sae.layer, cfg.batch_size, cfg.buffer_tokens, seed=cfg.seed,
                              starts=starts)
    sae, log = fit_sae(iter(stream), cfg, verbose)
    log["windows"] = "random offsets" if starts is None else f"story-aligned ({len(starts):,} possible starts)"
    return sae, log


def fit_sae(batches, cfg: SAETrainConfig, verbose: bool = True) -> tuple[TopKSAE, dict]:
    """Train on any iterator of (batch, d_in) activation tensors."""
    torch.manual_seed(cfg.seed)
    scale, mean = estimate_scale_and_mean((next(batches) for _ in range(8)), cfg.sae.d_in)
    sae_cfg = SAEConfig(**{**cfg.sae.to_dict(), "scale": scale})
    sae = TopKSAE(sae_cfg)
    sae.init_weights(torch.Generator().manual_seed(cfg.seed), mean)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr, betas=(0.9, 0.999))

    n_steps = cfg.total_tokens // cfg.batch_size
    since_fired = torch.zeros(sae_cfg.n_features, dtype=torch.long)
    log = {"config": {**asdict(cfg), "sae": sae_cfg.to_dict()}, "steps": n_steps, "history": []}
    t0 = time.time()
    run_mse = run_aux = 0.0
    for step in range(n_steps):
        x = next(batches)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, n_steps, cfg)
        dead = since_fired >= cfg.dead_tokens
        mse, aux, z = sae_losses(sae, x, dead, cfg.k_aux)
        loss = mse + cfg.aux_coef * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sae.remove_parallel_grad()
        opt.step()
        sae.normalize_decoder()

        fired = (z > 0).any(0)
        since_fired += x.shape[0]
        since_fired[fired] = 0
        run_mse += mse.item()
        run_aux += aux.item()
        if (step + 1) % cfg.log_every == 0 or step + 1 == n_steps:
            n = (step % cfg.log_every) + 1
            entry = {"step": step + 1, "tokens": (step + 1) * cfg.batch_size, "fvu": run_mse / n,
                     "aux": run_aux / n, "dead": int(dead.sum()), "elapsed_s": round(time.time() - t0, 1)}
            log["history"].append(entry)
            if verbose:
                print(f"step {step + 1:5d}/{n_steps} | fvu {entry['fvu']:.4f} | aux {entry['aux']:.3f} "
                      f"| dead {entry['dead']:4d} | {entry['tokens'] / (time.time() - t0):,.0f} tok/s")
            run_mse = run_aux = 0.0
    log["elapsed_s"] = round(time.time() - t0, 1)
    sae.eval()
    return sae, log


def save_sae(sae: TopKSAE, path: str | Path, **extra) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sae": sae.state_dict(), "sae_config": sae.cfg.to_dict(), **extra}, path)


def load_sae(path: str | Path) -> tuple[TopKSAE, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sae = TopKSAE(SAEConfig(**ckpt["sae_config"]))
    sae.load_state_dict(ckpt["sae"])
    sae.eval()
    return sae, ckpt


PILOT_LRS = [2e-4, 5e-4, 1e-3, 2e-3, 4e-3]
PILOT_MAX_EXTENSIONS = 3  # if the best lr is the largest tried, double it (at most this often)


def main() -> None:
    """
    python -m glassbox.sae pilot                      # short lr sweep -> results/sae_lr_pilot.json
    python -m glassbox.sae train --k 16 --out runs/sae_k16   # lr defaults to the pilot's best
    """
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["pilot", "train"])
    p.add_argument("--ckpt", type=Path, default=Path("runs/main/final.pt"))
    p.add_argument("--data-dir", type=Path, default=Path("data/tinystories"))
    p.add_argument("--out", type=Path, help="train: output directory")
    p.add_argument("--pilot-file", type=Path, default=Path("results/sae_lr_pilot.json"))
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--n-features", type=int, default=2048)
    p.add_argument("--layer", type=int, default=2)
    p.add_argument("--total-tokens", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="train: default is the pilot's best lr")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()

    model, _ = load_checkpoint(a.ckpt)
    tokens = load_tokens(a.data_dir / "train.bin")
    eot_id = BPETokenizer.load(a.data_dir / "tokenizer.json").eot_id
    starts = story_starts(tokens, eot_id, model.cfg.n_ctx)
    sae_cfg = SAEConfig(d_in=model.cfg.d_model, n_features=a.n_features, k=a.k, layer=a.layer)

    if a.command == "pilot":
        n_tokens = a.total_tokens or 1_000_000
        runs = []

        def run(lr):
            cfg = SAETrainConfig(total_tokens=n_tokens, lr=lr, seed=a.seed, threads=a.threads,
                                 warmup_steps=20, log_every=61, sae=sae_cfg)
            _, log = train_sae(model, tokens, cfg, verbose=False, starts=starts)
            last = log["history"][-1]
            runs.append({"lr": lr, "final_fvu": last["fvu"], "dead": last["dead"], "seconds": log["elapsed_s"]})
            print(f"lr {lr:g}: final train fvu {last['fvu']:.4f}, dead {last['dead']}", flush=True)

        for lr in PILOT_LRS:
            run(lr)
        extensions = 0
        while min(runs, key=lambda r: r["final_fvu"]) is runs[-1] and extensions < PILOT_MAX_EXTENSIONS:
            run(runs[-1]["lr"] * 2)  # best is at the edge of the grid: look further
            extensions += 1
        best = min(runs, key=lambda r: r["final_fvu"])
        out = {"k": a.k, "n_features": a.n_features, "layer": a.layer, "tokens_per_run": n_tokens,
               "seed": a.seed, "runs": runs, "best_lr": best["lr"],
               "best_is_largest_tried": best is runs[-1]}
        a.pilot_file.parent.mkdir(parents=True, exist_ok=True)
        a.pilot_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"best lr {best['lr']:g} -> {a.pilot_file}")
        return

    if a.out is None:
        p.error("train needs --out")
    lr = a.lr if a.lr is not None else json.loads(a.pilot_file.read_text(encoding="utf-8"))["best_lr"]
    cfg = SAETrainConfig(total_tokens=a.total_tokens or SAETrainConfig.total_tokens, lr=lr,
                         seed=a.seed, threads=a.threads, sae=sae_cfg)
    print(f"training k={a.k}, {a.n_features} features, layer {a.layer}, lr {lr:g}, {cfg.total_tokens:,} tokens")
    sae, log = train_sae(model, tokens, cfg, starts=starts)
    save_sae(sae, a.out / "sae.pt", source_checkpoint=a.ckpt.as_posix())
    (a.out / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"saved {a.out / 'sae.pt'}")


if __name__ == "__main__":
    main()
