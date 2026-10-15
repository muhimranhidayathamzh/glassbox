"""Training loop with log-spaced checkpoints (for the training-replay milestone).

    python -m glassbox.train --out-dir runs/main --max-steps 6000

Files written to out_dir:
    ckpt_step{N:06d}.pt   model weights after N optimizer steps, N log-spaced in [0, max_steps]
    final.pt              same as the last replay checkpoint
    latest.pt             model + optimizer + RNG state, for --resume
    train_log.json        config, train-loss history, val-loss evaluations
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

from .model import GPT, GPTConfig


@dataclass
class TrainConfig:
    data_dir: str = "data/tinystories"
    out_dir: str = "runs/main"
    batch_size: int = 16
    max_steps: int = 6000
    lr: float = 2e-3
    min_lr: float = 2e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_windows: int = 256
    log_interval: int = 25
    n_replay_checkpoints: int = 24
    seed: int = 1337
    threads: int = 0  # 0 = torch default
    model: GPTConfig = field(default_factory=GPTConfig)


def log_spaced_steps(max_steps: int, n: int) -> list[int]:
    """Step 0 plus ~n-1 roughly log-spaced steps in [1, max_steps], always including max_steps."""
    if max_steps < 1:
        return [0]
    steps = np.unique(np.round(np.geomspace(1, max_steps, max(n - 1, 1))).astype(int))
    return sorted({0, max_steps, *steps.tolist()})


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay from lr to min_lr at max_steps."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


def load_tokens(path: str | Path) -> np.ndarray:
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.ndarray, starts: np.ndarray, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    chunks = np.stack([data[s : s + seq_len + 1] for s in starts]).astype(np.int64)
    t = torch.from_numpy(chunks)
    return t[:, :-1], t[:, 1:]


def eval_starts(data: np.ndarray, n: int, seq_len: int) -> np.ndarray:
    """Fixed, evenly spaced validation windows, so val loss is comparable across steps."""
    return np.linspace(0, len(data) - seq_len - 1, n).astype(np.int64)


@torch.no_grad()
def evaluate(model: GPT, data: np.ndarray, starts: np.ndarray, batch_size: int) -> float:
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(starts), batch_size):
        x, y = get_batch(data, starts[i : i + batch_size], model.cfg.n_ctx)
        _, loss = model(x, y)
        total += loss.item() * x.shape[0]
        count += x.shape[0]
    model.train(was_training)
    return total / count


def make_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas)


def save_checkpoint(path: Path, model: GPT, step: int, **extra) -> None:
    torch.save(
        {"model": model.state_dict(), "model_config": model.cfg.to_dict(), "step": step, **extra},
        path,
    )


def load_checkpoint(path: str | Path) -> tuple[GPT, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def _config_dict(cfg: TrainConfig) -> dict:
    d = asdict(cfg)
    d["betas"] = list(cfg.betas)
    return d


def train(
    cfg: TrainConfig, resume: bool = False, stop_at: int | None = None, verbose: bool = True
) -> dict:
    """Train (or resume) a model. `stop_at` ends the run early after that step's checkpoint
    actions; use a multiple of eval_interval so latest.pt exists for a later --resume."""
    if cfg.threads > 0:
        torch.set_num_threads(cfg.threads)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg.data_dir)
    train_data = load_tokens(data_dir / "train.bin")
    val_data = load_tokens(data_dir / "val.bin")
    T = cfg.model.n_ctx

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = GPT(cfg.model)
    model.train()
    opt = make_optimizer(model, cfg)
    val_starts = eval_starts(val_data, cfg.eval_windows, T)
    replay_steps = set(log_spaced_steps(cfg.max_steps, cfg.n_replay_checkpoints))

    log = {
        "config": _config_dict(cfg),
        "n_params": model.num_params(),
        "n_params_non_embedding": model.num_params(include_embeddings=False),
        "replay_steps": sorted(replay_steps),
        "train": [],  # {step, loss, lr, tokens, elapsed_s}
        "val": [],  # {step, loss, tokens, elapsed_s}
    }
    start_step, elapsed_before = 0, 0.0
    if resume and (out / "latest.pt").exists():
        state = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["rng_state"]
        log = state["log"]
        start_step, elapsed_before = state["step"], state["elapsed_s"]
        if verbose:
            print(f"resumed from step {start_step}")

    tokens_per_step = cfg.batch_size * T
    t_start = time.time()
    running, n_running = 0.0, 0

    def elapsed() -> float:
        return elapsed_before + time.time() - t_start

    for step in range(start_step, cfg.max_steps + 1):
        # Checkpoint / eval actions describe the model *after* `step` updates. On resume they
        # already happened for start_step.
        if not (resume and step == start_step and start_step > 0):
            is_eval = step % cfg.eval_interval == 0 or step == cfg.max_steps
            val_loss = None
            if is_eval or step in replay_steps:
                val_loss = evaluate(model, val_data, val_starts, cfg.batch_size)
                log["val"].append(
                    {"step": step, "loss": val_loss, "tokens": step * tokens_per_step,
                     "elapsed_s": round(elapsed(), 1)}
                )
                if verbose:
                    print(f"step {step:6d} | val loss {val_loss:.4f} | {elapsed() / 60:.1f} min")
            if step in replay_steps:
                save_checkpoint(out / f"ckpt_step{step:06d}.pt", model, step, val_loss=val_loss)
            if is_eval:
                torch.save(
                    {"model": model.state_dict(), "optimizer": opt.state_dict(),
                     "rng_state": rng.bit_generator.state, "log": log, "step": step,
                     "elapsed_s": elapsed()},
                    out / "latest.pt",
                )
                (out / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
        if step == cfg.max_steps:
            break
        if stop_at is not None and step == stop_at:
            return log

        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        starts = rng.integers(0, len(train_data) - T - 1, size=cfg.batch_size)
        x, y = get_batch(train_data, starts, T)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        running += loss.item()
        n_running += 1
        if (step + 1) % cfg.log_interval == 0:
            log["train"].append(
                {"step": step + 1, "loss": running / n_running, "lr": lr,
                 "tokens": (step + 1) * tokens_per_step, "elapsed_s": round(elapsed(), 1)}
            )
            if verbose:
                tok_s = (step + 1 - start_step) * tokens_per_step / (time.time() - t_start)
                print(f"step {step + 1:6d} | loss {running / n_running:.4f} | lr {lr:.2e} | {tok_s:,.0f} tok/s")
            running, n_running = 0.0, 0

    final_step = max(replay_steps)
    (out / "final.pt").write_bytes((out / f"ckpt_step{final_step:06d}.pt").read_bytes())
    log["total_elapsed_s"] = round(elapsed(), 1)
    (out / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    return log


def main() -> None:
    p = argparse.ArgumentParser()
    defaults = TrainConfig()
    for f in ["data_dir", "out_dir"]:
        p.add_argument("--" + f.replace("_", "-"), default=getattr(defaults, f))
    for f in ["batch_size", "max_steps", "warmup_steps", "eval_interval", "eval_windows",
              "log_interval", "n_replay_checkpoints", "seed", "threads"]:
        p.add_argument("--" + f.replace("_", "-"), type=int, default=getattr(defaults, f))
    for f in ["lr", "min_lr", "weight_decay", "grad_clip"]:
        p.add_argument("--" + f.replace("_", "-"), type=float, default=getattr(defaults, f))
    mdef = GPTConfig()
    for f in ["n_ctx", "d_model", "n_layer", "n_head", "d_mlp"]:
        p.add_argument("--" + f.replace("_", "-"), type=int, default=getattr(mdef, f))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-at", type=int, default=None)
    a = p.parse_args()

    tok_meta = json.loads((Path(a.data_dir) / "tokenizer.json").read_text(encoding="utf-8"))
    model_cfg = GPTConfig(vocab_size=tok_meta["vocab_size"], n_ctx=a.n_ctx, d_model=a.d_model,
                          n_layer=a.n_layer, n_head=a.n_head, d_mlp=a.d_mlp)
    cfg = TrainConfig(**{k: getattr(a, k) for k in vars(a) if k in TrainConfig.__dataclass_fields__},
                      model=model_cfg)
    train(cfg, resume=a.resume, stop_at=a.stop_at)


if __name__ == "__main__":
    main()
