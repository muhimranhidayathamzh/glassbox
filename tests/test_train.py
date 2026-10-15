import json

import numpy as np
import pytest
import torch

from glassbox.model import GPTConfig
from glassbox.train import TrainConfig, load_checkpoint, log_spaced_steps, lr_at, train

VOCAB = 24


def test_log_spaced_steps():
    steps = log_spaced_steps(6000, 24)
    assert steps[0] == 0 and steps[-1] == 6000
    assert steps == sorted(set(steps)) and len(steps) <= 25
    gaps = np.diff(steps[1:])
    assert gaps[-1] > 50 * gaps[0]  # dense early, sparse late
    assert log_spaced_steps(0, 10) == [0]


def test_lr_schedule():
    cfg = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=100)
    lrs = [lr_at(s, cfg) for s in range(101)]
    assert all(a < b for a, b in zip(lrs[:10], lrs[1:10]))
    assert lrs[9] == pytest.approx(1e-3)
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:]))
    assert lrs[100] == pytest.approx(1e-4)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    """Learnable synthetic data: a noisy cycle through the vocabulary."""
    d = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    for name, n in [("train", 40_000), ("val", 4_000)]:
        seq = np.arange(n) % (VOCAB - 1)
        noise = rng.random(n) < 0.1
        seq[noise] = rng.integers(0, VOCAB, noise.sum())
        seq.astype(np.uint16).tofile(d / f"{name}.bin")
    return d


def tiny_cfg(data_dir, out_dir, **kw) -> TrainConfig:
    base = dict(
        data_dir=str(data_dir), out_dir=str(out_dir), batch_size=8, max_steps=60, lr=5e-3,
        min_lr=5e-4, warmup_steps=5, eval_interval=20, eval_windows=16, log_interval=10,
        n_replay_checkpoints=6, seed=0,
        model=GPTConfig(vocab_size=VOCAB, n_ctx=32, d_model=32, n_layer=2, n_head=2, d_mlp=64),
    )
    base.update(kw)
    return TrainConfig(**base)


def state(path):
    return load_checkpoint(path)[0].state_dict()


def assert_same_weights(a, b):
    assert a.keys() == b.keys()
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0, msg=k)


def test_training_reduces_loss_and_writes_checkpoints(data_dir, tmp_path):
    cfg = tiny_cfg(data_dir, tmp_path / "run")
    log = train(cfg, verbose=False)

    val = {e["step"]: e["loss"] for e in log["val"]}
    assert val[0] == pytest.approx(np.log(VOCAB), abs=0.15)
    assert val[60] < 0.6 * val[0]

    run = tmp_path / "run"
    for s in log_spaced_steps(60, 6):
        assert (run / f"ckpt_step{s:06d}.pt").exists()
    assert_same_weights(state(run / "final.pt"), state(run / "ckpt_step000060.pt"))
    saved = json.loads((run / "train_log.json").read_text())
    assert saved["replay_steps"] == log_spaced_steps(60, 6)
    assert saved["train"][-1]["step"] == 60

    model, ckpt = load_checkpoint(run / "final.pt")
    assert ckpt["step"] == 60 and ckpt["val_loss"] == pytest.approx(val[60])


def test_training_is_deterministic(data_dir, tmp_path):
    train(tiny_cfg(data_dir, tmp_path / "a", max_steps=20), verbose=False)
    train(tiny_cfg(data_dir, tmp_path / "b", max_steps=20), verbose=False)
    assert_same_weights(state(tmp_path / "a" / "final.pt"), state(tmp_path / "b" / "final.pt"))


def test_resume_matches_uninterrupted_run(data_dir, tmp_path):
    train(tiny_cfg(data_dir, tmp_path / "full", max_steps=40), verbose=False)

    cfg = tiny_cfg(data_dir, tmp_path / "resumed", max_steps=40)
    train(cfg, stop_at=20, verbose=False)
    assert not (tmp_path / "resumed" / "final.pt").exists()
    train(cfg, resume=True, verbose=False)

    assert_same_weights(
        state(tmp_path / "full" / "final.pt"), state(tmp_path / "resumed" / "final.pt")
    )
