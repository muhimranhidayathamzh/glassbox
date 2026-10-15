"""Decoder-only transformer (GPT-2 style, pre-LayerNorm), written with torch.nn primitives only.

Every operation is spelled out (attention, LayerNorm, GELU) so that the JavaScript engine can
reproduce it exactly:

    x = wte[idx] + wpe[pos]
    for each block:  x = x + attn(ln1(x));  x = x + mlp(ln2(x))
    logits = ln_f(x) @ wte.T        (tied input/output embedding)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4000
    n_ctx: int = 256
    d_model: int = 192
    n_layer: int = 4
    n_head: int = 4
    d_mlp: int = 768
    ln_eps: float = 1e-5

    def to_dict(self) -> dict:
        return asdict(self)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU, tanh approximation (as in GPT-2)."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class LayerNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        mask = torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        # (B, T, C) -> (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        pattern = torch.softmax(scores, dim=-1)
        y = pattern @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.proj = nn.Linear(cfg.d_mlp, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.d_model, cfg.ln_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.d_model, cfg.ln_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = LayerNorm(cfg.d_model, cfg.ln_eps)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if ".ln" in name or name.startswith("ln_f"):
                continue  # LayerNorm keeps weight=1, bias=0
            if name.endswith(".bias"):
                nn.init.zeros_(p)
            elif name.endswith("proj.weight"):  # residual-stream writers, scaled as in GPT-2
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layer))
            else:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def num_params(self, include_embeddings: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if not include_embeddings:
            n -= self.wte.weight.numel() + self.wpe.weight.numel()
        return n

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        if T > self.cfg.n_ctx:
            raise ValueError(f"sequence length {T} exceeds context {self.cfg.n_ctx}")
        return self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        resid_hooks: dict[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """resid_hooks maps a layer l in [0, n_layer] to a function applied to the residual
        stream entering block l (l == n_layer: the stream entering ln_f). The function returns
        the (possibly modified) residual; this is how activations are read, replaced or steered."""
        hooks = resid_hooks or {}
        x = self.embed(idx)
        for l, block in enumerate(self.blocks):
            if l in hooks:
                x = hooks[l](x)
            x = block(x)
        if self.cfg.n_layer in hooks:
            x = hooks[self.cfg.n_layer](x)
        x = self.ln_f(x)
        logits = x @ self.wte.weight.T
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def resid_at(self, idx: torch.Tensor, layer: int) -> torch.Tensor:
        """Residual stream entering block `layer`, computing only the blocks before it."""
        if not 0 <= layer <= self.cfg.n_layer:
            raise ValueError(f"layer must be in [0, {self.cfg.n_layer}]")
        x = self.embed(idx)
        for block in self.blocks[:layer]:
            x = block(x)
        return x

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        stop_token: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling for a single sequence (idx shape (1, T)).
        temperature == 0 means greedy decoding.

        When the sequence outgrows the context, the window restarts from the last n_ctx // 2
        tokens. The web engine uses the same rule, so its KV cache is rebuilt only once every
        n_ctx // 2 tokens and both produce identical text."""
        start = 0
        for _ in range(max_new_tokens):
            if idx.shape[1] - start > self.cfg.n_ctx:
                start = idx.shape[1] - self.cfg.n_ctx // 2
            logits, _ = self(idx[:, start:])
            logits = logits[:, -1, :]
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_token is not None and nxt.item() == stop_token:
                break
        return idx
