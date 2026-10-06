from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    context: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.0

    def validate(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if min(self.vocab_size, self.context, self.d_model, self.n_heads, self.n_layers, self.d_ff) <= 0:
            raise ValueError("model dimensions must be positive")


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        length = x.size(1)
        mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1
        )
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.ff(self.ln2(x))


class TinyRabbitLM(nn.Module):
    """Small causal LM used to prove the Foundry loop before scaling."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.token = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position = nn.Embedding(cfg.context, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.token.weight

    def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None):
        if ids.ndim != 2:
            raise ValueError("ids must have shape [batch, sequence]")
        if ids.size(1) > self.cfg.context:
            raise ValueError("sequence exceeds configured context")
        pos = torch.arange(ids.size(1), device=ids.device)
        x = self.token(ids) + self.position(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
