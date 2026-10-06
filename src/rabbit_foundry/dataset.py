from __future__ import annotations

import json
from pathlib import Path

import torch

from .split import source_split


def load_episode_manifest(path: str | Path, split: str) -> list[dict]:
    payload = json.loads(Path(path).read_text())
    rows = []
    for row in payload["episodes"]:
        actual = source_split(
            row["source_repository"], row["source_commit"], row["source_path"]
        )
        if actual == split:
            rows.append(row)
    return rows


def episode_tensors(row: dict, device: torch.device | str = "cpu"):
    x = torch.tensor(list(bytes.fromhex(row["prompt_hex"])), dtype=torch.long, device=device)
    y = torch.tensor(list(bytes.fromhex(row["target_hex"])), dtype=torch.long, device=device)
    return x, y
