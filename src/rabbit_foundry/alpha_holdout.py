from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .bunny_import import bunny_rows_to_episodes
from .bunny_lineage import ALPHA_HOLDOUT_IDS, normalize_capture_id, row_capture_id


def load_alpha_holdout(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = {normalize_capture_id(x) for x in filter(None, (row_capture_id(r) for r in rows))}
    expected = set(ALPHA_HOLDOUT_IDS)
    if ids != expected:
        missing = sorted(expected - ids)
        extra = sorted(ids - expected)
        raise ValueError(f"Alpha holdout mismatch: missing={missing}, extra={extra}")
    return rows


def build_alpha_holdout_manifest(path: str | Path, window: int = 128) -> dict:
    path = Path(path)
    rows = load_alpha_holdout(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    episodes = bunny_rows_to_episodes(rows, digest, path.name, window=window)
    for episode in episodes:
        episode["evaluation_only"] = True
        episode["split"] = "alpha_frozen_holdout"
    return {
        "version": 1,
        "kind": "alpha-frozen-holdout",
        "training_allowed": False,
        "source_path": str(path),
        "source_sha256": digest,
        "source_rows": len(rows),
        "expected_ids": sorted(ALPHA_HOLDOUT_IDS),
        "episodes": episodes,
    }


def write_alpha_holdout_manifest(source: str | Path, out: str | Path, window: int = 128) -> dict:
    payload = build_alpha_holdout_manifest(source, window=window)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
