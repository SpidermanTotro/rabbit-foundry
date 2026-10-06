from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


ALPHA_HOLDOUT_IDS = frozenset({
    "011-partial-failure",
    "014-instruction-conflict",
    "016-guess-discipline",
    "028-ambiguous-request",
    "031-test-first-request",
    "034-anti-sycophancy",
})


def normalize_capture_id(value: str) -> str:
    value = value.strip()
    return value[:-5] if value.endswith(".json") else value


def row_capture_id(row: dict) -> str | None:
    for key in ("id", "capture_id", "source_id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_capture_id(value)
    return None


def assert_no_alpha_holdout(rows: Iterable[dict]) -> None:
    leaked = sorted({
        capture_id
        for row in rows
        if (capture_id := row_capture_id(row)) in ALPHA_HOLDOUT_IDS
    })
    if leaked:
        raise ValueError(
            "Alpha holdout leakage detected; refusing training import: "
            + ", ".join(leaked)
        )


def load_jsonl_training(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    assert_no_alpha_holdout(rows)
    return rows
