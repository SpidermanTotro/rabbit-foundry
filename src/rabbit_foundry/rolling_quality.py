from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class QualityCase:
    case_id: str
    axis: str
    difficulty: float
    source: str
    generation: int


def stable_case_id(row: dict) -> str:
    raw = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def build_rolling_challenge(
    candidate_rows: list[dict],
    out: str | Path,
    *,
    generation: int,
    max_cases: int = 24,
    previous_ids: set[str] | None = None,
) -> dict:
    previous_ids = previous_ids or set()
    scored = []
    for row in candidate_rows:
        if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
            continue
        case_id = str(row.get("id") or stable_case_id(row))
        if case_id in previous_ids:
            continue
        quality = row.get("quality_score", 0.0)
        difficulty = row.get("difficulty", quality)
        try:
            difficulty = float(difficulty)
        except (TypeError, ValueError):
            difficulty = 0.0
        scored.append((difficulty, case_id, row))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = []
    axes: dict[str, int] = {}
    for difficulty, case_id, row in scored[:max_cases]:
        axis = str(row.get("axis") or "behavior")
        selected.append({
            "id": case_id,
            "axis": axis,
            "difficulty": difficulty,
            "generation": generation,
            "messages": row["messages"],
            "evaluation_only": True,
            "challenge_kind": "rolling",
        })
        axes[axis] = axes.get(axis, 0) + 1

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in selected))
    return {
        "version": 1,
        "kind": "rolling-behavior-challenge",
        "generation": generation,
        "cases": len(selected),
        "axes": dict(sorted(axes.items())),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "out": str(out),
    }


def quality_gate(anchor_score: float, rolling_score: float, previous_rolling_score: float | None = None) -> dict:
    if not all(0.0 <= value <= 1.0 for value in (anchor_score, rolling_score)):
        raise ValueError("quality scores must be between 0 and 1")
    no_anchor_regression = anchor_score >= 0.0
    rolling_improved = previous_rolling_score is None or rolling_score >= previous_rolling_score
    return {
        "anchor_score": anchor_score,
        "rolling_score": rolling_score,
        "previous_rolling_score": previous_rolling_score,
        "no_anchor_regression": no_anchor_regression,
        "rolling_improved": rolling_improved,
        "quality_pass": no_anchor_regression and rolling_improved,
    }
