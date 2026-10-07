from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .bunny_lineage import ALPHA_HOLDOUT_IDS, assert_no_alpha_holdout, row_capture_id


AXIS_ALIASES = {
    "coding": "code",
    "code": "code",
    "debugging": "debug",
    "debug": "debug",
    "self-correction": "selfcorr",
    "self_correction": "selfcorr",
    "selfcorr": "selfcorr",
    "tool-use": "tool",
    "tool_use": "tool",
    "tool": "tool",
    "persona": "persona",
}


@dataclass(frozen=True)
class CaptureAudit:
    captured: int
    eligible_training: int
    frozen_holdout: int
    rejected: int
    axes: dict[str, int]


def normalize_axis(value) -> str:
    if not isinstance(value, str) or not value.strip():
        return "behavior"
    key = value.strip().lower()
    return AXIS_ALIASES.get(key, key.replace(" ", "_"))


def normalize_message(message: dict) -> dict:
    if not isinstance(message, dict):
        raise ValueError("capture message must be an object")
    role = message.get("role")
    content = message.get("content")
    if not isinstance(role, str) or not role.strip():
        raise ValueError("capture message needs a role")
    if not isinstance(content, str):
        raise ValueError("capture message needs string content")
    return {"role": role.strip(), "content": content}


def normalize_capture(row: dict, index: int, source_name: str) -> dict:
    if not isinstance(row, dict):
        raise ValueError("capture must be an object")
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("capture needs a non-empty messages list")
    capture_id = row_capture_id(row) or f"{source_name}-row-{index:06d}"
    axis = normalize_axis(row.get("axis"))
    return {
        "id": capture_id,
        "axis": axis,
        "messages": [normalize_message(message) for message in messages],
        "source_capture_file": source_name,
        "source_metadata": {
            key: value for key, value in row.items()
            if key not in {"id", "capture_id", "source_id", "axis", "messages"}
        },
    }


def load_capture_file(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("captures"), list):
        return payload["captures"]
    if isinstance(payload, dict) and isinstance(payload.get("messages"), list):
        return [payload]
    raise ValueError(f"unsupported capture format: {path}")


def capture_course(paths: list[str | Path], out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    training, holdout, rejected = [], [], []
    seen = set()
    axis_counts: dict[str, int] = {}

    for raw_path in paths:
        path = Path(raw_path)
        for index, raw in enumerate(load_capture_file(path)):
            try:
                row = normalize_capture(raw, index, path.name)
                capture_id = row["id"]
                if capture_id in seen:
                    raise ValueError(f"duplicate capture id: {capture_id}")
                seen.add(capture_id)
                axis_counts[row["axis"]] = axis_counts.get(row["axis"], 0) + 1
                if capture_id in ALPHA_HOLDOUT_IDS:
                    holdout.append(row)
                else:
                    training.append(row)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                rejected.append({
                    "source": str(path),
                    "index": index,
                    "reason": str(exc),
                })

    assert_no_alpha_holdout(training)
    training_path = out_dir / "alpha_training.jsonl"
    holdout_path = out_dir / "alpha_holdout.jsonl"
    rejected_path = out_dir / "rejected.json"
    training_path.write_text("".join(json.dumps(row) + "\n" for row in training))
    holdout_path.write_text("".join(json.dumps(row) + "\n" for row in holdout))
    rejected_path.write_text(json.dumps(rejected, indent=2) + "\n")

    audit = CaptureAudit(
        captured=len(training) + len(holdout) + len(rejected),
        eligible_training=len(training),
        frozen_holdout=len(holdout),
        rejected=len(rejected),
        axes=dict(sorted(axis_counts.items())),
    )
    manifest = {
        "version": 1,
        "kind": "alpha-bunny-capture-course",
        "sources": [str(Path(p)) for p in paths],
        "training_jsonl": str(training_path),
        "holdout_jsonl": str(holdout_path),
        "rejected_json": str(rejected_path),
        "training_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
        "holdout_sha256": hashlib.sha256(holdout_path.read_bytes()).hexdigest(),
        "audit": audit.__dict__,
        "holdout_ids_expected": sorted(ALPHA_HOLDOUT_IDS),
        "holdout_complete": {row["id"] for row in holdout} == set(ALPHA_HOLDOUT_IDS),
    }
    (out_dir / "capture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
