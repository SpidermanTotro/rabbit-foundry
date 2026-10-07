from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .bunny_lineage import load_jsonl_training


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def message_text(row: dict) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Bunny training row needs a non-empty messages list")
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("message must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("message needs string role and content")
        parts.append(f"<|{role}|>\n{content}\n")
    return "".join(parts)


def bunny_rows_to_episodes(rows: list[dict], source_sha256: str, source_name: str, window: int = 128):
    if window < 2:
        raise ValueError("window must be >= 2")
    episodes = []
    for row_index, row in enumerate(rows):
        stream = message_text(row).encode("utf-8")
        if len(stream) <= window:
            continue
        capture_id = row.get("id") or row.get("capture_id") or f"row-{row_index}"
        axis = row.get("axis")
        skill = f"bunny_{axis}" if isinstance(axis, str) and axis.strip() else "bunny_behavior"
        for start in range(0, len(stream) - window, window):
            chunk = stream[start:start + window + 1]
            if len(chunk) < window + 1:
                continue
            episode_id = hashlib.sha256(
                f"{source_sha256}:{capture_id}:{start}".encode()
            ).hexdigest()
            episodes.append({
                "episode_id": episode_id,
                "skill": skill,
                "bunny_axis": axis if isinstance(axis, str) and axis.strip() else None,
                "lineage": "alpha-space-bunny",
                "source_name": source_name,
                "source_sha256": source_sha256,
                "source_capture_id": str(capture_id),
                "prompt_hex": chunk[:-1].hex(),
                "target_hex": chunk[1:].hex(),
            })
    return episodes


def import_bunny_jsonl(path: str | Path, window: int = 128) -> dict:
    path = Path(path)
    rows = load_jsonl_training(path)
    digest = sha256_file(path)
    episodes = bunny_rows_to_episodes(rows, digest, path.name, window=window)
    return {
        "version": 1,
        "kind": "bunny-lineage-episodes",
        "teacher_free_native_experiment": False,
        "source_path": str(path),
        "source_sha256": digest,
        "source_rows": len(rows),
        "episodes": episodes,
    }


def _ensure_small_dataset_splits(episodes: list[dict]) -> None:
    """Guarantee train+validation without splitting chunks from one capture family."""
    families: dict[str, list[dict]] = {}
    for row in episodes:
        families.setdefault(bunny_family_id(row), []).append(row)
    if len(families) < 2:
        return

    assignments = {family: deterministic_bunny_split(family) for family in families}
    present = set(assignments.values())

    if "train" not in present:
        family = sorted(families)[0]
        assignments[family] = "train"

    present = set(assignments.values())
    if "validation" not in present:
        candidates = [family for family, split in assignments.items() if split == "train"]
        if len(candidates) > 1:
            assignments[sorted(candidates)[-1]] = "validation"
        else:
            candidates = [family for family, split in assignments.items() if split != "train"]
            if candidates:
                assignments[sorted(candidates)[0]] = "validation"

    for family, rows in families.items():
        for row in rows:
            row["split"] = assignments[family]


def write_bunny_manifest(source: str | Path, out: str | Path, window: int = 128) -> dict:
    payload = import_bunny_jsonl(source, window=window)
    capture_ids = {row.get("source_capture_id") for row in payload["episodes"]}
    capture_ids.discard(None)
    if len(capture_ids) < 2:
        raise ValueError("Bunny training requires at least two independent usable captures for train/validation")
    _ensure_small_dataset_splits(payload["episodes"])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def bunny_family_id(row: dict) -> str:
    """Stable family key; legacy manifests fall back to their episode ID."""
    source_sha256 = row.get("source_sha256")
    capture_id = row.get("source_capture_id")
    if isinstance(source_sha256, str) and source_sha256 and isinstance(capture_id, str) and capture_id:
        return hashlib.sha256(f"{source_sha256}:{capture_id}".encode()).hexdigest()
    episode_id = row.get("episode_id")
    if isinstance(episode_id, str) and episode_id:
        return hashlib.sha256(f"legacy:{episode_id}".encode()).hexdigest()
    raise ValueError("Bunny episode is missing family provenance and episode_id")


def deterministic_bunny_split(family_id: str) -> str:
    bucket = int(family_id[:16], 16) % 1000
    if bucket < 800:
        return "train"
    if bucket < 900:
        return "validation"
    return "test"


def load_bunny_episode_manifest(path: str | Path, split: str) -> list[dict]:
    payload = json.loads(Path(path).read_text())
    if payload.get("kind") != "bunny-lineage-episodes":
        raise ValueError("not a Bunny lineage episode manifest")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    return [
        row for row in payload["episodes"]
        if row.get("split", deterministic_bunny_split(bunny_family_id(row))) == split
    ]
