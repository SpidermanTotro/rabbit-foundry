from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .provider_capture import redact_text


EVENT_KINDS = {
    "user", "assistant", "tool_call", "tool_result", "failure",
    "diagnosis", "revision", "test", "final",
}


def _text(value) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "output", "message", "error"):
            if isinstance(value.get(key), str) and value[key].strip():
                return value[key]
    return None


def normalize_trajectory(raw: dict, *, source: str, index: int) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("trajectory must be an object")
    events = raw.get("events") or raw.get("trajectory") or raw.get("steps")
    if not isinstance(events, list) or not events:
        raise ValueError("trajectory has no events")

    normalized = []
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"event {event_index} must be an object")
        kind = str(event.get("kind") or event.get("type") or event.get("role") or "").lower()
        aliases = {
            "human": "user", "ai": "assistant", "tool": "tool_result",
            "error": "failure", "debug": "diagnosis", "retry": "revision",
        }
        kind = aliases.get(kind, kind)
        if kind not in EVENT_KINDS:
            raise ValueError(f"event {event_index} has unsupported kind: {kind}")
        text = _text(event)
        if text is None:
            raise ValueError(f"event {event_index} has no text")
        normalized.append({"kind": kind, "text": redact_text(text)})

    kinds = {event["kind"] for event in normalized}
    if "user" not in kinds:
        raise ValueError("trajectory needs a user/task event")
    if not ({"assistant", "revision", "final"} & kinds):
        raise ValueError("trajectory needs an assistant/revision/final event")

    provider = str(raw.get("provider") or raw.get("source") or "unknown").lower()
    model = raw.get("model") or raw.get("model_id")
    digest_input = json.dumps(
        {"provider": provider, "model": model, "events": normalized},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(digest_input).hexdigest()

    messages = []
    for event in normalized:
        kind, text = event["kind"], event["text"]
        if kind == "user":
            messages.append({"role": "user", "content": text})
        elif kind in {"assistant", "revision", "final"}:
            messages.append({"role": "assistant", "content": f"[{kind}]\n{text}"})
        else:
            messages.append({"role": "tool", "content": f"[{kind}]\n{text}"})

    return {
        "id": str(raw.get("id") or raw.get("trace_id") or f"trajectory-{digest[:20]}"),
        "axis": raw.get("axis") or raw.get("skill") or "selfcorr",
        "messages": messages,
        "trajectory_events": normalized,
        "provider": provider,
        "model": str(model) if model is not None else None,
        "trace_sha256": digest,
        "source_trace_file": source,
        "source_trace_index": index,
        "capture_kind": "provider-trajectory",
        "has_failure": "failure" in kinds,
        "has_diagnosis": "diagnosis" in kinds,
        "has_revision": "revision" in kinds,
        "has_test": "test" in kinds,
    }


def import_trajectories(rows: Iterable[dict], out: str | Path, *, source: str = "live-preview") -> dict:
    captures, rejected = [], []
    for index, raw in enumerate(rows):
        try:
            captures.append(normalize_trajectory(raw, source=source, index=index))
        except (ValueError, TypeError) as exc:
            rejected.append({"index": index, "reason": str(exc)})

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in captures))
    return {
        "version": 1,
        "kind": "provider-trajectories",
        "captures": len(captures),
        "rejected": len(rejected),
        "failure_trajectories": sum(row["has_failure"] for row in captures),
        "repaired_trajectories": sum(row["has_revision"] for row in captures),
        "tested_trajectories": sum(row["has_test"] for row in captures),
        "out": str(out),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "rejections": rejected,
    }
