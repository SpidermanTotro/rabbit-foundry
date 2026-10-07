from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .environment import detect_environment
from .provider_capture import redact_text


ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


def normalize_alpha_message(message: dict) -> dict:
    if not isinstance(message, dict):
        raise ValueError("message must be an object")
    role = message.get("role")
    content = message.get("content")
    if role not in ALLOWED_ROLES or not isinstance(content, str):
        raise ValueError("message needs a supported role and string content")
    row = {"role": role, "content": redact_text(content)}
    if isinstance(message.get("name"), str):
        row["name"] = message["name"]
    if isinstance(message.get("tool_call_id"), str):
        row["tool_call_id"] = message["tool_call_id"]
    return row


def alpha_session_to_capture(session: dict, *, environment: dict | None = None) -> dict:
    messages = session.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Alpha Preview session needs messages")
    safe_messages = [normalize_alpha_message(m) for m in messages]
    canonical = json.dumps(safe_messages, sort_keys=True, separators=(",", ":"))
    capture_id = session.get("id")
    if not isinstance(capture_id, str) or not capture_id:
        capture_id = "alpha-preview-" + hashlib.sha256(canonical.encode()).hexdigest()[:20]
    metadata = {
        "source_kind": "observable_model_session",
        "provider": str(session.get("provider") or "alpha-preview"),
        "model": str(session.get("model") or "alpha-preview"),
        "weight_access": "unavailable",
        "environment": environment if environment is not None else detect_environment(),
    }
    for key in ("agent", "client", "outcome", "project"):
        if isinstance(session.get(key), str):
            metadata[key] = redact_text(session[key])
    return {
        "id": capture_id,
        "axis": str(session.get("axis") or "tool"),
        "messages": safe_messages,
        **metadata,
    }


def convert_alpha_export(source: str | Path, out: str | Path) -> dict:
    source, out = Path(source), Path(out)
    rows = [
        json.loads(line) for line in source.read_text().splitlines() if line.strip()
    ]
    environment = detect_environment()
    captures = [alpha_session_to_capture(row, environment=environment) for row in rows]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in captures))
    return {"source_rows": len(rows), "captures": len(captures), "output": str(out)}
