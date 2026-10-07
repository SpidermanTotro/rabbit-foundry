from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from .provider_capture import SECRET_KEYS, redact_text


def _safe(value):
    if isinstance(value, dict):
        return {
            key: ("<redacted>" if key.lower() in SECRET_KEYS else _safe(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


class SessionStore:
    def __init__(
        self,
        directory: str | Path,
        *,
        session_id: str | None = None,
        training_allowed: bool = False,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or uuid.uuid4().hex
        if not self.session_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("session_id contains unsafe characters")
        self.training_allowed = bool(training_allowed)
        self.path = self.directory / f"{self.session_id}.jsonl"
        self.resumed = self.path.exists()

    def record(self, kind: str, payload: dict) -> dict:
        if not kind:
            raise ValueError("event kind must not be empty")
        row = {
            "schema_version": 1,
            "session_id": self.session_id,
            "event_id": uuid.uuid4().hex,
            "timestamp": time.time(),
            "kind": kind,
            "source": "rabbit-code-runtime",
            "training_allowed": self.training_allowed,
            "payload": _safe(payload),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        return row

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for number, line in enumerate(self.path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid session JSON on line {number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"invalid session event on line {number}")
            rows.append(row)
        return rows

    def chat_history(self) -> list[dict]:
        history: list[dict] = []
        for row in self.events():
            kind = row.get("kind")
            if kind not in {"user", "assistant"}:
                continue
            payload = row.get("payload")
            if not isinstance(payload, dict):
                continue
            if payload.get("mode") == "agent":
                continue
            content = payload.get("content")
            if isinstance(content, str):
                history.append({"role": kind, "content": content})
        return history
