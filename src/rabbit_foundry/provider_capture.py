from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


SECRET_KEYS = {
    "authorization", "api_key", "apikey", "api-key", "token", "access_token",
    "openrouter_api_key",
}

SECRET_MARKERS = (
    "sk-", "Bearer ", "OPENROUTER_API_KEY=", "API_KEY=", "ACCESS_TOKEN=",
)


def redact_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    words = text.split()
    redacted = []
    hide_next = False
    for word in words:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        lower = word.lower()
        if lower == "authorization:":
            redacted.append(word)
            continue
        if lower in {"bearer", "token:", "api_key:", "api-key:"}:
            redacted.append(word)
            hide_next = True
            continue
        if any(marker.lower() in lower for marker in SECRET_MARKERS):
            if "=" in word:
                key = word.split("=", 1)[0]
                redacted.append(f"{key}=<redacted>")
            else:
                redacted.append("<redacted>")
            continue
        redacted.append(word)
    return " ".join(redacted)


def _redact(value):
    if isinstance(value, dict):
        return {
            key: ("<redacted>" if key.lower() in SECRET_KEYS else _redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _messages(value) -> list[dict] | None:
    if not isinstance(value, list) or not value:
        return None
    rows = []
    for item in value:
        if not isinstance(item, dict):
            return None
        role, content = item.get("role"), item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return None
        rows.append({"role": role, "content": redact_text(content)})
    return rows


def normalize_provider_trace(raw: dict, *, source: str, index: int) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("provider trace must be an object")

    provider = str(raw.get("provider") or raw.get("source") or "unknown").lower()
    model = raw.get("model") or raw.get("model_id")
    request = raw.get("request") if isinstance(raw.get("request"), dict) else raw
    response = raw.get("response")

    messages = _messages(request.get("messages"))
    if messages is None:
        messages = _messages(raw.get("messages"))
    if messages is None:
        raise ValueError("provider trace has no OpenAI-style messages")

    assistant_text = None
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    assistant_text = redact_text(message["content"])
    if assistant_text is None and isinstance(raw.get("output"), str):
        assistant_text = redact_text(raw["output"])
    if assistant_text is not None:
        if not messages or messages[-1].get("role") != "assistant" or messages[-1].get("content") != assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})

    safe = _redact(raw)
    trace_digest = hashlib.sha256(
        json.dumps(safe, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "id": str(raw.get("id") or raw.get("trace_id") or f"{provider}-{trace_digest[:20]}"),
        "axis": raw.get("axis") or raw.get("skill") or "behavior",
        "messages": messages,
        "provider": provider,
        "model": str(model) if model is not None else None,
        "trace_sha256": trace_digest,
        "source_trace_file": source,
        "source_trace_index": index,
        "capture_kind": "provider-preview",
    }


def load_provider_export(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("traces", "requests", "sessions", "captures"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    raise ValueError(f"unsupported provider export: {path}")


def import_provider_exports(paths: Iterable[str | Path], out: str | Path) -> dict:
    captures, rejected = [], []
    providers: dict[str, int] = {}
    models: dict[str, int] = {}
    for raw_path in paths:
        path = Path(raw_path)
        for index, raw in enumerate(load_provider_export(path)):
            try:
                row = normalize_provider_trace(raw, source=path.name, index=index)
                captures.append(row)
                providers[row["provider"]] = providers.get(row["provider"], 0) + 1
                if row["model"]:
                    models[row["model"]] = models.get(row["model"], 0) + 1
            except (ValueError, TypeError) as exc:
                rejected.append({"source": str(path), "index": index, "reason": str(exc)})

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in captures))
    return {
        "version": 1,
        "kind": "provider-preview-captures",
        "captures": len(captures),
        "rejected": len(rejected),
        "providers": dict(sorted(providers.items())),
        "models": dict(sorted(models.items())),
        "out": str(out),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "rejections": rejected,
    }
