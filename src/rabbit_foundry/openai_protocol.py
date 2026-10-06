from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ServedModel:
    id: str
    context: int
    output: int
    tools: bool = False


def models_response(models: list[ServedModel]) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": model.id,
                "object": "model",
                "owned_by": "rabbit-foundry",
            }
            for model in models
        ],
    }


def last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
    return ""
