from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .model import TinyRabbitLM


def _prompt_text(row: dict) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("behavior case needs messages")
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("behavior message must be an object")
        role, content = message.get("role"), message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("behavior message needs string role/content")
        parts.append(f"<|{role}|>\n{content}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


@torch.no_grad()
def generate_bytes(
    model: TinyRabbitLM,
    prompt: str,
    *,
    device: torch.device,
    max_new_bytes: int = 128,
) -> str:
    if max_new_bytes <= 0:
        raise ValueError("max_new_bytes must be positive")
    raw = prompt.encode("utf-8")
    ids = list(raw[-model.cfg.context:])
    generated: list[int] = []
    model.eval()
    for _ in range(max_new_bytes):
        x = torch.tensor([ids[-model.cfg.context:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        token = int(torch.argmax(logits[0, -1]).item())
        generated.append(token)
        ids.append(token)
    model.train()
    return bytes(generated).decode("utf-8", errors="replace")


def score_behavior_response(text: str) -> dict:
    if not isinstance(text, str):
        raise ValueError("response must be text")
    lowered = text.lower()
    signals = {
        "failure_detection": any(word in lowered for word in ("fail", "error", "broken", "incorrect")),
        "diagnosis": any(word in lowered for word in ("because", "cause", "reason", "diagnos")),
        "revision": any(word in lowered for word in ("fix", "repair", "revise", "change")),
        "testing": any(word in lowered for word in ("test", "verify", "check", "pytest")),
    }
    score = (
        0.20 * signals["failure_detection"]
        + 0.30 * signals["diagnosis"]
        + 0.30 * signals["revision"]
        + 0.20 * signals["testing"]
    )
    return {**signals, "behavior_score": float(score)}


def evaluate_behavior_cases(
    model: TinyRabbitLM,
    cases: list[dict],
    *,
    device: torch.device,
    max_new_bytes: int = 128,
) -> dict:
    if not cases:
        raise ValueError("behavior evaluation requires at least one case")
    rows = []
    for case in cases:
        prompt = _prompt_text(case)
        response = generate_bytes(model, prompt, device=device, max_new_bytes=max_new_bytes)
        scored = score_behavior_response(response)
        rows.append({
            "id": str(case.get("id") or "unknown"),
            "axis": str(case.get("axis") or "behavior"),
            "response": response,
            **scored,
        })
    overall = sum(row["behavior_score"] for row in rows) / len(rows)
    if not math.isfinite(overall):
        raise ValueError("non-finite behavior score")
    return {"behavior_score": overall, "cases": rows}


def load_behavior_cases(path: str | Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("behavior challenge file is empty")
    return rows
