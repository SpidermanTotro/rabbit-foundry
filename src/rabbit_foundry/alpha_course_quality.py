from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlphaQuality:
    score: float
    signals: tuple[str, ...]


def classify_message(content: str) -> set[str]:
    text = content.lower()
    signals = set()
    groups = {
        "failure": ("failed", "error", "traceback", "exception", "broken"),
        "diagnosis": ("because", "root cause", "caused by", "reason is", "diagnos"),
        "repair": ("fix", "fixed", "patch", "change", "replace", "repair"),
        "test": ("pytest", "test passed", "tests passed", "verify", "verified", "check"),
        "environment": ("linux", "fedora", "ubuntu", "cuda", "nvidia", "python", "gcc", "node"),
        "tool": ("tool", "command", "terminal", "shell"),
    }
    for name, words in groups.items():
        if any(word in text for word in words):
            signals.add(name)
    return signals


def score_alpha_capture(row: dict) -> AlphaQuality:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("capture needs messages")
    signals: set[str] = set()
    roles = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        roles.add(str(message.get("role") or ""))
        content = message.get("content")
        if isinstance(content, str):
            signals.update(classify_message(content))
    if "tool" in roles:
        signals.add("tool")
    if isinstance(row.get("environment"), dict):
        signals.add("environment")

    weights = {
        "failure": 0.15,
        "diagnosis": 0.20,
        "repair": 0.20,
        "test": 0.20,
        "tool": 0.15,
        "environment": 0.10,
    }
    score = sum(weights[s] for s in signals if s in weights)
    return AlphaQuality(min(1.0, score), tuple(sorted(signals)))


def enrich_alpha_capture(row: dict) -> dict:
    quality = score_alpha_capture(row)
    enriched = dict(row)
    enriched["training_quality"] = {
        "score": quality.score,
        "signals": list(quality.signals),
        "high_value_repair": all(
            signal in quality.signals
            for signal in ("failure", "diagnosis", "repair", "test")
        ),
    }
    return enriched
