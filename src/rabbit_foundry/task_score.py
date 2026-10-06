from __future__ import annotations

import hashlib


def exact_match(prediction: bytes | str, target: bytes | str) -> bool:
    return prediction == target


def content_digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def normalized_text_match(prediction: str, target: str) -> bool:
    def normalize(s: str) -> str:
        return "\n".join(line.rstrip() for line in s.strip().splitlines())
    return normalize(prediction) == normalize(target)
