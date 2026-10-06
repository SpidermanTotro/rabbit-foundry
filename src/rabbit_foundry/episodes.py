from __future__ import annotations

from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class Episode:
    episode_id: str
    skill: str
    prompt: bytes
    target: bytes
    source_repository: str
    source_commit: str
    source_path: str


def next_byte_episodes(repository: str, commit: str, path: str, content: bytes, window: int = 128):
    """Build deterministic self-supervised episodes without teacher-model output."""
    if window < 8:
        raise ValueError("window must be >= 8")
    if len(content) <= window:
        return []
    out = []
    stride = window
    for start in range(0, len(content) - window, stride):
        chunk = content[start : start + window + 1]
        if len(chunk) != window + 1:
            continue
        key = f"{repository}@{commit}:{path}:{start}".encode()
        episode_id = hashlib.sha256(key).hexdigest()[:24]
        out.append(
            Episode(
                episode_id=episode_id,
                skill="code_prediction",
                prompt=chunk[:-1],
                target=chunk[1:],
                source_repository=repository,
                source_commit=commit,
                source_path=path,
            )
        )
    return out
