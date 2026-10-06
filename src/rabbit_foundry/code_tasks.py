from __future__ import annotations

from dataclasses import dataclass
import difflib
import hashlib
import random


@dataclass(frozen=True)
class RepairEpisode:
    episode_id: str
    skill: str
    broken: str
    target: str
    source_repository: str
    source_commit: str
    source_path: str


def _id(repo: str, commit: str, path: str, skill: str, salt: str) -> str:
    raw = f"{repo}@{commit}:{path}:{skill}:{salt}".encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def corrupt_lines(
    repository: str,
    commit: str,
    path: str,
    text: str,
    *,
    seed: int = 0,
) -> RepairEpisode | None:
    """Remove one non-empty line. Target is the untouched source."""
    lines = text.splitlines(keepends=True)
    candidates = [i for i, line in enumerate(lines) if line.strip()]
    if len(candidates) < 3:
        return None
    rng = random.Random(seed)
    index = rng.choice(candidates)
    broken = "".join(lines[:index] + lines[index + 1 :])
    return RepairEpisode(
        _id(repository, commit, path, "code_repair", str(index)),
        "code_repair",
        broken,
        text,
        repository,
        commit,
        path,
    )


def hidden_diff(
    repository: str,
    before_commit: str,
    after_commit: str,
    path: str,
    before: str,
    after: str,
) -> RepairEpisode | None:
    """Ask Rabbit to reconstruct the later file from an earlier revision."""
    if before == after:
        return None
    diff = "".join(difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"{path}@{before_commit[:12]}",
        tofile=f"{path}@{after_commit[:12]}",
    ))
    # The diff is metadata for deterministic identity, not a teacher-model answer.
    salt = hashlib.sha256(diff.encode()).hexdigest()
    return RepairEpisode(
        _id(repository, after_commit, path, "hidden_diff", salt),
        "hidden_diff",
        before,
        after,
        repository,
        after_commit,
        path,
    )
