from __future__ import annotations

import hashlib


def split_bucket(key: str, buckets: int = 1000) -> int:
    if buckets < 2:
        raise ValueError("buckets must be >= 2")
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def source_split(repository: str, commit: str, path: str) -> str:
    """Stable source-level split so windows from one file cannot cross splits."""
    bucket = split_bucket(f"{repository}@{commit}:{path}")
    if bucket < 800:
        return "train"
    if bucket < 900:
        return "validation"
    return "test"
