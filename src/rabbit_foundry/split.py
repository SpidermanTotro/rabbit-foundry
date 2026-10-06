from __future__ import annotations

import hashlib


def split_bucket(key: str, buckets: int = 1000) -> int:
    if buckets < 2:
        raise ValueError("buckets must be >= 2")
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def source_split(repository: str, commit: str, path: str) -> str:
    """Stable file-family split.

    Commit is intentionally excluded from the split key. All revisions and
    windows of the same repository/path therefore remain in one partition,
    preventing historical variants of a file from leaking across train,
    validation, and test.
    """
    del commit  # retained in the API because manifests record exact provenance
    bucket = split_bucket(f"{repository}:{path}")
    if bucket < 800:
        return "train"
    if bucket < 900:
        return "validation"
    return "test"
