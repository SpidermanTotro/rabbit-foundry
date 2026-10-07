from __future__ import annotations

from dataclasses import asdict, dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class TrainerSourcePlan:
    source: str
    source_kind: str
    trainable: tuple[str, ...]
    evaluation_only: tuple[str, ...]
    excluded: tuple[str, ...]
    reasons: tuple[str, ...]


def inspect_trainer_source(url: str) -> TrainerSourcePlan:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")
    if parsed.scheme != "https":
        raise ValueError("trainer source URL must use https")

    if host == "github.com":
        parts = path.split("/")
        if len(parts) < 2:
            raise ValueError("GitHub trainer source must identify owner/repository")
        repo = f"{parts[0]}/{parts[1]}"
        trainable = (
            "licensed source-code structure and implementation patterns",
            "public documentation and agent/workflow configuration",
            "public tests, failure cases, fixes, and revision history",
        )
        evaluation_only = (
            "selected unseen repository tasks and bug cases",
        )
        excluded = (
            "provider/model weights not distributed by the repository",
            "secrets, credentials, private user data, and generated caches",
        )
        reasons = (
            f"{repo} is a public repository source; preserve its license and provenance per file/capture.",
            "Repository history can teach coding, debugging, tool-use and self-correction without pretending provider weights are available.",
            "Keep evaluation tasks separate so Greenlight measures generalization rather than memorization.",
        )
        return TrainerSourcePlan(url, "public-github-repository", trainable, evaluation_only, excluded, reasons)

    raise ValueError(f"unsupported trainer source host: {host}")


def plan_dict(url: str) -> dict:
    return asdict(inspect_trainer_source(url))
