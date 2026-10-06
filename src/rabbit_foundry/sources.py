from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

ALLOWED_LICENSES = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"}
CODE_SUFFIXES = {".py", ".rs", ".c", ".h", ".cpp", ".hpp", ".go", ".js", ".ts", ".java"}


@dataclass(frozen=True)
class RepositorySource:
    repository: str
    commit: str
    license: str

    def validate(self) -> None:
        if "/" not in self.repository:
            raise ValueError("repository must be owner/name")
        if not self.commit or len(self.commit) < 7:
            raise ValueError("commit must identify an immutable revision")
        if self.license not in ALLOWED_LICENSES:
            raise ValueError(f"license not allowlisted: {self.license}")


def is_training_path(path: str) -> bool:
    p = PurePosixPath(path)
    if any(part.startswith(".") for part in p.parts):
        return False
    if any(part in {"vendor", "node_modules", "dist", "build", "third_party"} for part in p.parts):
        return False
    return p.suffix.lower() in CODE_SUFFIXES
