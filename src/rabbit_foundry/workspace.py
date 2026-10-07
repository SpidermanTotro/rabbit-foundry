from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .permissions import PermissionPolicy


@dataclass
class Workspace:
    root: Path
    permissions: PermissionPolicy
    max_read_bytes: int = 2 * 1024 * 1024

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"workspace is not a directory: {self.root}")

    def resolve(self, relative: str | Path) -> Path:
        candidate = (self.root / Path(relative)).resolve(strict=False)
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError("path escapes workspace") from exc
        return candidate

    def read_text(self, relative: str | Path) -> str:
        self.permissions.require("read")
        path = self.resolve(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size > self.max_read_bytes:
            raise ValueError(f"file exceeds {self.max_read_bytes} byte read limit")
        return path.read_text(errors="replace")

    def list_files(self, pattern: str = "**/*", *, limit: int = 500) -> list[str]:
        self.permissions.require("search")
        rows: list[str] = []
        for path in self.root.glob(pattern):
            if len(rows) >= limit:
                break
            resolved = path.resolve(strict=False)
            try:
                relative = resolved.relative_to(self.root)
            except ValueError:
                continue
            if resolved.is_file():
                rows.append(str(relative))
        return sorted(rows)

    def grep(
        self,
        needle: str,
        pattern: str = "**/*",
        *,
        limit: int = 100,
    ) -> list[dict]:
        self.permissions.require("search")
        if not needle:
            raise ValueError("needle must not be empty")
        matches: list[dict] = []
        for relative in self.list_files(pattern, limit=2000):
            path = self.resolve(relative)
            if path.stat().st_size > self.max_read_bytes:
                continue
            try:
                lines = path.read_text(errors="replace").splitlines()
            except OSError:
                continue
            for number, line in enumerate(lines, 1):
                if needle in line:
                    matches.append({
                        "path": relative,
                        "line": number,
                        "text": line,
                    })
                    if len(matches) >= limit:
                        return matches
        return matches

    def write_text(
        self,
        relative: str | Path,
        content: str,
        *,
        approved: bool = False,
    ) -> Path:
        self.permissions.require("write", approved=approved)
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        path = self.resolve(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path
