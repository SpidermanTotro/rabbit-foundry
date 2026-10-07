from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .permissions import PermissionPolicy
from .sandbox import SandboxLimits, run_podman


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

    def replace_text(
        self,
        relative: str | Path,
        old: str,
        new: str,
        *,
        approved: bool = False,
    ) -> Path:
        self.permissions.require("write", approved=approved)
        if not old:
            raise ValueError("old text must not be empty")
        path = self.resolve(relative)
        original = self.read_text(relative)
        occurrences = original.count(old)
        if occurrences != 1:
            raise ValueError(
                f"expected exactly one match for edit, found {occurrences}"
            )
        path.write_text(original.replace(old, new, 1))
        return path

    def git_status(self) -> str:
        self.permissions.require("search")
        result = subprocess.run(
            ["git", "-C", str(self.root), "status", "--short"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "git status failed")
        return result.stdout

    def git_diff(self, relative: str | Path | None = None) -> str:
        self.permissions.require("read")
        command = ["git", "-C", str(self.root), "diff", "--"]
        if relative is not None:
            path = self.resolve(relative)
            command.append(str(path.relative_to(self.root)))
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "git diff failed")
        return result.stdout

    def run_sandbox(
        self,
        image: str,
        command: list[str],
        *,
        approved: bool = False,
        limits: SandboxLimits | None = None,
    ) -> dict:
        self.permissions.require("execute", approved=approved)
        if not command or not all(isinstance(item, str) and item for item in command):
            raise ValueError("sandbox command must be a non-empty argv list")
        result = run_podman(image, self.root, command, limits)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
