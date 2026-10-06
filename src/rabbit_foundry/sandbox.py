from __future__ import annotations

from dataclasses import dataclass
import shlex
import subprocess
from pathlib import Path


@dataclass(frozen=True)
class SandboxLimits:
    memory: str = "2g"
    cpus: float = 2.0
    pids: int = 256
    timeout_seconds: int = 120


def podman_command(image: str, source: Path, command: list[str], limits: SandboxLimits | None = None) -> list[str]:
    """Return a locked-down Podman command. Does not execute it."""
    limits = limits or SandboxLimits()
    if not image or any(c.isspace() for c in image):
        raise ValueError("invalid image")
    if not source.is_absolute():
        raise ValueError("source path must be absolute")
    if not command:
        raise ValueError("sandbox command is empty")
    return [
        "podman", "run", "--rm",
        "--network=none",
        "--read-only",
        "--cap-drop=all",
        "--security-opt=no-new-privileges",
        f"--memory={limits.memory}",
        f"--cpus={limits.cpus}",
        f"--pids-limit={limits.pids}",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=256m",
        f"--volume={source}:/workspace:ro,Z",
        "--workdir=/workspace",
        image,
        *command,
    ]


def run_podman(image: str, source: Path, command: list[str], limits: SandboxLimits | None = None):
    limits = limits or SandboxLimits()
    argv = podman_command(image, source, command, limits)
    return subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=limits.timeout_seconds,
        check=False,
    )
