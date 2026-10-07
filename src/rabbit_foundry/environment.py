from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path


SAFE_COMMANDS = {
    "python": ["python3", "--version"],
    "git": ["git", "--version"],
    "nvidia_smi": ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
    "gcc": ["gcc", "--version"],
    "node": ["node", "--version"],
}


def _run(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (result.stdout or result.stderr).strip()
    return text.splitlines()[0] if text else None


def detect_environment() -> dict:
    os_release = {}
    path = Path("/etc/os-release")
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                if key in {"ID", "VERSION_ID", "PRETTY_NAME"}:
                    os_release[key.lower()] = value.strip().strip('"')

    tools = {name: _run(command) for name, command in SAFE_COMMANDS.items()}
    return {
        "schema_version": 1,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "linux": os_release or None,
        "tools": tools,
    }


def write_environment_snapshot(out: str | Path) -> dict:
    snapshot = detect_environment()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2) + "\n")
    return snapshot
