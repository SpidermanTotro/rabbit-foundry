from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


@dataclass(frozen=True)
class BackupEntry:
    source: str
    backup: str
    sha256: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def backup_files(paths: list[Path], destination: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = destination / stamp
    root.mkdir(parents=True, exist_ok=False)
    entries = []
    for source in paths:
        if not source.is_file():
            continue
        target = root / source.name
        shutil.copy2(source, target)
        entries.append(BackupEntry(str(source), str(target), sha256_file(target)))
    manifest = {
        "created_utc": stamp,
        "entries": [entry.__dict__ for entry in entries],
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root
