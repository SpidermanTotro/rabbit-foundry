from __future__ import annotations

import hashlib
from pathlib import Path

DEFAULT_ARTIFACTS = [
    ("adapter", "~/SpaceBunny-Alpha/04-run-artifacts/checkpoint-53-adapter"),
    ("merged_model", "~/SpaceBunny-Alpha/06-model/bunny-qwen3-4b-ep1-merged"),
    ("provenance", "~/SpaceBunny-Alpha/06-model/bunny-qwen3-4b-ep1-merged/PROVENANCE.json"),
    ("gguf_f16", "~/spacebunny-clone/out/v14-4b/kilo-4b-v14-4b-f16.gguf"),
    ("gguf_q4km", "~/spacebunny-clone/out/v14-4b/kilo-4b-v14-4b-q4km.gguf"),
]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def inspect_artifact(kind: str, raw_path: str, hash_large: bool = False) -> dict:
    path = Path(raw_path).expanduser()
    result = {"kind": kind, "configured_path": raw_path, "resolved_path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    result["type"] = "directory" if path.is_dir() else "file"
    if path.is_file():
        size = path.stat().st_size
        result["bytes"] = size
        result["sha256"] = sha256_file(path) if hash_large or size <= 268435456 else None
    return result

def build_inventory(hash_large: bool = False) -> dict:
    artifacts = [inspect_artifact(k, p, hash_large) for k, p in DEFAULT_ARTIFACTS]
    return {"version": 1, "lineage": "space-bunny", "artifacts": artifacts,
            "all_configured_paths_exist": all(a["exists"] for a in artifacts)}
