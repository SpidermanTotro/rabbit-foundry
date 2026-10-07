from __future__ import annotations

import argparse
import json
from pathlib import Path

from rabbit_foundry.trajectory_capture import import_trajectories


def load(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("trajectories", "traces", "sessions"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    raise ValueError(f"unsupported trajectory export: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Kilo/OpenRouter repair trajectories")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/provider-preview/trajectories.jsonl"))
    args = parser.parse_args()

    rows = []
    for path in args.paths:
        rows.extend(load(path))
    result = import_trajectories(rows, args.out, source="provider-preview")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
