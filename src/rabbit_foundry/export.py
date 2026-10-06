from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_status(checkpoint: Path, out_dir: Path) -> dict:
    """Validate export prerequisites.

    Native Rabbit -> GGUF conversion is intentionally gated until the model
    architecture has a matching llama.cpp GGUF tensor mapping. Renaming a
    PyTorch checkpoint to .gguf is not a valid export.
    """
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "checkpoint": str(checkpoint),
        "gguf_ready": False,
        "reason": (
            "Rabbit tensor mapping and llama.cpp architecture registration "
            "are not implemented yet."
        ),
    }
    (out_dir / "export-status.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--out-dir", default="out")
    args = p.parse_args()
    print(json.dumps(export_status(Path(args.checkpoint), Path(args.out_dir)), indent=2))


if __name__ == "__main__":
    main()
