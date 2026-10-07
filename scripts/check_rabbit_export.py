from __future__ import annotations

import argparse
import json

from rabbit_foundry.export_readiness import write_export_readiness


def main():
    parser = argparse.ArgumentParser(
        description="Audit whether a Rabbit winner checkpoint is ready for GGUF/llama.cpp export"
    )
    parser.add_argument("--checkpoint", default="runs/latest/winner.pt")
    parser.add_argument("--out", default="runs/latest/export-readiness.json")
    args = parser.parse_args()
    report = write_export_readiness(args.checkpoint, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
