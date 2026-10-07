from __future__ import annotations

import argparse
import json

from rabbit_foundry.alpha_preview_capture import convert_alpha_export


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert exported original Alpha Preview/Kilo/OpenCode sessions into Rabbit capture JSONL"
    )
    parser.add_argument("source", help="exported JSONL sessions")
    parser.add_argument("--out", default="runs/alpha-preview/captures.jsonl")
    args = parser.parse_args()
    print(json.dumps(convert_alpha_export(args.source, args.out), indent=2))


if __name__ == "__main__":
    main()
