from __future__ import annotations

import argparse
import json

from rabbit_foundry.space_bunny_compare import compare_space_bunny


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current Space Bunny responses with the frozen Alpha behavioral fingerprint")
    parser.add_argument("responses", help="JSONL containing id and response for all six frozen cases")
    parser.add_argument("--out", default="runs/space-bunny/comparison.json")
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    print(json.dumps(compare_space_bunny(args.responses, args.out, args.threshold), indent=2))


if __name__ == "__main__":
    main()
