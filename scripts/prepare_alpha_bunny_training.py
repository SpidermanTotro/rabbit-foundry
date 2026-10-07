from __future__ import annotations

import argparse
import json

from rabbit_foundry.bunny_pipeline import prepare_alpha_bunny_training


def main():
    parser = argparse.ArgumentParser(
        description="Prepare captured Alpha/Bunny behavior for Rabbit Foundry training"
    )
    parser.add_argument("captures", nargs="+", help="JSON/JSONL capture files")
    parser.add_argument("--out-dir", default="runs/alpha-bunny")
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument(
        "--allow-incomplete-holdout",
        action="store_true",
        help="development only: do not require all six frozen Alpha holdout cases",
    )
    args = parser.parse_args()
    result = prepare_alpha_bunny_training(
        args.captures,
        args.out_dir,
        window=args.window,
        require_complete_holdout=not args.allow_incomplete_holdout,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
