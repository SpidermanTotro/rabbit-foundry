#!/usr/bin/env python3
import argparse
import json

from rabbit_foundry.alpha_holdout import write_alpha_holdout_manifest


def main():
    p = argparse.ArgumentParser(description="Freeze the six Alpha holdout captures as evaluation-only episodes")
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=128)
    args = p.parse_args()
    payload = write_alpha_holdout_manifest(args.source, args.out, args.window)
    print(json.dumps({
        "source_rows": payload["source_rows"],
        "episodes": len(payload["episodes"]),
        "source_sha256": payload["source_sha256"],
        "training_allowed": payload["training_allowed"],
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
