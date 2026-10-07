#!/usr/bin/env python3
import argparse
import json

from rabbit_foundry.bunny_import import write_bunny_manifest


def main():
    p = argparse.ArgumentParser(description="Import eligible Alpha/Space Bunny JSONL into Rabbit Code backend episodes")
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=128)
    args = p.parse_args()
    payload = write_bunny_manifest(args.source, args.out, window=args.window)
    print(json.dumps({
        "source_rows": payload["source_rows"],
        "episodes": len(payload["episodes"]),
        "source_sha256": payload["source_sha256"],
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
