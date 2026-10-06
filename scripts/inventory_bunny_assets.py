#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from rabbit_foundry.bunny_inventory import build_inventory

def main():
    p = argparse.ArgumentParser(description="Inventory preserved Space Bunny artifacts")
    p.add_argument("--out", default="runs/bunny_inventory.json")
    p.add_argument("--hash-large", action="store_true")
    args = p.parse_args()
    payload = build_inventory(hash_large=args.hash_large)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
