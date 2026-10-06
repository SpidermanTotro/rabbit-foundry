from __future__ import annotations

import argparse
import json
from pathlib import Path

from rabbit_foundry.ingest import build_episode_manifest, ingest_file, write_provenance
from rabbit_foundry.sources import RepositorySource


def main():
    p = argparse.ArgumentParser(description="Read-only pinned GitHub source ingester")
    p.add_argument("manifest")
    p.add_argument("--cache", default="runs/source-cache")
    p.add_argument("--out", default="runs/episodes.json")
    p.add_argument("--provenance", default="runs/provenance.json")
    p.add_argument("--window", type=int, default=128)
    args = p.parse_args()

    config = json.loads(Path(args.manifest).read_text())
    cache = Path(args.cache)
    ingested = []

    for entry in config["sources"]:
        source = RepositorySource(
            entry["repository"], entry["commit"], entry["license"]
        )
        for path in entry.get("paths", []):
            item = ingest_file(source, path, cache)
            ingested.append(item)
            print(f"cached {item.repository}@{item.commit[:12]}:{item.path} {item.bytes} bytes")

    payload = build_episode_manifest(ingested, Path(args.out), window=args.window)
    write_provenance(ingested, Path(args.provenance))
    print(f"episodes={len(payload['episodes'])}")


if __name__ == "__main__":
    main()
