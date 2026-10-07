from __future__ import annotations

import json
from pathlib import Path

from rabbit_foundry.environment import write_environment_snapshot


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Capture a safe Rabbit Code host-environment snapshot")
    parser.add_argument("--out", default="runs/environment.json")
    args = parser.parse_args()
    snapshot = write_environment_snapshot(Path(args.out))
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
