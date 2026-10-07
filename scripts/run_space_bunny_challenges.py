from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_space_bunny_fingerprint import DEFAULT_MODEL, run_case_with_failover


def main() -> None:
    p = argparse.ArgumentParser(description="Run authored non-anchor training challenges against Space Bunny")
    p.add_argument("challenges")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--preferred-route", choices=("zen", "go"), default="zen")
    p.add_argument("--out", default="runs/space-bunny/training-responses.jsonl")
    a = p.parse_args()

    rows = [json.loads(x) for x in Path(a.challenges).read_text().splitlines() if x.strip()]
    target = Path(a.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for row in rows:
            response, route = run_case_with_failover(row, a.model, a.preferred_route)
            handle.write(json.dumps({
                "id": row["id"], "response": response, "route": route,
                "source_model": a.model, "evaluation_only": False,
            }) + "\n")
            print(f"{row['id']}: captured via {route}")


if __name__ == "__main__":
    main()
