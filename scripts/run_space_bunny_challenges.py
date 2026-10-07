from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from run_space_bunny_fingerprint import DEFAULT_ENDPOINT, DEFAULT_MODEL, run_case


def main() -> None:
    p = argparse.ArgumentParser(description="Run authored non-anchor training challenges against Space Bunny")
    p.add_argument("challenges")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-key-env", default="OPENCODE_ZEN_API_KEY")
    p.add_argument("--out", default="runs/space-bunny/training-responses.jsonl")
    a = p.parse_args()

    rows = [json.loads(x) for x in Path(a.challenges).read_text().splitlines() if x.strip()]
    target = Path(a.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for row in rows:
            response = run_case(
                a.endpoint, a.model, row,
                api_key=os.environ.get(a.api_key_env),
            )
            handle.write(json.dumps({"id": row["id"], "response": response}) + "\n")
            print(f"{row['id']}: captured")


if __name__ == "__main__":
    main()
