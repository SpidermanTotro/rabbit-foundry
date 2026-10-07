from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Space Bunny -> Rabbit Foundry one-command pipeline")
    parser.add_argument("frozen_cases", help="six frozen Alpha cases JSONL")
    parser.add_argument("--workdir", default="runs/space-bunny/factory")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--endpoint", default="https://opencode.ai/zen/v1/chat/completions")
    parser.add_argument("--model", default="space-bunny-free")
    args = parser.parse_args()

    from rabbit_foundry.space_bunny_routes import routes_by_preference\n    routes = routes_by_preference(args.preferred_route)\n    route = next((r for r in routes if os.environ.get(r.api_key_env)), None)\n    if route is None:\n        raise SystemExit("No authorized Space Bunny route found. Set OPENCODE_ZEN_API_KEY or OPENCODE_GO_API_KEY.")\n    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    challenges = work / "training-challenges.jsonl"
    training_responses = work / "training-responses.jsonl"
    promoted = work / "training-promoted.jsonl"

    run([args.python, "scripts/run_space_bunny_fingerprint.py", args.frozen_cases,
         "--endpoint", route.endpoint, "--model", args.model, "--api-key-env", route.api_key_env,
         "--out-dir", str(work / "fingerprint")])
    run([args.python, "scripts/build_space_bunny_challenges.py", "--out", str(challenges)])
    run([args.python, "scripts/run_space_bunny_challenges.py", str(challenges),
         "--endpoint", route.endpoint, "--model", args.model, "--api-key-env", route.api_key_env,
         "--out", str(training_responses)])
    run([args.python, "scripts/promote_space_bunny_training.py", str(challenges),
         str(training_responses), "--out", str(promoted)])

    summary = {
        "model": args.model,
        "route": route.name,\n        "endpoint": route.endpoint,
        "fingerprint": str(work / "fingerprint" / "comparison.json"),
        "training_challenges": str(challenges),
        "training_responses": str(training_responses),
        "promoted_training": str(promoted),
        "next": "Feed promoted_training into the normal Rabbit course/training pipeline.",
    }
    (work / "factory-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
