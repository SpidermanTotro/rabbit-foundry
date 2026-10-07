from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--preferred-route", choices=("zen", "go"), default="zen")
    parser.add_argument("--model", default="space-bunny-free")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-quality", type=float, default=0.60)
    args = parser.parse_args()

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    challenges = work / "training-challenges.jsonl"
    training_responses = work / "training-responses.jsonl"
    promoted = work / "training-promoted.jsonl"
    manifest = work / "training-episodes.json"
    train_run = work / "rabbit-train"

    run([
        args.python, "scripts/run_space_bunny_fingerprint.py", args.frozen_cases,
        "--preferred-route", args.preferred_route,
        "--model", args.model,
        "--out-dir", str(work / "fingerprint"),
    ])
    run([
        args.python, "scripts/build_space_bunny_challenges.py",
        "--out", str(challenges),
    ])
    run([
        args.python, "scripts/run_space_bunny_challenges.py", str(challenges),
        "--preferred-route", args.preferred_route,
        "--model", args.model,
        "--out", str(training_responses),
    ])
    run([
        args.python, "scripts/promote_space_bunny_training.py",
        str(challenges), str(training_responses),
        "--out", str(promoted),
        "--min-quality", str(args.min_quality),
    ])
    run([
        args.python, "scripts/import_bunny_jsonl.py",
        "--source", str(promoted),
        "--out", str(manifest),
        "--window", str(args.window),
    ])
    run([
        args.python, "-m", "rabbit_foundry.train",
        "--episodes", str(manifest),
        "--episode-kind", "bunny",
        "--steps", str(args.steps),
        "--batch", str(args.batch),
        "--seq", str(args.seq),
        "--device", args.device,
        "--run-dir", str(train_run),
    ])

    winner = train_run / "winner.pt"
    summary = {
        "model": args.model,
        "preferred_route": args.preferred_route,
        "fingerprint": str(work / "fingerprint" / "comparison.json"),
        "training_challenges": str(challenges),
        "training_responses": str(training_responses),
        "promoted_training": str(promoted),
        "training_manifest": str(manifest),
        "training_run": str(train_run),
        "winner_checkpoint": str(winner) if winner.exists() else None,
        "status": "trained_and_promoted" if winner.exists() else "training_completed_no_greenlight_winner",
    }
    (work / "factory-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
