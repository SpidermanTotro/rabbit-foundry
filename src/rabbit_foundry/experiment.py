from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from .model import ModelConfig
from .train import device_from, manifest_episodes, train_one


@dataclass
class ArmResult:
    name: str
    compute_steps: int
    validation_loss: float
    finite: bool
    seeds: list[int]
    runs: list[dict]


def run_arm(name, sampling, seeds, cfg, train_rows, valid_rows, steps, batch, seq, device):
    runs = []
    for seed in seeds:
        _, val, history, finite, curriculum = train_one(
            seed, cfg, train_rows, valid_rows, steps, batch, seq, device,
            episode_mode=True, sampling=sampling,
        )
        runs.append({
            "seed": seed,
            "validation_loss": val,
            "finite": finite,
            "history": history,
            "curriculum": curriculum,
        })
    finite_runs = [r for r in runs if r["finite"]]
    mean_loss = (
        sum(r["validation_loss"] for r in finite_runs) / len(finite_runs)
        if len(finite_runs) == len(runs) and runs
        else float("inf")
    )
    return ArmResult(
        name=name,
        compute_steps=steps * len(seeds),
        validation_loss=mean_loss,
        finite=len(finite_runs) == len(runs),
        seeds=list(seeds),
        runs=runs,
    )


def choose_winner(results, minimum_relative_improvement: float = 0.0):
    eligible = [r for r in results if r.finite]
    if not eligible:
        return None
    best = min(eligible, key=lambda r: r.validation_loss)
    if minimum_relative_improvement <= 0 or len(eligible) < 2:
        return best.name
    runner_up = sorted(eligible, key=lambda r: r.validation_loss)[1]
    if runner_up.validation_loss <= 0:
        return None
    relative = (runner_up.validation_loss - best.validation_loss) / runner_up.validation_loss
    return best.name if relative >= minimum_relative_improvement else None


def run_experiment(
    episodes, out, *, steps=100, batch=8, seq=64, seeds=(1337, 7331, 2026),
    device_name="auto", minimum_relative_improvement=0.0,
):
    train_rows, valid_rows = manifest_episodes(episodes)
    skills = sorted({row.get("skill", "code_prediction") for row in train_rows})
    required = {"code_prediction", "code_repair", "hidden_diff"}
    missing = sorted(required - set(skills))
    if missing:
        raise ValueError(f"three-skill experiment is missing training skills: {', '.join(missing)}")

    device = device_from(device_name)
    cfg = ModelConfig(context=max(128, seq))
    results = [
        run_arm("fixed", "fixed", seeds, cfg, train_rows, valid_rows, steps, batch, seq, device),
        run_arm("adaptive", "adaptive", seeds, cfg, train_rows, valid_rows, steps, batch, seq, device),
    ]
    payload = {
        "version": 1,
        "experiment": "fixed-vs-adaptive-three-skill",
        "episodes_manifest": str(episodes),
        "skills": skills,
        "teacher_model": False,
        "device": str(device),
        "steps_per_seed": steps,
        "compute_steps_per_arm": steps * len(seeds),
        "seeds": list(seeds),
        "validation_sampling": "frozen_uniform",
        "primary_metric": "mean_validation_loss",
        "results": [asdict(r) for r in results],
        "minimum_relative_improvement": minimum_relative_improvement,
        "winner": choose_winner(results, minimum_relative_improvement),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main():
    p = argparse.ArgumentParser(description="Rabbit Foundry fixed-vs-adaptive three-skill experiment")
    p.add_argument("--episodes", required=True)
    p.add_argument("--out", default="runs/experiments/fixed-vs-adaptive.json")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--seeds", default="1337,7331,2026")
    p.add_argument(
        "--minimum-relative-improvement", type=float, default=0.0,
        help="require this fractional held-out loss improvement before declaring a winner",
    )
    args = p.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    payload = run_experiment(
        args.episodes, args.out, steps=args.steps, batch=args.batch, seq=args.seq,
        seeds=seeds, device_name=args.device,
        minimum_relative_improvement=args.minimum_relative_improvement,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
