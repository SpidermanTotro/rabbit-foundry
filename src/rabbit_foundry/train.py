from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import torch

from .curriculum import Curriculum, Outcome
from .dataset import episode_tensors, load_episode_manifest
from .greenlight import CandidateScore, decide
from .model import ModelConfig, TinyRabbitLM


def device_from(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bootstrap_stream():
    corpus = b"rabbit foundry learns by prediction, testing, and measured feedback.\n" * 4096
    tokens = torch.tensor(list(corpus), dtype=torch.long)
    cut = int(len(tokens) * 0.9)
    return tokens[:cut], tokens[cut:]


def manifest_episodes(path: str):
    train_rows = load_episode_manifest(path, "train")
    valid_rows = load_episode_manifest(path, "validation")
    if not train_rows or not valid_rows:
        raise ValueError("manifest needs both train and validation source splits")
    return train_rows, valid_rows


def rows_for_skill(rows, skill):
    selected = [row for row in rows if row.get("skill", "code_prediction") == skill]
    if not selected:
        raise ValueError(f"no training episodes for skill: {skill}")
    return selected


def episode_skills(rows):
    return sorted({row.get("skill", "code_prediction") for row in rows})


def make_episode_batch(rows, batch, seq, device):
    eligible = []
    for row in rows:
        x, y = episode_tensors(row)
        usable = min(len(x), len(y))
        if usable >= seq:
            eligible.append((x, y, usable))
    if not eligible:
        raise ValueError("no episode is long enough for the requested sequence length")

    picks = torch.randint(0, len(eligible), (batch,))
    xs, ys = [], []
    for pick in picks.tolist():
        x, y, usable = eligible[pick]
        start = int(torch.randint(0, usable - seq + 1, (1,)).item())
        xs.append(x[start:start + seq])
        ys.append(y[start:start + seq])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def make_batch(data, batch, seq, device):
    if len(data) <= seq + 1:
        raise ValueError("token stream is too short")
    starts = torch.randint(0, len(data) - seq - 1, (batch,))
    x = torch.stack([data[i:i+seq] for i in starts]).to(device)
    y = torch.stack([data[i+1:i+seq+1] for i in starts]).to(device)
    return x, y


def frozen_episode_batches(rows, batch, seq, seed, batches=8):
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        return [make_episode_batch(rows, batch, seq, torch.device("cpu")) for _ in range(batches)]
    finally:
        torch.random.set_rng_state(state)


@torch.no_grad()
def evaluate_episode_batches(model, frozen_batches, device):
    model.eval()
    losses = []
    for x, y in frozen_batches:
        _, loss = model(x.to(device), y.to(device))
        losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate(model, data, batch, seq, device, batches=8):
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = make_batch(data, batch, seq, device)
        _, loss = model(x, y)
        losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def train_one(
    seed, cfg, train_data, valid_data, steps, batch, seq, device,
    episode_mode=False, sampling="fixed",
):
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = TinyRabbitLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history = []
    finite = True
    curriculum = None
    if episode_mode and sampling == "adaptive":
        curriculum = Curriculum(skills=episode_skills(train_data))

    for step in range(1, steps + 1):
        skill = None
        sampled_rows = train_data
        if curriculum is not None:
            skill = curriculum.choose_skill(seed=seed + step)
            sampled_rows = rows_for_skill(train_data, skill)
        x, y = (
            make_episode_batch(sampled_rows, batch, seq, device)
            if episode_mode
            else make_batch(train_data, batch, seq, device)
        )
        logits, loss = model(x, y)
        if not torch.isfinite(loss):
            finite = False
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if curriculum is not None and skill is not None:
            # Better-than-uniform next-byte loss is a simple teacher-free pass signal.
            passed = float(loss.detach()) < math.log(cfg.vocab_size)
            curriculum.observe([Outcome(skill, passed)])
        if step == 1 or step == steps or step % max(1, steps // 5) == 0:
            record = {"step": step, "train_loss": float(loss.detach())}
            if skill is not None:
                record["skill"] = skill
            history.append(record)
    if finite and episode_mode:
        frozen = frozen_episode_batches(valid_data, batch, seq, seed=20261007, batches=8)
        val = evaluate_episode_batches(model, frozen, device)
    else:
        val = evaluate(model, valid_data, batch, seq, device) if finite else float("inf")
    curriculum_state = curriculum.state() if curriculum is not None else None
    return model, val, history, finite, curriculum_state


def main():
    p = argparse.ArgumentParser(description="Rabbit Foundry TwinTrain")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--run-dir", default="runs/latest")
    p.add_argument("--episodes", help="episode manifest produced by scripts/ingest_manifest.py")
    p.add_argument(
        "--sampling", choices=("fixed", "adaptive"), default="fixed",
        help="training episode sampling policy; validation is always frozen/uniform",
    )
    args = p.parse_args()

    cfg = ModelConfig(context=max(128, args.seq))
    device = device_from(args.device)
    if args.episodes:
        train_data, valid_data = manifest_episodes(args.episodes)
        objective = "pinned GitHub episode next-byte prediction"
        episode_mode = True
    else:
        train_data, valid_data = bootstrap_stream()
        objective = "next-byte prediction bootstrap"
        episode_mode = False

    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)

    results, models = {}, {}
    scores = []
    for name, seed in (("A", 1337), ("B", 7331)):
        model, val, history, finite, curriculum_state = train_one(
            seed, cfg, train_data, valid_data, args.steps, args.batch, args.seq, device,
            episode_mode=episode_mode, sampling=args.sampling,
        )
        models[name] = model
        results[name] = {
            "seed": seed, "validation_loss": val, "history": history, "finite": finite,
            "curriculum": curriculum_state,
        }
        scores.append(CandidateScore(name, val, 1.0, finite=finite))

    decision = decide(scores[0], scores[1])
    if decision.promoted:
        torch.save(
            {"config": asdict(cfg), "state_dict": models[decision.winner].state_dict()},
            run / "winner.pt",
        )

    ledger = {
        "version": "0.3.0",
        "device": str(device),
        "parameters": next(iter(models.values())).parameter_count(),
        "models": results,
        "promotion": asdict(decision),
        "objective": objective,
        "teacher_model": False,
        "episodes_manifest": args.episodes,
        "sampling": args.sampling,
        "validation_sampling": "frozen_uniform",
    }
    (run / "metrics.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
