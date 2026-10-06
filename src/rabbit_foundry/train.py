from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch

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


def train_one(seed, cfg, train_data, valid_data, steps, batch, seq, device, episode_mode=False):
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = TinyRabbitLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history = []
    finite = True
    for step in range(1, steps + 1):
        x, y = (
            make_episode_batch(train_data, batch, seq, device)
            if episode_mode
            else make_batch(train_data, batch, seq, device)
        )
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            finite = False
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step == steps or step % max(1, steps // 5) == 0:
            history.append({"step": step, "train_loss": float(loss.detach())})
    if finite and episode_mode:
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(8):
                x, y = make_episode_batch(valid_data, batch, seq, device)
                _, loss = model(x, y)
                losses.append(float(loss))
        model.train()
        val = sum(losses) / len(losses)
    else:
        val = evaluate(model, valid_data, batch, seq, device) if finite else float("inf")
    return model, val, history, finite


def main():
    p = argparse.ArgumentParser(description="Rabbit Foundry TwinTrain")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--run-dir", default="runs/latest")
    p.add_argument("--episodes", help="episode manifest produced by scripts/ingest_manifest.py")
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
        model, val, history, finite = train_one(
            seed, cfg, train_data, valid_data, args.steps, args.batch, args.seq, device,
            episode_mode=episode_mode,
        )
        models[name] = model
        results[name] = {
            "seed": seed, "validation_loss": val, "history": history, "finite": finite
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
    }
    (run / "metrics.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
