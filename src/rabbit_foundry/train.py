from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch

from .model import ModelConfig, TinyRabbitLM


def device_from(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_batch(data: torch.Tensor, batch: int, seq: int, device: torch.device):
    if len(data) <= seq + 1:
        raise ValueError("token stream is too short")
    starts = torch.randint(0, len(data) - seq - 1, (batch,))
    x = torch.stack([data[i : i + seq] for i in starts]).to(device)
    y = torch.stack([data[i + 1 : i + seq + 1] for i in starts]).to(device)
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


def train_one(seed, cfg, train_data, valid_data, steps, batch, seq, device):
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = TinyRabbitLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history = []
    for step in range(1, steps + 1):
        x, y = make_batch(train_data, batch, seq, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step == steps or step % max(1, steps // 5) == 0:
            history.append({"step": step, "train_loss": float(loss.detach())})
    val = evaluate(model, valid_data, batch, seq, device)
    return model, val, history


def main():
    p = argparse.ArgumentParser(description="Rabbit Foundry v0.1 twin-model experiment")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--run-dir", default="runs/latest")
    args = p.parse_args()

    cfg = ModelConfig(context=max(128, args.seq))
    device = device_from(args.device)

    # Bootstrap objective only: deterministic byte-token stream, no teacher model.
    corpus = (b"rabbit foundry learns by prediction, testing, and measured feedback.\n" * 4096)
    tokens = torch.tensor(list(corpus), dtype=torch.long)
    cut = int(len(tokens) * 0.9)
    train_data, valid_data = tokens[:cut], tokens[cut:]

    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)

    results = {}
    models = {}
    for name, seed in (("A", 1337), ("B", 7331)):
        model, val, history = train_one(
            seed, cfg, train_data, valid_data, args.steps, args.batch, args.seq, device
        )
        models[name] = model
        results[name] = {"seed": seed, "validation_loss": val, "history": history}

    winner = min(results, key=lambda n: results[n]["validation_loss"])
    torch.save(
        {"config": asdict(cfg), "state_dict": models[winner].state_dict()},
        run / "winner.pt",
    )
    ledger = {
        "version": "0.1.0",
        "device": str(device),
        "parameters": models[winner].parameter_count(),
        "models": results,
        "winner": winner,
        "objective": "next-byte prediction bootstrap",
        "teacher_model": False,
    }
    (run / "metrics.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
