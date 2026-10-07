from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import torch

from .bunny_import import load_bunny_episode_manifest
from .candidate_eval import evaluate_behavior_cases, load_behavior_cases
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


def manifest_episodes(path: str, kind: str = "github"):
    loader = load_bunny_episode_manifest if kind == "bunny" else load_episode_manifest
    train_rows = loader(path, "train")
    valid_rows = loader(path, "validation")
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
def heldout_next_byte_pass_rate(model, frozen_batches, device):
    """Measured token accuracy on the same frozen validation batches."""
    model.eval()
    correct = total = 0
    for x, y in frozen_batches:
        logits, _ = model(x.to(device), y.to(device))
        pred = logits.argmax(dim=-1)
        target = y.to(device)
        correct += int((pred == target).sum().item())
        total += int(target.numel())
    model.train()
    return correct / total if total else 0.0


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
    episode_mode=False, sampling="fixed", initial_state_dict=None,
    resume_state=None, checkpoint_state=None,
):
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = TinyRabbitLM(cfg).to(device)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict, strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history = []
    finite = True
    completed_steps = 0
    curriculum = None
    if episode_mode and sampling == "adaptive":
        curriculum = Curriculum(skills=episode_skills(train_data))

    if resume_state is not None:
        if not isinstance(resume_state, dict):
            raise ValueError("resume state must be a dictionary")
        if "optimizer_state_dict" not in resume_state:
            raise ValueError("resume checkpoint is missing optimizer state")
        opt.load_state_dict(resume_state["optimizer_state_dict"])
        completed_steps = int(resume_state.get("completed_steps", 0))
        if completed_steps < 0:
            raise ValueError("completed_steps must be non-negative")
        if "python_rng_state" in resume_state:
            random.setstate(resume_state["python_rng_state"])
        if "torch_rng_state" in resume_state:
            torch.random.set_rng_state(resume_state["torch_rng_state"])
        if device.type == "cuda" and resume_state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_state_all"])
        if curriculum is not None and resume_state.get("curriculum_state") is not None:
            curriculum.load_state(resume_state["curriculum_state"])

    last_completed_step = completed_steps
    for local_step in range(1, steps + 1):
        step = completed_steps + local_step
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
        last_completed_step = step
        if curriculum is not None and skill is not None:
            # Better-than-uniform next-byte loss is a simple teacher-free pass signal.
            passed = float(loss.detach()) < math.log(cfg.vocab_size)
            curriculum.observe([Outcome(skill, passed)])
        if step == 1 or step == steps or step % max(1, steps // 5) == 0:
            record = {"step": step, "train_loss": float(loss.detach())}
            if skill is not None:
                record["skill"] = skill
            history.append(record)
    heldout_pass_rate = 0.0
    if finite and episode_mode:
        frozen = frozen_episode_batches(valid_data, batch, seq, seed=20261007, batches=8)
        val = evaluate_episode_batches(model, frozen, device)
        heldout_pass_rate = heldout_next_byte_pass_rate(model, frozen, device)
    else:
        val = evaluate(model, valid_data, batch, seq, device) if finite else float("inf")
        heldout_pass_rate = 0.0
    curriculum_state = curriculum.state() if curriculum is not None else None
    if checkpoint_state is not None:
        checkpoint_state.clear()
        checkpoint_state.update({
            "optimizer_state_dict": opt.state_dict(),
            "completed_steps": last_completed_step,
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.random.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "curriculum_state": curriculum_state,
            "heldout_pass_rate": heldout_pass_rate,
        })
    return model, val, history, finite, curriculum_state


def main():
    p = argparse.ArgumentParser(description="Rabbit Foundry TwinTrain")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--run-dir", default="runs/latest")
    p.add_argument("--episodes", help="episode manifest produced by a Foundry importer")
    p.add_argument(
        "--episode-kind", choices=("github", "bunny"), default="github",
        help="manifest lineage: teacher-free GitHub episodes or Alpha/Space Bunny-derived episodes",
    )
    p.add_argument(
        "--sampling", choices=("fixed", "adaptive"), default="fixed",
        help="training episode sampling policy; validation is always frozen/uniform",
    )
    p.add_argument(
        "--minimum-relative-improvement", type=float, default=0.0,
        help="require this fractional held-out loss improvement before Greenlight promotion",
    )
    p.add_argument(
        "--minimum-behavior-score", type=float, default=0.0,
        help="require measured behavioral evaluation at or above this score; fails closed when evidence is absent",
    )
    p.add_argument(
        "--behavior-scores",
        help="JSON file mapping TwinTrain candidate names A/B to measured behavior scores in [0,1]",
    )
    p.add_argument(
        "--behavior-challenges",
        help="evaluation-only JSONL challenges used to generate and score A/B candidate responses automatically",
    )
    p.add_argument(
        "--behavior-max-new-bytes", type=int, default=128,
        help="maximum generated bytes per automatic behavior challenge",
    )
    p.add_argument(
        "--init-checkpoint",
        help="initialize model weights from a compatible owned Rabbit checkpoint; optimizer starts fresh",
    )
    p.add_argument(
        "--resume-checkpoint",
        help="exactly resume model, optimizer and RNG state from a Rabbit winner.pt checkpoint",
    )
    args = p.parse_args()

    if args.behavior_scores and args.behavior_challenges:
        raise ValueError("use either --behavior-scores or --behavior-challenges, not both")
    behavior_scores = {}
    behavior_cases = load_behavior_cases(args.behavior_challenges) if args.behavior_challenges else None
    if args.behavior_scores:
        payload = json.loads(Path(args.behavior_scores).read_text())
        if not isinstance(payload, dict):
            raise ValueError("behavior-scores file must be a JSON object")
        for name in ("A", "B"):
            if name in payload:
                value = payload[name]
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.0 <= float(value) <= 1.0:
                    raise ValueError(f"behavior score for {name} must be finite and in [0,1]")
                behavior_scores[name] = float(value)
    if args.minimum_behavior_score > 0 and set(behavior_scores) != {"A", "B"}:
        raise ValueError("behavioral Greenlight requires measured scores for both A and B")

    if args.init_checkpoint and args.resume_checkpoint:
        raise ValueError("--init-checkpoint and --resume-checkpoint are mutually exclusive")
    initial_state_dict = None
    resume_state = None
    init_checkpoint_sha256 = None
    checkpoint_arg = args.resume_checkpoint or args.init_checkpoint
    if checkpoint_arg:
        checkpoint_path = Path(checkpoint_arg)
        import hashlib
        init_checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict) or not isinstance(checkpoint.get("state_dict"), dict):
            raise ValueError("init checkpoint must contain config and state_dict")
        cfg = ModelConfig(**checkpoint["config"])
        if args.seq > cfg.context:
            raise ValueError("requested sequence exceeds init checkpoint context")
        initial_state_dict = checkpoint["state_dict"]
        if args.resume_checkpoint:
            resume_state = checkpoint.get("training_state")
            if not isinstance(resume_state, dict):
                raise ValueError("resume checkpoint has no training_state; use --init-checkpoint for weight-only continuation")
    else:
        cfg = ModelConfig(context=max(128, args.seq))
    device = device_from(args.device)
    if args.episodes:
        train_data, valid_data = manifest_episodes(args.episodes, args.episode_kind)
        if args.episode_kind == "bunny":
            objective = "Alpha/Space Bunny lineage next-byte behavior modeling"
            lineage = "alpha-space-bunny"
            teacher_free_native = False
        else:
            objective = "pinned GitHub episode next-byte prediction"
            lineage = "rabbit-native"
            teacher_free_native = True
        episode_mode = True
    else:
        train_data, valid_data = bootstrap_stream()
        objective = "next-byte prediction bootstrap"
        lineage = "rabbit-native-bootstrap"
        teacher_free_native = True
        episode_mode = False

    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)

    results, models, training_states = {}, {}, {}
    scores = []
    for name, seed in (("A", 1337), ("B", 7331)):
        candidate_checkpoint_state = {}
        model, val, history, finite, curriculum_state = train_one(
            seed, cfg, train_data, valid_data, args.steps, args.batch, args.seq, device,
            episode_mode=episode_mode, sampling=args.sampling,
            initial_state_dict=initial_state_dict,
            resume_state=resume_state,
            checkpoint_state=candidate_checkpoint_state,
        )
        training_states[name] = candidate_checkpoint_state
        heldout_pass_rate = float(candidate_checkpoint_state.get("heldout_pass_rate", 0.0))
        models[name] = model
        results[name] = {
            "seed": seed, "validation_loss": val, "heldout_pass_rate": heldout_pass_rate, "history": history, "finite": finite,
            "curriculum": curriculum_state,
        }
        if behavior_cases is not None:
            behavior_eval = evaluate_behavior_cases(
                model, behavior_cases, device=device,
                max_new_bytes=args.behavior_max_new_bytes,
            )
            behavior_score = behavior_eval["behavior_score"]
            results[name]["behavior_evaluation"] = behavior_eval
            behavior_scores[name] = behavior_score
        else:
            behavior_score = behavior_scores.get(name)
        results[name]["behavior_score"] = behavior_score
        scores.append(CandidateScore(name, val, heldout_pass_rate, finite=finite, behavior_score=behavior_score))

    decision = decide(
        scores[0], scores[1],
        minimum_relative_improvement=args.minimum_relative_improvement,
        minimum_behavior_score=args.minimum_behavior_score,
    )
    if decision.promoted:
        torch.save(
            {
                "checkpoint_version": 2,
                "config": asdict(cfg),
                "state_dict": models[decision.winner].state_dict(),
                "training_state": training_states[decision.winner],
            },
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
        "teacher_free_native_experiment": teacher_free_native,
        "lineage": lineage,
        "episode_kind": args.episode_kind if args.episodes else None,
        "episodes_manifest": args.episodes,
        "sampling": args.sampling,
        "validation_sampling": "frozen_uniform",
        "minimum_relative_improvement": args.minimum_relative_improvement,
        "minimum_behavior_score": args.minimum_behavior_score,
        "behavior_scores_file": args.behavior_scores,
        "behavior_challenges": args.behavior_challenges,
        "behavior_max_new_bytes": args.behavior_max_new_bytes if args.behavior_challenges else None,
        "behavior_evidence": "generated_candidate_responses" if args.behavior_challenges else ("measured" if behavior_scores else "not_provided"),
        "initialization": "exact_resume" if args.resume_checkpoint else ("owned_checkpoint" if args.init_checkpoint else "random"),
        "init_checkpoint": args.init_checkpoint,
        "resume_checkpoint": args.resume_checkpoint,
        "init_checkpoint_sha256": init_checkpoint_sha256,
    }
    (run / "metrics.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
