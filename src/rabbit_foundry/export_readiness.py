from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import torch

from .model import ModelConfig, TinyRabbitLM


SUPPORTED_LLAMA_CPP_ARCHITECTURES = {
    "llama", "qwen2", "qwen3", "mistral", "gemma", "gemma2", "phi3",
}


def inspect_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Rabbit checkpoint must be a mapping")
    config = payload.get("config")
    state = payload.get("state_dict")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError("Rabbit checkpoint needs config and state_dict")

    cfg = ModelConfig(**config)
    model = TinyRabbitLM(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    architecture = "tiny_rabbit_lm"
    blockers = [
        "custom TinyRabbitLM architecture is not registered in llama.cpp",
        "byte-level vocabulary has no llama.cpp tokenizer/model architecture mapping",
        "PyTorch MultiheadAttention parameter layout needs an explicit GGUF tensor mapping",
        "learned absolute position embeddings need runtime support or an architecture change",
    ]
    return {
        "version": 1,
        "checkpoint": str(path),
        "architecture": architecture,
        "config": asdict(cfg),
        "parameters": model.parameter_count(),
        "state_dict_keys": len(state),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "checkpoint_loads": not missing and not unexpected,
        "llama_cpp_supported_architecture": architecture in SUPPORTED_LLAMA_CPP_ARCHITECTURES,
        "gguf_export_ready": False,
        "blockers": blockers,
        "next_step": (
            "implement a llama.cpp TinyRabbit architecture/tokenizer mapping, or train a "
            "Rabbit-native model on a llama.cpp-supported architecture before GGUF conversion"
        ),
    }


def write_export_readiness(checkpoint: str | Path, out: str | Path) -> dict:
    report = inspect_checkpoint(checkpoint)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report
