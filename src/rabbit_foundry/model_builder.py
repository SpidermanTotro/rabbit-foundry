from __future__ import annotations

from dataclasses import asdict, dataclass
from .model_recipe import ModelRecipe


@dataclass(frozen=True)
class BuildPlan:
    stages: tuple[str, ...]
    checkpoint_ready: bool
    gguf_ready: bool
    gguf_blocker: str | None


def plan_model(recipe: ModelRecipe, *, architecture: str = "auto") -> BuildPlan:
    recipe.validate()
    gguf_requested = "gguf" in recipe.export
    if architecture == "auto":
        architecture = "llama-compatible" if gguf_requested else "tiny-rabbit"
    stages = ["provenance", "privacy_redaction"]
    mapping = {
        "behavior": "behavior_course",
        "coding": "code_course",
        "debugging": "repair_course",
        "tool_use": "trajectory_course",
        "style": "style_course",
        "knowledge": "knowledge_course",
    }
    stages.extend(mapping[x] for x in recipe.preserve)
    stages.extend(["family_split", "train", "heldout_eval", "greenlight", "checkpoint"])

    gguf_ready = gguf_requested and architecture != "tiny-rabbit"
    blocker = None
    if gguf_requested and architecture == "tiny-rabbit":
        blocker = (
            "TinyRabbit uses a custom byte vocabulary, MultiheadAttention mapping, "
            "and learned absolute positions; native llama.cpp GGUF support or an "
            "architecture migration is required before export."
        )
    if gguf_requested:
        stages.append("gguf_export" if gguf_ready else "gguf_readiness_blocked")
    return BuildPlan(tuple(stages), True, gguf_ready, blocker)


def plan_dict(recipe: ModelRecipe, *, architecture: str = "auto") -> dict:
    return asdict(plan_model(recipe, architecture=architecture))
