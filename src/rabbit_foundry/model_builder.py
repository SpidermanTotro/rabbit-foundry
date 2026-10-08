from __future__ import annotations

from dataclasses import asdict, dataclass

from .model_recipe import ModelRecipe


# These are possible future conversion targets, not verified Rabbit outputs.
# The real GGUF gate lives in export_readiness and must inspect a checkpoint.
GGUF_ARCHITECTURE_CANDIDATES = frozenset({
    "llama", "qwen2", "qwen3", "mistral", "gemma", "gemma2", "phi3",
})


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
        # Do not invent a compatible architecture merely because GGUF
        # was requested: today's Rabbit training backend is TinyRabbit.
        architecture = "tiny-rabbit"

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

    # A plan is not an exported artifact. There is no GGUF conversion/runtime
    # verification in plan_model, so this must never be a passing readiness gate.
    gguf_ready = False
    blocker = None
    if gguf_requested:
        if architecture in {"tiny-rabbit", "tiny_rabbit_lm"}:
            blocker = (
                "TinyRabbit uses a custom byte vocabulary, MultiheadAttention mapping, "
                "and learned absolute positions; native llama.cpp GGUF support or an "
                "architecture migration is required before export."
            )
            stages.append("gguf_readiness_blocked")
        elif architecture in GGUF_ARCHITECTURE_CANDIDATES:
            blocker = (
                f"{architecture} is only a candidate architecture, not a verified GGUF "
                "export. Train or supply a matching checkpoint, convert it using the "
                "correct tokenizer/tensor mapping, then pass a llama.cpp runtime "
                "load and inference smoke test."
            )
            stages.append("gguf_conversion_candidate")
        else:
            blocker = (
                f"Architecture {architecture!r} is not a verified export target; "
                "select a concrete supported architecture and verify conversion "
                "and llama.cpp inference before declaring GGUF ready."
            )
            stages.append("gguf_readiness_blocked")

    return BuildPlan(tuple(stages), True, gguf_ready, blocker)


def plan_dict(recipe: ModelRecipe, *, architecture: str = "auto") -> dict:
    return asdict(plan_model(recipe, architecture=architecture))
