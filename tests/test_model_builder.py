from rabbit_foundry.model_builder import plan_model
from rabbit_foundry.model_recipe import ModelRecipe, recipe_from_answers


def test_preservation_recipe_builds_selected_courses_without_fake_export_pass():
    recipe = recipe_from_answers(
        "debug-bunny",
        ["behavior", "debugging", "tool_use"],
        export_gguf=True,
        source_model="space-bunny-free",
    )
    plan = plan_model(recipe)
    assert plan.gguf_ready is False
    assert "architecture migration" in plan.gguf_blocker
    assert "behavior_course" in plan.stages
    assert "repair_course" in plan.stages
    assert "trajectory_course" in plan.stages
    assert "style_course" not in plan.stages


def test_tiny_rabbit_never_claims_fake_gguf_readiness():
    recipe = ModelRecipe(
        name="rabbit",
        preserve=["coding"],
        export=["checkpoint", "gguf"],
    )
    plan = plan_model(recipe, architecture="tiny-rabbit")
    assert plan.checkpoint_ready is True
    assert plan.gguf_ready is False
    assert "architecture migration" in plan.gguf_blocker
    assert plan.stages[-1] == "gguf_readiness_blocked"


def test_concrete_architecture_is_a_candidate_not_a_verified_export():
    recipe = ModelRecipe(
        name="rabbit",
        preserve=["coding"],
        export=["checkpoint", "gguf"],
    )
    plan = plan_model(recipe, architecture="llama")
    assert plan.gguf_ready is False
    assert "llama.cpp runtime" in plan.gguf_blocker
    assert plan.stages[-1] == "gguf_conversion_candidate"


def test_generic_compatibility_label_cannot_bypass_readiness_gate():
    recipe = ModelRecipe(
        name="rabbit",
        preserve=["coding"],
        export=["checkpoint", "gguf"],
    )
    plan = plan_model(recipe, architecture="llama-compatible")
    assert plan.gguf_ready is False
    assert "not a verified export target" in plan.gguf_blocker
    assert plan.stages[-1] == "gguf_readiness_blocked"


def test_auto_mode_does_not_infer_llama_compatibility_from_gguf_request():
    recipe = ModelRecipe(
        name="portable-rabbit",
        preserve=["coding", "debugging"],
        export=["checkpoint", "gguf"],
    )
    plan = plan_model(recipe)
    assert plan.gguf_ready is False
    assert "TinyRabbit" in plan.gguf_blocker
    assert plan.stages[-1] == "gguf_readiness_blocked"


def test_plan_without_gguf_request_has_no_export_blocker():
    recipe = ModelRecipe(name="rabbit", preserve=["coding"], export=["checkpoint"])
    plan = plan_model(recipe)
    assert plan.gguf_ready is False
    assert plan.gguf_blocker is None
    assert plan.stages[-1] == "checkpoint"
