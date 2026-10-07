from rabbit_foundry.model_builder import plan_model
from rabbit_foundry.model_recipe import ModelRecipe, recipe_from_answers


def test_preservation_recipe_builds_selected_courses():
    recipe = recipe_from_answers(
        "debug-bunny",
        ["behavior", "debugging", "tool_use"],
        export_gguf=True,
        source_model="space-bunny-free",
    )
    plan = plan_model(recipe)
    assert "behavior_course" in plan.stages
    assert "repair_course" in plan.stages
    assert "trajectory_course" in plan.stages
    assert "style_course" not in plan.stages


def test_tiny_rabbit_never_claims_fake_gguf_readiness():
    recipe = ModelRecipe(name="rabbit", preserve=["coding"], export=["checkpoint", "gguf"])
    plan = plan_model(recipe, architecture="tiny-rabbit")
    assert plan.checkpoint_ready is True
    assert plan.gguf_ready is False
    assert "architecture migration" in plan.gguf_blocker


def test_compatible_architecture_can_plan_gguf():
    recipe = ModelRecipe(name="rabbit", preserve=["coding"], export=["checkpoint", "gguf"])
    plan = plan_model(recipe, architecture="llama-compatible")
    assert plan.gguf_ready is True
    assert plan.gguf_blocker is None
    assert plan.stages[-1] == "gguf_export"
