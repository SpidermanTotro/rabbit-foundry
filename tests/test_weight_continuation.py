import torch

from rabbit_foundry.model import ModelConfig, TinyRabbitLM
from rabbit_foundry.train import train_one


def rows():
    prompt = bytes(range(32)).hex()
    target = bytes((x + 1) % 256 for x in range(32)).hex()
    return [{
        "episode_id": "one",
        "skill": "bunny_behavior",
        "prompt_hex": prompt,
        "target_hex": target,
    }]


def test_train_one_can_continue_from_compatible_owned_state():
    cfg = ModelConfig(context=32, d_model=32, n_heads=4, n_layers=1, d_ff=64)
    base = TinyRabbitLM(cfg)
    state = {key: value.detach().clone() for key, value in base.state_dict().items()}
    model, val, history, finite, _ = train_one(
        1, cfg, rows(), rows(), 1, 1, 16, torch.device("cpu"),
        episode_mode=True, initial_state_dict=state,
    )
    assert finite
    assert history
    assert val >= 0
    assert model.parameter_count() == base.parameter_count()


def test_train_one_exact_resume_restores_optimizer_and_step_state():
    cfg = ModelConfig(context=32, d_model=32, n_heads=4, n_layers=1, d_ff=64)
    first_state = {}
    first_model, _, first_history, finite, _ = train_one(
        11, cfg, rows(), rows(), 2, 1, 16, torch.device("cpu"),
        episode_mode=True, checkpoint_state=first_state,
    )
    assert finite
    assert first_state["completed_steps"] == 2
    assert first_state["optimizer_state_dict"]["state"]
    assert first_history[-1]["step"] == 2

    resumed_state = {}
    resumed_model, _, resumed_history, finite, _ = train_one(
        11, cfg, rows(), rows(), 1, 1, 16, torch.device("cpu"),
        episode_mode=True,
        initial_state_dict={
            key: value.detach().clone()
            for key, value in first_model.state_dict().items()
        },
        resume_state=first_state,
        checkpoint_state=resumed_state,
    )
    assert finite
    assert resumed_history[-1]["step"] == 3
    assert resumed_state["completed_steps"] == 3
    assert resumed_state["optimizer_state_dict"]["state"]
    assert any(
        state.get("step", 0) != 0
        for state in resumed_state["optimizer_state_dict"]["state"].values()
    )
    assert resumed_model.parameter_count() == first_model.parameter_count()


def test_exact_resume_rejects_missing_optimizer_state():
    cfg = ModelConfig(context=32, d_model=32, n_heads=4, n_layers=1, d_ff=64)
    try:
        train_one(
            11, cfg, rows(), rows(), 1, 1, 16, torch.device("cpu"),
            episode_mode=True, resume_state={"completed_steps": 2},
        )
    except ValueError as exc:
        assert "optimizer state" in str(exc)
    else:
        raise AssertionError("resume without optimizer state must fail closed")
