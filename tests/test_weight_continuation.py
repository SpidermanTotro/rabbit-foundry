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
