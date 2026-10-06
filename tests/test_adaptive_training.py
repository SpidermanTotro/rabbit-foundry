import torch

from rabbit_foundry.train import episode_skills, rows_for_skill, train_one
from rabbit_foundry.model import ModelConfig


def row(skill: str, byte: int):
    prompt = bytes([byte]) * 8
    target = bytes([(byte + 1) % 256]) * 8
    return {"skill": skill, "prompt_hex": prompt.hex(), "target_hex": target.hex()}


def test_skill_helpers_keep_episode_classes_separate():
    rows = [row("code_prediction", 65), row("code_repair", 66), row("hidden_diff", 67)]
    assert episode_skills(rows) == ["code_prediction", "code_repair", "hidden_diff"]
    selected = rows_for_skill(rows, "code_repair")
    assert len(selected) == 1
    assert selected[0]["skill"] == "code_repair"


def test_adaptive_training_records_curriculum_without_changing_validation_policy():
    rows = [
        row("code_prediction", 65),
        row("code_repair", 66),
        row("hidden_diff", 67),
    ]
    cfg = ModelConfig(vocab_size=256, context=8, d_model=16, heads=2, layers=1, d_ff=32)
    _, val, history, finite, state = train_one(
        123, cfg, rows, rows, steps=3, batch=1, seq=8, device=torch.device("cpu"),
        episode_mode=True, sampling="adaptive",
    )
    assert finite
    assert val >= 0.0
    assert state is not None
    assert sum(item["attempts"] for item in state.values()) == 3
    assert all("skill" in item for item in history)


def test_fixed_training_has_no_adaptive_curriculum_state():
    rows = [row("code_prediction", 65)]
    cfg = ModelConfig(vocab_size=256, context=8, d_model=16, heads=2, layers=1, d_ff=32)
    _, _, _, finite, state = train_one(
        123, cfg, rows, rows, steps=1, batch=1, seq=8, device=torch.device("cpu"),
        episode_mode=True, sampling="fixed",
    )
    assert finite
    assert state is None
