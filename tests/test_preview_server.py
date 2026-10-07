import json

import torch

from rabbit_foundry.model import ModelConfig, TinyRabbitLM
from rabbit_foundry.preview_server import RabbitPreview


def checkpoint(path):
    cfg = ModelConfig(context=32, d_model=32, n_heads=4, n_layers=1, d_ff=64)
    model = TinyRabbitLM(cfg)
    torch.save({"config": cfg.__dict__, "state_dict": model.state_dict()}, path)


def test_preview_loads_owned_checkpoint_and_captures_redacted_session(tmp_path, monkeypatch):
    ckpt = tmp_path / "winner.pt"
    captures = tmp_path / "captures.jsonl"
    checkpoint(ckpt)
    monkeypatch.setattr(
        "rabbit_foundry.preview_server.detect_environment",
        lambda: {"schema_version": 1, "platform": {"system": "Linux"}},
    )
    preview = RabbitPreview(ckpt, captures)
    answer = preview.complete(
        [{"role": "user", "content": "debug API_KEY=super-secret"}],
        max_tokens=2,
    )
    assert isinstance(answer, str)
    row = json.loads(captures.read_text().strip())
    assert row["provider"] == "rabbit-preview"
    assert row["environment"]["platform"]["system"] == "Linux"
    assert "super-secret" not in captures.read_text()
    assert "<redacted>" in captures.read_text()


def test_preview_rejects_malformed_messages(tmp_path):
    ckpt = tmp_path / "winner.pt"
    checkpoint(ckpt)
    preview = RabbitPreview(ckpt)
    try:
        preview.complete([{"role": "user"}])
    except ValueError as exc:
        assert "role/content" in str(exc)
    else:
        raise AssertionError("malformed messages must fail closed")
