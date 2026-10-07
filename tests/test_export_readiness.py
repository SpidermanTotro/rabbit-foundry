import torch

from rabbit_foundry.export_readiness import inspect_checkpoint, write_export_readiness
from rabbit_foundry.model import ModelConfig, TinyRabbitLM


def checkpoint(tmp_path):
    cfg = ModelConfig(context=32, d_model=32, n_heads=4, n_layers=1, d_ff=64)
    model = TinyRabbitLM(cfg)
    path = tmp_path / "winner.pt"
    torch.save({"config": cfg.__dict__, "state_dict": model.state_dict()}, path)
    return path


def test_export_readiness_loads_checkpoint_but_blocks_fake_gguf(tmp_path):
    report = inspect_checkpoint(checkpoint(tmp_path))
    assert report["checkpoint_loads"] is True
    assert report["architecture"] == "tiny_rabbit_lm"
    assert report["gguf_export_ready"] is False
    assert report["llama_cpp_supported_architecture"] is False
    assert report["blockers"]


def test_export_readiness_writes_machine_readable_report(tmp_path):
    out = tmp_path / "export-readiness.json"
    report = write_export_readiness(checkpoint(tmp_path), out)
    assert out.exists()
    assert report["parameters"] > 0
