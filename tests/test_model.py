import torch

from rabbit_foundry.model import ModelConfig, TinyRabbitLM


def test_forward_and_loss():
    cfg = ModelConfig(vocab_size=256, context=16, d_model=32, n_heads=4, n_layers=2, d_ff=64)
    model = TinyRabbitLM(cfg)
    x = torch.randint(0, 256, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 256)
    assert loss is not None and torch.isfinite(loss)


def test_parameter_count_positive():
    model = TinyRabbitLM(ModelConfig(d_model=32, n_heads=4, n_layers=1, d_ff=64))
    assert model.parameter_count() > 0
