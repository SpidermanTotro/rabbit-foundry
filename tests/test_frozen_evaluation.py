import torch

from rabbit_foundry.train import frozen_episode_batches


def row(byte: int):
    prompt = bytes([byte + i for i in range(16)])
    target = bytes([(byte + i + 1) % 256 for i in range(16)])
    return {"skill": "code_prediction", "prompt_hex": prompt.hex(), "target_hex": target.hex()}


def test_frozen_episode_batches_repeat_exactly():
    rows = [row(10), row(40), row(70)]
    a = frozen_episode_batches(rows, batch=3, seq=8, seed=99, batches=4)
    b = frozen_episode_batches(rows, batch=3, seq=8, seed=99, batches=4)
    assert len(a) == len(b) == 4
    for (ax, ay), (bx, by) in zip(a, b):
        assert torch.equal(ax, bx)
        assert torch.equal(ay, by)


def test_frozen_evaluation_does_not_consume_training_rng():
    rows = [row(10), row(40)]
    torch.manual_seed(123)
    expected = torch.randint(0, 1000, (5,))
    torch.manual_seed(123)
    frozen_episode_batches(rows, batch=2, seq=8, seed=77, batches=2)
    actual = torch.randint(0, 1000, (5,))
    assert torch.equal(actual, expected)
