import torch

from rabbit_foundry.train import make_episode_batch


def row(prompt: bytes, target: bytes):
    return {
        "prompt_hex": prompt.hex(),
        "target_hex": target.hex(),
    }


def test_episode_batch_uses_stored_targets_not_concatenated_prompts():
    rows = [
        row(b"AAAA", b"BBBB"),
        row(b"CCCC", b"DDDD"),
    ]
    torch.manual_seed(0)
    x, y = make_episode_batch(rows, batch=16, seq=4, device="cpu")

    pairs = {(bytes(a.tolist()), bytes(b.tolist())) for a, b in zip(x, y)}
    assert pairs <= {(b"AAAA", b"BBBB"), (b"CCCC", b"DDDD")}
    assert pairs


def test_episode_batch_rejects_too_short_rows():
    rows = [row(b"abc", b"bcd")]
    try:
        make_episode_batch(rows, batch=1, seq=4, device="cpu")
    except ValueError as exc:
        assert "long enough" in str(exc)
    else:
        raise AssertionError("expected ValueError")
