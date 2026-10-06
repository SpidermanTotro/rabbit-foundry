from rabbit_foundry.code_tasks import corrupt_lines, hidden_diff


def test_corruption_is_reproducible():
    text = "a = 1\nb = 2\nprint(a + b)\n"
    a = corrupt_lines("o/r", "abcdef012345", "x.py", text, seed=7)
    b = corrupt_lines("o/r", "abcdef012345", "x.py", text, seed=7)
    assert a is not None
    assert a.broken == b.broken
    assert a.target == text
    assert a.broken != a.target


def test_hidden_diff_uses_real_before_after():
    ep = hidden_diff(
        "o/r", "aaaaaaa", "bbbbbbb", "x.py",
        "answer = 41\n", "answer = 42\n"
    )
    assert ep is not None
    assert ep.skill == "hidden_diff"
    assert ep.broken == "answer = 41\n"
    assert ep.target == "answer = 42\n"
