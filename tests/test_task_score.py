from rabbit_foundry.task_score import exact_match, normalized_text_match


def test_exact_match():
    assert exact_match(b"rabbit", b"rabbit")
    assert not exact_match(b"rabbit", b"bunny")


def test_normalized_text_match_ignores_trailing_space():
    assert normalized_text_match("x = 1   \n", "x = 1\n")
