from rabbit_foundry.split import source_split


def test_source_split_is_stable():
    args = ("owner/repo", "abcdef012345", "src/a.py")
    assert source_split(*args) == source_split(*args)
    assert source_split(*args) in {"train", "validation", "test"}
