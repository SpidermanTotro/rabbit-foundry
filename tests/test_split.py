from rabbit_foundry.split import source_split


def test_source_split_is_stable():
    args = ("owner/repo", "abcdef012345", "src/a.py")
    assert source_split(*args) == source_split(*args)
    assert source_split(*args) in {"train", "validation", "test"}


def test_revisions_of_same_file_cannot_cross_splits():
    first = source_split("owner/repo", "a" * 40, "src/a.py")
    later = source_split("owner/repo", "b" * 40, "src/a.py")
    assert first == later
