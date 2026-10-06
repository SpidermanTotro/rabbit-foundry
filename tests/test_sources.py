import pytest

from rabbit_foundry.sources import RepositorySource, is_training_path


def test_source_requires_allowlisted_license():
    RepositorySource("owner/repo", "1234567890abcdef", "MIT").validate()
    with pytest.raises(ValueError):
        RepositorySource("owner/repo", "1234567890abcdef", "GPL-3.0").validate()


def test_training_path_filter():
    assert is_training_path("src/main.py")
    assert not is_training_path("node_modules/x/index.js")
    assert not is_training_path(".github/workflows/test.yml")
