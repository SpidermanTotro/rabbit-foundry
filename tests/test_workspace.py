import pytest

from rabbit_foundry.permissions import Decision, PermissionPolicy
from rabbit_foundry.workspace import Workspace


def test_workspace_read_search_and_escape_guard(tmp_path):
    (tmp_path / "a.py").write_text("one\nneedle\n")
    workspace = Workspace(tmp_path, PermissionPolicy())
    assert workspace.read_text("a.py").startswith("one")
    assert workspace.list_files("*.py") == ["a.py"]
    assert workspace.grep("needle", "*.py")[0]["line"] == 2
    with pytest.raises(PermissionError):
        workspace.read_text("../outside.txt")


def test_workspace_write_requires_explicit_approval(tmp_path):
    workspace = Workspace(tmp_path, PermissionPolicy())
    with pytest.raises(PermissionError):
        workspace.write_text("new.txt", "hello")

    saved = workspace.write_text("new.txt", "hello", approved=True)
    assert saved.read_text() == "hello"


def test_workspace_can_be_configured_for_write_allow(tmp_path):
    policy = PermissionPolicy(write=Decision.ALLOW)
    workspace = Workspace(tmp_path, policy)
    workspace.write_text("new.txt", "hello")
    assert (tmp_path / "new.txt").read_text() == "hello"
