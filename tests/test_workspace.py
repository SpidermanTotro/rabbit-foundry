import subprocess

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


def test_workspace_write_and_exact_edit_require_explicit_approval(tmp_path):
    workspace = Workspace(tmp_path, PermissionPolicy())
    with pytest.raises(PermissionError):
        workspace.write_text("new.txt", "hello")

    workspace.write_text("new.txt", "hello", approved=True)
    workspace.replace_text("new.txt", "hello", "world", approved=True)
    assert (tmp_path / "new.txt").read_text() == "world"


def test_exact_edit_rejects_ambiguous_match(tmp_path):
    (tmp_path / "a.txt").write_text("same same")
    workspace = Workspace(tmp_path, PermissionPolicy())
    with pytest.raises(ValueError):
        workspace.replace_text("a.txt", "same", "x", approved=True)


def test_workspace_can_be_configured_for_write_allow(tmp_path):
    policy = PermissionPolicy(write=Decision.ALLOW)
    workspace = Workspace(tmp_path, policy)
    workspace.write_text("new.txt", "hello")
    assert (tmp_path / "new.txt").read_text() == "hello"


def test_sandbox_execution_uses_existing_locked_down_runner(tmp_path, monkeypatch):
    policy = PermissionPolicy(execute=Decision.ALLOW)
    workspace = Workspace(tmp_path, policy)

    def fake_run(image, source, command, limits):
        assert image == "python:3.12"
        assert source == tmp_path.resolve()
        assert command == ["python", "-m", "pytest"]
        return subprocess.CompletedProcess(command, 0, "ok\n", "")

    monkeypatch.setattr("rabbit_foundry.workspace.run_podman", fake_run)
    result = workspace.run_sandbox(
        "python:3.12",
        ["python", "-m", "pytest"],
    )
    assert result == {"returncode": 0, "stdout": "ok\n", "stderr": ""}
