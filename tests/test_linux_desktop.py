"""Integration checks for the per-user Rabbit Code Linux launcher."""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


def _installer():
    source = Path(__file__).resolve().parents[1] / "scripts" / "install_linux_desktop.py"
    spec = importlib.util.spec_from_file_location("rabbit_linux_desktop_installer", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_desktop_install_and_uninstall_only_touch_user_paths(tmp_path):
    installer = _installer()
    repo_root = Path(__file__).resolve().parents[1]
    home = tmp_path / "test home"
    workspace = tmp_path / "project space"
    workspace.mkdir()
    executable, desktop = installer.install(
        repo_root=repo_root, workspace=workspace, home=home,
    )
    assert executable.is_file()
    assert desktop.is_file()
    script = executable.read_text()
    assert str(workspace) in script
    assert "exec /usr/bin/env bash" in script
    assert subprocess.run(["bash", "-n", str(executable)], check=False).returncode == 0

    entry = desktop.read_text()
    assert "[Desktop Entry]" in entry
    assert "Name=Rabbit Code" in entry
    assert "Terminal=false" in entry
    assert "Exec=" in entry
    assert str(executable) in entry

    installer.uninstall(home)
    assert not executable.exists()
    assert not desktop.exists()
    assert (repo_root / "scripts" / "rabbit_code_linux_ui.sh").exists()


def test_missing_workspace_refuses_install(tmp_path):
    installer = _installer()
    repo_root = Path(__file__).resolve().parents[1]
    try:
        installer.install(
            repo_root=repo_root,
            workspace=tmp_path / "not-created",
            home=tmp_path / "home",
        )
    except NotADirectoryError as exc:
        assert "workspace" in str(exc)
    else:
        raise AssertionError("nonexistent workspace should fail")


def test_launcher_argument_is_quoted(tmp_path):
    installer = _installer()
    assert installer.desktop_argument(Path('/tmp/a "quote"')) == '"/tmp/a \\"quote\\""'
