"""Verify the auditable Rabbit Code Bash installer is safe and reversible."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "install_rabbit.sh"


def _run(*args, home: Path, bin_dir: Path):
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["RABBIT_BIN_DIR"] = str(bin_dir)
    env["PYTHON"] = "python3"
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=env, text=True, capture_output=True,
        timeout=30, check=False,
    )


def test_installer_bash_syntax():
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_install_update_help_and_uninstall(tmp_path):
    home = tmp_path / "home with spaces"
    home.mkdir()
    bin_dir = home / ".local" / "bin"
    install = _run(home=home, bin_dir=bin_dir)
    assert install.returncode == 0, install.stderr
    assert "No dependency packages, downloads" in install.stdout
    launcher = bin_dir / "rabbit"
    assert launcher.is_file()
    assert launcher.stat().st_mode & 0o111
    assert "${PYTHONPATH:+:$PYTHONPATH}" in launcher.read_text()

    env = os.environ.copy()
    env["HOME"] = str(home)
    output = subprocess.run(
        [str(launcher), "--help"],
        capture_output=True, text=True,
        timeout=20, env=env, check=False,
    )
    assert output.returncode == 0, output.stderr
    assert "rabbit" in output.stdout
    assert "doctor" in output.stdout

    update = _run("--update", home=home, bin_dir=bin_dir)
    assert update.returncode == 0, update.stderr
    uninstall = _run("--uninstall", home=home, bin_dir=bin_dir)
    assert uninstall.returncode == 0, uninstall.stderr
    assert not launcher.exists()
    assert not (home / ".bashrc").exists()


def test_installer_refuses_unmanaged_launcher(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    bin_dir = home / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    launcher = bin_dir / "rabbit"
    launcher.write_text("#!/bin/sh\necho other-program\n", encoding="utf-8")
    install = _run(home=home, bin_dir=bin_dir)
    assert install.returncode == 3
    assert "Refusing" in install.stderr
    removal = _run("--uninstall", home=home, bin_dir=bin_dir)
    assert removal.returncode == 3
    assert launcher.read_text() == "#!/bin/sh\necho other-program\n"


def test_installer_rejects_unknown_mode(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    result = _run("--wipe-home", home=home, bin_dir=home / "bin")
    assert result.returncode == 2
    assert "Usage:" in result.stderr
