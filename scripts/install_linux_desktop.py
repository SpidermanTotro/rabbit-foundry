#!/usr/bin/env python3
"""Install or remove a per-user Rabbit Code app-menu launcher on Linux.

Does not use sudo or edit system application directories. The launcher starts
the local browser UI from this checkout; the terminal CLI is unchanged.
"""
from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path


def desktop_argument(path: Path) -> str:
    """Quote a desktop-entry Exec argument, including embedded special chars."""
    value = str(path).replace("\\", "\\\\").replace('"', '\\"')
    return '"' + value + '"'


def install(*, repo_root: Path, workspace: Path, home: Path) -> tuple[Path, Path]:
    repo_root = repo_root.expanduser().resolve()
    workspace = workspace.expanduser().resolve()
    home = home.expanduser().resolve()
    source_script = repo_root / "scripts" / "rabbit_code_linux_ui.sh"
    if not source_script.is_file():
        raise FileNotFoundError("Rabbit Code UI launcher script was not found")
    if not workspace.is_dir():
        raise NotADirectoryError("workspace does not exist: " + str(workspace))

    bin_dir = home / ".local" / "bin"
    app_dir = home / ".local" / "share" / "applications"
    bin_dir.mkdir(parents=True, exist_ok=True)
    app_dir.mkdir(parents=True, exist_ok=True)

    executable = bin_dir / "rabbit-code-gui"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "exec /usr/bin/env bash "
        + shlex.quote(str(source_script)) + " "
        + shlex.quote(str(workspace)) + "\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)

    entry = app_dir / "rabbit-code.desktop"
    entry.write_text(
        "[Desktop Entry]\n"
        "Version=1.0\n"
        "Type=Application\n"
        "Name=Rabbit Code\n"
        "GenericName=Local Coding Assistant\n"
        "Comment=Local-first AI coding workspace for Linux\n"
        "Exec=" + desktop_argument(executable) + "\n"
        "Icon=applications-development\n"
        "Terminal=false\n"
        "Categories=Development;IDE;\n"
        "StartupNotify=true\n",
        encoding="utf-8",
    )
    entry.chmod(0o644)
    return executable, entry


def uninstall(home: Path) -> tuple[Path, Path]:
    home = home.expanduser().resolve()
    executable = home / ".local" / "bin" / "rabbit-code-gui"
    entry = home / ".local" / "share" / "applications" / "rabbit-code.desktop"
    # Only fixed per-user destinations are removed, never the source checkout.
    entry.unlink(missing_ok=True)
    executable.unlink(missing_ok=True)
    return executable, entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, help="folder to open; defaults to Rabbit repository")
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    home = Path.home()
    if args.uninstall:
        _, entry = uninstall(home)
        print("Removed Rabbit Code desktop entry:", entry)
    else:
        _, entry = install(
            repo_root=repo_root,
            workspace=args.workspace if args.workspace is not None else repo_root,
            home=home,
        )
        print("Installed Rabbit Code app launcher:", entry)
        print("Find 'Rabbit Code' in your Linux application launcher.")
        print("This is local-only; Ollama or Rabbit gateway must run on your PC.")


if __name__ == "__main__":
    main()
