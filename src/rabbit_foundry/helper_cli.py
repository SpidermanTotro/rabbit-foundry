"""Friendly, local-first Rabbit Code helper CLI.

The main `rabbit-code` engine remains unchanged. The `rabbit` entry point
offers diagnostics and shortcuts without requiring a cloud account or Bun.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_TAGS = "http://127.0.0.1:11434/api/tags"
GATEWAY_HEALTH = "http://127.0.0.1:8765/health"


def _loopback_json(url: str) -> dict | None:
    # Only fixed literal loopback URLs are permitted by this helper.
    if url not in (OLLAMA_TAGS, GATEWAY_HEALTH):
        raise ValueError("diagnostics only query known loopback endpoints")
    try:
        with urllib.request.urlopen(url, timeout=1.5) as response:
            raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            return None
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except (urllib.error.URLError, TimeoutError, OSError,
            ValueError, UnicodeDecodeError):
        return None


def diagnostic_report() -> dict:
    ollama = _loopback_json(OLLAMA_TAGS)
    gateway = _loopback_json(GATEWAY_HEALTH)
    local_bin = Path.home() / ".local" / "bin"
    path_entries = [Path(v).expanduser() for v in os.environ.get("PATH", "").split(os.pathsep) if v]
    data = {
        "os": platform.system(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "commands": {
            tool: shutil.which(tool)
            for tool in ("git", "ollama", "podman", "bun", "node", "pipx")
        },
        "path_local_bin": local_bin in path_entries,
        "ollama_running": bool(ollama is not None and isinstance(ollama.get("models"), list)),
        "ollama_installed_models": [
            m.get("name")
            for m in (ollama or {}).get("models", [])[:200]
            if isinstance(m, dict) and isinstance(m.get("name"), str)
        ] if isinstance((ollama or {}).get("models", []), list) else [],
        "rabbit_gateway_running": bool(
            isinstance(gateway, dict)
            and gateway.get("status") == "ok"
            and gateway.get("mode") == "rabbit-code-model-gateway"
        ),
        "rabbit_gateway_upstream_model": (
            gateway.get("upstream_model") if isinstance(gateway, dict) else None
        ),
    }
    return data


def print_doctor(*, as_json: bool = False) -> None:
    data = diagnostic_report()
    if as_json:
        print(json.dumps(data, indent=2))
        return
    print("Rabbit Code · Linux Doctor")
    print("=" * 31)
    print(f"System: {data['os']} · Python {data['python']}")
    for tool, found in data["commands"].items():
        print(f"{tool:8} {'OK  ' + found if found else 'not found'}")
    print(f"~/.local/bin on PATH: {'yes' if data['path_local_bin'] else 'NO'}")
    print(f"Local Ollama: {'running' if data['ollama_running'] else 'not reachable'}")
    print(f"Rabbit gateway: {'running' if data['rabbit_gateway_running'] else 'not reachable'}")
    if data["ollama_installed_models"]:
        print("Installed local models: " + ", ".join(data["ollama_installed_models"]))
    if not data["path_local_bin"]:
        print('PATH tip for Bash: export PATH="$HOME/.local/bin:$PATH"')
    if not data["ollama_running"]:
        print("Ollama tip: start your existing local Ollama service (ollama serve).")
    print("Doctor is read-only. Nothing was installed or downloaded.")


def _main_engine(args: list[str]) -> None:
    from . import rabbit_code
    parser = rabbit_code.make_parser()
    parsed = parser.parse_args(args)
    if parsed.ui and (parsed.allow_network or parsed.allow_write or parsed.allow_exec):
        parser.error("browser mode cannot enable network, writes, or execution")
    runtime = rabbit_code.build_runtime(parsed)
    if parsed.ui:
        if not runtime.router.provider(runtime.provider_id).local:
            parser.error("browser mode requires a loopback provider")
        from .web_ui import serve_browser_ui
        serve_browser_ui(runtime, port=parsed.ui_port, open_browser=parsed.open_browser)
    elif parsed.command:
        rabbit_code.execute_command(
            runtime, parsed, [parsed.command, *parsed.command_args]
        )
    else:
        # Preserve original interactive terminal CLI.
        rabbit_code.main()


def _common_workspace(parser):
    parser.add_argument("-w", "--workspace", type=Path, default=Path.cwd(),
                        help="project folder (defaults to current directory)")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rabbit",
        description="Rabbit Code: your local coding helper (CLI + Linux interface)",
        epilog="Examples: rabbit doctor | rabbit models | rabbit ui | rabbit chat 'hello'",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("version", help="show Rabbit Code version")
    doctor = sub.add_parser("doctor", help="check local Python, tools, Ollama and PATH")
    doctor.add_argument("--json", action="store_true", help="machine-readable diagnostics")
    sub.add_parser("models", help="list already-installed local Ollama models")

    ui = sub.add_parser("ui", help="start local Linux browser workspace")
    _common_workspace(ui)
    ui.add_argument("--port", type=int, default=8766)
    ui.add_argument("--no-browser", action="store_true")

    for kind in ("chat", "agent"):
        command = sub.add_parser(kind, help=("ask the local model" if kind == "chat"
                                            else "inspect a project with read-only agent tools"))
        _common_workspace(command)
        command.add_argument("--model", help="installed local Ollama model ID")
        command.add_argument("message", nargs="+", help="message or task text")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "version":
        from . import __version__
        print(f"rabbit-code {__version__}")
        return 0
    if args.command == "doctor":
        print_doctor(as_json=args.json)
        return 0
    if args.command == "models":
        from .ui_models import installed_ollama_models
        names = installed_ollama_models()
        if not names:
            print("No installed models found, or local Ollama is not running.")
        else:
            print("Installed local Ollama models:")
            for name in names:
                print("  " + name)
        return 0

    cli_args = ["--workspace", str(args.workspace)]
    if args.command == "ui":
        cli_args += ["--ui", "--ui-port", str(args.port)]
        if not args.no_browser:
            cli_args.append("--open-browser")
    elif args.command in ("chat", "agent"):
        if args.model:
            from .ui_models import installed_ollama_models
            if args.model not in installed_ollama_models():
                parser.error("model is not installed in local Ollama")
            cli_args += [
                "--base-url", "http://127.0.0.1:11434/v1",
                "--model", args.model,
            ]
            if args.command == "chat":
                cli_args.append("--no-native-tools")
        cli_args += [args.command, " ".join(args.message)]
    _main_engine(cli_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
