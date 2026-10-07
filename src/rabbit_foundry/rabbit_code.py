from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from .agent_runtime import RabbitCodeRuntime
from .model_router import ModelRouter, ProviderConfig
from .permissions import Decision, PermissionPolicy
from .session_store import SessionStore
from .workspace import Workspace


HELP = """Commands:
  /help
  /capabilities
  /list [glob]
  /read PATH
  /grep NEEDLE [glob]
  /write PATH TEXT
  /edit PATH OLD NEW
  /git-status
  /git-diff [PATH]
  /test IMAGE COMMAND [ARG ...]
  /quit

Any other line is sent to the selected model.

Writes require --allow-write.
Sandbox execution requires --allow-exec and always runs with Podman network
disabled through the Rabbit backend sandbox.
"""


def build_runtime(args) -> RabbitCodeRuntime:
    policy = PermissionPolicy(
        read=Decision.ALLOW,
        search=Decision.ALLOW,
        write=Decision.ALLOW if args.allow_write else Decision.ASK,
        execute=Decision.ALLOW if args.allow_exec else Decision.ASK,
        network=Decision.ALLOW if args.allow_network else Decision.DENY,
    )
    workspace = Workspace(Path(args.workspace), policy)
    router = ModelRouter(allow_network=args.allow_network)
    router.register(ProviderConfig(
        provider_id=args.provider,
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
    ))
    session_root = Path(args.session_dir)
    if not session_root.is_absolute():
        session_root = workspace.root / session_root
    session = SessionStore(session_root)
    return RabbitCodeRuntime(router, args.provider, workspace, session)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rabbit Code local-first coding agent")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--provider", default="rabbit-local")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765/v1")
    parser.add_argument("--model", default="rabbit-code")
    parser.add_argument("--api-key-env")
    parser.add_argument("--session-dir", default=".rabbit-code/sessions")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--allow-write", action="store_true")
    parser.add_argument("--allow-exec", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    runtime = build_runtime(args)
    print("Rabbit Code")
    print(f"workspace: {runtime.workspace.root}")
    print(f"provider: {args.provider} -> {args.model}")
    print(f"session: {runtime.session.session_id}")
    print("Type /help for commands.")

    while True:
        try:
            line = input("rabbit> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in {"/quit", "/exit"}:
            return
        if line == "/help":
            print(HELP)
            continue

        try:
            parts = shlex.split(line)
            command = parts[0] if parts else ""
            if command == "/capabilities":
                print(json.dumps(runtime.capabilities(), indent=2))
            elif command == "/list":
                pattern = parts[1] if len(parts) > 1 else "**/*"
                print("\n".join(runtime.list(pattern)))
            elif command == "/read":
                if len(parts) != 2:
                    raise ValueError("usage: /read PATH")
                print(runtime.read(parts[1]))
            elif command == "/grep":
                if len(parts) < 2 or len(parts) > 3:
                    raise ValueError("usage: /grep NEEDLE [glob]")
                pattern = parts[2] if len(parts) == 3 else "**/*"
                print(json.dumps(runtime.grep(parts[1], pattern), indent=2))
            elif command == "/write":
                if len(parts) < 3:
                    raise ValueError("usage: /write PATH TEXT")
                print(runtime.write(
                    parts[1],
                    " ".join(parts[2:]),
                    approved=args.allow_write,
                ))
            elif command == "/edit":
                if len(parts) != 4:
                    raise ValueError("usage: /edit PATH OLD NEW")
                print(runtime.edit(
                    parts[1], parts[2], parts[3],
                    approved=args.allow_write,
                ))
            elif command == "/git-status":
                print(runtime.git_status(), end="")
            elif command == "/git-diff":
                if len(parts) > 2:
                    raise ValueError("usage: /git-diff [PATH]")
                print(runtime.git_diff(parts[1] if len(parts) == 2 else None), end="")
            elif command == "/test":
                if len(parts) < 3:
                    raise ValueError("usage: /test IMAGE COMMAND [ARG ...]")
                result = runtime.run_sandbox(
                    parts[1],
                    parts[2:],
                    approved=args.allow_exec,
                )
                print(json.dumps(result, indent=2))
            elif command.startswith("/"):
                raise ValueError(f"unknown command: {command}")
            else:
                print(runtime.ask(line, max_tokens=args.max_tokens))
        except Exception as exc:
            print(f"error: {exc}")


if __name__ == "__main__":
    main()
