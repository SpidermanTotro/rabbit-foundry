from __future__ import annotations

import json
from dataclasses import dataclass

from .agent_runtime import RabbitCodeRuntime
from .model_router import assistant_text


TOOL_PROTOCOL_PROMPT = """You are Rabbit Code operating a controlled workspace.

Respond with exactly one JSON object and no prose outside it.

To request a tool:
{"type":"tool","tool":"read","args":{"path":"README.md"}}

To finish:
{"type":"final","content":"your answer"}

Available tools:
- read {"path": string}
- list {"pattern": string, optional}
- grep {"needle": string, "pattern": string, optional}
- write {"path": string, "content": string} [permission gated]
- edit {"path": string, "old": string, "new": string} [permission gated]
- git_status {}
- git_diff {"path": string, optional}
- sandbox {"image": string, "command": [string, ...]} [permission gated]

Never invent tool results. There is no raw shell tool. Sandbox execution has
network disabled and a read-only workspace.
"""


def parse_action(text: str) -> dict:
    raw = text.strip()
    fence = chr(96) * 3
    if raw.startswith(fence):
        lines = raw.splitlines()
        if lines and lines[0].startswith(fence):
            lines = lines[1:]
        if lines and lines[-1].strip() == fence:
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"type": "final", "content": text}
    if not isinstance(value, dict):
        raise ValueError("agent action must be a JSON object")
    kind = value.get("type")
    if kind not in {"tool", "final"}:
        raise ValueError("agent action type must be tool or final")
    if kind == "final":
        if not isinstance(value.get("content"), str):
            raise ValueError("final action requires string content")
        return value
    if not isinstance(value.get("tool"), str):
        raise ValueError("tool action requires tool name")
    args = value.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("tool args must be an object")
    value["args"] = args
    return value


def _string(args: dict, key: str, *, default: str | None = None) -> str:
    value = args.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


@dataclass
class AgentLoop:
    runtime: RabbitCodeRuntime
    max_steps: int = 8
    approve_write: bool = False
    approve_exec: bool = False
    max_tokens: int = 1024

    def run(self, task: str) -> str:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must not be empty")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        self.runtime.session.record("user", {
            "content": task,
            "mode": "agent",
        })
        messages = [
            {"role": "system", "content": TOOL_PROTOCOL_PROMPT},
            {"role": "user", "content": task},
        ]

        for step in range(1, self.max_steps + 1):
            response = self.runtime.router.complete(
                self.runtime.provider_id,
                messages,
                max_tokens=self.max_tokens,
            )
            raw = assistant_text(response)
            self.runtime.session.record("assistant", {
                "content": raw,
                "mode": "agent",
                "protocol": "rabbit-json-tool-v1",
                "step": step,
            })

            try:
                action = parse_action(raw)
            except Exception as exc:
                self.runtime.session.record("failure", {
                    "operation": "agent.parse_action",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "step": step,
                })
                raise

            if action["type"] == "final":
                content = action["content"]
                self.runtime.session.record("final", {
                    "content": content,
                    "mode": "agent",
                    "steps": step,
                })
                return content

            tool = action["tool"]
            args = action["args"]
            try:
                result = self._dispatch(tool, args)
                tool_result = {"ok": True, "tool": tool, "result": result}
            except Exception as exc:
                tool_result = {
                    "ok": False,
                    "tool": tool,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }

            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": "RABBIT_TOOL_RESULT\n" + json.dumps(
                    tool_result, ensure_ascii=False
                ),
            })

        error = f"agent exceeded maximum tool steps ({self.max_steps})"
        self.runtime.session.record("failure", {
            "operation": "agent.loop",
            "error_type": "StepLimitExceeded",
            "message": error,
        })
        raise RuntimeError(error)

    def _dispatch(self, tool: str, args: dict):
        if tool == "read":
            return self.runtime.read(_string(args, "path"))
        if tool == "list":
            return self.runtime.list(_string(args, "pattern", default="**/*"))
        if tool == "grep":
            return self.runtime.grep(
                _string(args, "needle"),
                _string(args, "pattern", default="**/*"),
            )
        if tool == "write":
            return self.runtime.write(
                _string(args, "path"),
                _string(args, "content"),
                approved=self.approve_write,
            )
        if tool == "edit":
            return self.runtime.edit(
                _string(args, "path"),
                _string(args, "old"),
                _string(args, "new"),
                approved=self.approve_write,
            )
        if tool == "git_status":
            if args:
                raise ValueError("git_status takes no arguments")
            return self.runtime.git_status()
        if tool == "git_diff":
            path = args.get("path")
            if path is not None and not isinstance(path, str):
                raise ValueError("path must be a string")
            return self.runtime.git_diff(path)
        if tool == "sandbox":
            image = _string(args, "image")
            command = args.get("command")
            if not isinstance(command, list) or not command or not all(
                isinstance(item, str) and item for item in command
            ):
                raise ValueError("sandbox command must be a non-empty string list")
            return self.runtime.run_sandbox(
                image,
                command,
                approved=self.approve_exec,
            )
        raise ValueError(f"unknown Rabbit Code tool: {tool}")
