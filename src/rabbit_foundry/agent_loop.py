from __future__ import annotations

import json
from dataclasses import dataclass

from .agent_runtime import RabbitCodeRuntime
from .model_router import assistant_message, assistant_text, assistant_tool_calls


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

You DO have access to the user's current workspace through the tools above.
For repository inspection, debugging, review, or fixing tasks, use workspace
tools before giving a final answer. Do not claim you cannot access local files
until you have actually attempted an appropriate Rabbit Code tool.

Never invent tool results. There is no raw shell tool. Sandbox execution has
network disabled and a read-only workspace.
"""


NATIVE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a UTF-8 text file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list",
            "description": "List files inside the workspace using a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search workspace text files for an exact substring.",
            "parameters": {
                "type": "object",
                "properties": {
                    "needle": {"type": "string"},
                    "pattern": {"type": "string"},
                },
                "required": ["needle"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write a text file. Permission gated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Replace exactly one matching text fragment. Permission gated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Read git status --short for the workspace.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Read the unstaged git diff, optionally for one path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sandbox",
            "description": (
                "Run an argv command inside the locked-down Podman sandbox. "
                "Network is disabled and the workspace is read-only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {"type": "string"},
                    "command": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                },
                "required": ["image", "command"],
            },
        },
    },
]


def _object_args(value: dict, *reserved: str) -> dict:
    args = value.get("args", value.get("arguments", value.get("parameters")))
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError as exc:
            raise ValueError("tool arguments string must contain JSON") from exc
        args = parsed
    if args is None:
        args = {
            key: item
            for key, item in value.items()
            if key not in set(reserved) | {
                "type", "tool", "action", "name", "function",
                "args", "arguments", "parameters",
            }
        }
    if not isinstance(args, dict):
        raise ValueError("tool args must be an object")
    return args


def _canonicalize_action(value: dict) -> dict:
    kind = value.get("type")

    if kind == "final":
        content = value.get("content", value.get("answer", value.get("final")))
        if not isinstance(content, str):
            raise ValueError("final action requires string content")
        return {"type": "final", "content": content}

    if kind == "tool":
        tool = value.get("tool", value.get("name"))
        if not isinstance(tool, str):
            raise ValueError("tool action requires tool name")
        return {
            "type": "tool",
            "tool": tool,
            "args": _object_args(value),
        }

    if kind in {"function", "tool_call"} and isinstance(value.get("function"), dict):
        function = value["function"]
        tool = function.get("name")
        if not isinstance(tool, str):
            raise ValueError("function action requires function.name")
        return {
            "type": "tool",
            "tool": tool,
            "args": _object_args(function),
        }

    tool = value.get("tool")
    if isinstance(tool, str):
        return {
            "type": "tool",
            "tool": tool,
            "args": _object_args(value),
        }

    action = value.get("action")
    if isinstance(action, str):
        normalized = action.strip().lower()
        if normalized in {"final", "finish", "done", "answer", "respond"}:
            content = value.get(
                "content",
                value.get("answer", value.get("final", value.get("message"))),
            )
            if not isinstance(content, str):
                raise ValueError("final action requires string content")
            return {"type": "final", "content": content}
        return {
            "type": "tool",
            "tool": action,
            "args": _object_args(value),
        }

    name = value.get("name")
    if isinstance(name, str):
        return {
            "type": "tool",
            "tool": name,
            "args": _object_args(value),
        }

    for key in ("final", "answer"):
        if isinstance(value.get(key), str):
            return {"type": "final", "content": value[key]}

    if isinstance(value.get("content"), str):
        return {"type": "final", "content": value["content"]}

    keys = ", ".join(sorted(map(str, value.keys())))
    raise ValueError(
        "unrecognized agent action object"
        + (f" (keys: {keys})" if keys else "")
    )


def parse_action(text: str, *, allow_plain_final: bool = True) -> dict:
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
    except json.JSONDecodeError as exc:
        if allow_plain_final:
            return {"type": "final", "content": text}
        raise ValueError(
            "agent mode requires exactly one JSON action object"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("agent action must be a JSON object")
    return _canonicalize_action(value)


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
        tool_attempts = 0
        fallback_context_used = False

        for step in range(1, self.max_steps + 1):
            provider = self.runtime.router.provider(self.runtime.provider_id)
            response = self.runtime.router.complete(
                self.runtime.provider_id,
                messages,
                max_tokens=self.max_tokens,
                tools=NATIVE_TOOL_SCHEMAS if provider.supports_tools else None,
            )

            native_calls = (
                assistant_tool_calls(response)
                if provider.supports_tools
                else []
            )
            if native_calls:
                assistant = assistant_message(response)
                self.runtime.session.record("assistant", {
                    "content": assistant.get("content") or "",
                    "tool_calls": native_calls,
                    "mode": "agent",
                    "protocol": "openai-tool-calls",
                    "step": step,
                })
                messages.append(assistant)
                for index, call in enumerate(native_calls):
                    tool_attempts += 1
                    call_id = call.get("id") or f"rabbit-call-{step}-{index}"
                    function = call.get("function")
                    if not isinstance(function, dict):
                        tool_result = {
                            "ok": False,
                            "error_type": "ValueError",
                            "error": "native tool call is missing function object",
                        }
                    else:
                        tool = function.get("name")
                        arguments = function.get("arguments", {})
                        try:
                            if isinstance(arguments, str):
                                arguments = json.loads(arguments or "{}")
                            if not isinstance(tool, str) or not tool:
                                raise ValueError("native tool call has no function name")
                            if not isinstance(arguments, dict):
                                raise ValueError("native tool arguments must be an object")
                            result = self._dispatch(tool, arguments)
                            tool_result = {
                                "ok": True,
                                "tool": tool,
                                "result": result,
                            }
                        except Exception as exc:
                            tool_result = {
                                "ok": False,
                                "tool": tool if isinstance(tool, str) else None,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                    messages.append({
                        "role": "tool",
                        "tool_call_id": str(call_id),
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    })
                continue

            raw = assistant_text(response)
            self.runtime.session.record("assistant", {
                "content": raw,
                "mode": "agent",
                "protocol": (
                    "openai-tools-fallback-json"
                    if provider.supports_tools
                    else "rabbit-json-tool-v1"
                ),
                "step": step,
            })

            try:
                action = parse_action(
                    raw,
                    allow_plain_final=(tool_attempts > 0),
                )
            except Exception as exc:
                self.runtime.session.record("failure", {
                    "operation": "agent.parse_action",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "step": step,
                    "recoverable": True,
                })
                messages.append({"role": "assistant", "content": raw})

                if tool_attempts == 0 and not fallback_context_used:
                    observations, attempts = self._bootstrap_readonly_context()
                    tool_attempts += attempts
                    fallback_context_used = True
                    messages.append({
                        "role": "user",
                        "content": (
                            "RABBIT_FALLBACK_CONTEXT\n"
                            + json.dumps({
                                "instruction": (
                                    "Rabbit Code inspected the workspace for you. "
                                    "Use these real observations. You may now either "
                                    "request another Rabbit Code tool with JSON or "
                                    "give a concise final answer."
                                ),
                                "observations": observations,
                            }, ensure_ascii=False)
                        ),
                    })
                    continue

                messages.append({
                    "role": "user",
                    "content": (
                        "RABBIT_PROTOCOL_ERROR\n"
                        + json.dumps({
                            "error": str(exc),
                            "required": [
                                {
                                    "type": "tool",
                                    "tool": "read",
                                    "args": {"path": "README.md"},
                                },
                                {
                                    "type": "final",
                                    "content": "your answer",
                                },
                            ],
                            "instruction": (
                                "Reply again with exactly one supported JSON "
                                "object and no prose outside it."
                            ),
                        })
                    ),
                })
                continue

            if action["type"] == "final":
                if tool_attempts == 0:
                    self.runtime.session.record("failure", {
                        "operation": "agent.final_without_tool",
                        "error_type": "ToolRequired",
                        "message": (
                            "agent attempted to finish before using a workspace tool"
                        ),
                        "step": step,
                        "recoverable": True,
                    })
                    messages.append({"role": "assistant", "content": raw})
                    if not fallback_context_used:
                        observations, attempts = self._bootstrap_readonly_context()
                        tool_attempts += attempts
                        fallback_context_used = True
                        messages.append({
                            "role": "user",
                            "content": (
                                "RABBIT_FALLBACK_CONTEXT\n"
                                + json.dumps({
                                    "instruction": (
                                        "Rabbit Code inspected the workspace for you. "
                                        "Use these real observations and answer the "
                                        "user's task. You may request more tools if "
                                        "needed."
                                    ),
                                    "observations": observations,
                                }, ensure_ascii=False)
                            ),
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": (
                                "RABBIT_TOOL_REQUIRED\n"
                                + json.dumps({
                                    "error": (
                                        "You have not used any Rabbit Code workspace "
                                        "tool yet."
                                    ),
                                    "instruction": (
                                        "Use git_status, list, read, grep, or git_diff "
                                        "before giving a final answer."
                                    ),
                                })
                            ),
                        })
                    continue
                content = action["content"]
                self.runtime.session.record("final", {
                    "content": content,
                    "mode": "agent",
                    "steps": step,
                    "tool_attempts": tool_attempts,
                })
                return content

            tool = action["tool"]
            args = action["args"]
            tool_attempts += 1
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

    def _bootstrap_readonly_context(self) -> tuple[list[dict], int]:
        observations: list[dict] = []
        attempts = 0

        def observe(tool: str, action):
            nonlocal attempts
            attempts += 1
            try:
                result = action()
                observations.append({
                    "tool": tool,
                    "ok": True,
                    "result": result,
                })
            except Exception as exc:
                observations.append({
                    "tool": tool,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })

        observe("git_status", self.runtime.git_status)
        observe(
            "list",
            lambda: self.runtime.list("*")[:120],
        )

        for candidate in ("README.md", "pyproject.toml", "package.json"):
            try:
                path = self.runtime.workspace.resolve(candidate)
            except Exception:
                continue
            if path.is_file():
                observe(
                    "read",
                    lambda candidate=candidate: self.runtime.read(candidate)[:16000],
                )

        self.runtime.session.record("tool_result", {
            "tool": "fallback_context",
            "result": observations,
        })
        return observations, attempts

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
