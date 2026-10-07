from __future__ import annotations

import shutil
from dataclasses import dataclass

from .model_router import ModelRouter, assistant_text
from .session_store import SessionStore
from .workspace import Workspace


@dataclass
class RabbitCodeRuntime:
    router: ModelRouter
    provider_id: str
    workspace: Workspace
    session: SessionStore
    system_prompt: str = (
        "You are Rabbit Code, a local-first coding assistant. "
        "Be precise about what you inspected and do not claim tool actions "
        "that were not actually executed."
    )

    def ask(self, text: str, *, max_tokens: int = 1024) -> str:
        if not text.strip():
            raise ValueError("user text must not be empty")
        self.session.record("user", {"content": text})
        try:
            response = self.router.complete(
                self.provider_id,
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text},
                ],
                max_tokens=max_tokens,
            )
            answer = assistant_text(response)
        except Exception as exc:
            self.session.record("failure", {
                "operation": "model.complete",
                "error_type": type(exc).__name__,
                "message": str(exc),
            })
            raise
        self.session.record("assistant", {"content": answer})
        return answer

    def read(self, path: str) -> str:
        return self._tool("read", {"path": path}, lambda: self.workspace.read_text(path))

    def list(self, pattern: str = "**/*") -> list[str]:
        return self._tool(
            "list",
            {"pattern": pattern},
            lambda: self.workspace.list_files(pattern),
        )

    def grep(self, needle: str, pattern: str = "**/*") -> list[dict]:
        return self._tool(
            "grep",
            {"needle": needle, "pattern": pattern},
            lambda: self.workspace.grep(needle, pattern),
        )

    def write(self, path: str, content: str, *, approved: bool = False) -> str:
        saved = self._tool(
            "write",
            {"path": path},
            lambda: self.workspace.write_text(path, content, approved=approved),
            result_mapper=lambda p: {
                "path": str(p.relative_to(self.workspace.root)),
                "bytes": len(content.encode()),
            },
        )
        return str(saved.relative_to(self.workspace.root))

    def edit(
        self,
        path: str,
        old: str,
        new: str,
        *,
        approved: bool = False,
    ) -> str:
        saved = self._tool(
            "edit",
            {"path": path, "old": old, "new": new},
            lambda: self.workspace.replace_text(
                path, old, new, approved=approved
            ),
            result_mapper=lambda p: {
                "path": str(p.relative_to(self.workspace.root)),
            },
        )
        return str(saved.relative_to(self.workspace.root))

    def git_status(self) -> str:
        return self._tool("git_status", {}, self.workspace.git_status)

    def git_diff(self, path: str | None = None) -> str:
        return self._tool(
            "git_diff",
            {"path": path},
            lambda: self.workspace.git_diff(path),
        )

    def run_sandbox(
        self,
        image: str,
        command: list[str],
        *,
        approved: bool = False,
    ) -> dict:
        return self._tool(
            "sandbox",
            {"image": image, "command": command},
            lambda: self.workspace.run_sandbox(
                image, command, approved=approved
            ),
        )

    def capabilities(self) -> dict:
        provider = self.router.provider(self.provider_id)
        policy = self.workspace.permissions
        return {
            "product": "Rabbit Code",
            "provider": {
                "id": provider.provider_id,
                "model": provider.model,
                "base_url": provider.base_url,
                "local": provider.local,
            },
            "permissions": {
                "read": policy.read.value,
                "search": policy.search.value,
                "write": policy.write.value,
                "execute": policy.execute.value,
                "network": policy.network.value,
            },
            "tools": {
                "read": True,
                "list": True,
                "grep": True,
                "write": True,
                "edit": True,
                "git_status": True,
                "git_diff": True,
                "sandbox_execute": shutil.which("podman") is not None,
            },
            "model_protocol": {
                "streaming": False,
                "tool_calls": False,
            },
            "session": {
                "id": self.session.session_id,
                "training_allowed": self.session.training_allowed,
            },
        }

    def _tool(self, name, call_payload, action, result_mapper=None):
        self.session.record("tool_call", {"tool": name, **call_payload})
        try:
            result = action()
        except Exception as exc:
            self.session.record("failure", {
                "operation": f"tool.{name}",
                "error_type": type(exc).__name__,
                "message": str(exc),
            })
            raise
        saved_result = result_mapper(result) if result_mapper else result
        self.session.record("tool_result", {
            "tool": name,
            "result": saved_result,
        })
        return result
