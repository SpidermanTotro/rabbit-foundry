from __future__ import annotations

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
        self.session.record("tool_call", {"tool": "read", "path": path})
        try:
            result = self.workspace.read_text(path)
        except Exception as exc:
            self._tool_failure("read", exc)
            raise
        self.session.record("tool_result", {
            "tool": "read",
            "path": path,
            "content": result,
        })
        return result

    def list(self, pattern: str = "**/*") -> list[str]:
        self.session.record("tool_call", {"tool": "list", "pattern": pattern})
        try:
            result = self.workspace.list_files(pattern)
        except Exception as exc:
            self._tool_failure("list", exc)
            raise
        self.session.record("tool_result", {"tool": "list", "items": result})
        return result

    def grep(self, needle: str, pattern: str = "**/*") -> list[dict]:
        self.session.record("tool_call", {
            "tool": "grep",
            "needle": needle,
            "pattern": pattern,
        })
        try:
            result = self.workspace.grep(needle, pattern)
        except Exception as exc:
            self._tool_failure("grep", exc)
            raise
        self.session.record("tool_result", {"tool": "grep", "matches": result})
        return result

    def write(self, path: str, content: str, *, approved: bool = False) -> str:
        self.session.record("tool_call", {"tool": "write", "path": path})
        try:
            saved = self.workspace.write_text(path, content, approved=approved)
        except Exception as exc:
            self._tool_failure("write", exc)
            raise
        relative = str(saved.relative_to(self.workspace.root))
        self.session.record("tool_result", {
            "tool": "write",
            "path": relative,
            "bytes": len(content.encode()),
        })
        return relative

    def _tool_failure(self, tool: str, exc: Exception) -> None:
        self.session.record("failure", {
            "operation": f"tool.{tool}",
            "error_type": type(exc).__name__,
            "message": str(exc),
        })
