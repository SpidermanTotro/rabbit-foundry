import json

from rabbit_foundry.agent_runtime import RabbitCodeRuntime
from rabbit_foundry.model_router import ProviderConfig
from rabbit_foundry.permissions import PermissionPolicy
from rabbit_foundry.session_store import SessionStore
from rabbit_foundry.workspace import Workspace


class FakeRouter:
    def __init__(self):
        self.config = ProviderConfig(
            "fake",
            "http://127.0.0.1:8765/v1",
            "rabbit-code",
        )

    def provider(self, provider_id):
        assert provider_id == "fake"
        return self.config

    def complete(self, provider_id, messages, *, max_tokens=1024):
        assert provider_id == "fake"
        assert messages[-1]["content"] == "hello"
        return {
            "choices": [{
                "message": {"role": "assistant", "content": "hi from rabbit"}
            }]
        }


def test_runtime_chat_tool_events_and_capabilities(tmp_path):
    (tmp_path / "code.py").write_text("print('hello')\n")
    sessions = tmp_path / "sessions"
    runtime = RabbitCodeRuntime(
        FakeRouter(),
        "fake",
        Workspace(tmp_path, PermissionPolicy()),
        SessionStore(sessions, session_id="run"),
    )

    assert runtime.ask("hello") == "hi from rabbit"
    assert "print" in runtime.read("code.py")
    assert runtime.capabilities()["provider"]["local"] is True
    assert runtime.capabilities()["model_protocol"]["tool_calls"] is False

    rows = [json.loads(line) for line in runtime.session.path.read_text().splitlines()]
    kinds = [row["kind"] for row in rows]
    assert kinds[:2] == ["user", "assistant"]
    assert "tool_call" in kinds
    assert "tool_result" in kinds
