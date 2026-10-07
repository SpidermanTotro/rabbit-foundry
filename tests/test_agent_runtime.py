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
            supports_streaming=True,
            supports_tools=True,
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
    assert runtime.capabilities()["model_protocol"]["tool_calls"] is True
    assert runtime.capabilities()["model_protocol"]["streaming"] is True

    rows = [json.loads(line) for line in runtime.session.path.read_text().splitlines()]
    kinds = [row["kind"] for row in rows]
    assert kinds[:2] == ["user", "assistant"]
    assert "tool_call" in kinds
    assert "tool_result" in kinds


class HistoryRouter(FakeRouter):
    def __init__(self):
        super().__init__()
        self.calls = []

    def complete(self, provider_id, messages, *, max_tokens=1024):
        self.calls.append(messages)
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": f"answer-{len(self.calls)}",
                }
            }]
        }


def test_runtime_preserves_multi_turn_chat_history(tmp_path):
    router = HistoryRouter()
    store = SessionStore(tmp_path / "sessions", session_id="history")
    first = RabbitCodeRuntime(
        router,
        "fake",
        Workspace(tmp_path, PermissionPolicy()),
        store,
    )
    assert first.ask("one") == "answer-1"
    assert first.ask("two") == "answer-2"
    assert {"role": "assistant", "content": "answer-1"} in router.calls[1]

    resumed_router = HistoryRouter()
    resumed = RabbitCodeRuntime(
        resumed_router,
        "fake",
        Workspace(tmp_path, PermissionPolicy()),
        SessionStore(tmp_path / "sessions", session_id="history"),
    )
    assert resumed.session.resumed is True
    assert resumed.history[-1] == {
        "role": "assistant",
        "content": "answer-2",
    }


class StreamRouter(FakeRouter):
    def stream_text(self, provider_id, messages, *, max_tokens=1024):
        assert provider_id == "fake"
        assert messages[-1]["content"] == "stream hello"
        yield "Rabbit "
        yield "stream"


def test_runtime_streaming_chat_is_captured_and_added_to_history(tmp_path):
    runtime = RabbitCodeRuntime(
        StreamRouter(),
        "fake",
        Workspace(tmp_path, PermissionPolicy()),
        SessionStore(tmp_path / "sessions", session_id="stream"),
    )
    assert "".join(runtime.ask_stream("stream hello")) == "Rabbit stream"
    assert runtime.history[-1] == {
        "role": "assistant",
        "content": "Rabbit stream",
    }
    rows = [json.loads(line) for line in runtime.session.path.read_text().splitlines()]
    assert rows[-1]["payload"]["streamed"] is True
