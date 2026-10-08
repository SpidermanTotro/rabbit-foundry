"""Hermetic native-tool transport tests: router -> gateway -> mock Ollama.

No real model, external network access, or GPU is required.
"""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rabbit_foundry.model_gateway as gateway
from rabbit_foundry.agent_loop import AgentLoop
from rabbit_foundry.agent_runtime import RabbitCodeRuntime
from rabbit_foundry.model_router import ModelRouter, ProviderConfig
from rabbit_foundry.permissions import PermissionPolicy
from rabbit_foundry.session_store import SessionStore
from rabbit_foundry.workspace import Workspace


class MockOllama(BaseHTTPRequestHandler):
    requests: list[dict] = []
    scenario: str = "read"

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        self.requests.append(payload)
        if payload.get("stream"):
            chunks = [
                {"choices": [{"delta": {"content": "Rabbit "}}]},
                {"choices": [{"delta": {"content": "stream works"}}]},
            ]
            data = "".join(
                "data: " + json.dumps(item) + "\n\n" for item in chunks
            ) + "data: [DONE]\n\n"
            raw = data.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        tools = payload.get("tools")
        assert tools, "The agent must send OpenAI tool definitions"
        messages = payload["messages"]
        tool_messages = [m for m in messages if m["role"] == "tool"]
        if not tool_messages:
            if self.scenario == "read":
                function = {
                    "name": "read",
                    "arguments": json.dumps({"path": "README.md"}),
                }
            else:
                function = {
                    "name": "write",
                    "arguments": json.dumps({
                        "path": "must-not-exist.txt",
                        "content": "bad write",
                    }),
                }
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": function,
                }],
            }
        else:
            result = json.loads(tool_messages[-1]["content"])
            if self.scenario == "read":
                assert result["ok"] is True
                assert "RABBIT_MARKER_2026" in result["result"]
                answer = "Inspected README.md and found RABBIT_MARKER_2026."
            else:
                assert result["ok"] is False
                assert "ApprovalRequired" in result["error_type"]
                answer = "Write was denied by the Rabbit Code permission policy."
            msg = {"role": "assistant", "content": answer}

        raw = json.dumps({
            "model": "mock-ollama",
            "choices": [{"message": msg}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@contextmanager
def local_server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def stack(monkeypatch, scenario):
    # Subclass isolates per-test request history, even if tests are reordered.
    mock = type("PerTestMockOllama", (MockOllama,), {
        "requests": [],
        "scenario": scenario,
    })
    with local_server(mock) as upstream:
        monkeypatch.setattr(gateway, "UPSTREAM", upstream + "/v1")
        monkeypatch.setattr(gateway, "UPSTREAM_MODEL", "qwen2.5-coder:7b")
        with local_server(gateway.Handler) as gateway_url:
            yield gateway_url, mock


def runtime(tmp_path, gateway_url):
    router = ModelRouter()
    router.register(ProviderConfig(
        "rabbit-local",
        gateway_url + "/v1",
        "rabbit-code",
        supports_tools=True,
        supports_streaming=True,
    ))
    return RabbitCodeRuntime(
        router,
        "rabbit-local",
        Workspace(tmp_path, PermissionPolicy()),
        SessionStore(tmp_path / "sessions", session_id="integration"),
    )


def test_native_read_crosses_router_gateway_and_model(monkeypatch, tmp_path):
    (tmp_path / "README.md").write_text("RABBIT_MARKER_2026\n")
    with stack(monkeypatch, "read") as (gateway_url, mock):
        agent = AgentLoop(runtime(tmp_path, gateway_url), max_steps=4)
        result = agent.run("Read README.md and report its unique marker.")
        assert "RABBIT_MARKER_2026" in result
        assert len(mock.requests) == 2
        assert mock.requests[0]["model"] == "qwen2.5-coder:7b"
        assert any(
            item["function"]["name"] == "read"
            for item in mock.requests[0]["tools"]
        )
        assert any(
            m["role"] == "tool"
            and m["tool_call_id"] == "call-1"
            and "RABBIT_MARKER_2026" in m["content"]
            for m in mock.requests[1]["messages"]
        )


def test_native_write_is_denied_through_complete_stack(monkeypatch, tmp_path):
    with stack(monkeypatch, "write") as (gateway_url, mock):
        agent = AgentLoop(runtime(tmp_path, gateway_url), max_steps=4)
        result = agent.run("Attempt to write must-not-exist.txt")
        assert "denied" in result.lower()
        assert len(mock.requests) == 2
        assert not (tmp_path / "must-not-exist.txt").exists()
        tool_result = next(
            m for m in mock.requests[1]["messages"] if m["role"] == "tool"
        )
        assert json.loads(tool_result["content"])["ok"] is False


def test_streaming_crosses_router_gateway_and_model(monkeypatch, tmp_path):
    with stack(monkeypatch, "read") as (gateway_url, mock):
        client = runtime(tmp_path, gateway_url)
        chunks = list(client.ask_stream("Say the streaming phrase"))
        assert chunks == ["Rabbit ", "stream works"]
        assert client.history[-1]["content"] == "Rabbit stream works"
        assert mock.requests[0]["stream"] is True
        assert mock.requests[0]["model"] == "qwen2.5-coder:7b"
