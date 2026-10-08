"""HTTP integration tests for the local-only Rabbit Code Linux interface."""
from __future__ import annotations

import hashlib
import http.client
import json
import threading
from contextlib import contextmanager

import pytest

from rabbit_foundry.agent_runtime import RabbitCodeRuntime
from rabbit_foundry.model_router import ProviderConfig
from rabbit_foundry.permissions import PermissionPolicy
from rabbit_foundry.session_store import SessionStore
from rabbit_foundry.web_ui import RabbitWebServer, chat_messages
from rabbit_foundry.workspace import Workspace


class FakeRouter:
    allow_network = False

    def provider(self, provider_id):
        assert provider_id == "fake"
        return ProviderConfig(
            provider_id="fake",
            base_url="http://127.0.0.1:8765/v1",
            model="test-local",
            supports_tools=False,
        )

    def complete(self, provider_id, messages, *, max_tokens=1024, tools=None):
        return {"choices": [{
            "message": {"role": "assistant", "content": "Fake local answer"}
        }]}


@contextmanager
def ui_server(tmp_path):
    (tmp_path / "README.md").write_text("# Rabbit Code\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("answer = 42\n", encoding="utf-8")
    runtime = RabbitCodeRuntime(
        FakeRouter(),
        "fake",
        Workspace(tmp_path, PermissionPolicy()),
        SessionStore(tmp_path / ".rabbit-code" / "sessions", session_id="first"),
    )
    server = RabbitWebServer(("127.0.0.1", 0), runtime)
    runner = threading.Thread(target=server.serve_forever, daemon=True)
    runner.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        runner.join(timeout=5)


def req(server, method, path, payload=None, *, token=None, origin=None, host=None):
    port = server.server_port
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Host": host or "127.0.0.1:" + str(port)}
    if token is not None:
        headers["X-Rabbit-Token"] = token
    if origin is not None:
        headers["Origin"] = origin
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        data = response.read()
        content_type = response.getheader("Content-Type")
        status = response.status
        if content_type and "application/json" in content_type:
            data = json.loads(data)
        return status, data, dict(response.getheaders())
    finally:
        connection.close()


def test_ui_page_is_local_and_contains_interface(tmp_path):
    with ui_server(tmp_path) as server:
        status, page, headers = req(server, "GET", "/")
        assert status == 200
        assert b"Rabbit Code Assistant" in page
        assert b"Workspace explorer" in page
        assert server.csrf_token.encode() in page
        assert headers["X-Frame-Options"] == "DENY"
        assert "default-src 'self'" in headers["Content-Security-Policy"]
        assert server.server_address[0] == "127.0.0.1"


def test_state_and_file_read_are_bounded_to_workspace(tmp_path):
    with ui_server(tmp_path) as server:
        status, result, _ = req(server, "GET", "/api/state")
        assert status == 200
        assert result["provider"] == "fake"
        assert result["local"] is True
        assert result["session"] == "first"

        status, result, _ = req(server, "GET", "/api/files")
        assert status == 200
        assert "README.md" in result["files"]

        status, result, _ = req(server, "GET", "/api/file?path=README.md")
        assert status == 200
        assert result["content"] == "# Rabbit Code\n"

        status, result, _ = req(server, "GET", "/api/file?path=..%2Fetc%2Fpasswd")
        assert status == 400
        assert "escapes workspace" in result["error"]


def test_csrf_and_origin_checks_block_cross_site_calls(tmp_path):
    with ui_server(tmp_path) as server:
        status, _, _ = req(server, "GET", "/api/state", host="evil.invalid")
        assert status == 403
        status, _, _ = req(server, "GET", "/api/state",
                           origin="https://evil.invalid")
        assert status == 403
        status, _, _ = req(server, "POST", "/api/send",
                           payload={"prompt": "hello"})
        assert status == 403
        status, _, _ = req(server, "POST", "/api/send",
                           payload={"prompt": "hello"}, token="bad")
        assert status == 403
        status, result, _ = req(server, "POST", "/api/send",
                                payload={"prompt": "hello"},
                                token=server.csrf_token,
                                origin="http://127.0.0.1:" + str(server.server_port))
        assert status == 200
        assert result["response"] == "Fake local answer"
        status, state, _ = req(server, "GET", "/api/state")
        assert [m["role"] for m in state["messages"]] == ["user", "assistant"]


def test_file_save_requires_approval_and_matching_disk_hash(tmp_path):
    original = "answer = 42\n"
    with ui_server(tmp_path) as server:
        data = {"path": "app.py", "content": "answer = 43\n",
                "expected_sha256": hashlib.sha256(original.encode()).hexdigest()}
        status, _, _ = req(server, "POST", "/api/save", payload=data,
                           token=server.csrf_token)
        assert status == 400
        assert (tmp_path / "app.py").read_text() == original
        data["approved"] = True
        bad = dict(data, expected_sha256="f" * 64)
        status, _, _ = req(server, "POST", "/api/save", payload=bad,
                           token=server.csrf_token)
        assert status == 409
        status, result, _ = req(server, "POST", "/api/save", payload=data,
                                token=server.csrf_token)
        assert status == 200
        assert result["saved"] == "app.py"
        assert (tmp_path / "app.py").read_text() == "answer = 43\n"


def test_session_new_and_resume(tmp_path):
    with ui_server(tmp_path) as server:
        status, result, _ = req(server, "POST", "/api/send",
                               payload={"prompt": "hello"}, token=server.csrf_token)
        assert status == 200
        status, result, _ = req(server, "POST", "/api/session/new",
                               payload={}, token=server.csrf_token)
        assert status == 200
        assert result["session"] != "first"
        status, state, _ = req(server, "GET", "/api/state")
        assert state["messages"] == []
        status, result, _ = req(server, "POST", "/api/session/resume",
                               payload={"session": "first"}, token=server.csrf_token)
        assert status == 200
        status, state, _ = req(server, "GET", "/api/state")
        assert state["session"] == "first"
        assert len(state["messages"]) == 2


def test_ui_refuses_unsafe_permission_configurations(tmp_path):
    from rabbit_foundry.permissions import Decision
    from rabbit_foundry.model_router import ModelRouter
    runtime = RabbitCodeRuntime(
        FakeRouter(), "fake",
        Workspace(tmp_path, PermissionPolicy(write=Decision.ALLOW)),
        SessionStore(tmp_path / "sessions"),
    )
    with pytest.raises(ValueError, match="approval-gated writes"):
        RabbitWebServer(("127.0.0.1", 0), runtime)

    router = ModelRouter(allow_network=True)
    router.register(ProviderConfig("fake", "http://127.0.0.1:8765/v1", "fake"))
    runtime = RabbitCodeRuntime(
        router, "fake", Workspace(tmp_path, PermissionPolicy()),
        SessionStore(tmp_path / "sessions"),
    )
    with pytest.raises(ValueError, match="network routing"):
        RabbitWebServer(("127.0.0.1", 0), runtime)


def test_agent_reply_visible_in_session_history(tmp_path):
    with ui_server(tmp_path) as server:
        server.runtime.session.record("user", {
            "content": "inspect this", "mode": "agent",
        })
        server.runtime.session.record("assistant", {
            "content": "internal protocol", "mode": "agent",
        })
        server.runtime.session.record("ui_reply", {
            "content": "read-only answer", "mode": "agent",
        })
        assert chat_messages(server.runtime.session) == [
            {"role": "user", "content": "inspect this", "mode": "agent"},
            {"role": "assistant", "content": "read-only answer", "mode": "agent"},
        ]
