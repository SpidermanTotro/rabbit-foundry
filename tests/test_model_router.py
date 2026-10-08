import json
import urllib.request

import pytest

from rabbit_foundry.model_router import (
    ModelRouter,
    ProviderConfig,
    assistant_text,
    assistant_tool_calls,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class FakeStreamResponse:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self.lines)


def test_router_blocks_nonlocal_provider_without_network_permission():
    router = ModelRouter()
    router.register(ProviderConfig("remote", "https://example.com/v1", "model"))
    with pytest.raises(PermissionError):
        router.complete("remote", [{"role": "user", "content": "hello"}])


def test_router_calls_loopback_provider_and_extracts_text(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        return FakeResponse({
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        })

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    router = ModelRouter()
    router.register(ProviderConfig(
        "local",
        "http://127.0.0.1:8765/v1",
        "rabbit-code",
    ))
    response = router.complete(
        "local",
        [{"role": "user", "content": "hello"}],
    )
    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["body"]["model"] == "rabbit-code"
    assert assistant_text(response) == "ok"


def test_router_sends_native_tools_when_enabled(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["body"] = json.loads(request.data)
        return FakeResponse({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"path":"README.md"}',
                        },
                    }],
                }
            }]
        })

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    router = ModelRouter()
    router.register(ProviderConfig(
        "local",
        "http://127.0.0.1:8765/v1",
        "rabbit-code",
        supports_tools=True,
    ))
    tools = [{
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {"type": "object"},
        },
    }]
    response = router.complete(
        "local",
        [{"role": "user", "content": "inspect"}],
        tools=tools,
    )
    assert seen["body"]["tools"] == tools
    assert assistant_text(response) == ""
    assert assistant_tool_calls(response)[0]["function"]["name"] == "read"


def test_router_streams_openai_sse_text(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["body"] = json.loads(request.data)
        return FakeStreamResponse([
            b'data: {"choices":[{"delta":{"content":"Rab"}}]}\n',
            b'\n',
            b'data: {"choices":[{"delta":{"content":"bit"}}]}\n',
            b'data: [DONE]\n',
        ])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    router = ModelRouter()
    router.register(ProviderConfig(
        "local",
        "http://localhost:8765/v1",
        "rabbit-code",
        supports_streaming=True,
    ))
    assert "".join(router.stream_text(
        "local",
        [{"role": "user", "content": "hello"}],
    )) == "Rabbit"
    assert seen["body"]["stream"] is True


def test_router_respects_declared_capabilities():
    router = ModelRouter()
    router.register(ProviderConfig(
        "local",
        "http://localhost:8765/v1",
        "rabbit-code",
        supports_streaming=False,
        supports_tools=False,
    ))
    with pytest.raises(NotImplementedError):
        router.complete(
            "local",
            [{"role": "user", "content": "hello"}],
            tools=[{"type": "function"}],
        )
    with pytest.raises(NotImplementedError):
        list(router.stream_text(
            "local",
            [{"role": "user", "content": "hello"}],
        ))
    with pytest.raises(ValueError):
        router.complete(
            "local",
            [{"role": "user", "content": "hello"}],
            stream=True,
        )


def test_router_probe_reads_gateway_health_from_v1_base(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        return FakeResponse({
            "status": "ok",
            "mode": "rabbit-code-model-gateway",
            "protocol_version": 2,
            "model": "rabbit-code",
            "streaming": True,
            "tools": True,
        })

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    router = ModelRouter()
    router.register(ProviderConfig(
        "local",
        "http://127.0.0.1:8765/v1",
        "rabbit-code",
        supports_tools=True,
    ))
    result = router.probe("local")
    assert seen["url"] == "http://127.0.0.1:8765/health"
    assert result["reachable"] is True
    assert result["health"]["protocol_version"] == 2
    assert result["health"]["tools"] is True


def test_router_stream_error_event_does_not_become_empty_success(monkeypatch):
    def fake_urlopen(request, timeout):
        return FakeStreamResponse([
            b'data: {"error":{"message":"gateway streaming interrupted"}}\n',
            b'\n',
        ])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    router = ModelRouter()
    router.register(ProviderConfig(
        "local",
        "http://127.0.0.1:8765/v1",
        "rabbit-code",
        supports_streaming=True,
    ))
    with pytest.raises(RuntimeError, match="streaming interrupted"):
        list(router.stream_text(
            "local", [{"role": "user", "content": "hello"}]
        ))
