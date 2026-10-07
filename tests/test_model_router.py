import json
import urllib.request

import pytest

from rabbit_foundry.model_router import (
    ModelRouter,
    ProviderConfig,
    assistant_text,
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


def test_router_refuses_unimplemented_tools_and_streaming():
    router = ModelRouter()
    router.register(ProviderConfig(
        "local",
        "http://localhost:8765/v1",
        "rabbit-code",
    ))
    with pytest.raises(NotImplementedError):
        router.complete(
            "local",
            [{"role": "user", "content": "hello"}],
            tools=[{"type": "function"}],
        )
    with pytest.raises(NotImplementedError):
        router.complete(
            "local",
            [{"role": "user", "content": "hello"}],
            stream=True,
        )
