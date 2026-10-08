import io
import json
import threading
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer

import rabbit_foundry.model_gateway as model_gateway
from rabbit_foundry.model_gateway import GATEWAY_PROTOCOL, Handler, MODEL_ID


def serve():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_gateway_advertises_current_protocol_and_model():
    server, thread = serve()
    try:
        host, port = server.server_address
        with urllib.request.urlopen(
            f"http://{host}:{port}/health",
            timeout=5,
        ) as response:
            health = json.loads(response.read())
        assert health["status"] == "ok"
        assert health["protocol_version"] == GATEWAY_PROTOCOL == 2
        assert health["tools"] is True
        assert health["streaming"] is True

        with urllib.request.urlopen(
            f"http://{host}:{port}/v1/models",
            timeout=5,
        ) as response:
            body = json.loads(response.read())
        assert MODEL_ID == "rabbit-code"
        assert body["data"][0]["id"] == "rabbit-code"
        assert body["data"][0]["owned_by"] == "rabbit-code"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_gateway_passes_native_tools_to_upstream(monkeypatch):
    seen = {}

    def fake_request_json(url, payload=None):
        seen["url"] = url
        seen["payload"] = payload
        return 200, {
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
        }

    monkeypatch.setattr(model_gateway, "request_json", fake_request_json)
    server, thread = serve()
    try:
        host, port = server.server_address
        tools = [{
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {"type": "object"},
            },
        }]
        payload = json.dumps({
            "model": "rabbit-code",
            "messages": [{"role": "user", "content": "inspect"}],
            "tools": tools,
            "stream": False,
        }).encode()
        request = urllib.request.Request(
            f"http://{host}:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = json.loads(response.read())

        assert seen["url"].endswith("/v1/chat/completions")
        assert seen["payload"]["model"] == model_gateway.UPSTREAM_MODEL
        assert seen["payload"]["tools"] == tools
        assert body["model"] == "rabbit-code"
        assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "read"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_gateway_returns_upstream_stream_http_error_before_sse(monkeypatch):
    def reject_stream(url, payload):
        raise urllib.error.HTTPError(
            url, 503, "Service Unavailable", {},
            io.BytesIO(b'{"error":{"message":"model unavailable"}}'),
        )

    monkeypatch.setattr(model_gateway, "stream_proxy", reject_stream)
    server, thread = serve()
    try:
        host, port = server.server_address
        payload = json.dumps({
            "model": "rabbit-code",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }).encode()
        request = urllib.request.Request(
            f"http://{host}:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as exc:
            assert exc.code == 503
            assert exc.headers.get("Content-Type") == "application/json"
            body = json.loads(exc.read())
            assert body["error"]["message"] == "model unavailable"
        else:
            raise AssertionError("gateway must report upstream 503")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)



def test_gateway_returns_502_when_nonstreaming_upstream_is_unreachable(monkeypatch):
    def unreachable(url, payload=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(model_gateway, "request_json", unreachable)
    server, thread = serve()
    try:
        host, port = server.server_address
        payload = json.dumps({
            "model": "rabbit-code",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }).encode()
        request = urllib.request.Request(
            f"http://{host}:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as exc:
            assert exc.code == 502
            body = json.loads(exc.read())
            assert "unavailable" in body["error"]["message"]
        else:
            raise AssertionError("unavailable upstream must return HTTP 502")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
