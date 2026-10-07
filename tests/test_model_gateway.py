import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from rabbit_foundry.model_gateway import Handler, MODEL_ID


def serve():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_gateway_advertises_rabbit_code_model():
    server, thread = serve()
    try:
        host, port = server.server_address
        with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=5) as response:
            body = json.loads(response.read())
        assert MODEL_ID == "rabbit-code"
        assert body["data"][0]["id"] == "rabbit-code"
        assert body["data"][0]["owned_by"] == "rabbit-code"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_gateway_rejects_tool_requests_until_supported():
    server, thread = serve()
    try:
        host, port = server.server_address
        payload = json.dumps({
            "model": "rabbit-code",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "noop"}}],
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
            assert exc.code == 400
            body = json.loads(exc.read())
        else:
            raise AssertionError("tool requests must fail closed")

        assert "tool calling not enabled" in body["error"]["message"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
