from __future__ import annotations

import argparse
import errno
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "rabbit-code"
UPSTREAM = "http://127.0.0.1:11434/v1"
UPSTREAM_MODEL = "qwen2.5-coder:7b"


def request_json(url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body)
        except Exception:
            return exc.code, {"error": {"message": body.decode(errors="replace")}}


def stream_proxy(url: str, payload: dict):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=300)


class Handler(BaseHTTPRequestHandler):
    server_version = "RabbitCodeGateway/0.1"

    def _json(self, status: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _sse_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {
                "status": "ok",
                "mode": "rabbit-code-model-gateway",
                "model": MODEL_ID,
                "upstream": UPSTREAM,
                "streaming": True,
                "tools": False,
            })
            return

        if self.path == "/v1/models":
            self._json(200, {
                "object": "list",
                "data": [{
                    "id": MODEL_ID,
                    "object": "model",
                    "owned_by": "rabbit-code",
                }],
            })
            return

        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found"}})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": {"message": str(exc)}})
            return

        if request.get("tools"):
            self._json(400, {"error": {"message": "tool calling not enabled yet"}})
            return

        forwarded = dict(request)
        forwarded["model"] = UPSTREAM_MODEL

        if request.get("stream", False):
            self._handle_streaming(forwarded)
        else:
            self._handle_non_streaming(forwarded)

    def _handle_non_streaming(self, forwarded: dict) -> None:
        status, response = request_json(f"{UPSTREAM}/chat/completions", forwarded)
        if status < 400:
            response["model"] = MODEL_ID
        self._json(status, response)

    def _handle_streaming(self, forwarded: dict) -> None:
        self._sse_headers()
        try:
            upstream_resp = stream_proxy(f"{UPSTREAM}/chat/completions", forwarded)
            for raw_line in upstream_resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue

                data_content = line[6:].strip()
                if data_content == "[DONE]":
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return

                try:
                    parsed = json.loads(data_content)
                except json.JSONDecodeError:
                    continue

                if "model" in parsed:
                    parsed["model"] = MODEL_ID
                self.wfile.write(f"data: {json.dumps(parsed)}\n\n".encode())
                self.wfile.flush()
        except Exception as exc:
            error = {
                "error": {"message": f"gateway streaming error: {exc}"},
                "model": MODEL_ID,
            }
            self.wfile.write(f"data: {json.dumps(error)}\n\n".encode())
            self.wfile.flush()

    def log_message(self, fmt, *args):
        print("[rabbit-code-gateway]", fmt % args)


def main() -> None:
    global UPSTREAM, UPSTREAM_MODEL
    parser = argparse.ArgumentParser(description="Rabbit Code local model gateway")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--upstream", default=UPSTREAM)
    parser.add_argument("--upstream-model", default=UPSTREAM_MODEL)
    args = parser.parse_args()

    UPSTREAM = args.upstream.rstrip("/")
    UPSTREAM_MODEL = args.upstream_model

    print(f"Rabbit Code gateway: http://{args.host}:{args.port}")
    print(f"Upstream: {UPSTREAM} model={UPSTREAM_MODEL}")
    print("Streaming: enabled")
    print("Tools: disabled until verified")
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            raise SystemExit(
                f"Rabbit Code gateway cannot bind {args.host}:{args.port}: "
                "address already in use. Check /health or choose another port."
            ) from exc
        raise
    server.serve_forever()


if __name__ == "__main__":
    main()
