from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "rabbit-absorber"
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


class Handler(BaseHTTPRequestHandler):
    server_version = "RabbitAbsorber/0.1"

    def _json(self, status: int, body: dict):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {
                "status": "ok",
                "mode": "qwen-plumbing-test",
                "model": MODEL_ID,
                "upstream": UPSTREAM,
            })
            return
        if self.path == "/v1/models":
            self._json(200, {
                "object": "list",
                "data": [{"id": MODEL_ID, "object": "model", "owned_by": "rabbit-foundry"}],
            })
            return
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found"}})
            return

        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")

        # First boot intentionally supports non-streaming text only.
        if request.get("stream"):
            self._json(400, {"error": {"message": "streaming not enabled in first boot"}})
            return
        if request.get("tools"):
            self._json(400, {"error": {"message": "tool calling not enabled in first boot"}})
            return

        forwarded = dict(request)
        forwarded["model"] = UPSTREAM_MODEL
        status, response = request_json(f"{UPSTREAM}/chat/completions", forwarded)

        # Hide the upstream model identity from clients; routing is Rabbit's concern.
        if status < 400:
            response["model"] = MODEL_ID
        self._json(status, response)

    def log_message(self, fmt, *args):
        print("[absorber]", fmt % args)


def main():
    global UPSTREAM, UPSTREAM_MODEL
    p = argparse.ArgumentParser(description="Rabbit Absorber first-boot proxy")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--upstream", default=UPSTREAM)
    p.add_argument("--upstream-model", default=UPSTREAM_MODEL)
    args = p.parse_args()
    UPSTREAM = args.upstream.rstrip("/")
    UPSTREAM_MODEL = args.upstream_model

    print(f"Rabbit Absorber: http://{args.host}:{args.port}")
    print(f"Upstream: {UPSTREAM} model={UPSTREAM_MODEL}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
