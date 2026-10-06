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


def stream_proxy(url: str, payload: dict) -> urllib.request.addinfourl:
    """Create a streaming request to upstream and return the response object."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=300)


class Handler(BaseHTTPRequestHandler):
    server_version = "RabbitAbsorber/0.2"

    def _json(self, status: int, body: dict):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _sse_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {
                "status": "ok",
                "mode": "qwen-plumbing-test",
                "model": MODEL_ID,
                "upstream": UPSTREAM,
                "streaming": True,
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

        is_streaming = request.get("stream", False)
        has_tools = bool(request.get("tools"))

        # Tool calling still not enabled in this phase
        if has_tools:
            self._json(400, {"error": {"message": "tool calling not enabled yet"}})
            return

        forwarded = dict(request)
        forwarded["model"] = UPSTREAM_MODEL

        if is_streaming:
            self._handle_streaming(forwarded)
        else:
            self._handle_non_streaming(forwarded)

    def _handle_non_streaming(self, forwarded: dict):
        status, response = request_json(f"{UPSTREAM}/chat/completions", forwarded)

        # Hide the upstream model identity from clients; routing is Rabbit's concern.
        if status < 400:
            response["model"] = MODEL_ID
        self._json(status, response)

    def _handle_streaming(self, forwarded: dict):
        """Proxy streaming response from Ollama."""
        self._sse_headers()

        try:
            upstream_resp = stream_proxy(f"{UPSTREAM}/chat/completions", forwarded)

            # Stream each line from upstream - SSE format has "data: " prefix
            for raw_line in upstream_resp:
                line = raw_line.decode("utf-8", errors="replace")
                
                # SSE lines end with \n\n (blank line after data)
                if line.strip() == "":
                    # Pass through blank lines (SSE event separator)
                    self.wfile.write(b"\n")
                    self.wfile.flush()
                    continue
                
                if line.startswith("data: "):
                    data_content = line[6:].strip()  # Remove "data: " prefix
                    
                    if data_content == "[DONE]":
                        # End of stream
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        break
                    
                    try:
                        parsed = json.loads(data_content)
                        # Replace model ID in the stream
                        if "model" in parsed:
                            parsed["model"] = MODEL_ID
                        
                        rewritten = f"data: {json.dumps(parsed)}\n\n"
                        self.wfile.write(rewritten.encode("utf-8"))
                        self.wfile.flush()
                    except json.JSONDecodeError:
                        # Pass through non-JSON data lines
                        self.wfile.write(f"{line}\n".encode("utf-8"))
                        self.wfile.flush()
                else:
                    # Pass through other lines (comments, etc.)
                    self.wfile.write(f"{line}\n".encode("utf-8"))
                    self.wfile.flush()

        except urllib.error.HTTPError as exc:
            body = exc.read()
            try:
                error_data = json.loads(body)
            except Exception:
                error_data = {"error": {"message": body.decode(errors="replace")}}
            error_data["model"] = MODEL_ID
            self.wfile.write(f"data: {json.dumps(error_data)}\n\n".encode("utf-8"))
            self.wfile.flush()
        except Exception as e:
            print(f"[absorber] Streaming error: {e}")
            error_data = {"error": {"message": f"streaming error: {e}"}, "model": MODEL_ID}
            self.wfile.write(f"data: {json.dumps(error_data)}\n\n".encode("utf-8"))
            self.wfile.flush()

    def log_message(self, fmt, *args):
        print("[absorber]", fmt % args)


def main():
    global UPSTREAM, UPSTREAM_MODEL
    p = argparse.ArgumentParser(description="Rabbit Absorber proxy with streaming")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--upstream", default=UPSTREAM)
    p.add_argument("--upstream-model", default=UPSTREAM_MODEL)
    args = p.parse_args()
    UPSTREAM = args.upstream.rstrip("/")
    UPSTREAM_MODEL = args.upstream_model

    print(f"Rabbit Absorber: http://{args.host}:{args.port}")
    print(f"Upstream: {UPSTREAM} model={UPSTREAM_MODEL}")
    print(f"Streaming: enabled")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()