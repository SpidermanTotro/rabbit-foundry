from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from .environment import detect_environment
from .model import ModelConfig, TinyRabbitLM
from .provider_capture import redact_text


class RabbitNative:
    def __init__(self, checkpoint: str | Path, capture_path: str | Path | None = None):
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
            raise ValueError("checkpoint must contain config")
        self.model = TinyRabbitLM(ModelConfig(**payload["config"]))
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.eval()
        self.capture_path = Path(capture_path) if capture_path else None
        self.environment = detect_environment()

    @torch.no_grad()
    def complete(self, messages: list[dict], max_tokens: int = 128) -> str:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        parts = []
        safe_messages = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError("each message needs string role/content")
            content = redact_text(content)
            safe_messages.append({"role": role, "content": content})
            parts.append(f"<|{role}|>\n{content}\n")
        parts.append("<|assistant|>\n")
        ids = list("".join(parts).encode("utf-8"))[-self.model.cfg.context:]
        generated = []
        for _ in range(max(1, min(int(max_tokens), 2048))):
            x = torch.tensor([ids[-self.model.cfg.context:]], dtype=torch.long)
            logits, _ = self.model(x)
            token = int(torch.argmax(logits[0, -1]).item())
            generated.append(token)
            ids.append(token)
        text = redact_text(bytes(generated).decode("utf-8", errors="replace"))
        self._capture(safe_messages, text)
        return text

    def _capture(self, messages: list[dict], response: str) -> None:
        if self.capture_path is None:
            return
        self.capture_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "id": str(uuid.uuid4()),
            "axis": "tool",
            "provider": "rabbit-native",
            "environment": self.environment,
            "messages": messages + [{"role": "assistant", "content": response}],
        }
        with self.capture_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


def handler_for(preview: RabbitNative):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": "rabbit-native", "object": "model", "owned_by": "rabbit-foundry"}
                ]})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                answer = preview.complete(
                    payload.get("messages"),
                    max_tokens=payload.get("max_tokens", 128),
                )
                now = int(time.time())
                self._json(200, {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": now,
                    "model": "rabbit-native",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "length",
                    }],
                })
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": {"message": str(exc)}})

        def log_message(self, format, *args):
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Local OpenAI-compatible Rabbit Native server")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--capture", default="runs/preview/captures.jsonl")
    args = parser.parse_args()
    preview = RabbitNative(args.checkpoint, args.capture)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(preview))
    print(f"Rabbit Native listening on http://{args.host}:{args.port}/v1")
    server.serve_forever()


if __name__ == "__main__":
    main()
