"""Local-only browser interface for the existing Rabbit Code runtime.

No cloud service, browser CDN, shell endpoint, or autonomous write approval.
The browser API binds to loopback, requires a per-process CSRF token for
mutations, checks Host/Origin, and serializes runtime/session operations.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

from .agent_loop import AgentLoop
from .permissions import Decision
from .session_store import SessionStore
from .ui_models import available_model_options, installed_ollama_models, select_model

MAX_REQUEST_BYTES = 256 * 1024
MAX_FILE_BYTES = 128 * 1024
MAX_MESSAGE_CHARS = 8192


def _as_message(row):
    payload = row.get("payload")
    if not isinstance(payload, dict):
        return None
    kind = row.get("kind")
    mode = payload.get("mode")
    if kind == "user" and mode in {"chat", "agent"}:
        role = "user"
    elif kind == "assistant" and mode == "chat":
        role = "assistant"
    elif kind == "ui_reply" and mode == "agent":
        role = "assistant"
    else:
        return None
    content = payload.get("content")
    return {"role": role, "mode": mode, "content": content} if isinstance(content, str) else None


def chat_messages(session):
    return [message for row in session.events()
            if (message := _as_message(row)) is not None][-200:]


def list_sessions(runtime):
    directory = runtime.session.directory
    result = []
    for path in directory.glob("*.jsonl"):
        if path.is_symlink() or not path.is_file():
            continue
        session_id = path.stem
        if not session_id.replace("-", "").replace("_", "").isalnum():
            continue
        result.append((path.stat().st_mtime, session_id))
    return [sid for _, sid in sorted(result, reverse=True)[:100]]


class RabbitWebServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, runtime):
        if address[0] != "127.0.0.1":
            raise ValueError("Rabbit Code UI must bind to IPv4 loopback")
        if runtime.workspace.permissions.write is not Decision.ASK:
            raise ValueError("UI requires approval-gated writes")
        if runtime.workspace.permissions.execute is not Decision.ASK:
            raise ValueError("UI requires approval-gated sandbox execution")
        if runtime.router.allow_network:
            raise ValueError("UI network routing must be disabled")
        if not runtime.router.provider(runtime.provider_id).local:
            raise ValueError("UI requires a loopback model endpoint")
        self.runtime = runtime
        self.initial_provider = runtime.router.provider(runtime.provider_id)
        self.active_model_choice = "original"
        self.csrf_token = secrets.token_urlsafe(32)
        self.runtime_lock = threading.RLock()
        super().__init__(address, RabbitWebHandler)


class RabbitWebHandler(BaseHTTPRequestHandler):
    server_version = "RabbitCodeUI/0.1"

    def _headers(self, code, kind, length):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; img-src 'self' data:; "
                         "script-src 'self' 'unsafe-inline'; "
                         "style-src 'self' 'unsafe-inline'; "
                         "connect-src 'self'; object-src 'none'; "
                         "frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()

    def _send(self, code, body, kind="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self._headers(code, kind, len(body))
        self.wfile.write(body)

    def _allowed_origin(self):
        port = self.server.server_port
        allowed = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
        host = self.headers.get("Host", "")
        if host not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in allowed:
            return False
        return True

    def _check_request(self, *, mutate=False):
        if not self._allowed_origin():
            self._send(403, {"error": "loopback Host/Origin required"})
            return False
        if mutate and not secrets.compare_digest(
            self.headers.get("X-Rabbit-Token", ""),
            self.server.csrf_token,
        ):
            self._send(403, {"error": "invalid UI token"})
            return False
        return True

    def _read_payload(self):
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("chunked request bodies are not supported")
        if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            raise ValueError("Content-Type must be application/json")
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if size < 2 or size > MAX_REQUEST_BYTES:
            raise ValueError("invalid request size")
        value = json.loads(self.rfile.read(size))
        if not isinstance(value, dict):
            raise ValueError("request body must be an object")
        return value

    def do_GET(self):
        if not self._check_request():
            return
        parsed = urllib.parse.urlsplit(self.path)
        try:
            with self.server.runtime_lock:
                runtime = self.server.runtime
                if parsed.path == "/":
                    template = resources.files("rabbit_foundry").joinpath(
                        "ui/index.html").read_text(encoding="utf-8")
                    page = template.replace("__RABBIT_CSRF_TOKEN__",
                                            self.server.csrf_token)
                    return self._send(200, page, "text/html; charset=utf-8")
                if parsed.path == "/api/state":
                    provider = runtime.router.provider(runtime.provider_id)
                    return self._send(200, {
                        "workspace": str(runtime.workspace.root),
                        "session": runtime.session.session_id,
                        "model": provider.model,
                        "provider": provider.provider_id,
                        "local": provider.local,
                        "messages": chat_messages(runtime.session),
                    })
                if parsed.path == "/api/models":
                    return self._send(200, {
                        "options": available_model_options(self.server.initial_provider),
                        "selected": self.server.active_model_choice,
                    })
                if parsed.path == "/api/health":
                    if self.server.active_model_choice.startswith("ollama:"):
                        current = runtime.router.provider(runtime.provider_id).model
                        return self._send(200, {
                            "reachable": current in installed_ollama_models(),
                            "upstream_model": current,
                            "protocol_version": None,
                        })
                    try:
                        probe = (
                            runtime.router.probe(runtime.provider_id)
                            if callable(getattr(runtime.router, "probe", None))
                            else {"reachable": False}
                        )
                    except (OSError, ValueError, RuntimeError) as exc:
                        probe = {"reachable": False, "error": str(exc)[:250]}
                    health = probe.get("health") if isinstance(probe, dict) else None
                    return self._send(200, {
                        "reachable": bool(probe.get("reachable")) if isinstance(probe, dict) else False,
                        "upstream_model": (
                            health.get("upstream_model", health.get("model"))
                            if isinstance(health, dict) else None
                        ),
                        "protocol_version": (
                            health.get("protocol_version")
                            if isinstance(health, dict) else None
                        ),
                    })
                if parsed.path == "/api/files":
                    return self._send(200, {"files": runtime.list("**/*")})
                if parsed.path == "/api/file":
                    query = urllib.parse.parse_qs(parsed.query)
                    path = query.get("path", [""])[0]
                    if not path or len(path) > 1024:
                        raise ValueError("file path is required")
                    runtime.read(path)  # enforce workspace read policy and audit
                    data = runtime.workspace.resolve(path).read_bytes()
                    if len(data) > MAX_FILE_BYTES or b"\x00" in data:
                        raise ValueError("file is too large or not a text file")
                    content = data.decode("utf-8")  # do not silently replace bytes
                    return self._send(200, {
                        "path": path,
                        "content": content,
                        "sha256": hashlib.sha256(data).hexdigest(),
                    })
                if parsed.path == "/api/git":
                    return self._send(200, {
                        "status": runtime.git_status()[:100000],
                        "diff": runtime.git_diff()[:100000],
                    })
                if parsed.path == "/api/sessions":
                    return self._send(200, {
                        "sessions": list_sessions(runtime),
                        "current": runtime.session.session_id,
                    })
            self._send(404, {"error": "not found"})
        except (ValueError, OSError, PermissionError, RuntimeError) as exc:
            self._send(400, {"error": str(exc)[:500]})

    def do_POST(self):
        if not self._check_request(mutate=True):
            return
        try:
            payload = self._read_payload()
            with self.server.runtime_lock:
                runtime = self.server.runtime
                if self.path == "/api/model":
                    config = select_model(
                        runtime, self.server.initial_provider, payload.get("selection")
                    )
                    self.server.active_model_choice = payload["selection"]
                    return self._send(200, {
                        "selected": self.server.active_model_choice,
                        "model": config.model,
                    })
                if self.path == "/api/send":
                    prompt = payload.get("prompt")
                    mode = payload.get("mode", "chat")
                    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_MESSAGE_CHARS:
                        raise ValueError("prompt must contain 1–8192 characters")
                    if mode not in {"chat", "agent"}:
                        raise ValueError("mode must be chat or agent")
                    if mode == "chat":
                        response = runtime.ask(prompt)
                    else:
                        response = AgentLoop(runtime, approve_write=False,
                                             approve_exec=False, max_steps=8).run(prompt)
                        runtime.session.record("ui_reply", {
                            "mode": "agent", "content": response,
                        })
                    return self._send(200, {"response": response,
                                            "session": runtime.session.session_id})
                if self.path == "/api/save":
                    if payload.get("approved") is not True:
                        raise PermissionError("explicit approval required")
                    path = payload.get("path")
                    content = payload.get("content")
                    expected = payload.get("expected_sha256")
                    if (not isinstance(path, str) or not path or len(path) > 1024
                            or not isinstance(content, str)
                            or len(content.encode("utf-8")) > MAX_FILE_BYTES
                            or not isinstance(expected, str)):
                        raise ValueError("invalid save request")
                    resolved = runtime.workspace.resolve(path)
                    if not resolved.is_file():
                        raise ValueError("web editor can only save existing files")
                    try:
                        resolved.relative_to(runtime.session.directory)
                    except ValueError:
                        pass
                    else:
                        raise PermissionError("session logs cannot be edited from UI")
                    runtime.read(path)  # enforce read policy and audit
                    current_bytes = resolved.read_bytes()
                    if len(current_bytes) > MAX_FILE_BYTES or b"\x00" in current_bytes:
                        raise ValueError("file is too large or not a text file")
                    current_bytes.decode("utf-8")
                    if hashlib.sha256(current_bytes).hexdigest() != expected:
                        return self._send(409, {"error": "file changed on disk; reopen before saving"})
                    runtime.write(path, content, approved=True)
                    return self._send(200, {"saved": path,
                        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()})
                if self.path == "/api/session/new":
                    runtime.session = SessionStore(runtime.session.directory)
                    runtime.history = []
                    return self._send(200, {"session": runtime.session.session_id})
                if self.path == "/api/session/resume":
                    sid = payload.get("session")
                    if not isinstance(sid, str) or sid not in list_sessions(runtime):
                        raise ValueError("unknown session")
                    runtime.session = SessionStore(runtime.session.directory, session_id=sid)
                    runtime.history = runtime.session.chat_history()
                    return self._send(200, {"session": sid})
            self._send(404, {"error": "not found"})
        except (ValueError, OSError, PermissionError, RuntimeError,
                json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)[:500]})

    def log_message(self, fmt, *args):
        print("[rabbit-ui]", fmt % args)


def serve_browser_ui(runtime, port=8766, *, open_browser=False):
    if not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    with RabbitWebServer(("127.0.0.1", port), runtime) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Rabbit Code Linux UI: {url}")
        print("Local-only interface; model inference uses the configured provider.")
        print("Stop with Ctrl-C. Terminal commands remain available in another shell.")
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
