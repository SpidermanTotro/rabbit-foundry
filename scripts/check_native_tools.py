"""Read-only live probe of native OpenAI-style tool calling through Rabbit Code.

No tool is executed, no workspace files are read, and no output is used for
training. Defaults to the local Rabbit Code gateway on 127.0.0.1:8765.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import sys
import urllib.error
import urllib.parse
import urllib.request


PROBE_TOKEN = "rabbit-native-tool-probe"
PROBE_NAME = "rabbit_probe"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code, "redirects disabled for local probe", headers, fp
        )


def local_endpoint(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return False
    host = parsed.hostname
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def probe_payload(model: str) -> dict:
    return {
        "model": model,
        "stream": False,
        "max_tokens": 192,
        "messages": [
            {
                "role": "system",
                "content": (
                    "This is a diagnostic of native function calls. "
                    "Call the supplied rabbit_probe function with the exact "
                    "token value. Do not answer in ordinary prose."
                ),
            },
            {
                "role": "user",
                "content": "Call rabbit_probe with token rabbit-native-tool-probe.",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": PROBE_NAME,
                    "description": "Record an inert local test token; no side effects.",
                    "parameters": {
                        "type": "object",
                        "properties": {"token": {"type": "string"}},
                        "required": ["token"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
    }


def inspect_response(body: dict) -> dict:
    if not isinstance(body, dict):
        return {"native_tool_calls": False, "reason": "invalid_response"}
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return {"native_tool_calls": False, "reason": "missing_choices"}
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    if not isinstance(message, dict):
        return {"native_tool_calls": False, "reason": "missing_message"}
    calls = message.get("tool_calls", [])
    if not isinstance(calls, list) or not calls:
        return {"native_tool_calls": False, "reason": "model_returned_no_native_calls"}
    for call in calls:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        if not isinstance(function, dict) or function.get("name") != PROBE_NAME:
            continue
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return {"native_tool_calls": False, "reason": "invalid_tool_arguments_json"}
        if isinstance(arguments, dict) and arguments.get("token") == PROBE_TOKEN:
            return {"native_tool_calls": True, "reason": "verified_tool_call"}
        return {"native_tool_calls": False, "reason": "wrong_tool_arguments"}
    return {"native_tool_calls": False, "reason": "wrong_tool_name"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe Rabbit Code native tool calls")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8765/v1/chat/completions",
    )
    parser.add_argument("--model", default="rabbit-code")
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args(argv)
    if not local_endpoint(args.endpoint):
        parser.error("endpoint must be localhost/loopback (local-only probe)")
    if args.timeout <= 0:
        parser.error("timeout must be positive")

    request = urllib.request.Request(
        args.endpoint,
        data=json.dumps(probe_payload(args.model)).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), NoRedirect()
    )
    try:
        with opener.open(request, timeout=args.timeout) as response:
            body = json.loads(response.read())
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        print(json.dumps({
            "native_tool_calls": False,
            "reason": "probe_request_failed",
            "error_type": type(exc).__name__,
        }))
        return 1

    report = inspect_response(body)
    report["model"] = args.model
    report["endpoint"] = args.endpoint
    print(json.dumps(report, indent=2))
    return 0 if report["native_tool_calls"] else 2


if __name__ == "__main__":
    sys.exit(main())
