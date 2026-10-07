from __future__ import annotations

import ipaddress
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterator


def _is_loopback_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    if host is None:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class ProviderConfig:
    provider_id: str
    base_url: str
    model: str
    api_key_env: str | None = None
    timeout: float = 300.0
    supports_streaming: bool = True
    supports_tools: bool = False

    @property
    def local(self) -> bool:
        return _is_loopback_url(self.base_url)


class ModelRouter:
    def __init__(self, *, allow_network: bool = False):
        self.allow_network = allow_network
        self._providers: dict[str, ProviderConfig] = {}

    def register(self, config: ProviderConfig) -> None:
        if not config.provider_id:
            raise ValueError("provider_id must not be empty")
        parsed = urllib.parse.urlparse(config.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an http(s) URL")
        self._providers[config.provider_id] = config

    def provider(self, provider_id: str) -> ProviderConfig:
        try:
            return self._providers[provider_id]
        except KeyError as exc:
            raise KeyError(f"unknown provider: {provider_id}") from exc

    def _check_route(self, config: ProviderConfig) -> None:
        if not config.local and not self.allow_network:
            raise PermissionError(
                f"network access disabled for provider {config.provider_id}"
            )

    def _headers(self, config: ProviderConfig) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if config.api_key_env:
            secret = os.environ.get(config.api_key_env)
            if not secret:
                raise RuntimeError(
                    f"required API key environment variable is not set: "
                    f"{config.api_key_env}"
                )
            headers["Authorization"] = f"Bearer {secret}"
        return headers

    @staticmethod
    def _validate_messages(messages: list[dict]) -> None:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("messages must contain objects")
            if not isinstance(message.get("role"), str):
                raise ValueError("each message needs a string role")
            content = message.get("content")
            if content is not None and not isinstance(content, (str, list)):
                raise ValueError("message content must be string, list, or null")

    def _payload(
        self,
        config: ProviderConfig,
        messages: list[dict],
        *,
        max_tokens: int,
        stream: bool,
        tools: list[dict] | None,
    ) -> dict:
        self._validate_messages(messages)
        if tools and not config.supports_tools:
            raise NotImplementedError(
                f"provider {config.provider_id} has native tool calls disabled"
            )
        if stream and not config.supports_streaming:
            raise NotImplementedError(
                f"provider {config.provider_id} has streaming disabled"
            )
        payload = {
            "model": config.model,
            "messages": messages,
            "max_tokens": max(1, int(max_tokens)),
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        return payload

    def complete(
        self,
        provider_id: str,
        messages: list[dict],
        *,
        max_tokens: int = 1024,
        stream: bool = False,
        tools: list[dict] | None = None,
    ) -> dict:
        if stream:
            raise ValueError("use stream_text() for streaming responses")
        config = self.provider(provider_id)
        self._check_route(config)
        payload = self._payload(
            config,
            messages,
            max_tokens=max_tokens,
            stream=False,
            tools=tools,
        )
        request = urllib.request.Request(
            config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers=self._headers(config),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=config.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(
                f"provider {provider_id} returned HTTP {exc.code}: {body[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"provider {provider_id} is unreachable: {exc.reason}"
            ) from exc

    def stream_text(
        self,
        provider_id: str,
        messages: list[dict],
        *,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        config = self.provider(provider_id)
        self._check_route(config)
        payload = self._payload(
            config,
            messages,
            max_tokens=max_tokens,
            stream=True,
            tools=None,
        )
        request = urllib.request.Request(
            config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers=self._headers(config),
            method="POST",
        )
        try:
            response = urllib.request.urlopen(request, timeout=config.timeout)
            with response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = event.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    text = delta.get("content")
                    if isinstance(text, str) and text:
                        yield text
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(
                f"provider {provider_id} returned HTTP {exc.code}: {body[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"provider {provider_id} is unreachable: {exc.reason}"
            ) from exc


def assistant_message(response: dict) -> dict:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("provider response has no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("provider choice is malformed")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("provider response has no assistant message")
    return message


def assistant_text(response: dict) -> str:
    message = assistant_message(response)
    content = message.get("content")
    if content is None and isinstance(message.get("tool_calls"), list):
        return ""
    if not isinstance(content, str):
        raise ValueError("provider response has no assistant text")
    return content


def assistant_tool_calls(response: dict) -> list[dict]:
    message = assistant_message(response)
    calls = message.get("tool_calls")
    if calls is None:
        return []
    if not isinstance(calls, list):
        raise ValueError("provider tool_calls must be a list")
    return [call for call in calls if isinstance(call, dict)]
