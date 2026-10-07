from __future__ import annotations

import ipaddress
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


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

    def complete(
        self,
        provider_id: str,
        messages: list[dict],
        *,
        max_tokens: int = 1024,
        stream: bool = False,
        tools: list[dict] | None = None,
    ) -> dict:
        config = self.provider(provider_id)
        if not config.local and not self.allow_network:
            raise PermissionError(
                f"network access disabled for provider {provider_id}"
            )
        if stream:
            raise NotImplementedError("Rabbit Code router streaming is not wired yet")
        if tools:
            raise NotImplementedError("Rabbit Code router tool calls are not wired yet")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("messages must contain objects")
            if not isinstance(message.get("role"), str) or not isinstance(
                message.get("content"), str
            ):
                raise ValueError("each message needs string role/content")

        headers = {"Content-Type": "application/json"}
        if config.api_key_env:
            secret = os.environ.get(config.api_key_env)
            if not secret:
                raise RuntimeError(
                    f"required API key environment variable is not set: "
                    f"{config.api_key_env}"
                )
            headers["Authorization"] = f"Bearer {secret}"

        payload = {
            "model": config.model,
            "messages": messages,
            "max_tokens": max(1, int(max_tokens)),
            "stream": False,
        }
        request = urllib.request.Request(
            config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
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


def assistant_text(response: dict) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("provider response has no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("provider choice is malformed")
    message = first.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("provider response has no assistant text")
    return message["content"]
