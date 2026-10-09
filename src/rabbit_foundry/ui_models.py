"""Safe model selection for the loopback-only Rabbit Code browser workspace.

Models are enumerated from the local Ollama API, never downloaded
automatically, and switching never accepts a caller-provided URL.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from .model_router import ProviderConfig

OLLAMA_TAGS = "http://127.0.0.1:11434/api/tags"
OLLAMA_V1 = "http://127.0.0.1:11434/v1"


def installed_ollama_models() -> list[str]:
    """Return installed local model identifiers or [] if Ollama is offline."""
    req = urllib.request.Request(OLLAMA_TAGS, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=2) as response:
            raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            return []
        data = json.loads(raw)
    except (urllib.error.URLError, TimeoutError, OSError,
            ValueError, UnicodeDecodeError):
        return []
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        return []
    names = set()
    for item in data["models"][:500]:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and 0 < len(name) <= 150:
            names.add(name)
    return sorted(names)[:200]


def available_model_options(initial_provider: ProviderConfig) -> list[dict]:
    options = [{
        "id": "original",
        "name": f"Startup provider · {initial_provider.model}",
        "type": "startup",
    }]
    options += [
        {"id": "ollama:" + name, "name": name, "type": "ollama"}
        for name in installed_ollama_models()
    ]
    return options


def select_model(runtime, initial_provider: ProviderConfig, selection: str):
    """Select only the original trusted provider or an installed local model."""
    if not isinstance(selection, str) or len(selection) > 160:
        raise ValueError("invalid model selection")
    if selection == "original":
        config = initial_provider
    elif selection.startswith("ollama:"):
        name = selection.removeprefix("ollama:")
        if name not in installed_ollama_models():
            raise ValueError("model is not installed or local Ollama is unavailable")
        config = ProviderConfig(
            provider_id=runtime.provider_id,
            base_url=OLLAMA_V1,
            model=name,
            supports_streaming=True,
            supports_tools=True,
        )
    else:
        raise ValueError("unknown model source")
    if not config.local or runtime.router.allow_network:
        raise PermissionError("browser model must remain on loopback")
    runtime.router.register(config)
    runtime.session.record("model_selected", {
        "model": config.model,
        "provider": config.provider_id,
        "local": True,
        "source": "startup" if selection == "original" else "ollama",
    })
    return config
