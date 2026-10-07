from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpaceBunnyRoute:
    name: str
    endpoint: str
    model: str
    api_key_env: str


ROUTES = (
    SpaceBunnyRoute(
        "zen",
        "https://opencode.ai/zen/v1/chat/completions",
        "space-bunny-free",
        "OPENCODE_ZEN_API_KEY",
    ),
    SpaceBunnyRoute(
        "go",
        "https://opencode.ai/zen/go/v1/chat/completions",
        "space-bunny-free",
        "OPENCODE_GO_API_KEY",
    ),
)


def routes_by_preference(preferred: str = "zen") -> tuple[SpaceBunnyRoute, ...]:
    if preferred not in {"zen", "go"}:
        raise ValueError("preferred route must be zen or go")
    return tuple(sorted(ROUTES, key=lambda route: route.name != preferred))
