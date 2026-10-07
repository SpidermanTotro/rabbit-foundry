from __future__ import annotations

import os
import urllib.error

from .space_bunny_routes import SpaceBunnyRoute, routes_by_preference


RETRYABLE_STATUS = {404, 408, 429, 500, 502, 503, 504}


def authorized_routes(preferred: str = "zen") -> tuple[SpaceBunnyRoute, ...]:
    return tuple(
        route for route in routes_by_preference(preferred)
        if os.environ.get(route.api_key_env)
    )


def run_with_failover(run_one, *, preferred: str = "zen"):
    routes = authorized_routes(preferred)
    if not routes:
        raise RuntimeError(
            "No authorized Space Bunny route. Set OPENCODE_ZEN_API_KEY or OPENCODE_GO_API_KEY."
        )
    failures = []
    for route in routes:
        try:
            result = run_one(route, os.environ[route.api_key_env])
            return result, route
        except urllib.error.HTTPError as exc:
            failures.append((route.name, exc.code))
            if exc.code not in RETRYABLE_STATUS:
                raise
        except (TimeoutError, ConnectionError) as exc:
            failures.append((route.name, type(exc).__name__))
    raise RuntimeError(f"all authorized Space Bunny routes failed: {failures}")
