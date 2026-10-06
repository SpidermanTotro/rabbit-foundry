from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Route(str, Enum):
    RABBIT = "rabbit"
    FALLBACK = "fallback"
    DENY = "deny"


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    reason: str


@dataclass
class RouterPolicy:
    fallback_model: str | None = None
    allow_fallback: bool = False
    minimum_confidence: float = 0.55

    def decide(self, *, confidence: float, rabbit_failed: bool = False) -> RouteDecision:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not rabbit_failed and confidence >= self.minimum_confidence:
            return RouteDecision(Route.RABBIT, "rabbit accepted")
        if self.allow_fallback and self.fallback_model:
            return RouteDecision(Route.FALLBACK, "rabbit below gate; fallback permitted")
        return RouteDecision(Route.DENY, "rabbit below gate; fallback not permitted")
