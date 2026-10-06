from __future__ import annotations

from dataclasses import dataclass, field
import random


@dataclass
class SkillState:
    weight: float = 1.0
    attempts: int = 0
    failures: int = 0

    @property
    def failure_rate(self) -> float:
        return self.failures / self.attempts if self.attempts else 0.0


@dataclass
class AdaptiveSampler:
    """Small weakness-driven curriculum primitive for Foundry v0.1."""

    skills: dict[str, SkillState] = field(default_factory=dict)
    floor: float = 0.10

    def add(self, name: str, weight: float = 1.0) -> None:
        self.skills[name] = SkillState(weight=max(self.floor, weight))

    def record(self, name: str, passed: bool) -> None:
        state = self.skills[name]
        state.attempts += 1
        state.failures += int(not passed)
        # Failed skills become more likely; consistently passing skills cool down.
        target = 1.0 + 3.0 * state.failure_rate
        state.weight = max(self.floor, 0.75 * state.weight + 0.25 * target)

    def choose(self, rng: random.Random | None = None) -> str:
        if not self.skills:
            raise ValueError("no skills registered")
        rng = rng or random
        names = list(self.skills)
        weights = [self.skills[n].weight for n in names]
        return rng.choices(names, weights=weights, k=1)[0]

    def snapshot(self) -> dict:
        return {
            name: {
                "weight": state.weight,
                "attempts": state.attempts,
                "failures": state.failures,
                "failure_rate": state.failure_rate,
            }
            for name, state in self.skills.items()
        }
