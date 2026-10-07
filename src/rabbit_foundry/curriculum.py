from __future__ import annotations

from dataclasses import dataclass
import random

from .sampler import AdaptiveSampler


@dataclass(frozen=True)
class Outcome:
    skill: str
    passed: bool


class Curriculum:
    """Failure-driven skill scheduler.

    It changes *which kind* of episode is sampled, not the evaluation set.
    """

    def __init__(self, skills=("code_prediction", "code_repair", "hidden_diff")):
        self.sampler = AdaptiveSampler()
        for skill in skills:
            self.sampler.add(skill)

    def observe(self, outcomes: list[Outcome]) -> None:
        for outcome in outcomes:
            if outcome.skill not in self.sampler.skills:
                self.sampler.add(outcome.skill)
            self.sampler.record(outcome.skill, outcome.passed)

    def choose_skill(self, seed: int | None = None) -> str:
        return self.sampler.choose(random.Random(seed) if seed is not None else None)

    def state(self) -> dict:
        return self.sampler.snapshot()

    def load_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            raise ValueError("curriculum state must be a dictionary")
        expected = set(self.sampler.skills)
        if set(state) != expected:
            raise ValueError("curriculum state skills do not match current training skills")
        for name, saved in state.items():
            if not isinstance(saved, dict):
                raise ValueError(f"invalid curriculum state for {name}")
            current = self.sampler.skills[name]
            weight = float(saved.get("weight", 1.0))
            attempts = int(saved.get("attempts", 0))
            failures = int(saved.get("failures", 0))
            if weight < self.sampler.floor or attempts < 0 or failures < 0 or failures > attempts:
                raise ValueError(f"invalid curriculum counters for {name}")
            current.weight = weight
            current.attempts = attempts
            current.failures = failures
