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
