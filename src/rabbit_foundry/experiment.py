from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .curriculum import Curriculum, Outcome


@dataclass
class ArmResult:
    name: str
    compute_steps: int
    score: float
    curriculum: dict


class SamplingExperiment:
    """Ledger for an equal-compute fixed-vs-adaptive comparison."""

    def __init__(self, compute_steps: int):
        if compute_steps <= 0:
            raise ValueError("compute_steps must be positive")
        self.compute_steps = compute_steps
        self.curriculum = Curriculum()
        self.results: list[ArmResult] = []

    def observe_adaptive(self, skill: str, passed: bool) -> None:
        self.curriculum.observe([Outcome(skill, passed)])

    def record_fixed(self, score: float) -> None:
        self.results.append(ArmResult("fixed", self.compute_steps, score, {}))

    def record_adaptive(self, score: float) -> None:
        self.results.append(
            ArmResult("adaptive", self.compute_steps, score, self.curriculum.state())
        )

    def winner(self) -> str | None:
        if len(self.results) < 2:
            return None
        # Score convention: larger is better.
        return max(self.results, key=lambda r: r.score).name

    def write(self, path: str | Path) -> None:
        payload = {
            "compute_steps_per_arm": self.compute_steps,
            "results": [asdict(r) for r in self.results],
            "winner": self.winner(),
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")
