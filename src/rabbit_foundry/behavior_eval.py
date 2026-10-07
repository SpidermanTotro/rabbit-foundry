from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BehaviorScore:
    failure_detection: float
    diagnosis: float
    revision: float
    testing: float

    def __post_init__(self):
        values = (self.failure_detection, self.diagnosis, self.revision, self.testing)
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError("behavior scores must be finite values between 0 and 1")

    @property
    def overall(self) -> float:
        # Diagnosis/revision are the core self-correction signal; testing protects
        # against plausible-looking repairs that were never verified.
        return (
            self.failure_detection * 0.20
            + self.diagnosis * 0.30
            + self.revision * 0.30
            + self.testing * 0.20
        )


def score_trajectory_events(events: list[dict]) -> BehaviorScore:
    if not isinstance(events, list) or not events:
        raise ValueError("trajectory events are required")
    kinds = {
        str(event.get("kind", "")).lower()
        for event in events
        if isinstance(event, dict)
    }
    return BehaviorScore(
        failure_detection=1.0 if "failure" in kinds else 0.0,
        diagnosis=1.0 if "diagnosis" in kinds else 0.0,
        revision=1.0 if "revision" in kinds else 0.0,
        testing=1.0 if "test" in kinds else 0.0,
    )


def aggregate_behavior(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("at least one trajectory is required")
    scores = [score_trajectory_events(row.get("trajectory_events")) for row in rows]
    overall = sum(score.overall for score in scores) / len(scores)
    return {
        "trajectories": len(scores),
        "behavior_score": overall,
        "failure_detection": sum(s.failure_detection for s in scores) / len(scores),
        "diagnosis": sum(s.diagnosis for s in scores) / len(scores),
        "revision": sum(s.revision for s in scores) / len(scores),
        "testing": sum(s.testing for s in scores) / len(scores),
    }
