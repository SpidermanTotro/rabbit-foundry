from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CandidateScore:
    name: str
    validation_loss: float
    heldout_pass_rate: float
    finite: bool = True
    behavior_score: float = 1.0


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    winner: str | None
    reason: str


def decide(
    a: CandidateScore,
    b: CandidateScore,
    minimum_pass_rate: float = 0.0,
    minimum_relative_improvement: float = 0.0,
    minimum_behavior_score: float = 0.0,
) -> PromotionDecision:
    valid = [
        x for x in (a, b)
        if x.finite
        and math.isfinite(x.validation_loss)
        and math.isfinite(x.heldout_pass_rate)
        and math.isfinite(x.behavior_score)
        and x.heldout_pass_rate >= minimum_pass_rate
        and x.behavior_score >= minimum_behavior_score
    ]
    if not valid:
        return PromotionDecision(False, None, "no candidate passed Greenlight")

    ranked = sorted(valid, key=lambda x: (-x.behavior_score, x.validation_loss))
    winner = ranked[0]
    if minimum_relative_improvement > 0 and len(ranked) > 1:
        runner_up = ranked[1]
        if runner_up.validation_loss <= 0:
            return PromotionDecision(False, None, "Greenlight improvement threshold not measurable")
        relative = (
            runner_up.validation_loss - winner.validation_loss
        ) / runner_up.validation_loss
        if relative < minimum_relative_improvement:
            return PromotionDecision(
                False, None,
                "best candidate did not clear Greenlight relative-improvement threshold",
            )
    return PromotionDecision(True, winner.name, "best eligible behavioral score and held-out validation loss")
