from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateScore:
    name: str
    validation_loss: float
    heldout_pass_rate: float
    finite: bool = True


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    winner: str | None
    reason: str


def decide(a: CandidateScore, b: CandidateScore, minimum_pass_rate: float = 0.0) -> PromotionDecision:
    valid = [
        x for x in (a, b)
        if x.finite and x.heldout_pass_rate >= minimum_pass_rate
    ]
    if not valid:
        return PromotionDecision(False, None, "no candidate passed Greenlight")
    winner = min(valid, key=lambda x: x.validation_loss)
    return PromotionDecision(True, winner.name, "best eligible held-out validation loss")
