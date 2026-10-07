from rabbit_foundry.greenlight import CandidateScore, decide


def test_greenlight_selects_lower_loss_eligible_candidate():
    result = decide(
        CandidateScore("A", 2.0, 1.0),
        CandidateScore("B", 1.5, 1.0),
    )
    assert result.promoted
    assert result.winner == "B"


def test_greenlight_rejects_nonfinite_candidates():
    result = decide(
        CandidateScore("A", 1.0, 1.0, finite=False),
        CandidateScore("B", 1.0, 1.0, finite=False),
    )
    assert not result.promoted


def test_greenlight_rejects_nonfinite_metric_even_if_flag_is_true():
    result = decide(
        CandidateScore("A", float("nan"), 1.0, finite=True),
        CandidateScore("B", float("inf"), 1.0, finite=True),
    )
    assert not result.promoted


def test_greenlight_can_require_material_relative_improvement():
    result = decide(
        CandidateScore("A", 0.4000, 1.0),
        CandidateScore("B", 0.3999, 1.0),
        minimum_relative_improvement=0.01,
    )
    assert not result.promoted
    assert result.winner is None
