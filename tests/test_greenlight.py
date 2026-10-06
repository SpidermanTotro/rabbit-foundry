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
