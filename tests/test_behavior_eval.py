from rabbit_foundry.behavior_eval import aggregate_behavior, score_trajectory_events


def test_complete_repair_trajectory_scores_one():
    score = score_trajectory_events([
        {"kind": "user"}, {"kind": "failure"}, {"kind": "diagnosis"},
        {"kind": "revision"}, {"kind": "test"}, {"kind": "final"},
    ])
    assert score.overall == 1.0


def test_unverified_repair_is_penalized():
    score = score_trajectory_events([
        {"kind": "failure"}, {"kind": "diagnosis"}, {"kind": "revision"},
    ])
    assert score.overall == 0.8


def test_aggregate_reports_behavior_dimensions():
    result = aggregate_behavior([
        {"trajectory_events": [
            {"kind": "failure"}, {"kind": "diagnosis"},
            {"kind": "revision"}, {"kind": "test"},
        ]},
        {"trajectory_events": [
            {"kind": "failure"}, {"kind": "diagnosis"},
        ]},
    ])
    assert result["trajectories"] == 2
    assert result["failure_detection"] == 1.0
    assert result["diagnosis"] == 1.0
    assert result["revision"] == 0.5
    assert result["testing"] == 0.5
    assert result["behavior_score"] == 0.75
