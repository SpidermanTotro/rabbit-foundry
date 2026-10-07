import torch

from rabbit_foundry.candidate_eval import (
    evaluate_behavior_cases,
    score_behavior_response,
)
from rabbit_foundry.model import ModelConfig, TinyRabbitLM


def test_behavior_response_scoring_rewards_repair_sequence():
    scored = score_behavior_response(
        "The test failed because state was lost. Fix the state handling and verify with pytest."
    )
    assert scored["failure_detection"] is True
    assert scored["diagnosis"] is True
    assert scored["revision"] is True
    assert scored["testing"] is True
    assert scored["behavior_score"] == 1.0


def test_candidate_evaluation_uses_model_generated_response(monkeypatch):
    cfg = ModelConfig(context=64, d_model=32, n_heads=4, n_layers=1, d_ff=64)
    model = TinyRabbitLM(cfg)

    def fake_generate(*args, **kwargs):
        return "error because parser state is lost; fix it and test the repair"

    monkeypatch.setattr("rabbit_foundry.candidate_eval.generate_bytes", fake_generate)
    result = evaluate_behavior_cases(
        model,
        [{"id": "anchor-1", "axis": "selfcorr", "messages": [
            {"role": "user", "content": "debug this parser"}
        ]}],
        device=torch.device("cpu"),
    )
    assert result["behavior_score"] == 1.0
    assert result["cases"][0]["response"].startswith("error because")


def test_behavior_response_without_evidence_scores_zero():
    assert score_behavior_response("looks fine to me")["behavior_score"] == 0.0
