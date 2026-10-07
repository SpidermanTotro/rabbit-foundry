import pytest

from rabbit_foundry.rolling_quality import build_rolling_challenge, quality_gate


def case(case_id, difficulty, axis="selfcorr"):
    return {
        "id": case_id,
        "axis": axis,
        "difficulty": difficulty,
        "messages": [{"role": "user", "content": "challenge"}],
    }


def test_rolling_challenge_prefers_harder_unseen_cases(tmp_path):
    rows = [case("easy", 0.2), case("hard", 0.9), case("medium", 0.5)]
    result = build_rolling_challenge(
        rows, tmp_path / "challenge.jsonl", generation=2,
        max_cases=2, previous_ids={"medium"},
    )
    text = (tmp_path / "challenge.jsonl").read_text()
    assert result["cases"] == 2
    assert "hard" in text
    assert "easy" in text
    assert "medium" not in text


def test_quality_gate_requires_rolling_quality_not_to_regress():
    assert quality_gate(0.8, 0.75, 0.70, anchor_floor=0.75)["quality_pass"] is True
    assert quality_gate(0.8, 0.65, 0.70, anchor_floor=0.75)["quality_pass"] is False
    assert quality_gate(0.70, 0.80, 0.70, anchor_floor=0.75)["quality_pass"] is False


def test_quality_gate_rejects_invalid_scores():
    with pytest.raises(ValueError):
        quality_gate(1.1, 0.5)
