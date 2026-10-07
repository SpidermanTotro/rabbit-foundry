import json

import pytest

from rabbit_foundry.bunny_lineage import ALPHA_HOLDOUT_IDS
from rabbit_foundry.capture_course import capture_course


def row(capture_id, axis="self-correction", content=None):
    return {
        "id": capture_id,
        "axis": axis,
        "messages": [
            {"role": "user", "content": "debug this"},
            {"role": "assistant", "content": content or ("revise hypothesis " * 20)},
        ],
    }


def test_capture_course_routes_frozen_alpha_away_from_training(tmp_path):
    source = tmp_path / "captures.jsonl"
    rows = [row("training-001", "debugging")]
    rows += [row(capture_id) for capture_id in sorted(ALPHA_HOLDOUT_IDS)]
    source.write_text("".join(json.dumps(item) + "\n" for item in rows))

    result = capture_course([source], tmp_path / "course")
    training = (tmp_path / "course" / "alpha_training.jsonl").read_text()
    holdout = (tmp_path / "course" / "alpha_holdout.jsonl").read_text()

    assert "training-001" in training
    assert not any(capture_id in training for capture_id in ALPHA_HOLDOUT_IDS)
    assert all(capture_id in holdout for capture_id in ALPHA_HOLDOUT_IDS)
    assert result["holdout_complete"] is True
    assert result["audit"]["eligible_training"] == 1
    assert result["audit"]["frozen_holdout"] == 6


def test_capture_course_normalizes_behavior_axis(tmp_path):
    source = tmp_path / "capture.json"
    source.write_text(json.dumps(row("training-002", "self-correction")))
    capture_course([source], tmp_path / "course")
    saved = json.loads((tmp_path / "course" / "alpha_training.jsonl").read_text())
    assert saved["axis"] == "selfcorr"


def test_capture_course_quarantines_malformed_capture(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text(json.dumps({"id": "broken", "messages": []}))
    result = capture_course([source], tmp_path / "course")
    assert result["audit"]["rejected"] == 1
    assert result["audit"]["eligible_training"] == 0
