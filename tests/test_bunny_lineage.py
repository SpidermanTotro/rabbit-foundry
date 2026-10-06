import json

import pytest

from rabbit_foundry.bunny_lineage import (
    ALPHA_HOLDOUT_IDS,
    assert_no_alpha_holdout,
    load_jsonl_training,
    normalize_capture_id,
)


def test_normalize_capture_id():
    assert normalize_capture_id("011-partial-failure.json") == "011-partial-failure"
    assert normalize_capture_id("011-partial-failure") == "011-partial-failure"


def test_alpha_holdout_is_rejected():
    holdout = next(iter(ALPHA_HOLDOUT_IDS))
    with pytest.raises(ValueError, match="Alpha holdout leakage"):
        assert_no_alpha_holdout([{"id": holdout}])


def test_eligible_alpha_training_row_is_allowed():
    assert_no_alpha_holdout([{"id": "003-self-correct-midstream"}])


def test_jsonl_loader_rejects_holdout(tmp_path):
    path = tmp_path / "training.jsonl"
    path.write_text(json.dumps({"capture_id": "034-anti-sycophancy"}) + "\n")
    with pytest.raises(ValueError, match="034-anti-sycophancy"):
        load_jsonl_training(path)
