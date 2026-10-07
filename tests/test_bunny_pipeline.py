import json

import pytest

from rabbit_foundry.bunny_lineage import ALPHA_HOLDOUT_IDS
from rabbit_foundry.bunny_pipeline import prepare_alpha_bunny_training


def capture(capture_id, axis="debug"):
    return {
        "id": capture_id,
        "axis": axis,
        "messages": [
            {"role": "user", "content": "x" * 160},
            {"role": "assistant", "content": "y" * 160},
        ],
    }


def test_pipeline_builds_training_and_frozen_holdout_manifests(tmp_path):
    source = tmp_path / "captures.jsonl"
    rows = [capture(f"training-{i:03d}", "debug" if i % 2 else "tool") for i in range(20)]
    rows += [capture(capture_id, "selfcorr") for capture_id in sorted(ALPHA_HOLDOUT_IDS)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = prepare_alpha_bunny_training([source], tmp_path / "run", window=32)
    assert result["training_rows"] == 20
    assert result["holdout_rows"] == 6
    assert result["training_episodes"] > 0
    assert result["holdout_episodes"] > 0
    assert result["training_allowed_for_holdout"] is False
    assert set(result["holdout_ids"]) == set(ALPHA_HOLDOUT_IDS)


def test_pipeline_requires_complete_frozen_holdout_by_default(tmp_path):
    source = tmp_path / "captures.jsonl"
    source.write_text(json.dumps(capture("training-only")) + "\n")
    with pytest.raises(ValueError, match="complete six-case frozen Alpha holdout"):
        prepare_alpha_bunny_training([source], tmp_path / "run", window=32)
