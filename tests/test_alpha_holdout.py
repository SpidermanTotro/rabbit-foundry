import json

import pytest

from rabbit_foundry.alpha_holdout import build_alpha_holdout_manifest
from rabbit_foundry.bunny_lineage import ALPHA_HOLDOUT_IDS


def row(capture_id):
    return {
        "id": capture_id,
        "axis": "selfcorr",
        "messages": [{"role": "assistant", "content": "frozen evaluation " * 30}],
    }


def test_alpha_holdout_requires_exact_frozen_set(tmp_path):
    path = tmp_path / "holdout.jsonl"
    path.write_text("".join(json.dumps(row(x)) + "\n" for x in sorted(ALPHA_HOLDOUT_IDS)))
    payload = build_alpha_holdout_manifest(path, window=16)
    assert payload["source_rows"] == 6
    assert payload["training_allowed"] is False
    assert payload["episodes"]
    assert all(e["evaluation_only"] for e in payload["episodes"])
    assert all(e["split"] == "alpha_frozen_holdout" for e in payload["episodes"])


def test_alpha_holdout_rejects_incomplete_set(tmp_path):
    path = tmp_path / "holdout.jsonl"
    path.write_text(json.dumps(row("011-partial-failure")) + "\n")
    with pytest.raises(ValueError, match="Alpha holdout mismatch"):
        build_alpha_holdout_manifest(path, window=16)
