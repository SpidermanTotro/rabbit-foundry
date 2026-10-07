import json

import pytest

from rabbit_foundry.space_bunny_fingerprint import FROZEN_ALPHA_IDS, build_fingerprint_manifest


def test_manifest_requires_all_six_frozen_cases(tmp_path):
    source = tmp_path / "cases.jsonl"
    source.write_text("".join(
        json.dumps({"id": case_id, "messages": [{"role": "user", "content": case_id}]}) + "\n"
        for case_id in FROZEN_ALPHA_IDS
    ))
    out = tmp_path / "manifest.json"
    result = build_fingerprint_manifest(source, out)
    assert result["candidate_model"] == "space-bunny-free"
    assert result["identity_status"] == "unverified-continuation-candidate"
    assert result["case_ids"] == list(FROZEN_ALPHA_IDS)
    assert out.exists()


def test_manifest_fails_closed_when_anchor_missing(tmp_path):
    source = tmp_path / "cases.jsonl"
    source.write_text(json.dumps({"id": FROZEN_ALPHA_IDS[0], "messages": []}) + "\n")
    with pytest.raises(ValueError, match="missing frozen Alpha cases"):
        build_fingerprint_manifest(source, tmp_path / "manifest.json")
