import json

from rabbit_foundry.space_bunny_fingerprint import FROZEN_ALPHA_IDS
from rabbit_foundry.space_bunny_training_source import build_space_bunny_training_source


def test_frozen_fingerprint_never_becomes_training(tmp_path):
    cases = tmp_path / "cases.jsonl"
    responses = tmp_path / "responses.jsonl"
    out = tmp_path / "source.jsonl"
    cases.write_text("".join(json.dumps({
        "id": case_id,
        "messages": [{"role": "user", "content": "challenge " + case_id}],
    }) + "\n" for case_id in FROZEN_ALPHA_IDS))
    responses.write_text("".join(json.dumps({
        "id": case_id,
        "response": "candidate response",
    }) + "\n" for case_id in FROZEN_ALPHA_IDS))

    result = build_space_bunny_training_source(cases, responses, out)
    assert result["trainable_rows"] == 0
    assert result["evaluation_rows"] == 6
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert all(row["evaluation_only"] is True for row in rows)
    assert all(row["trainable"] is False for row in rows)
