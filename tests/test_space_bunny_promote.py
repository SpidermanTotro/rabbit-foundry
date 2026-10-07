import json

from rabbit_foundry.space_bunny_challenges import build_challenges
from rabbit_foundry.space_bunny_fingerprint import FROZEN_ALPHA_IDS
from rabbit_foundry.space_bunny_promote import promote_verified_repairs


def test_authored_training_challenges_do_not_overlap_frozen_alpha(tmp_path):
    out = tmp_path / "challenges.jsonl"
    build_challenges(out)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    ids = {row["id"] for row in rows}
    assert ids.isdisjoint(FROZEN_ALPHA_IDS)
    assert all(row["evaluation_only"] is False for row in rows)


def test_only_quality_repairs_are_promoted(tmp_path):
    challenges = tmp_path / "challenges.jsonl"
    responses = tmp_path / "responses.jsonl"
    out = tmp_path / "training.jsonl"
    challenges.write_text(
        json.dumps({"id": "good", "messages": [{"role": "user", "content": "debug"}]}) + "\n" +
        json.dumps({"id": "weak", "messages": [{"role": "user", "content": "debug"}]}) + "\n"
    )
    responses.write_text(
        json.dumps({"id": "good", "response": "The error failed because state is stale. Fix it, then run pytest and verify tests passed."}) + "\n" +
        json.dumps({"id": "weak", "response": "Try changing it."}) + "\n"
    )
    result = promote_verified_repairs(challenges, responses, out, min_quality=0.60)
    assert result["promoted"] == 1
    assert result["rejected"] == 1
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert rows[0]["id"] == "good"
    assert rows[0]["trainable"] is True
