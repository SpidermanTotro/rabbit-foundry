import json

import pytest

from rabbit_foundry.bunny_import import import_bunny_jsonl


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_import_builds_provenanced_next_byte_episodes(tmp_path):
    src = tmp_path / "alpha.jsonl"
    write_rows(src, [{
        "id": "003-self-correct-midstream",
        "messages": [
            {"role": "user", "content": "x" * 40},
            {"role": "assistant", "content": "y" * 40},
        ],
    }])
    payload = import_bunny_jsonl(src, window=16)
    assert payload["source_rows"] == 1
    assert payload["episodes"]
    assert payload["teacher_free_native_experiment"] is False
    assert {e["skill"] for e in payload["episodes"]} == {"bunny_behavior"}
    assert all(e["source_sha256"] == payload["source_sha256"] for e in payload["episodes"])
    assert all(len(bytes.fromhex(e["prompt_hex"])) == 16 for e in payload["episodes"])
    assert all(len(bytes.fromhex(e["target_hex"])) == 16 for e in payload["episodes"])


def test_import_refuses_alpha_holdout(tmp_path):
    src = tmp_path / "bad.jsonl"
    write_rows(src, [{
        "id": "011-partial-failure",
        "messages": [{"role": "assistant", "content": "must stay held out" * 10}],
    }])
    with pytest.raises(ValueError, match="Alpha holdout leakage"):
        import_bunny_jsonl(src, window=16)
