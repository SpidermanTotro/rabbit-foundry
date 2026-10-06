import json

import pytest

from rabbit_foundry.bunny_import import (
    bunny_family_id,
    deterministic_bunny_split,
    import_bunny_jsonl,
    load_bunny_episode_manifest,
)


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


def test_bunny_manifest_keeps_capture_families_in_one_split(tmp_path):
    src = tmp_path / "alpha.jsonl"
    write_rows(src, [
        {
            "id": f"capture-{i}",
            "messages": [{"role": "assistant", "content": ("behavior-%d " % i) * 200}],
        }
        for i in range(30)
    ])
    payload = import_bunny_jsonl(src, window=16)
    manifest = tmp_path / "episodes.json"
    manifest.write_text(json.dumps(payload))

    groups = {
        name: load_bunny_episode_manifest(manifest, name)
        for name in ("train", "validation", "test")
    }
    episode_sets = {
        name: {row["episode_id"] for row in rows}
        for name, rows in groups.items()
    }
    assert episode_sets["train"].isdisjoint(episode_sets["validation"])
    assert episode_sets["train"].isdisjoint(episode_sets["test"])
    assert episode_sets["validation"].isdisjoint(episode_sets["test"])
    assert set().union(*episode_sets.values()) == {r["episode_id"] for r in payload["episodes"]}

    capture_splits = {}
    for split, rows in groups.items():
        for row in rows:
            capture_splits.setdefault(row["source_capture_id"], set()).add(split)
            assert deterministic_bunny_split(bunny_family_id(row)) == split
    assert capture_splits
    assert all(len(splits) == 1 for splits in capture_splits.values())


def test_alpha_axis_becomes_curriculum_skill(tmp_path):
    src = tmp_path / "axis.jsonl"
    write_rows(src, [{
        "id": "003-self-correct-midstream",
        "axis": "selfcorr",
        "messages": [{"role": "assistant", "content": "revise hypothesis " * 40}],
    }])
    payload = import_bunny_jsonl(src, window=16)
    assert {e["skill"] for e in payload["episodes"]} == {"bunny_selfcorr"}
    assert {e["bunny_axis"] for e in payload["episodes"]} == {"selfcorr"}
