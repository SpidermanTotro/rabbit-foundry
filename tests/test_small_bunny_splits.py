import json

from rabbit_foundry.bunny_import import load_bunny_episode_manifest, write_bunny_manifest


def test_small_bunny_manifest_has_train_and_validation(tmp_path):
    source = tmp_path / "captures.jsonl"
    rows = []
    for i in range(3):
        rows.append({
            "id": f"capture-{i}",
            "axis": "debug",
            "messages": [
                {"role": "user", "content": ("find the bug " * 30) + str(i)},
                {"role": "assistant", "content": ("diagnose repair test verify " * 30) + str(i)},
            ],
        })
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "episodes.json"
    write_bunny_manifest(source, manifest, window=32)
    train = load_bunny_episode_manifest(manifest, "train")
    validation = load_bunny_episode_manifest(manifest, "validation")
    assert train
    assert validation
    train_ids = {row["source_capture_id"] for row in train}
    validation_ids = {row["source_capture_id"] for row in validation}
    assert train_ids.isdisjoint(validation_ids)
