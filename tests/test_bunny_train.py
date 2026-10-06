import json

from rabbit_foundry.train import manifest_episodes


def test_bunny_manifest_routes_without_github_source_fields(tmp_path):
    episodes = []
    for i in range(200):
        episode_id = f"{i:064x}"
        episodes.append({
            "episode_id": episode_id,
            "skill": "bunny_behavior",
            "lineage": "alpha-space-bunny",
            "prompt_hex": "00" * 16,
            "target_hex": "01" * 16,
        })
    path = tmp_path / "bunny.json"
    path.write_text(json.dumps({
        "kind": "bunny-lineage-episodes",
        "episodes": episodes,
    }))
    train, valid = manifest_episodes(path, "bunny")
    assert train
    assert valid
    assert all(row["lineage"] == "alpha-space-bunny" for row in train + valid)
