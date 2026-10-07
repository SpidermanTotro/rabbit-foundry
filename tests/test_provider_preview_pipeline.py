import json

from rabbit_foundry.bunny_import import load_bunny_episode_manifest
from rabbit_foundry.bunny_lineage import ALPHA_HOLDOUT_IDS
from rabbit_foundry.bunny_pipeline import prepare_alpha_bunny_training
from rabbit_foundry.provider_capture import import_provider_exports


def _holdout(capture_id):
    return {
        "id": capture_id,
        "axis": "selfcorr",
        "messages": [
            {"role": "user", "content": "diagnose the failure " * 12},
            {"role": "assistant", "content": "revise, test, and verify the repair " * 12},
        ],
    }


def test_provider_preview_reaches_training_without_holdout_leakage(tmp_path):
    provider = tmp_path / "provider.jsonl"
    traces = []
    for i in range(12):
        traces.append({
            "id": f"preview-{i:03d}",
            "provider": "openrouter" if i % 2 == 0 else "kilo",
            "model": "preview-model",
            "axis": "debug" if i % 2 == 0 else "tool",
            "request": {
                "messages": [
                    {"role": "user", "content": ("inspect this failing program " * 10) + str(i)}
                ]
            },
            "response": {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": ("find the cause, repair it, and test it " * 10) + str(i),
                    }
                }]
            },
        })
    provider.write_text("".join(json.dumps(row) + "\n" for row in traces))

    preview_captures = tmp_path / "preview.jsonl"
    imported = import_provider_exports([provider], preview_captures)
    assert imported["captures"] == 12
    assert imported["providers"] == {"kilo": 6, "openrouter": 6}

    anchors = tmp_path / "anchors.jsonl"
    anchors.write_text(
        "".join(json.dumps(_holdout(case_id)) + "\n" for case_id in sorted(ALPHA_HOLDOUT_IDS))
    )

    result = prepare_alpha_bunny_training(
        [preview_captures, anchors],
        tmp_path / "prepared",
        window=32,
    )
    assert result["training_rows"] == 12
    assert result["holdout_rows"] == 6
    assert result["training_episodes"] > 0

    manifest_text = (tmp_path / "prepared" / "training_episodes.json").read_text()
    assert "preview-" in manifest_text
    assert not any(case_id in manifest_text for case_id in ALPHA_HOLDOUT_IDS)

    train_rows = load_bunny_episode_manifest(
        tmp_path / "prepared" / "training_episodes.json", "train"
    )
    validation_rows = load_bunny_episode_manifest(
        tmp_path / "prepared" / "training_episodes.json", "validation"
    )
    test_rows = load_bunny_episode_manifest(
        tmp_path / "prepared" / "training_episodes.json", "test"
    )
    assert train_rows or validation_rows or test_rows
    assert all(row["lineage"] == "alpha-space-bunny" for row in train_rows + validation_rows + test_rows)
