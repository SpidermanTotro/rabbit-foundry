from __future__ import annotations

import json
from pathlib import Path

from .alpha_holdout import write_alpha_holdout_manifest
from .bunny_import import write_bunny_manifest
from .capture_course import capture_course


def prepare_alpha_bunny_training(
    captures: list[str | Path],
    out_dir: str | Path,
    *,
    window: int = 128,
    require_complete_holdout: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    course = capture_course(captures, out_dir / "course")
    if require_complete_holdout and not course["holdout_complete"]:
        raise ValueError(
            "capture course does not contain the complete six-case frozen Alpha holdout"
        )

    training_manifest_path = out_dir / "training_episodes.json"
    holdout_manifest_path = out_dir / "alpha_holdout_episodes.json"
    training = write_bunny_manifest(
        course["training_jsonl"], training_manifest_path, window=window
    )
    holdout = write_alpha_holdout_manifest(
        course["holdout_jsonl"], holdout_manifest_path, window=window
    )
    payload = {
        "version": 1,
        "kind": "alpha-bunny-training-pipeline",
        "lineage": "alpha-space-bunny",
        "teacher_free_native_experiment": False,
        "capture_manifest": str(out_dir / "course" / "capture_manifest.json"),
        "training_manifest": str(training_manifest_path),
        "frozen_holdout_manifest": str(holdout_manifest_path),
        "training_rows": training["source_rows"],
        "training_episodes": len(training["episodes"]),
        "holdout_rows": holdout["source_rows"],
        "holdout_episodes": len(holdout["episodes"]),
        "holdout_ids": holdout["expected_ids"],
        "holdout_coverage": holdout["episode_coverage"],
        "training_allowed_for_holdout": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pipeline.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
