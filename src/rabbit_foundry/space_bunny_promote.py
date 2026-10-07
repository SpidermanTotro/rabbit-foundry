from __future__ import annotations

import json
from pathlib import Path

from .alpha_course_quality import enrich_alpha_capture


def promote_verified_repairs(
    challenges_path: str | Path,
    responses_path: str | Path,
    out: str | Path,
    min_quality: float = 0.60,
) -> dict:
    challenges = [json.loads(x) for x in Path(challenges_path).read_text().splitlines() if x.strip()]
    responses = [json.loads(x) for x in Path(responses_path).read_text().splitlines() if x.strip()]
    response_by_id = {str(row.get("id")): row for row in responses}
    promoted, rejected = [], []
    for challenge in challenges:
        cid = str(challenge.get("id"))
        response = response_by_id.get(cid)
        if response is None or not isinstance(response.get("response"), str):
            rejected.append({"id": cid, "reason": "missing_response"})
            continue
        row = enrich_alpha_capture({
            "id": cid,
            "axis": "debug",
            "source_kind": "observable_model_session",
            "source_model": response.get("source_model", "space-bunny-free"),
            "source_route": response.get("route"),
            "provenance": "opencode-space-bunny-non-anchor-challenge",
            "messages": challenge.get("messages", []) + [
                {"role": "assistant", "content": response["response"]}
            ],
        })
        score = float(row["training_quality"]["score"])
        if score < min_quality:
            rejected.append({"id": cid, "reason": "quality_below_threshold", "score": score})
            continue
        row["trainable"] = True
        row["evaluation_only"] = False
        promoted.append(row)

    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row) + "\n" for row in promoted))
    rejected_path = target.with_suffix(".rejected.json")
    rejected_path.write_text(json.dumps(rejected, indent=2) + "\n")
    return {
        "promoted": len(promoted),
        "rejected": len(rejected),
        "output": str(target),
        "rejected_output": str(rejected_path),
    }
