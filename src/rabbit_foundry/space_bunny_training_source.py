from __future__ import annotations

import json
from pathlib import Path

from .space_bunny_fingerprint import FROZEN_ALPHA_IDS


def build_space_bunny_training_source(
    cases_path: str | Path,
    responses_path: str | Path,
    out: str | Path,
) -> dict:
    cases = [json.loads(x) for x in Path(cases_path).read_text().splitlines() if x.strip()]
    responses = [json.loads(x) for x in Path(responses_path).read_text().splitlines() if x.strip()]
    case_by_id = {str(x.get("id")): x for x in cases}
    response_by_id = {str(x.get("id")): x for x in responses}

    missing = [x for x in FROZEN_ALPHA_IDS if x not in case_by_id or x not in response_by_id]
    if missing:
        raise ValueError("incomplete Space Bunny fingerprint: " + ", ".join(missing))

    # Frozen identity anchors remain evaluation-only. They must never leak into training.
    rows = []
    for case_id in FROZEN_ALPHA_IDS:
        response = response_by_id[case_id].get("response")
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"{case_id} has no usable response")
        rows.append({
            "id": "space-bunny-live-" + case_id,
            "source_model": "space-bunny-free",
            "source_kind": "observable_model_session",
            "provenance": "opencode-current-space-bunny",
            "evaluation_only": True,
            "trainable": False,
            "messages": case_by_id[case_id].get("messages", [])
                + [{"role": "assistant", "content": response}],
        })

    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {
        "rows": len(rows),
        "trainable_rows": 0,
        "evaluation_rows": len(rows),
        "reason": "frozen Alpha fingerprint anchors are evaluation-only",
        "output": str(target),
    }
