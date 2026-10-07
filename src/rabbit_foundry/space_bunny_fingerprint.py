from __future__ import annotations

import json
from pathlib import Path


FROZEN_ALPHA_IDS = (
    "011-partial-failure",
    "014-instruction-conflict",
    "016-guess-discipline",
    "028-ambiguous-request",
    "031-test-first-request",
    "034-anti-sycophancy",
)


def build_fingerprint_manifest(cases_path: str | Path, out: str | Path) -> dict:
    rows = [
        json.loads(line)
        for line in Path(cases_path).read_text().splitlines()
        if line.strip()
    ]
    by_id = {str(row.get("id")): row for row in rows}
    missing = [case_id for case_id in FROZEN_ALPHA_IDS if case_id not in by_id]
    if missing:
        raise ValueError("missing frozen Alpha cases: " + ", ".join(missing))
    manifest = {
        "schema_version": 1,
        "candidate_model": "space-bunny-free",
        "identity_status": "unverified-continuation-candidate",
        "case_ids": list(FROZEN_ALPHA_IDS),
        "cases": [by_id[case_id] for case_id in FROZEN_ALPHA_IDS],
    }
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
