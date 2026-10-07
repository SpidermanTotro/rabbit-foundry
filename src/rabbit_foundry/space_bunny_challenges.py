from __future__ import annotations

import hashlib
import json
from pathlib import Path


CHALLENGES = (
    ("debug-state-resume", "A Python training loop resumes model weights correctly but AdamW crashes after restart because optimizer state entries are incomplete. Diagnose the failure, propose a minimal repair, and describe the regression test you would run."),
    ("debug-off-by-one", "A byte-level language-model dataset creates input and target chunks of window+1 bytes, but training occasionally sees mismatched lengths. Find the likely boundary mistake and propose a test-first fix."),
    ("debug-secret-log", "A local coding agent writes request headers and messages to a JSONL training capture. Design a fix that prevents credentials from entering training data while preserving useful debugging context."),
    ("debug-heldout-leak", "A model-training pipeline accidentally includes evaluation challenge IDs in its training JSONL. Explain how to detect the leakage, remove it, and add a fail-closed regression test."),
    ("debug-nonfinite-loss", "A candidate training run produces NaN validation loss but has a high behavioral score. Explain why promotion must be blocked and give a robust gate condition."),
    ("tool-test-first", "You inherit a failing Python repository. Describe a tool-using workflow that reproduces the failure, inspects only relevant files, makes the smallest patch, and verifies it without claiming success before tests run."),
)


def build_challenges(out: str | Path) -> dict:
    rows = []
    for name, prompt in CHALLENGES:
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        rows.append({
            "id": f"space-bunny-train-{name}-{digest}",
            "source": "rabbit-code-authored",
            "evaluation_only": False,
            "messages": [{"role": "user", "content": prompt}],
        })
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {"rows": len(rows), "output": str(target)}
