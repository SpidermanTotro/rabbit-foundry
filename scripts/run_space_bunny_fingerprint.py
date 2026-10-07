from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

from rabbit_foundry.space_bunny_compare import compare_space_bunny
from rabbit_foundry.space_bunny_fingerprint import FROZEN_ALPHA_IDS


DEFAULT_ENDPOINT = "https://opencode.ai/inference/openai/v1/chat/completions"
DEFAULT_MODEL = "space-bunny-free"


def load_cases(path: str | Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    by_id = {str(row.get("id")): row for row in rows}
    missing = [case_id for case_id in FROZEN_ALPHA_IDS if case_id not in by_id]
    if missing:
        raise ValueError("missing frozen Alpha cases: " + ", ".join(missing))
    return by_id


def run_case(endpoint: str, model: str, case: dict, timeout: int = 120) -> str:
    messages = case.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{case.get('id')} needs messages")
    payload = json.dumps({"model": model, "messages": messages}).encode()
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Space Bunny HTTP {exc.code}: {detail}") from exc
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Space Bunny response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("Space Bunny response has no assistant content")
    return message["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen six-case fingerprint against current Space Bunny Free")
    parser.add_argument("cases", help="JSONL containing the six frozen Alpha cases")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", default="runs/space-bunny/live-fingerprint")
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    cases = load_cases(args.cases)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    responses_path = out_dir / "responses.jsonl"
    with responses_path.open("w") as handle:
        for case_id in FROZEN_ALPHA_IDS:
            response = run_case(args.endpoint, args.model, cases[case_id])
            handle.write(json.dumps({"id": case_id, "response": response}) + "\n")
            print(f"{case_id}: captured")

    report = compare_space_bunny(
        responses_path,
        out_dir / "comparison.json",
        threshold=args.threshold,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
