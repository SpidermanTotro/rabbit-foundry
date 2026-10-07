from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

from rabbit_foundry.space_bunny_compare import compare_space_bunny
from rabbit_foundry.space_bunny_failover import run_with_failover
from rabbit_foundry.space_bunny_fingerprint import FROZEN_ALPHA_IDS

DEFAULT_ENDPOINT = "https://opencode.ai/zen/v1/chat/completions"
DEFAULT_MODEL = "space-bunny-free"


def load_cases(path: str | Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    by_id = {str(row.get("id")): row for row in rows}
    missing = [case_id for case_id in FROZEN_ALPHA_IDS if case_id not in by_id]
    if missing:
        raise ValueError("missing frozen Alpha cases: " + ", ".join(missing))
    return by_id


def run_case(endpoint: str, model: str, case: dict, timeout: int = 120, api_key: str | None = None) -> str:
    messages = case.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{case.get('id')} needs messages")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps({"model": model, "messages": messages}).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Space Bunny response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("Space Bunny response has no assistant content")
    return message["content"]


def run_case_with_failover(case: dict, model: str, preferred: str) -> tuple[str, str]:
    def call(route, key):
        return run_case(route.endpoint, model, case, api_key=key)
    response, route = run_with_failover(call, preferred=preferred)
    return response, route.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Alpha fingerprint against current Space Bunny")
    parser.add_argument("cases")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--preferred-route", choices=("zen", "go"), default="zen")
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
            response, route = run_case_with_failover(cases[case_id], args.model, args.preferred_route)
            handle.write(json.dumps({
                "id": case_id, "response": response, "route": route,
                "evaluation_only": True, "trainable": False,
            }) + "\n")
            print(f"{case_id}: captured via {route}")

    report = compare_space_bunny(responses_path, out_dir / "comparison.json", threshold=args.threshold)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
