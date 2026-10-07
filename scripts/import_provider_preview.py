from __future__ import annotations

import argparse
import json

from rabbit_foundry.provider_capture import import_provider_exports


def main():
    parser = argparse.ArgumentParser(
        description="Import exported Kilo/OpenRouter/OpenAI-compatible preview traces"
    )
    parser.add_argument("exports", nargs="+", help="JSON/JSONL trace exports")
    parser.add_argument("--out", default="runs/provider-preview/captures.jsonl")
    args = parser.parse_args()
    print(json.dumps(import_provider_exports(args.exports, args.out), indent=2))


if __name__ == "__main__":
    main()
