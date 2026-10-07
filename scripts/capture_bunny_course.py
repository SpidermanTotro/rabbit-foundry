from __future__ import annotations

import argparse
import json

from rabbit_foundry.capture_course import capture_course


def main():
    parser = argparse.ArgumentParser(
        description="Normalize Alpha/Bunny behavioral captures into training and frozen holdout courses"
    )
    parser.add_argument("captures", nargs="+", help="JSON/JSONL capture files")
    parser.add_argument("--out-dir", default="runs/capture-course")
    args = parser.parse_args()
    result = capture_course(args.captures, args.out_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
