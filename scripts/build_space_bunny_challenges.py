from __future__ import annotations
import argparse, json
from rabbit_foundry.space_bunny_challenges import build_challenges
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out", default="runs/space-bunny/training-challenges.jsonl")
    a=p.parse_args()
    print(json.dumps(build_challenges(a.out), indent=2))
if __name__ == "__main__": main()
