from __future__ import annotations
import argparse, json
from rabbit_foundry.space_bunny_promote import promote_verified_repairs
def main():
    p=argparse.ArgumentParser()
    p.add_argument("challenges")
    p.add_argument("responses")
    p.add_argument("--out", default="runs/space-bunny/training-promoted.jsonl")
    p.add_argument("--min-quality", type=float, default=0.60)
    a=p.parse_args()
    print(json.dumps(promote_verified_repairs(a.challenges,a.responses,a.out,a.min_quality), indent=2))
if __name__ == "__main__": main()
