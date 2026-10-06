# Rabbit Foundry 🐇🏭

Rabbit Foundry is an experimental, local-first model-building and Bunny-lineage research system.

## Research tracks

### Rabbit-native

The native experiment starts from random weights and does not require pretrained weights or teacher-model answers. It learns from pinned GitHub-derived byte tasks:

- code prediction
- deterministic code repair
- hidden revision differences

TwinTrain compares candidates with frozen held-out evaluation, Greenlight promotion, and fixed-vs-adaptive equal-compute experiments. Hidden-diff byte training is deliberately restricted to equal-length changed spans so insertions/deletions cannot create false offset-aligned labels.

### Alpha / Space Bunny lineage

Foundry also preserves and imports eligible behavioral training material from the Alpha → Space Bunny project. This is a **separately labeled lineage**, not the teacher-free native experiment.

The six frozen Alpha holdout captures are mechanically blocked from training imports. Large model weights and GGUFs remain outside Git; their provenance is cataloged in `configs/bunny_lineage.json`.

See `docs/BUNNY_LINEAGE.md`.

## Safety and evaluation boundaries

Repository code is untrusted input. Build/test execution belongs in the locked Podman sandbox with networking disabled, read-only source mounts, dropped capabilities, and resource/time limits.

Historical revisions of the same repository/path stay in one dataset partition to reduce evaluation leakage. Bunny training imports preserve source SHA-256, keep every capture family in one train/validation/test partition, preserve Alpha behavior axes, and keep the six frozen Alpha holdouts separate.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python -m rabbit_foundry.train --steps 20 --device auto
```

### Import Bunny lineage data

```bash
python scripts/import_bunny_jsonl.py \
  --source /path/to/eligible-bunny-training.jsonl \
  --out runs/bunny/episodes.json \
  --window 128

python -m rabbit_foundry.train \
  --episodes runs/bunny/episodes.json \
  --episode-kind bunny \
  --sampling adaptive \
  --run-dir runs/bunny/adaptive
```

Outputs are written beneath `runs/`.

### Run the Rabbit-native fixed-vs-adaptive experiment

```bash
python -m rabbit_foundry.experiment \
  --episodes runs/episodes.json \
  --out runs/experiments/fixed-vs-adaptive.json \
  --steps 100 \
  --batch 8 \
  --seq 64 \
  --minimum-relative-improvement 0.01 \
  --device auto
```

The optional improvement threshold prevents a numerically tiny held-out loss difference from being reported as a meaningful winner.

## Current capability

- [x] Tiny causal LM
- [x] Twin A/B training controller
- [x] Held-out evaluation and winner selection
- [x] JSON experiment ledger
- [x] Adaptive weakness-driven resampling
- [x] Read-only pinned GitHub provenance/episode builder
- [x] Prediction + repair + hidden-diff curricula
- [x] File-family split protection across revisions
- [x] Frozen held-out episode batches
- [x] Equal-compute fixed-vs-adaptive experiment runner
- [x] Locked Podman sandbox runner
- [x] Greenlight promotion gate
- [x] Alpha/Space Bunny lineage inventory
- [x] Alpha holdout contamination guard
- [x] Bunny conversational JSONL → episode importer
- [x] Bunny-lineage TwinTrain mode
- [ ] Run full local regression suite after newest integration commits
- [ ] Import and hash the real local Bunny training corpus into a run manifest
- [x] Preserve per-row Bunny axes (code/debug/persona/selfcorr/tool) in curriculum
- [x] Frozen Alpha holdout evaluation manifest with exact-set and per-case coverage guards
- [x] Space Bunny checkpoint/GGUF structural inventory (directory stats, provenance JSON, GGUF magic)
- [ ] Run the Space Bunny inventory against the real local artifacts and record hashes
- [ ] Rabbit-native GGUF conversion adapter
- [ ] llama.cpp post-export verification
