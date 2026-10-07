# Rabbit Foundry 🐇🏭

Rabbit Foundry is a local-first model research and behavioral-training foundry.

Its primary workflow now understands **captured Alpha/Bunny behavior as a training course**: raw behavioral captures are normalized, audited, separated into eligible training material and frozen evaluation material, converted into deterministic training episodes, trained with TwinTrain, and promoted only through Greenlight.

Rabbit-native teacher-free GitHub learning remains a separate research track.

## Alpha/Bunny capture → training pipeline

```text
raw Alpha/Bunny captures
        ↓
capture-course normalization
        ↓
axis + provenance audit
        ↓
┌──────────────────────┬────────────────────────┐
│ eligible behavior    │ six frozen Alpha cases │
│ training             │ evaluation only        │
└──────────┬───────────┴────────────┬───────────┘
           ↓                        ↓
 Bunny training episodes       frozen holdout
           ↓                        │
           └──────── TwinTrain ─────┘
                       ↓
                   Greenlight
                       ↓
                   winner.pt
```

The six frozen Alpha IDs are never permitted into training:

- `011-partial-failure`
- `014-instruction-conflict`
- `016-guess-discipline`
- `028-ambiguous-request`
- `031-test-first-request`
- `034-anti-sycophancy`

### Prepare captured behavior

Pass one or more JSON/JSONL capture files:

```bash
python scripts/prepare_alpha_bunny_training.py \
  captures/*.jsonl \
  --out-dir runs/alpha-bunny \
  --window 128
```

This writes:

- `course/alpha_training.jsonl`
- `course/alpha_holdout.jsonl`
- `course/rejected.json`
- `course/capture_manifest.json`
- `training_episodes.json`
- `alpha_holdout_episodes.json`
- `pipeline.json`

Malformed captures are quarantined. Duplicate capture IDs are rejected. Behavior axes such as coding/debugging/self-correction/tool/persona are normalized. Source files and generated courses are hashed.

By default the pipeline requires the complete six-case Alpha holdout before calling the preparation complete.

### Capture Kilo / OpenRouter previews

Rabbit can import **exported** Kilo, OpenRouter, or other OpenAI-compatible request/response traces as behavioral preview data. Provider/model provenance is retained; API credentials are not training data.

```bash
python scripts/import_provider_preview.py \
  /path/to/kilo-or-openrouter-exports/*.jsonl \
  --out runs/provider-preview/captures.jsonl

python scripts/prepare_alpha_bunny_training.py \
  runs/provider-preview/captures.jsonl \
  /path/to/preserved-alpha-holdout.jsonl \
  --out-dir runs/provider-bunny \
  --window 128
```

This trains on **observable outputs you intentionally captured/exported**. It does not extract provider weights and does not imply ownership of Kilo/OpenRouter model weights.

### Preserve repair trajectories

For richer preview exports, Rabbit can preserve the whole repair path rather than only the final answer:

```text
task → attempt → tool call/result → failure → diagnosis → revision → test → final
```

Import trajectory JSON/JSONL:

```bash
python scripts/import_provider_trajectories.py \
  /path/to/provider-trajectories/*.jsonl \
  --out runs/provider-preview/trajectories.jsonl
```

Trajectory events are normalized into training-compatible messages while retaining structured event metadata. Incomplete trajectories are quarantined. This gives self-correction/debugging training a causal repair sequence instead of a flattened answer pair.

### Train the captured course

```bash
python -m rabbit_foundry.train \
  --episodes runs/alpha-bunny/training_episodes.json \
  --episode-kind bunny \
  --sampling adaptive \
  --minimum-relative-improvement 0.01 \
  --run-dir runs/alpha-bunny/train \
  --device auto
```

To continue from a compatible Rabbit checkpoint that you own/control:

```bash
python -m rabbit_foundry.train \
  --episodes runs/alpha-bunny/training_episodes.json \
  --episode-kind bunny \
  --sampling adaptive \
  --init-checkpoint /path/to/our/winner.pt \
  --run-dir runs/alpha-bunny/continued \
  --device auto
```

The checkpoint SHA-256 and initialization mode are recorded in the run ledger. Provider preview outputs and owned model weights remain separate provenance concepts.

Alpha/Space Bunny lineage training is explicitly labeled `alpha-space-bunny`; it is **not** reported as Rabbit-native teacher-free training.

## Rabbit-native research track

The separate native experiment starts from random weights and learns from pinned GitHub-derived byte tasks without teacher-model answers:

- code prediction
- deterministic code repair
- safely aligned hidden revision differences

Historical revisions of one repository/path remain in one dataset partition. Hidden-diff byte training uses only equal-length changed spans; insertion/deletion shifts are not treated as aligned targets.

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

Fixed and adaptive arms receive equal compute and identical frozen validation. A tiny numerical loss difference does not need to be declared a meaningful winner.

## Greenlight

Greenlight rejects non-finite candidates and can require a minimum relative held-out improvement before promotion. Behavioral quality is never silently assumed: without measured evidence it is recorded as unknown. When `--minimum-behavior-score` is enabled, both TwinTrain candidates must have measured scores or training fails closed.

Measured candidate scores can be supplied as a JSON object such as `{"A": 0.82, "B": 0.91}`:

```bash
python -m rabbit_foundry.train \
  --episodes runs/alpha-bunny/training_episodes.json \
  --episode-kind bunny \
  --behavior-scores runs/eval/behavior-scores.json \
  --minimum-behavior-score 0.75 \
  --run-dir runs/alpha-bunny/train
```

The run ledger records whether behavioral evidence was actually provided. A promoted model is saved as `winner.pt`.

## Export boundary

TinyRabbitLM is currently a custom byte-level PyTorch architecture, not a llama.cpp-supported Llama/Qwen architecture. Rabbit Foundry therefore audits export readiness instead of pretending generic GGUF conversion works:

```bash
python scripts/check_rabbit_export.py \
  --checkpoint runs/latest/winner.pt \
  --out runs/latest/export-readiness.json
```

A real native GGUF requires either a TinyRabbit llama.cpp architecture/tokenizer implementation or migration of the native brain to an architecture llama.cpp already supports.

## Safety and provenance

- Alpha provider weights are unavailable and are not extracted or bypassed.
- Observable eligible Alpha behavior may be used as behavioral training material.
- The six designated Alpha holdout captures remain evaluation-only.
- Space Bunny checkpoints/GGUFs stay external to Git and retain their own lineage.
- Rabbit-native experiments remain separately labeled.
- Repository code is untrusted input and belongs in the locked sandbox for execution.

Machine-readable lineage policy: `configs/bunny_lineage.json`.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Current capability

- [x] Raw Alpha/Bunny JSON + JSONL capture ingestion
- [x] Exported Kilo/OpenRouter/OpenAI-compatible preview-trace ingestion
- [x] Structured failure → diagnosis → revision → test trajectory capture
- [x] Provider/model provenance with secret redaction
- [x] Compatible owned Rabbit checkpoint continuation
- [x] Capture normalization and quarantine
- [x] Behavior-axis normalization
- [x] Capture provenance + SHA-256 course manifests
- [x] Automatic eligible-training / frozen-holdout routing
- [x] Mechanical Alpha holdout contamination guard
- [x] Complete-six holdout requirement for normal pipeline preparation
- [x] Capture-family train/validation/test isolation
- [x] Bunny behavioral episodes
- [x] Adaptive Bunny curriculum
- [x] TwinTrain + Greenlight promotion
- [x] Rabbit-native GitHub prediction/repair/hidden-diff research
- [x] Leak-resistant file-family splitting
- [x] Equal-compute fixed-vs-adaptive experiment
- [x] Meaningful-winner threshold
- [x] Space Bunny artifact structural inventory
- [x] Rabbit-native checkpoint/export-readiness audit
- [ ] Live provider proxy/logger capture (current provider ingestion uses intentional exports)
- [x] Behavioral score gate wired into Greenlight with fail-closed evidence handling
- [ ] Generate candidate behavior scores automatically from model responses on frozen + rolling challenges
- [ ] True optimizer/scheduler/RNG checkpoint resume
- [ ] Run the newest complete regression suite locally
- [ ] Run capture pipeline against the real preserved Alpha/Bunny capture collection
- [ ] Record the real course hashes and episode counts
- [ ] Evaluate trained candidates against the six frozen Alpha cases with behavioral metrics
- [ ] TinyRabbit llama.cpp architecture/tokenizer mapping or supported architecture migration
- [ ] Rabbit-native GGUF + llama.cpp verification

See `docs/BUNNY_LINEAGE.md` for the preserved lineage boundary.
