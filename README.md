# Rabbit Code 🐇💻

**Rabbit Code** is a local-first, open coding-agent platform built around your own models, your own tools, and your own machine.

It is being built as an alternative to depending on paywalled coding-agent platforms. Rabbit Code itself has no subscription gate. Local models can run through Rabbit-native checkpoints, Ollama, llama.cpp, or any compatible local endpoint. External providers can remain optional adapters when the user deliberately chooses them.

The existing `rabbit_foundry` Python package remains the model-training and research backend while the new Rabbit Code agent runtime is built on top.

## Vision

```text
                        Rabbit Code
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
        Chat            Agent Loop        Model Router
          │                 │                 │
          │          ┌──────┼──────┐          │
          │          │      │      │          │
          │        read    edit   bash        │
          │        grep    test   git         │
          │                                   │
          └─────────────────┬─────────────────┘
                            │
                     Provider Interface
                            │
              ┌─────────────┼──────────────┐
              │             │              │
         Rabbit local     Ollama        llama.cpp
              │
         owned checkpoints

              optional external providers
              only when explicitly enabled
```

The goal is not to clone every feature of Kilo, OpenCode, or a full IDE at once. The first target is a small, reliable coding agent with strong local-model support, explicit permissions, reproducible tool execution, and native trajectory capture.

## Product principles

- **Local first** — local inference is a first-class path, not a fallback.
- **No Rabbit Code paywall** — the application does not require a subscription.
- **Bring your own model** — Rabbit, Ollama, llama.cpp, and OpenAI-compatible endpoints.
- **Explicit permissions** — read operations can be allowed while edits, execution, and network access remain separately controlled.
- **Observable agent behavior** — tool calls, failures, corrections, tests, and final answers can be captured with provenance.
- **No credential harvesting** — provider authentication remains provider-owned and is never training data.
- **Honest capability flags** — tools, streaming, vision, export, and context support are only advertised when they really work.
- **Training/evaluation separation** — frozen evaluation cases never silently become training material.

## Architecture

Rabbit Code is the user-facing coding-agent platform.

`rabbit_foundry` is currently its backend for:

- owned-model training
- checkpoint/resume
- behavioral capture
- provenance
- Greenlight promotion
- Rabbit-native inference
- Alpha/Space Bunny lineage research
- OpenAI-compatible local serving

That lets us evolve the product without breaking the already-working training namespace.

## v0.1 build target

Rabbit Code v0.1 should provide:

- [ ] interactive CLI/TUI
- [ ] persistent coding sessions
- [ ] project/workspace selection
- [ ] model router
- [ ] Rabbit-native provider
- [ ] Ollama provider
- [ ] generic OpenAI-compatible provider
- [ ] read tool
- [ ] write/edit tool
- [ ] grep/glob/list tools
- [ ] bash tool
- [ ] test runner integration
- [ ] Git status/diff support
- [ ] permission policy
- [ ] streaming responses
- [ ] native trajectory capture
- [ ] safe secret redaction
- [ ] selectable local/external network policy

Later targets:

- desktop/web UI
- VS Code integration
- richer tool-call protocol
- multi-agent workflows
- model benchmarking
- local training-from-approved-trajectories
- optional OpenCode/Kilo compatibility adapters

## Existing Rabbit backend

The repository already contains a substantial working backend.

### Alpha/Bunny capture → training pipeline

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

```bash
python scripts/prepare_alpha_bunny_training.py \
  captures/*.jsonl \
  --out-dir runs/alpha-bunny \
  --window 128
```

The pipeline normalizes captures, quarantines malformed inputs, hashes source/course files, separates eligible training from frozen holdout data, and produces deterministic training episodes.

### Train an owned Rabbit checkpoint

```bash
python -m rabbit_foundry.train \
  --episodes runs/alpha-bunny/training_episodes.json \
  --episode-kind bunny \
  --sampling adaptive \
  --minimum-relative-improvement 0.01 \
  --run-dir runs/alpha-bunny/train \
  --device auto
```

Exact resume preserves model weights, AdamW state, completed steps, Python/Torch/CUDA RNG, and adaptive curriculum state.

### Local Rabbit endpoint

An owned Rabbit checkpoint can be exposed through the existing OpenAI-compatible preview server:

```bash
python -m rabbit_foundry.preview_server \
  --checkpoint runs/latest/winner.pt \
  --capture runs/preview/captures.jsonl
```

Default endpoint:

```text
http://127.0.0.1:8765/v1
```

Implemented routes:

- `GET /v1/models`
- `POST /v1/chat/completions`

This server becomes one of Rabbit Code's native model-provider paths.

## Space Bunny / Alpha research

Historical Space Bunny Alpha:

```text
stealth/space-bunny-alpha
```

Current OpenCode candidate under investigation:

```text
opencode/space-bunny-free
```

The current model is treated only as an **unverified continuation candidate**. Behavioral similarity does not prove model identity.

See:

- `docs/BUNNY_LINEAGE.md`
- `docs/SPACE_BUNNY_EXTERNAL_AGENT_RESEARCH.md`

The six frozen Alpha fingerprint cases remain evaluation-only.

## Greenlight

Greenlight promotes candidates only from measured evidence. Non-finite candidates fail, behavior scores can be required, and held-out evidence is not silently replaced with a fake pass value.

A promoted owned model is saved as:

```text
winner.pt
```

## Export boundary

The current TinyRabbitLM is a custom byte-level PyTorch architecture. It is **not** automatically GGUF-compatible.

A real GGUF pass requires either:

1. actual TinyRabbit support in llama.cpp, or
2. migration to an architecture/tokenizer llama.cpp already supports.

Rabbit Code must not report GGUF readiness unless conversion and runtime verification genuinely succeed.

## Safety and provenance

- Provider weights are not extracted or claimed as owned.
- Credentials are never training data.
- Provider/model provenance is retained.
- Frozen Alpha evaluation cases never enter training.
- Repository code is untrusted input for execution.
- Network access is independently permissioned.
- Current provider outputs are not automatically promoted into training.
- External-output training must be explicitly permitted by applicable source policy.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The Python import namespace remains:

```python
import rabbit_foundry
```

even though the public product name is now **Rabbit Code**.

## Current backend capability

- [x] Alpha/Bunny JSON + JSONL capture ingestion
- [x] provider-preview trace ingestion
- [x] structured repair trajectory capture
- [x] provider/model provenance
- [x] secret redaction
- [x] compatible checkpoint continuation
- [x] exact optimizer/RNG/curriculum resume
- [x] frozen-holdout contamination guard
- [x] family-safe train/validation splitting
- [x] adaptive Bunny curriculum
- [x] TwinTrain + Greenlight
- [x] Rabbit-native GitHub prediction/repair/hidden-diff research
- [x] local OpenAI-compatible Rabbit endpoint
- [x] export-readiness audit
- [ ] Rabbit Code agent loop
- [ ] Rabbit Code local model router
- [ ] Rabbit Code tool runtime
- [ ] Rabbit Code session store
- [ ] Rabbit Code CLI/TUI
- [ ] automatic behavior scoring from real candidate responses
- [ ] verified native GGUF runtime

## Repository naming

The repository is still named `rabbit-foundry` during the transition so existing clones, imports, scripts, and links keep working.

Public product: **Rabbit Code**

Backend/import namespace: **rabbit_foundry**

A future repository/package rename can happen after the new agent runtime has its own stable entry point.
