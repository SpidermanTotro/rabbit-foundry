# Milestones

## v0.1 — Foundry heartbeat

Twin random-init models, native checkpointing, held-out loss, experiment ledger.

## v0.2 — Controlled learning substrate

- deterministic source episodes
- immutable commit provenance
- conservative license allowlist
- adaptive curriculum primitive
- Greenlight candidate gate
- Podman sandbox contract

## v0.3 — Live read-only GitHub learner

The next milestone will fetch explicitly configured repositories at pinned commits,
verify declared license metadata, cache source text, split by repository/commit
without leaking evaluation episodes, and feed byte/token episodes to TwinTrain.

No repository code will execute during ingestion.

## v0.4 — Objective coding episodes

Add corruption/repair and hidden-diff reconstruction. Repository test execution
is permitted only through the sandbox runner.

## v0.5 — GGUF-compatible model line

For the first real GGUF deployment, introduce a small Llama-compatible decoder
configuration rather than inventing an unsupported GGUF architecture. Train the
tokenizer on the allowed corpus, export through the official llama.cpp conversion
path, then compare native and GGUF evaluations before promotion.
