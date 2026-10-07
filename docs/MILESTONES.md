# Rabbit Code milestones

## v0.1 — Local agent foundation

Build the first usable Rabbit Code runtime:

- interactive CLI
- project/workspace selection
- persistent sessions
- model router
- Rabbit-native provider
- Ollama provider
- generic OpenAI-compatible provider
- read/search/edit/bash tools
- explicit permission policy
- trajectory capture with secret redaction

## v0.2 — Reliable coding loop

- streaming responses
- structured tool calls/results
- test-run integration
- Git status/diff support
- failure → diagnosis → revision loop
- reproducible session/run IDs
- provider/model provenance

## v0.3 — Local model platform

- llama.cpp provider
- model discovery
- per-model capability registry
- context/output limits
- local benchmark harness
- automatic candidate behavior evaluation

## v0.4 — Training integration

- approved Rabbit Code trajectories → training course
- strict source-policy gate
- frozen-evaluation contamination checks
- Greenlight behavior scoring
- exact resume from owned checkpoints

## v0.5 — Native model deployment

- supported architecture/tokenizer path
- real GGUF conversion
- llama.cpp runtime verification
- native-vs-exported evaluation parity

## Compatibility track

OpenCode and Kilo remain optional adapters only. They may be useful for
interoperability or Space Bunny research, but Rabbit Code must remain usable
without either product.

## Non-goals

Rabbit Code v0.x does not claim AGI and does not pretend unsupported tool,
vision, export, or provider capabilities work.
