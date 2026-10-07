# Rabbit Code agent and provider integration

**Rabbit Code** is the user-facing coding-agent platform.

The existing `rabbit_foundry` Python namespace remains its model-training,
provenance, evaluation, and local-inference backend during the transition.

## Core boundary

Rabbit Code uses an **OpenAI-compatible HTTP boundary** for model providers when
that protocol is available.

This lets one agent runtime talk to multiple model backends without rebuilding
the tool loop for every model.

Minimum provider target:

- `GET /v1/models`
- `POST /v1/chat/completions`
- model IDs
- system/user/assistant messages
- streaming
- usage accounting

Agent-capable target:

- tool/function definitions in requests
- tool calls in responses
- tool-call IDs
- tool-result messages
- reliable stop reasons
- sufficiently large context windows

Rabbit Code must advertise only capabilities that actually work.

## Native provider targets

The intended local-first provider set is:

1. owned Rabbit checkpoints through the existing local server
2. Ollama
3. llama.cpp / compatible local servers
4. generic OpenAI-compatible endpoints

External hosted providers are optional connectors, not a requirement for using
Rabbit Code.

## Existing Rabbit backend

The current Rabbit server binds locally and exposes an OpenAI-compatible
interface. That server can become a native Rabbit Code provider without routing
through OpenCode or Kilo.

Default development endpoint:

`http://127.0.0.1:8765/v1`

## OpenCode and Kilo

OpenCode and Kilo are now treated as **optional compatibility integrations**,
not as foundations Rabbit Code depends on.

The project may still provide adapters for them when useful, including research
around current Space Bunny access, but Rabbit Code's agent loop, permissions,
sessions, model routing, and trajectory capture should remain independently
usable.

## Permissions

The agent runtime should separate at least:

- read
- write/edit
- command execution
- network access

A local-model path must not imply unrestricted shell or network permissions.

## Training capture boundary

Native Rabbit Code trajectories can capture:

```text
user task
  → assistant attempt
  → tool call
  → tool result
  → failure
  → diagnosis
  → revision
  → test
  → final
```

Credentials and secrets must be redacted and must never become training data.

Frozen evaluation material remains evaluation-only.

## Current limitation

The existing tiny Rabbit training model is not yet strong enough to be treated
as a production coding agent. Protocol/runtime work can proceed independently
of model capability.

Tool support must remain false for any model/provider that cannot produce valid
tool calls.
