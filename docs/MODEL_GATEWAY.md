# Rabbit Code model gateway

Rabbit Code uses a local model gateway when a provider already exposes an
OpenAI-compatible interface or can be adapted safely.

The gateway is infrastructure, not the product itself. Rabbit Code owns the
agent loop, sessions, permissions, tool execution, and trajectory capture.

## Development path

```text
Rabbit Code
    |
    +-- agent runtime
    |
    +-- model router
            |
            +-- Rabbit-native endpoint
            +-- Ollama
            +-- llama.cpp
            +-- generic OpenAI-compatible endpoint
            +-- optional compatibility adapters
```

## Current compatibility server

The existing `rabbit_foundry.absorber_server` is retained temporarily as the
legacy implementation behind the gateway concept. It can proxy a local Ollama
OpenAI-compatible endpoint and currently supports streaming text while refusing
tool requests.

It should be migrated/replaced behind the new Rabbit Code router instead of
becoming a permanent public abstraction.

## Permission boundary

Routing a model must never grant tools automatically.

At minimum Rabbit Code controls these independently:

- read/search
- write/edit
- command execution
- network
- external-directory access

## Provenance

A routed response should preserve safe metadata such as:

- selected Rabbit Code provider ID
- model ID
- route type
- timestamp/run ID
- local/external source classification

Credentials, authorization headers, and secrets are never trajectory or
training content.

## Training boundary

Provider reachability does not imply training permission. External-model output
must remain non-trainable until the relevant source policy is explicitly
satisfied.

Frozen evaluation cases remain evaluation-only.
