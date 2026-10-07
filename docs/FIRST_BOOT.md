# Rabbit Code first boot

This boot tests the Rabbit Code local model-gateway path, not Rabbit model quality.

## Backend

The first boot routes text-only requests through a local Ollama model
(`qwen2.5-coder:7b` by default) while Rabbit Code exposes the local model ID:

`rabbit-code`

The gateway binds to `127.0.0.1`. Streaming is available on the gateway path.
Tool calling remains disabled until the tool protocol is implemented and tested.

## Run

From an installed Rabbit Code development environment:

```bash
bash scripts/rabbit_code_boot.sh
```

In another terminal:

```bash
bash scripts/smoke_gateway.sh
```

Expected checkpoints:

- `/health` returns `status: ok`
- `/v1/models` lists `rabbit-code`
- the completion request reaches the selected local backend
- the response is returned through the Rabbit Code gateway

## Compatibility clients

OpenCode and Kilo are optional compatibility clients, not dependencies of
Rabbit Code itself.

Current example files:

- `configs/opencode.compat.jsonc`
- `configs/kilo.compat.jsonc`

Only point an external client at port 8765 after the smoke checks pass.

## Legacy transition

The older `scripts/first_boot.sh` and
`rabbit_foundry.absorber_server` entrypoints are legacy transition files.
They are not the current documented Rabbit Code path.

They remain temporarily because existing local work may still depend on them.
The active replacement is:

```text
scripts/rabbit_code_boot.sh
        ↓
rabbit_foundry.model_gateway
        ↓
local model provider
```

## Configuration safety

Back up any real client configuration before merging compatibility provider
settings. Rabbit Code must not copy or expose provider credentials.

Tool calling and mutation permissions remain disabled until verified.
