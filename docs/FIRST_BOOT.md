# Rabbit Code first boot

This boot tests the Rabbit Code integration path, not Rabbit model quality.

## Backend

The first boot routes text-only requests through a local Ollama model
(`qwen2.5-coder:7b` by default) while compatibility clients see
`rabbit-absorber`.

The proxy binds to `127.0.0.1` and refuses streaming and tool requests until
those capabilities are verified. That makes failures obvious instead of
silently pretending agent support exists.

## Run

From an installed Rabbit Code development environment:

```bash
bash scripts/first_boot.sh
```

In another terminal:

```bash
bash scripts/smoke_absorber.sh
```

Expected checkpoints:

- `/health` returns `status: ok`
- `/v1/models` lists `rabbit-absorber`
- the completion request reaches the selected local backend and returns through Rabbit

## Compatibility clients

OpenCode and Kilo are optional compatibility clients now, not dependencies of
Rabbit Code itself.

Only point an external client at port 8765 after the smoke checks pass.

## Do not overwrite configs blindly

Use the example first-boot configs as references. Back up any real client
configuration before merging provider settings.

Streaming compatibility is a separate gate. Tool calling comes after valid
streaming/tool protocol support, and mutation permissions remain disabled until
verified.
