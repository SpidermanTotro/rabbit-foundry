# First boot

This boot tests the Rabbit integration machine, not Rabbit model quality.

## Backend

The first boot routes text-only requests through a local Ollama model
(`qwen2.5-coder:7b` by default) while OpenCode/Kilo see only
`rabbit-absorber`.

The proxy binds to `127.0.0.1` and refuses streaming and tool requests.
That makes failures obvious instead of silently pretending agent support exists.

## Run

From an installed Rabbit Foundry environment:

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
- the completion request reaches local Qwen and returns through Rabbit

Only after those pass should OpenCode or Kilo be pointed at port 8765.

## Do not overwrite configs yet

Use the example first-boot configs as references. Back up the real agent
configuration before replacing or merging provider settings.

The next gate is streaming compatibility. Tool calling comes after streaming,
and mutation permissions remain disabled until both are verified.
