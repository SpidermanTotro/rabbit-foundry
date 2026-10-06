# Feeding Rabbit into OpenCode and Kilo

Rabbit Foundry's integration boundary is an **OpenAI-compatible HTTP server**.

## Why this boundary

Both OpenCode and Kilo can be configured with a custom OpenAI-compatible provider.
This lets Rabbit serve one model endpoint instead of implementing a separate
inference adapter for every coding agent.

Minimum compatibility target:

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
- sufficiently large context window

## OpenCode

Use provider ID `rabbit`, point `settings.baseURL` at Rabbit's `/v1` endpoint,
and map `rabbit-local` to the server model ID.

## Kilo

Use provider ID `rabbit`, `@ai-sdk/openai-compatible`, and the same local
`/v1` endpoint. Kilo also supports model discovery from an OpenAI-compatible
models endpoint.

## Important

The current tiny training model is not yet capable enough to be a coding agent.
The protocol adapter can be built and tested before the final model exists.
Tool support must remain advertised as false until Rabbit actually produces
valid tool calls; lying about this capability would make agent debugging much
harder.
