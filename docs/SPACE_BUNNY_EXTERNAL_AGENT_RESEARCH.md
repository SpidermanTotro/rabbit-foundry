# External Space Bunny agent research

Date: 2026-10-07

Purpose: preserve external evidence relevant to Rabbit Foundry's Space Bunny lineage and agent-integration work. This is an examination note, not an identity proof or an authentication recipe.

## 1. Agent Solo / Lynx

Source: https://lynx.luxferre.top/

The Lynx author describes **Agent Solo**, a deliberately minimal coding agent built around a single Bash tool. The documented runtime dependencies are Bash plus `curl`, `jq`, and `timeout`.

The author documents configuration variables including:

- `SOLO_ENDPOINT`
- `SOLO_MODEL`
- `SOLO_API_KEY`
- `SOLO_TOOLS`
- `SOLO_SKILLS`

Most importantly for this repository, the author states that Solo itself was developed using free models from the Kilo Code endpoint, predominantly:

`stealth/space-bunny-alpha`

The author also says the bootstrap path used **Bantam** and then **Solo itself** with Kilo's endpoint.

### Why this matters

This is independent evidence that the historical Space Bunny Alpha model was practically usable for agentic coding through Kilo, including a very small client architecture.

It does **not** establish that today's `opencode/space-bunny-free` is the same model.

## 2. Historical Kilo model record

Source: https://kilo.ai/models/stealth-space-bunny-alpha

Kilo's model page records:

- model ID: `stealth/space-bunny-alpha`
- created: 2026-09-23
- context: 1,000,000 tokens
- maximum output: 524,288 tokens
- inputs: text, image, video
- function calling: supported
- structured output: supported
- reasoning: supported
- historical price: free

The page labels the model as retiring October 5.

This corroborates the historical alias already recorded in `configs/space_bunny_sources.json`.

## 3. Relationship to current OpenCode candidate

Rabbit Foundry currently treats:

`opencode/space-bunny-free`

as a **continuation candidate**, not proven historical Alpha identity.

The local OpenCode CLI has separately been observed successfully invoking this current model. That observation should be preserved separately from the external sources above.

Identity/lineage should continue to be assessed using the six frozen Alpha behavioral cases:

- `011-partial-failure`
- `014-instruction-conflict`
- `016-guess-discipline`
- `028-ambiguous-request`
- `031-test-first-request`
- `034-anti-sycophancy`

Those cases remain evaluation-only and must never enter training.

## 4. Architecture lesson worth testing

Agent Solo demonstrates that a useful coding agent does not require a large integration layer. Its general shape is:

```text
coding agent
    |
    +-- tool loop
    |
    +-- HTTP/model endpoint
            |
            +-- model
```

For Rabbit Foundry, a supported experiment is therefore:

```text
Kilo
  |
  +-- supported local/OpenAI-compatible bridge
          |
          +-- OpenCode-supported model interface
                  |
                  +-- opencode/space-bunny-free
```

This is only an architecture hypothesis until the interfaces are verified end-to-end.

Do not extract, scrape, copy, or hard-code OpenCode/Kilo credentials to make this work. Prefer documented provider/server interfaces, and keep Rabbit's existing local provider available alongside any Space Bunny route.

## 5. Examination questions

1. Can Kilo talk directly to a documented OpenCode server/provider interface?
2. If not, can Rabbit Foundry expose a small local OpenAI-compatible bridge while OpenCode remains responsible for its own supported authentication?
3. Can tool calls, tool results, stop reasons, streaming, and context limits survive that bridge correctly?
4. Does current `opencode/space-bunny-free` match the six frozen Alpha behavioral signals strongly enough to support a continuation-lineage claim?
5. What provenance must be captured so historical Alpha, current Space Bunny, and local Rabbit are never conflated?
6. What provider terms/permissions apply before any current provider output could be promoted into training data?

## 6. Safety / provenance boundary

- Historical Alpha provider weights are not owned and are not to be extracted.
- Frozen Alpha fingerprint cases are evaluation-only.
- Behavioral similarity is evidence of lineage/continuation, not proof of model identity.
- Provider credentials are never training data and must not be logged.
- Current provider outputs must not be promoted to training merely because an endpoint is technically reachable; source permission must be explicitly established and the pipeline should fail closed otherwise.

## Next engineering task

Inspect the documented OpenCode server/interface and Kilo custom-provider interface, then build a minimal proof-of-concept bridge only if needed. Test it with a harmless current-model request first. Keep local Rabbit and Space Bunny as separate selectable providers/models.
