# Rabbit Absorber experiment

The absorber is a reversible compatibility and routing layer between coding
agents and model backends.

## Phase 0 — backup

Back up the active OpenCode and Kilo configuration files and record SHA-256
digests before changing provider/model/permission settings.

## Phase 1 — dry Rabbit

Rabbit receives normal coding-agent prompts but has no mutation permissions.

- read/search: allow
- edit/write: deny
- shell: deny
- network: deny
- external directories: deny
- fallback model: disabled

Record what Rabbit attempted, where it failed, and whether its response obeyed
the agent protocol.

## Phase 2 — supervised Rabbit

Enable edit and shell as **ask**, never automatic allow. Every mutating request
requires user approval.

## Phase 3 — explicit Qwen fallback

If Rabbit fails its confidence/evaluation gate, the router may use a configured
local fallback such as Qwen. Fallback is opt-in and logged. It must never happen
silently.

This gives three measurements:

1. Rabbit alone.
2. Rabbit with supervised tools.
3. Rabbit versus/with an explicitly permitted local fallback.

## Principle

The absorber should not copy hidden reasoning or weights from another model.
It records observable prompts, responses, tool requests, tool results when
permitted, scores, routes, and failures. Those artifacts can become controlled
training/evaluation episodes later.
