# Rabbit Code architecture

Rabbit Code is the user-facing coding-agent platform. The existing
`rabbit_foundry` namespace remains the training, evaluation, provenance, and
owned-model backend.

## Product layer

```text
user / workspace
      |
  Rabbit Code
      |
  +---+-------------------+
  |                       |
agent loop             model router
  |                       |
tools + permissions   local/external providers
  |                       |
  +-----------+-----------+
              |
       trajectory capture
              |
       rabbit_foundry
       training backend
```

The first product milestone is a small, dependable local coding agent rather
than a full IDE clone.

## Existing training backend

```text
bootstrap token stream
        |
        +----> Model A (fresh seed)
        |
        +----> Model B (fresh seed)
                  |
             held-out eval
                  |
            select winner
                  |
        checkpoint + metrics
```

The current backend proves orchestration, reproducibility, evaluation, exact
resume, and promotion mechanics. It does not claim AGI.

## Research/training layer

```text
read-only GitHub sources
        |
license/provenance gate
        |
episode generator
        |
adaptive sampler
        |
  +-----+-----+
  |           |
Model A     Model B
  |           |
  +-----+-----+
        |
held-out Greenlight
        |
 winner checkpoint
        |
 honest export-readiness gate
```

GGUF is not considered ready until a supported architecture/tokenizer mapping
and llama.cpp runtime verification really succeed.

## Training principles

- Random initialization is supported.
- No teacher LLM output is required for the native Rabbit experiment.
- Source code used for learning is training data and must be tracked.
- Evaluation episodes must be held out from training.
- Executable tests/compilers are preferred as objective judges.
- Promotion is based on held-out measurements, not training loss alone.
- External provider output must not become training data without explicit source permission.

## Agent principles

- Local inference is first-class.
- Read, write, execute, and network permissions are separate.
- Tool capability is never advertised before it works.
- Credentials and secrets are never trajectory/training content.
- Local Rabbit, Ollama, llama.cpp, and generic OpenAI-compatible providers share one router boundary.

## Security principles

Repositories are untrusted. Build/test execution belongs in a disposable
sandbox/container with networking disabled by default, read-only source where
possible, bounded CPU/RAM/time, and no host secrets mounted into the container.
