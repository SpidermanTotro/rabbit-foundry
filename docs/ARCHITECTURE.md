# Rabbit Foundry architecture

## Current v0.1

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

The current code proves orchestration, reproducibility, evaluation, and promotion mechanics. It is **not yet DreamRabbit** and does not claim AGI.

## Next layer

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
 GGUF converter
        |
 llama.cpp verification
```

## Training principles

- Random initialization is supported.
- No teacher LLM output is required.
- Source code used for learning is still training data and must be tracked.
- Evaluation episodes must be held out from training.
- Executable tests/compilers are preferred as objective judges.
- Promotion is based on held-out measurements, not training loss alone.

## Security principles

GitHub repositories are untrusted. Build/test execution belongs in a disposable
Podman container with networking disabled, read-only source where possible,
bounded CPU/RAM/time, and no host secrets mounted into the container.
