# Rabbit Foundry 🐇🏭

Rabbit Foundry is an experimental local-first model-building system.

## v0.1 goal

Prove a small, falsifiable loop:

1. create two randomly initialized language models;
2. train them independently on the same token stream;
3. evaluate both on held-out data;
4. select the better checkpoint;
5. preserve provenance and metrics;
6. later add adaptive curriculum, sandboxed GitHub episodes, Greenlight gates, and GGUF export.

No pretrained model weights or teacher-model answers are required by the core experiment.

## Safety boundary

Repository code is untrusted input. Future GitHub-derived build/test episodes must run in disposable containers with networking disabled and resource/time limits. Rabbit Foundry must not autonomously push, open issues, or create pull requests in source repositories.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m rabbit_foundry.train --steps 20 --device auto
```

Outputs are written beneath `runs/`.

## Roadmap

- [x] Tiny causal LM
- [x] Twin A/B training controller
- [x] Held-out evaluation and winner selection
- [x] JSON experiment ledger
- [ ] Adaptive weakness-driven resampling
- [ ] Read-only GitHub provenance/episode builder
- [ ] Podman sandbox runner
- [ ] Greenlight promotion gates
- [ ] GGUF conversion adapter
- [ ] llama.cpp post-export verification
