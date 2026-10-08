# Rabbit Code 🐇💻

**Rabbit Code** is a local-first, open coding-agent platform built around your own models, your own tools, and your own machine.

It is being built as an alternative to depending on paywalled coding-agent platforms. Rabbit Code itself has no subscription gate. Local models can run through Rabbit-native checkpoints, Ollama, llama.cpp, or any compatible local endpoint. External providers can remain optional adapters when the user deliberately chooses them.

The existing `rabbit_foundry` Python package remains the model-training and research backend while the new Rabbit Code agent runtime is built on top.

## Vision

```text
                        Rabbit Code
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
        Chat            Agent Loop        Model Router
          │                 │                 │
          │          ┌──────┼──────┐          │
          │          │      │      │          │
          │        read    edit   bash        │
          │        grep    test   git         │
          │                                   │
          └─────────────────┬─────────────────┘
                            │
                     Provider Interface
                            │
              ┌─────────────┼──────────────┐
              │             │              │
         Rabbit local     Ollama        llama.cpp
              │
         owned checkpoints

              optional external providers
              only when explicitly enabled
```

The goal is not to clone every feature of Kilo, OpenCode, or a full IDE at once. The first target is a small, reliable coding agent with strong local-model support, explicit permissions, reproducible tool execution, and native trajectory capture.

## Product principles

- **Local first** — local inference is a first-class path, not a fallback.
- **No Rabbit Code paywall** — the application does not require a subscription.
- **Bring your own model** — Rabbit, Ollama, llama.cpp, and OpenAI-compatible endpoints.
- **Explicit permissions** — read operations can be allowed while edits, execution, and network access remain separately controlled.
- **Observable agent behavior** — tool calls, failures, corrections, tests, and final answers can be captured with provenance.
- **No credential harvesting** — provider authentication remains provider-owned and is never training data.
- **Honest capability flags** — tools, streaming, vision, export, and context support are only advertised when they really work.
- **Training/evaluation separation** — frozen evaluation cases never silently become training material.

## Architecture

Rabbit Code is the user-facing coding-agent platform.

`rabbit_foundry` is currently its backend for:

- owned-model training
- checkpoint/resume
- behavioral capture
- provenance
- Greenlight promotion
- Rabbit-native inference
- Alpha/Space Bunny lineage research
- OpenAI-compatible local serving

That lets us evolve the product without breaking the already-working training namespace.

## Linux workspace interface (experimental v0.1)

Rabbit Code now has a responsive **local browser interface** alongside the
original terminal application. It is a GUI for the same local Python runtime,
not a remote coding service. The first version includes chat, a read-only agent,
a workspace file explorer/editor, Git status/diff, and resumable sessions.

**Installation** (from this checkout):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

**Optional — start the original Rabbit model gateway** (requires local Ollama):

```bash
bash scripts/rabbit_code_boot.sh
```

**Launch the interface**:

```bash
rabbit-code --workspace "$HOME/your-project" --ui --open-browser
```

Or run directly from this source checkout without relying on a globally installed
Rabbit Code version:

```bash
bash scripts/rabbit_code_linux_ui.sh "$HOME/your-project"
```

The browser opens at `http://127.0.0.1:8766/`. Without `--open-browser`,
open that address yourself. Pick a free port with `--ui-port 8877`.
Use the **Local model** selector to switch to any model already installed
in your loopback Ollama service, or switch back to the original Rabbit gateway.
Direct Ollama selection does not require the separate Rabbit gateway process.
Neither selection pulls or downloads weights. Ollama must already be running
locally. To keep using the terminal engine, run `rabbit-code --workspace .`.

**Safety:** The interface binds to IPv4 loopback only, rejects nonlocal
model endpoints, uses a per-process anti-CSRF token, checks browser
Host/Origin, and blocks network routing. Agent mode cannot approve writes or
sandbox execution. File-editor changes need explicit confirmation and matching
on-disk SHA-256; concurrent edits are rejected instead of overwritten. The
file editor is for existing UTF-8 text files up to 128 KiB; it is not yet a
full Monaco/VS Code editor. Browser chats are request/response in v0.1; CLI
streaming remains available.

### Fedora and Linux desktop app-menu launcher

From the Rabbit Code repository checkout, install a **user-local** launcher:

```bash
python3 scripts/install_linux_desktop.py --workspace "$HOME/your-project"
```

Search for **Rabbit Code** in the GNOME/KDE application launcher.
It opens this local web UI in your normal browser without using a terminal.
The installer writes only to `~/.local/bin/rabbit-code-gui` and
`~/.local/share/applications/rabbit-code.desktop`. No `sudo` is needed.
To remove the launcher:

```bash
python3 scripts/install_linux_desktop.py --uninstall
```

This is a locally served web GUI with a Linux app-menu shortcut,
**not yet a packaged GTK/Qt desktop app**.
Do not open the server through a reverse proxy, public tunnel, or LAN bind.
There is no cloud account, CDN, or browser-side API key.

### Linux interface tests

```bash
python -m pytest tests/test_web_ui.py tests/test_linux_desktop.py -q
python -m pytest -q
```

Long-running model calls are synchronous in v0.1; use smaller tasks if
the UI appears busy, and use the terminal for advanced or batch operations.

## Rabbit Code runtime

The first real Rabbit Code runtime is now in-tree.

Implemented:

- [x] interactive `rabbit-code` CLI
- [x] persistent/resumable coding sessions
- [x] project/workspace boundary with path-escape protection
- [x] local-first model router
- [x] generic OpenAI-compatible provider path
- [x] Rabbit Code local gateway model
- [x] read tool
- [x] write + exact-edit tools
- [x] grep/glob/list tools
- [x] Git status/diff inspection
- [x] explicit read/write/execute/network permission policy
- [x] model-driven JSON tool loop with a hard step limit
- [x] no-network, read-only Podman sandbox execution
- [x] native session/trajectory capture
- [x] safe secret redaction
- [x] local-vs-remote provider network gate
- [x] OpenAI-style native function/tool-call transport
- [x] streaming through the Rabbit Code model router and CLI
- [x] live gateway protocol/capability probing
- [x] automatic JSON/context fallback for models that do not use native tools reliably
- [x] local-only Linux browser workspace (experimental v0.1)
- [x] installed Ollama model selection (no downloads or remote inference)
- [x] per-user GNOME/KDE app-menu launcher
- [ ] native GTK/Qt app, advanced code editor, and integrated terminal

Native tool calling is a transport capability, not a guarantee that every local
model will choose or format tool calls correctly. Rabbit Code prefers native
OpenAI-style `tool_calls` when the provider and running gateway advertise
support. If a model ignores the protocol, Rabbit Code falls back to its JSON
tool loop and safe read-only repository context.

The gateway health response includes a protocol version. A gateway process
started from older code must be restarted before new transport features become
active.

### Native tool-call diagnostic

To check whether the **currently running local model** emits native OpenAI-style
tool calls (not merely whether the gateway forwards tool definitions), run:

```bash
python3 scripts/check_native_tools.py
```

The probe uses an inert `rabbit_probe` function definition. It never executes
a tool or reads workspace files. It sends a fixed synthetic prompt only to a
loopback endpoint; check `/health` to verify the gateway's upstream is your
local Ollama instance. Exit code `0` means the expected tool call was returned;
`2` means the model did not return a matching call; `1` means the probe
request failed.
A nonpassing result does not disable Rabbit Code's JSON/context fallback.

### Quick start

Start the local gateway:

```bash
bash scripts/rabbit_code_boot.sh
```

Then, from a project you want Rabbit Code to inspect:

```bash
rabbit-code --workspace /path/to/project
```

Useful commands inside the interactive `rabbit>` prompt:

```text
/capabilities
/list *.py
/read README.md
/grep TODO **/*.py
/git-status
/git-diff
/agent inspect this project and explain the most important bug
```

The same commands can be run directly from Bash without entering the interactive
prompt:

```bash
rabbit-code --workspace . capabilities
rabbit-code --workspace . git-status
rabbit-code --workspace . list "*.py"
rabbit-code --workspace . read README.md
rabbit-code --workspace . agent "inspect this repository and tell me what needs fixing"
rabbit-code --workspace . --stream chat "summarize the project"
```

Typing `Ctrl-C` at `rabbit>` exits Rabbit Code and returns to Bash. A bare
`/read` or `/agent` typed after that is a shell path, not a Rabbit Code
command.

Writes remain approval-gated unless Rabbit Code is launched with
`--allow-write`.

Sandbox execution remains approval-gated unless launched with
`--allow-exec`, and it is still routed through the existing Podman sandbox
with networking disabled and the workspace mounted read-only.

Resume an existing conversation with:

```bash
rabbit-code --workspace /path/to/project --session-id SESSION_ID
```

Normal chat history is restored. Agent protocol/tool chatter is kept separate
from normal chat context.

Later targets:

- desktop/web UI
- VS Code integration
- richer tool-call protocol
- multi-agent workflows
- model benchmarking
- local training-from-approved-trajectories
- optional OpenCode/Kilo compatibility adapters

## Existing Rabbit backend

The repository already contains a substantial working backend.

### Alpha/Bunny capture → training pipeline

```text
raw Alpha/Bunny captures
        ↓
capture-course normalization
        ↓
axis + provenance audit
        ↓
┌──────────────────────┬────────────────────────┐
│ eligible behavior    │ six frozen Alpha cases │
│ training             │ evaluation only        │
└──────────┬───────────┴────────────┬───────────┘
           ↓                        ↓
 Bunny training episodes       frozen holdout
           ↓                        │
           └──────── TwinTrain ─────┘
                       ↓
                   Greenlight
                       ↓
                   winner.pt
```

The six frozen Alpha IDs are never permitted into training:

- `011-partial-failure`
- `014-instruction-conflict`
- `016-guess-discipline`
- `028-ambiguous-request`
- `031-test-first-request`
- `034-anti-sycophancy`

### Prepare captured behavior

```bash
python scripts/prepare_alpha_bunny_training.py \
  captures/*.jsonl \
  --out-dir runs/alpha-bunny \
  --window 128
```

The pipeline normalizes captures, quarantines malformed inputs, hashes source/course files, separates eligible training from frozen holdout data, and produces deterministic training episodes.

### Train an owned Rabbit checkpoint

```bash
python -m rabbit_foundry.train \
  --episodes runs/alpha-bunny/training_episodes.json \
  --episode-kind bunny \
  --sampling adaptive \
  --minimum-relative-improvement 0.01 \
  --run-dir runs/alpha-bunny/train \
  --device auto
```

Exact resume preserves model weights, AdamW state, completed steps, Python/Torch/CUDA RNG, and adaptive curriculum state.

### Local Rabbit endpoint

An owned Rabbit checkpoint can be exposed through the existing OpenAI-compatible preview server:

```bash
python -m rabbit_foundry.preview_server \
  --checkpoint runs/latest/winner.pt \
  --capture runs/preview/captures.jsonl
```

Default endpoint:

```text
http://127.0.0.1:8765/v1
```

Implemented routes:

- `GET /v1/models`
- `POST /v1/chat/completions`

This server becomes one of Rabbit Code's native model-provider paths.

## Space Bunny / Alpha research

Historical Space Bunny Alpha:

```text
stealth/space-bunny-alpha
```

Current OpenCode candidate under investigation:

```text
opencode/space-bunny-free
```

The current model is treated only as an **unverified continuation candidate**. Behavioral similarity does not prove model identity.

See:

- `docs/BUNNY_LINEAGE.md`
- `docs/SPACE_BUNNY_EXTERNAL_AGENT_RESEARCH.md`

The six frozen Alpha fingerprint cases remain evaluation-only.

## Greenlight

Greenlight promotes candidates only from measured evidence. Non-finite candidates fail, behavior scores can be required, and held-out evidence is not silently replaced with a fake pass value.

A promoted owned model is saved as:

```text
winner.pt
```

## Export boundary

The current TinyRabbitLM is a custom byte-level PyTorch architecture. It is **not** automatically GGUF-compatible.

A real GGUF pass requires either:

1. actual TinyRabbit support in llama.cpp, or
2. migration to an architecture/tokenizer llama.cpp already supports.

Rabbit Code must not report GGUF readiness unless conversion and runtime verification genuinely succeed.

## Safety and provenance

- Provider weights are not extracted or claimed as owned.
- Credentials are never training data.
- Provider/model provenance is retained.
- Frozen Alpha evaluation cases never enter training.
- Repository code is untrusted input for execution.
- Network access is independently permissioned.
- Current provider outputs are not automatically promoted into training.
- External-output training must be explicitly permitted by applicable source policy.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The Python import namespace remains:

```python
import rabbit_foundry
```

even though the public product name is now **Rabbit Code**.

## Current backend capability

- [x] Alpha/Bunny JSON + JSONL capture ingestion
- [x] provider-preview trace ingestion
- [x] structured repair trajectory capture
- [x] provider/model provenance
- [x] secret redaction
- [x] compatible checkpoint continuation
- [x] exact optimizer/RNG/curriculum resume
- [x] frozen-holdout contamination guard
- [x] family-safe train/validation splitting
- [x] adaptive Bunny curriculum
- [x] TwinTrain + Greenlight
- [x] Rabbit-native GitHub prediction/repair/hidden-diff research
- [x] local OpenAI-compatible Rabbit endpoint
- [x] export-readiness audit
- [x] Rabbit Code JSON agent loop
- [x] Rabbit Code local-first model router
- [x] Rabbit Code workspace/tool runtime
- [x] Rabbit Code resumable session store
- [x] Rabbit Code CLI
- [x] no-network Podman sandbox execution
- [x] Git status/diff inspection
- [x] native function-call transport + streaming
- [x] live gateway protocol probing + stale-gateway fallback
- [ ] automatic behavior scoring from real candidate responses
- [ ] verified native GGUF runtime

## Repository naming

Public product: **Rabbit Code**

Current GitHub slug: `rabbit-foundry`

Target GitHub slug: `rabbit-code`

Python backend/import namespace: `rabbit_foundry`

The package distribution is already named `rabbit-code`. The GitHub repository
slug still needs to be renamed through GitHub repository settings because the
current connector does not expose repository-admin rename actions.

After that rename, existing local clones can be repointed with:

```bash
git remote set-url origin https://github.com/SpidermanTotro/rabbit-code.git
```

The `rabbit_foundry` Python namespace remains temporarily for import
compatibility while the new Rabbit Code runtime is introduced.
