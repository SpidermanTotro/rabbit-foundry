# Rabbit Foundry — DreamRabbit Architecture

An experimental 2026-style model architecture: **Recursive Core + Expert Pool + Neural Verifier + Episodic Memory**.

```
Input → Encoder → RECURSIVE CORE → Router → Expert Pool
                    ↑      │          │
                    │      ↓          ↓
                    │   Verifier   Tool/Code Head
                    │      │          │
                    └──── retry ←─────┘
                          │
                       Output
```

## Architecture Components

| Component | File | Description |
|-----------|------|-------------|
| **RecursiveCore** | `src/recursive_core.py` | Dynamic-depth reasoning (1-4 recursions), learned controller |
| **ExpertPool** | `src/experts.py` | 16 specialized experts (Python, debugging, planning, shell, git, math, language, tool_use, memory, 7×general) |
| **MoELayer** | `src/experts.py` | Top-2 routing with load balancing |
| **NeuralVerifier** | `src/verifier.py` | 5 heads: confidence, contradiction, tool_needed, retry, hallucination |
| **CompressedMemory** | `src/memory.py` | Learned compression, importance-weighted retrieval |
| **WorkingMemory** | `src/memory.py` | Context buffer across recursive steps |
| **DreamRabbit** | `src/model.py` | Integrated model (~14M active params tiny, ~25M small) |

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run tests
```bash
python tests/test_model.py
```

### 3. Train a model (coming soon)
```bash
python -m src.train --config tiny --steps 10000
```

### 4. Export to GGUF
```bash
python -m src.gguf_export --checkpoint checkpoints/dreamrabbit-tiny.pt --output models/dreamrabbit.gguf --quantization Q4_K
```

### 5. Start inference server
```bash
./scripts/launch_server.sh ./checkpoints/dreamrabbit-tiny
# Server runs at http://localhost:8080/v1 (OpenAI-compatible)
```

### 6. Connect Alpha Rabbit / Kilo / OpenRouter client
```bash
python scripts/connect_client.py --test
# Or generate configs:
python scripts/connect_client.py --output-dir ./configs
```

## Connecting External Clients

### Alpha Rabbit
```bash
export ALPHA_RABBIT_MODEL_ENDPOINT=http://localhost:8080/v1
export ALPHA_RABBIT_MODEL=dreamrabbit
alpha-rabbit run
```

### Kilo
```bash
# Use generated kilo.jsonc or add to your config:
{
  "provider": {
    "local-dreamrabbit": {
      "options": { "baseURL": "http://localhost:8080/v1", "apiKey": "not-needed" },
      "models": { "dreamrabbit": { "name": "DreamRabbit Local" } }
    }
  },
  "enabled_providers": ["local-dreamrabbit"]
}
```

### OpenRouter-compatible
```bash
# Point any OpenRouter client to local endpoint
export OPENROUTER_BASE_URL=http://localhost:8080/v1
export OPENROUTER_API_KEY=local
```

### curl / OpenAI SDK
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"dreamrabbit","messages":[{"role":"user","content":"Write a Python function"}],"max_tokens":200}'
```

## Docker Deployment

```bash
# Build and run full stack
docker-compose up --build

# Services:
# - dreamrabbit:8080  — Inference server (GPU)
# - ollama:11434      — Comparison models
# - evaluator         — Sandboxed evaluation (no network)
# - trainer           — Training worker
```

## Project Structure

```
rabbit-foundry/
├── src/
│   ├── config.py           # Model configurations
│   ├── recursive_core.py   # Dynamic-depth reasoning core
│   ├── experts.py          # Expert pool + MoE routing
│   ├── verifier.py         # Neural verifier (self-correction)
│   ├── memory.py           # Episodic + working memory
│   ├── model.py            # Integrated DreamRabbit model
│   ├── training.py         # Losses, optimizer, scheduler
│   ├── inference_server.py # OpenAI-compatible API server
│   └── gguf_export.py      # GGUF conversion for llama.cpp
├── tests/
│   └── test_model.py       # Architecture tests
├── scripts/
│   ├── launch_server.sh    # Start inference server
│   └── connect_client.py   # Generate client configs
├── docker-compose.yml      # Full stack deployment
├── Dockerfile              # Sandboxed evaluation container
├── requirements.txt        # Python dependencies
└── kilo.jsonc              # Kilo configuration
```

## Next Steps (Rabbit Foundry v0.1)

- [ ] Training loop with TwinTrain (Model A vs B)
- [ ] GitHub repository ingestion → episode generator
- [ ] Sandboxed evaluator (disposable containers, no network)
- [ ] Adaptive resampling based on failure patterns
- [ ] Greenlight promotion gate (frozen held-out tests)
- [ ] Automated GGUF export + llama.cpp verification

## License

MIT