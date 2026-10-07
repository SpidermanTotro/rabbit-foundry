#!/usr/bin/env bash
set -euo pipefail

MODEL="${RABBIT_UPSTREAM_MODEL:-qwen2.5-coder:7b}"
PORT="${RABBIT_PORT:-8765}"

echo "== Rabbit Code boot =="
echo "Checking Ollama..."
curl -fsS http://127.0.0.1:11434/api/tags >/dev/null || {
  echo "Ollama is not reachable on localhost:11434"
  echo "Start it with: ollama serve"
  exit 1
}

echo "Checking model: $MODEL"
if ! ollama list | awk 'NR>1 {print $1}' | grep -Fxq "$MODEL"; then
  echo "Model is not installed: $MODEL"
  echo "No download was started automatically."
  exit 2
fi

echo "Starting Rabbit Code model gateway on localhost:$PORT"
python -m rabbit_foundry.model_gateway   --port "$PORT"   --upstream http://127.0.0.1:11434/v1   --upstream-model "$MODEL"
