#!/usr/bin/env bash
set -euo pipefail

MODEL="${RABBIT_UPSTREAM_MODEL:-qwen2.5-coder:7b}"
PORT="${RABBIT_PORT:-8765}"
BASE="http://127.0.0.1:$PORT"

echo "== Rabbit Code boot =="

health="$(curl -fsS --max-time 2 "$BASE/health" 2>/dev/null || true)"
if [[ -n "$health" ]]; then
  if printf '%s' "$health" | python3 -c '
import json, sys
try:
    payload = json.load(sys.stdin)
except Exception:
    raise SystemExit(1)
ok = (
    payload.get("status") == "ok"
    and payload.get("mode") == "rabbit-code-model-gateway"
    and payload.get("model") == "rabbit-code"
)
raise SystemExit(0 if ok else 1)
'; then
    echo "Rabbit Code gateway is already running on localhost:$PORT"
    echo "$health"
    exit 0
  fi

  echo "Port $PORT is already serving HTTP, but it is not the Rabbit Code gateway."
  echo "Response: $health"
  exit 3
fi

if python3 - "$PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    occupied = sock.connect_ex(("127.0.0.1", port)) == 0
raise SystemExit(0 if occupied else 1)
PY
then
  echo "Port $PORT is already in use by another process."
  echo "Inspect it with: ss -ltnp 'sport = :$PORT'"
  exit 3
fi

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
