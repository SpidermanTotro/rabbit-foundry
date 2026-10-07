#!/usr/bin/env bash
set -euo pipefail
BASE="${RABBIT_BASE:-http://127.0.0.1:8765}"

echo "== Rabbit Code gateway health =="
curl -fsS "$BASE/health"
echo

echo "== models =="
curl -fsS "$BASE/v1/models"
echo

echo "== completion =="
curl -fsS "$BASE/v1/chat/completions"   -H 'Content-Type: application/json'   -d '{
    "model":"rabbit-code",
    "stream":false,
    "messages":[
      {"role":"system","content":"Reply briefly. Do not request tools."},
      {"role":"user","content":"Return exactly: RABBIT_CODE_GATEWAY_OK"}
    ]
  }'
echo
