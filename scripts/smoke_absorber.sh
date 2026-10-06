#!/usr/bin/env bash
set -euo pipefail
BASE="${RABBIT_BASE:-http://127.0.0.1:8765}"

echo "== health =="
curl -fsS "$BASE/health"
echo

echo "== models =="
curl -fsS "$BASE/v1/models"
echo

echo "== completion =="
curl -fsS "$BASE/v1/chat/completions"   -H 'Content-Type: application/json'   -d '{
    "model":"rabbit-absorber",
    "stream":false,
    "messages":[
      {"role":"system","content":"Reply briefly. Do not request tools."},
      {"role":"user","content":"Return exactly: RABBIT_FIRST_BOOT_OK"}
    ]
  }'
echo
