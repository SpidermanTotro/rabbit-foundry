#!/usr/bin/env bash
# Start Rabbit Code local Linux UI from the repository checkout.
# Model gateway runs independently in another terminal.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="${1:-$PWD}"

if [[ ! -d "$WORKSPACE" ]]; then
  echo "Rabbit Code: workspace does not exist: $WORKSPACE" >&2
  exit 2
fi

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

# Import from this checkout instead of an older system installation.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Opening Rabbit Code Linux UI for: $WORKSPACE"
echo "Start the gateway separately with: bash scripts/rabbit_code_boot.sh"
exec "$PYTHON" -m rabbit_foundry.rabbit_code \
  --workspace "$WORKSPACE" --ui --open-browser
