#!/bin/bash
# launch_dreamrabbit.sh - Start DreamRabbit inference server and connect Alpha Rabbit

set -e

# Configuration
MODEL_PATH="${1:-./checkpoints/dreamrabbit-tiny}"
TOKENIZER_PATH="${2:-gpt2}"
PORT="${3:-8080}"
DEVICE="${4:-auto}"

echo "=========================================="
echo "DreamRabbit Inference Server Launcher"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Port: $PORT"
echo "Device: $DEVICE"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ] && [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found at $MODEL_PATH"
    echo "Run training first or specify path: $0 <model_path> [tokenizer] [port] [device]"
    exit 1
fi

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Install dependencies if needed
echo "Checking dependencies..."
pip install -q fastapi uvicorn transformers accelerate 2>/dev/null || true

# Start inference server
echo ""
echo "Starting DreamRabbit inference server on http://localhost:$PORT/v1"
echo "Press Ctrl+C to stop"
echo ""

cd "$(dirname "$0")"
PYTHONPATH=src python -m src.inference_server "$MODEL_PATH" "$TOKENIZER_PATH" "$DEVICE" --port "$PORT"