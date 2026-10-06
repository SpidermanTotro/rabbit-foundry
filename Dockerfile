# Dockerfile for Rabbit Foundry sandboxed evaluation
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash sandbox

# Set up working directory
WORKDIR /workspace
RUN chown sandbox:sandbox /workspace

# Copy requirements first for caching
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /workspace/src/
COPY tests/ /workspace/tests/
COPY scripts/ /workspace/scripts/

# Switch to sandbox user
USER sandbox

# Default command
CMD ["python", "-m", "src.inference_server"]