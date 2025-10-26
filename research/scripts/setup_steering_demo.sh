#!/bin/bash

# Setup script for temporal steering demo

set -e

echo "=========================================="
echo "Temporal Steering Demo Setup"
echo "=========================================="
echo ""

# Activate venv
if [ -d "venv" ]; then
    echo "✓ Activating virtual environment..."
    source venv_steering/bin/activate
else
    echo "✗ Virtual environment not found. Please create one first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install torch transformers flask numpy"
    exit 1
fi

# Check required packages
echo "Checking required packages..."
python3 -c "import torch; import transformers; import flask; import numpy" 2>/dev/null || {
    echo "Installing required packages..."
    pip install torch transformers flask numpy
}

echo ""
echo "=========================================="
echo "Step 1: Extract Steering Vectors"
echo "=========================================="
echo ""

# Use a subset of test prompts for faster extraction
echo "Extracting steering vectors from test prompts (first 20 pairs)..."
python3 temporal_steering/extract_steering_vectors.py \
    --pairs data_download/test_prompts.json \
    --output steering_vectors/temporal_steering.json \
    --model gpt2 \
    --max-pairs 20

echo ""
echo "=========================================="
echo "Step 2: Launch Demo Server"
echo "=========================================="
echo ""
echo "Starting demo server on http://localhost:5000"
echo "Press Ctrl+C to stop"
echo ""

python3 temporal_steering/temporal_steering_demo.py \
    --steering steering_vectors/temporal_steering.json \
    --model gpt2 \
    --port 5000
