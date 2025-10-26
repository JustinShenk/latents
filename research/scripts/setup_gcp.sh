#!/bin/bash

# Setup script for GCP instance with PyTorch pre-installed
# Run this ONCE after creating the instance

set -e  # Exit on error

echo "=========================================="
echo "SETUP: Temporal Grounding Experiment"
echo "=========================================="

# Download code from GCS
echo "1. Downloading code from GCS bucket..."
cd ~
mkdir -p temporal-grounding-gpt2
gsutil -m rsync -r ${GCS_BUCKET:-gs://temporal-grounding-gpt2-82feb}/code/ temporal-grounding-gpt2/
cd temporal-grounding-gpt2

# Install additional dependencies (PyTorch already installed in the image!)
echo ""
echo "2. Installing additional Python packages..."
pip install --quiet transformer-lens transformers scikit-learn google-cloud-storage

echo ""
echo "3. Verifying GPU and PyTorch..."
python3 -c "import torch; print('✓ CUDA available:', torch.cuda.is_available()); print('✓ GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Ready to run Phase 0:"
echo "  cd ~/temporal-grounding-gpt2"
echo "  ./run_on_gcp.sh"
echo ""
