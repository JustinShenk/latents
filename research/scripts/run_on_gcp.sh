#!/bin/bash

# Run Phase 0 on GCP instance
# This script should be run ON the GCP instance after setup

set -e  # Exit on error

echo "========================================"
echo "PHASE 0: SANITY CHECK ON GCP"
echo "========================================"

# Verify GPU
echo "Checking GPU availability..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo ""
echo "========================================"
echo "Step 1/2: Extracting Activations"
echo "========================================"
python3 research/tools/extract_activations.py \
    --prompts data/sanity_check_prompts.json \
    --output activations/sanity_check.npz \
    --model gpt2-small

echo ""
echo "========================================"
echo "Step 2/2: Training Probes (Parallelized)"
echo "========================================"
python3 research/tools/train_probes.py \
    --input activations/sanity_check.npz \
    --output results/sanity_check_results.csv \
    --prefix sanity

echo ""
echo "========================================"
echo "PHASE 0 COMPLETE!"
echo "========================================"
echo ""
echo "Results have been synced to:"
echo "  gs://temporal-grounding-gpt2-82feb/"
echo ""
echo "To view results:"
echo "  cat results/sanity_check_results.csv"
echo ""
echo "To download to local machine:"
echo "  gsutil -m rsync -r gs://temporal-grounding-gpt2-82feb/results/ results/"
echo "  gsutil -m rsync -r gs://temporal-grounding-gpt2-82feb/probes/ probes/"
echo ""
