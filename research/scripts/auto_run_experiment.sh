#!/bin/bash

# Automated experiment runner
# Monitors dataset generation and launches full pipeline when ready

set -e

echo "=========================================="
echo "AUTOMATED EXPERIMENT RUNNER"
echo "=========================================="
echo ""

export OPENAI_API_KEY="${OPENAI_API_KEY}"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set!"
    exit 1
fi

cd ~/temporal-grounding-gpt2

# Wait for dataset generation to complete
echo "Waiting for dataset generation to complete..."
echo "(This takes ~15-20 minutes)"
echo ""

while true; do
    # Check if process is still running
    if ! ps aux | grep -v grep | grep "dataset.py" > /dev/null; then
        # Check if dataset file exists
        if [ -f "data/raw_prompts.json" ]; then
            echo "✓ Dataset generation complete!"
            break
        else
            echo "✗ Dataset generation failed!"
            echo "Check dataset_generation.log for errors"
            exit 1
        fi
    fi

    # Show progress every 30 seconds
    if [ -f "dataset_generation.log" ]; then
        echo "[$(date +%H:%M:%S)] Still generating..."
        tail -3 dataset_generation.log | grep -E "(Progress|Generated)" || true
    fi

    sleep 30
done

echo ""
echo "=========================================="
echo "STARTING FULL EXPERIMENT (Phases 2-8)"
echo "=========================================="
echo ""

# Phase 2: Split Dataset
echo "=== Phase 2: Splitting Dataset ==="
python3 research/tools/split_dataset.py --input data/raw_prompts.json
echo ""

# Phase 3: Generate Controls
echo "=== Phase 3: Generating Control Datasets ==="
python3 research/tools/generate_controls.py --base-prompts data/test_prompts.json
echo ""

# Phase 4: Extract Activations
echo "=== Phase 4: Extracting Activations ==="
for split in train val test; do
    echo "Extracting ${split}..."
    python3 research/tools/extract_activations.py \
        --prompts data/${split}_prompts.json \
        --output activations/${split}_activations.npz \
        --model gpt2-small
done

echo "Extracting control activations..."
python3 research/tools/extract_activations.py \
    --prompts data/control_ablated.json \
    --output activations/control_ablated.npz \
    --model gpt2-small
echo ""

# Phase 5: Train Probes (use same script as Phase 0, but with train/val/test)
echo "=== Phase 5: Training Probes on Full Dataset ==="
python3 research/tools/train_probes.py \
    --input activations/train_activations.npz \
    --output results/full_train_results.csv \
    --prefix full_train

echo "Evaluating on validation set..."
python3 research/tools/train_probes.py \
    --input activations/val_activations.npz \
    --output results/full_val_results.csv \
    --prefix full_val

echo "Final evaluation on test set..."
python3 research/tools/train_probes.py \
    --input activations/test_activations.npz \
    --output results/full_test_results.csv \
    --prefix full_test
echo ""

# Phase 6: Control Experiments
echo "=== Phase 6: Running Control Experiments ==="
echo "Testing keyword ablation..."
python3 research/tools/train_probes.py \
    --input activations/control_ablated.npz \
    --output results/control_ablated_results.csv \
    --prefix control_ablated
echo ""

echo "=========================================="
echo "EXPERIMENT COMPLETE!"
echo "=========================================="
echo ""
echo "Results synced to: ${GCS_BUCKET:-gs://temporal-grounding-gpt2-82feb}/"
echo ""
echo "Key files:"
echo "  - results/full_*_results.csv"
echo "  - activations/*.npz"
echo "  - probes/full_*.pkl"
echo ""
echo "Download locally with:"
echo "  ./gcp_manager.sh download"
echo ""
echo "⚠️ Remember to delete GCP instance when done:"
echo "  ./gcp_manager.sh delete"
echo ""
