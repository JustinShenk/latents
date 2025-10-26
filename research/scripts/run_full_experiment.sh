#!/bin/bash

# Master script to run the complete experiment (Phases 1-8)
# Run this ON the GCP instance after setup

set -e  # Exit on error

echo "=========================================="
echo "FULL TEMPORAL GROUNDING EXPERIMENT"
echo "Phases 1-8"
echo "=========================================="
echo ""

# Configuration
export OPENAI_API_KEY="${OPENAI_API_KEY}"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set!"
    echo "Run: export OPENAI_API_KEY='your-key'"
    exit 1
fi

echo "✓ OpenAI API key set"
echo "✓ GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""

# PHASE 1: Dataset Generation
echo "=========================================="
echo "PHASE 1: Dataset Generation"
echo "=========================================="

# Download pre-generated dataset from GCS (faster than regenerating)
if [ ! -f "data/raw_prompts.json" ]; then
    echo "Downloading pre-generated dataset..."
    gsutil cp gs://temporal-grounding-gpt2-82feb/data/raw_prompts.json data/ || {
        echo "Dataset not found in GCS, generating locally..."
        python3 temporal_steering/dataset.py --mode full
    }
fi

echo "✓ Dataset ready"
echo ""

# PHASE 2: Split Dataset
echo "=========================================="
echo "PHASE 2: Split Dataset"
echo "=========================================="
python3 research/tools/split_dataset.py --input data/raw_prompts.json
echo ""

# PHASE 3: Generate Controls
echo "=========================================="
echo "PHASE 3: Generate Control Datasets"
echo "=========================================="
python3 research/tools/generate_controls.py --base-prompts data/test_prompts.json
echo ""

# PHASE 4: Extract Activations (Train/Val/Test)
echo "=========================================="
echo "PHASE 4: Extract Activations"
echo "=========================================="

for split in train val test; do
    echo "Extracting ${split} activations..."
    python3 research/tools/extract_activations.py \
        --prompts data/${split}_prompts.json \
        --output activations/${split}_activations.npz \
        --model gpt2-small
    echo ""
done

# Extract control activations
echo "Extracting control activations..."
python3 research/tools/extract_activations.py \
    --prompts data/control_ablated.json \
    --output activations/control_ablated.npz \
    --model gpt2-small
echo ""

# PHASE 5: Train Probes
echo "=========================================="
echo "PHASE 5: Train Probes (Full Dataset)"
echo "=========================================="
python3 src/train_probes_full.py \
    --train activations/train_activations.npz \
    --val activations/val_activations.npz \
    --test activations/test_activations.npz \
    --output results/full_probe_results.csv
echo ""

# PHASE 6: Control Experiments
echo "=========================================="
echo "PHASE 6: Run Control Experiments"
echo "=========================================="
python3 src/control_experiments.py \
    --test activations/test_activations.npz \
    --ablated activations/control_ablated.npz \
    --results results/control_results.json
echo ""

# PHASE 7: Visualizations
echo "=========================================="
echo "PHASE 7: Generate Visualizations"
echo "=========================================="
python3 src/visualization.py --all
echo ""

# PHASE 8: Final Report
echo "=========================================="
echo "PHASE 8: Generate Final Report"
echo "=========================================="
python3 src/generate_report.py
echo ""

echo "=========================================="
echo "EXPERIMENT COMPLETE!"
echo "=========================================="
echo ""
echo "All results synced to:"
echo "  gs://temporal-grounding-gpt2-82feb/"
echo ""
echo "Key files:"
echo "  - results/full_probe_results.csv"
echo "  - results/control_results.json"
echo "  - results/figures/*.png"
echo "  - results/FINAL_REPORT.md"
echo ""
echo "Download locally:"
echo "  ./gcp_manager.sh download"
echo ""
