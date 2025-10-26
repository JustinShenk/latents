# Full Experiment Plan: Temporal Grounding in GPT-2

**Status**: Phase 0 Complete ✓ | Full Experiment In Progress
**Start Date**: 2025-10-25
**Phase 0 Result**: 78% accuracy - STRONG SIGNAL ✓

---

## Overview

Based on Phase 0's strong results (78% accuracy), we're proceeding with the complete 8-phase experimental protocol to comprehensively study temporal representation in GPT-2.

### Phase 0 Summary
- **Dataset**: 50 prompt pairs (100 samples)
- **Best Layer**: Layer 8
- **Accuracy**: 78.0% (±6.8%)
- **Recommendation**: PROCEED CONFIDENTLY ✓

---

## Full Experiment Timeline

### ✅ Phase 0: Sanity Check (COMPLETE)
- Duration: 15 minutes
- Cost: $0.10
- **Status**: ✓ Complete
- **Result**: Strong signal detected (78% accuracy)

### 🔄 Phase 1: Full Dataset Generation (IN PROGRESS)
- **Objective**: Generate 300 high-quality prompt pairs
- **Actions**:
  - Generate 300 pairs across 5 domains (60 each)
  - Include length matching (±15 tokens)
  - Quality validation
- **Duration**: 15-20 minutes
- **Cost**: ~$1.50 (OpenAI API)
- **Status**: Running on GCP
- **Output**: `data/raw_prompts.json`

### Phase 2: Dataset Preparation
- **Objective**: Split and prepare datasets
- **Actions**:
  - Split into train/val/test (200/50/50)
  - Generate control datasets:
    - Keyword ablation (50 pairs)
    - Trap prompts (50 prompts)
    - Non-planning temporal (50 pairs)
- **Duration**: 10-15 minutes
- **Cost**: ~$0.50 (API for trap/non-planning)
- **Output**:
  - `data/train_prompts.json`
  - `data/val_prompts.json`
  - `data/test_prompts.json`
  - `data/control_*.json`

### Phase 3: Full Activation Extraction
- **Objective**: Extract GPT-2 activations for all datasets
- **Actions**:
  - Extract train set (200 pairs = 400 samples)
  - Extract val set (50 pairs = 100 samples)
  - Extract test set (50 pairs = 100 samples)
  - Extract controls (150 samples)
- **Duration**: 20-30 minutes (GPU)
- **Cost**: ~$0.20
- **Output**: `activations/*.npz` (~15-20 MB total)

### Phase 4: Probe Training & Optimization
- **Objective**: Train probes on full dataset with proper train/val/test split
- **Actions**:
  - Train probes on all 12 layers
  - Use train set for training
  - Validate on val set
  - Final evaluation on test set
  - Hyperparameter tuning if needed
- **Duration**: 5-10 minutes (parallelized)
- **Cost**: negligible
- **Output**:
  - `results/full_probe_results.csv`
  - `probes/full_layer_*.pkl`

### Phase 5: Control Experiments
- **Objective**: Test robustness of findings
- **Experiments**:
  1. **Keyword Ablation**: Does probe rely on temporal words?
  2. **Trap Prompts**: Is probe fooled by misleading keywords?
  3. **Cross-Domain**: Does it generalize across domains?
- **Duration**: 10 minutes
- **Cost**: negligible
- **Output**: `results/control_results.json`

### Phase 6: Circuit Analysis
- **Objective**: Identify which components causally encode temporal info
- **Experiments**:
  1. **Activation Patching**: Test each layer's causal importance
  2. **Attention Head Ablation**: Find critical attention heads
- **Duration**: 30-60 minutes (computationally intensive)
- **Cost**: ~$0.40
- **Output**:
  - `results/patching_results.csv`
  - `results/head_ablation_results.csv`

### Phase 7: Visualization
- **Objective**: Create publication-quality figures
- **Plots**:
  1. Accuracy by layer (line plot)
  2. Confusion matrices
  3. Control experiment comparison (bar chart)
  4. Activation patching heatmap
  5. Head ablation scatter plot
  6. Probe weight distribution
- **Duration**: 5 minutes
- **Cost**: negligible
- **Output**: `results/figures/*.png`

### Phase 8: Final Report
- **Objective**: Comprehensive analysis and writeup
- **Contents**:
  - Executive summary
  - All results with interpretations
  - Statistical analysis
  - Discussion of findings
  - Limitations and future work
- **Duration**: 5 minutes (automated)
- **Cost**: negligible
- **Output**: `results/FINAL_REPORT.md`

---

## Total Estimates

| Metric | Estimate |
|--------|----------|
| **Total Duration** | 2-3 hours |
| **GPU Time** | ~1.5 hours |
| **Total Cost** | $3-5 |
| **Dataset Size** | 300 pairs (600 samples) |
| **Activations** | ~20 MB |
| **Probes** | 12 trained models |
| **Figures** | 6+ visualizations |

---

## Current Status

### Phase 1 Progress
- ✅ Scripts created and uploaded to GCS
- 🔄 300 prompt generation running on GCP
- ⏳ Estimated completion: 10-15 minutes

### Monitoring
```bash
# Check dataset generation progress
gcloud compute ssh temporal-gpt2-experiment \
  --project=new-one-82feb \
  --zone=us-central1-a \
  --command="tail -f ~/temporal-grounding-gpt2/dataset_generation.log"
```

---

## Key Decision Points

### After Phase 4 (Probe Training)
**Decision**: Proceed to circuits if accuracy > 70%
- If < 55%: Stop, signal too weak
- If 55-70%: Skip circuits, focus on controls
- If > 70%: Full circuit analysis

### After Phase 5 (Controls)
**Decision**: Interpret robustness
- Keyword ablation drop > 30%: Probe is lexical
- Trap prompts fooled > 70%: Probe uses keywords only
- Cross-domain drop > 20%: Weak generalization

---

## Files Structure

```
results/
├── full_probe_results.csv        # Accuracy by layer
├── control_results.json           # Control experiment outcomes
├── patching_results.csv           # Causal importance by component
├── head_ablation_results.csv      # Critical attention heads
├── statistical_summary.json       # Effect sizes, p-values
├── FINAL_REPORT.md                # Complete writeup
└── figures/
    ├── accuracy_by_layer.png
    ├── confusion_matrix.png
    ├── control_results.png
    ├── patching_heatmap.png
    ├── head_ablations.png
    └── probe_weights.png
```

---

## GCP Instance Management

**Current Instance**: `temporal-gpt2-experiment`
- Type: n1-standard-4 + T4 GPU
- Status: Running
- Cost: $0.40/hour (preemptible)

### Important Commands
```bash
# Check status
./gcp_manager.sh status

# Download results
./gcp_manager.sh download

# Delete instance (IMPORTANT!)
./gcp_manager.sh delete
```

**⚠️ Remember to delete instance when done to stop charges!**

---

## Next Steps

1. ⏳ Wait for Phase 1 dataset generation to complete (~10 min remaining)
2. Run Phase 2-8 sequentially on GCP
3. Download all results locally
4. Review final report
5. Delete GCP instance

**Estimated total time to completion**: 2-3 hours from now

---

## Expected Outcomes

Based on Phase 0's strong results, we expect:

1. **Confirmed Signal**: 75-80% accuracy on full dataset
2. **Robust to Controls**: < 20% accuracy drop with keyword ablation
3. **Circuit Identified**: Specific layers/heads causally encode temporal info
4. **Publication-Ready**: Complete analysis with visualizations

**Success threshold**: Accuracy > 70% with robust controls

---

**Last Updated**: 2025-10-25 21:57 UTC
**Next Checkpoint**: Phase 1 completion (ETA: 10 minutes)
