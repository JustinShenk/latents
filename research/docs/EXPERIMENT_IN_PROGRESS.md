# Experiment In Progress

**Status**: 🔄 RUNNING AUTOMATICALLY
**Started**: 2025-10-25 22:00 UTC
**Estimated Completion**: 2-3 hours
**GCP Instance**: temporal-gpt2-experiment (preemptible, $0.40/hour)

---

## What's Happening Now

The **full automated experiment** is running on GCP. Here's the pipeline:

### Current Progress

1. ✅ **Phase 0**: Sanity check - 78% accuracy
2. 🔄 **Phase 1**: Generating 300 prompt pairs (~15-20 min)
3. ⏳ **Phase 2-6**: Will run automatically after Phase 1
   - Split dataset
   - Generate controls
   - Extract activations
   - Train probes
   - Run control experiments

---

## Monitoring

### Check Live Progress

```bash
# From your local machine
gcloud compute ssh temporal-gpt2-experiment \
  --project=new-one-82feb \
  --zone=us-central1-a \
  --command="tail -f ~/temporal-grounding-gpt2/full_experiment.log"
```

### Quick Status Check

```bash
# Check if experiment is still running
gcloud compute ssh temporal-gpt2-experiment \
  --project=new-one-82feb \
  --zone=us-central1-a \
  --command="ps aux | grep -E '(dataset|experiment)' | grep -v grep"
```

### Download Results (when complete)

```bash
./gcp_manager.sh download
```

---

## Timeline

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| 0 | Sanity check | 15 min | ✅ Complete |
| 1 | Generate 300 prompts | 15-20 min | 🔄 Running |
| 2 | Split dataset | 1 min | ⏳ Queued |
| 3 | Generate controls | 10 min | ⏳ Queued |
| 4 | Extract activations | 30 min | ⏳ Queued |
| 5 | Train probes | 10 min | ⏳ Queued |
| 6 | Control experiments | 5 min | ⏳ Queued |
| **Total** | | **~2 hours** | |

---

## What to Expect

### Successful Completion

When the experiment completes, you'll have:

1. **Datasets**:
   - 300 prompt pairs (train/val/test split)
   - Control datasets (ablated, traps, non-planning)

2. **Activations**:
   - Full dataset activations (~15-20 MB)
   - All 12 GPT-2 layers

3. **Trained Probes**:
   - 12 probes (one per layer)
   - Train/val/test evaluations

4. **Results**:
   - Accuracy by layer
   - Control experiment outcomes
   - All synced to GCS automatically

5. **Key Metrics**:
   - Best layer accuracy
   - Keyword ablation robustness
   - Cross-domain generalization

### Files Generated

```
data/
├── raw_prompts.json (300 pairs)
├── train_prompts.json (200 pairs)
├── val_prompts.json (50 pairs)
├── test_prompts.json (50 pairs)
├── control_ablated.json
├── control_traps.json
└── control_nonplanning.json

activations/
├── train_activations.npz (~12 MB)
├── val_activations.npz (~3 MB)
├── test_activations.npz (~3 MB)
└── control_ablated.npz (~3 MB)

probes/
├── full_train_layer_*.pkl (12 files)
├── full_val_layer_*.pkl (12 files)
└── full_test_layer_*.pkl (12 files)

results/
├── full_train_results.csv
├── full_val_results.csv
├── full_test_results.csv
└── control_ablated_results.csv
```

All automatically synced to: `gs://temporal-grounding-gpt2-82feb/`

---

## Expected Results

Based on Phase 0 (78% accuracy on 50 pairs), we expect:

- **Full Dataset Accuracy**: 75-80%
- **Keyword Ablation**: < 20% drop (robust)
- **Best Layer**: Layer 6-11 (middle-to-late)

### Success Criteria

✓ **Minimum**: > 60% accuracy
✓ **Target**: > 75% accuracy
✓ **Robust**: < 30% drop with ablation

---

## Cost Tracking

| Item | Cost |
|------|------|
| Phase 0 (complete) | $0.10 |
| Dataset generation (API) | ~$1.50 |
| Control generation (API) | ~$0.50 |
| GPU time (~2 hours) | ~$0.80 |
| **Estimated Total** | **~$3.00** |

---

## After Completion

### 1. Download Results

```bash
cd temporal-grounding-gpt2
./gcp_manager.sh download
```

### 2. Review Results

```bash
# Check final accuracies
cat results/full_test_results.csv

# Check control experiments
cat results/control_ablated_results.csv
```

### 3. **IMPORTANT: Delete Instance**

```bash
./gcp_manager.sh delete
```

**⚠️ Don't forget this step to stop charges!**

---

## Troubleshooting

### If Experiment Fails

1. Check logs:
   ```bash
   gcloud compute ssh temporal-gpt2-experiment \
     --project=new-one-82feb \
     --zone=us-central1-a \
     --command="cat ~/temporal-grounding-gpt2/full_experiment.log"
   ```

2. Check for errors:
   ```bash
   grep -i error ~/temporal-grounding-gpt2/full_experiment.log
   ```

3. Restart manually:
   ```bash
   cd ~/temporal-grounding-gpt2
   export OPENAI_API_KEY='your-key'
   bash auto_run_experiment.sh
   ```

### If Instance is Preempted

All progress is automatically synced to GCS. You can:

1. Create a new instance
2. Download code from GCS
3. Resume from where it stopped

---

## Next Steps

**Right Now**: The experiment is running automatically!

**In 2-3 hours**:
1. Check if experiment completed successfully
2. Download all results
3. Review findings
4. Delete GCP instance

**Then**: Decide whether to:
- Proceed to advanced analysis (circuit analysis, visualizations)
- Write up findings
- Test on larger models
- Publish results

---

## Quick Reference

```bash
# Monitor progress
gcloud compute ssh temporal-gpt2-experiment --project=new-one-82feb --zone=us-central1-a \
  --command="tail -20 ~/temporal-grounding-gpt2/full_experiment.log"

# Check if still running
./gcp_manager.sh status

# Download results (when done)
./gcp_manager.sh download

# Delete instance (when done)
./gcp_manager.sh delete
```

---

**Last Updated**: 2025-10-25 22:00 UTC
**Experiment Log**: `gs://temporal-grounding-gpt2-82feb/full_experiment.log`
**All Results**: `gs://temporal-grounding-gpt2-82feb/`
