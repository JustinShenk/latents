# Temporal Grounding in GPT-2 - Status

## ✅ PHASE 0 SETUP COMPLETE (Local)

### What's Done

1. **Project Structure Created**
   - All directories set up
   - Source code modules implemented
   - GCS sync utilities integrated

2. **Dataset Generated**
   - **50 prompt pairs** across 5 domains
   - Balanced distribution (10 per domain)
   - Temporal horizons: short (days-months) vs long (years-decades)
   - Stored in: `data/sanity_check_prompts.json`

3. **Files Uploaded to GCS**
   - All code synced to: `${GCS_BUCKET:-gs://temporal-grounding-gpt2-82feb}/code/`
   - Prompts uploaded and ready

### Domain Distribution

- Business planning: 10 pairs
- Scientific research: 10 pairs
- Personal projects: 10 pairs
- Technical/engineering: 10 pairs
- Creative/artistic: 10 pairs

**Total: 50 prompt pairs = 100 samples (50 short + 50 long)**

## Next Steps: Run on GCP

### Option 1: Quick Setup (Recommended)

Follow the instructions in `GCP_SETUP.md`:

```bash
# 1. Create GCP instance (from local machine)
gcloud compute instances create temporal-gpt2-experiment \
    --project=new-one-82feb \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=50GB \
    --maintenance-policy=TERMINATE \
    --preemptible \
    --scopes=cloud-platform

# 2. SSH into instance
gcloud compute ssh temporal-gpt2-experiment --project=new-one-82feb --zone=us-central1-a

# 3. On the instance:
cd ~
gsutil -m rsync -r ${GCS_BUCKET:-gs://temporal-grounding-gpt2-82feb}/code/ temporal-grounding-gpt2/
cd temporal-grounding-gpt2

# 4. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformer-lens transformers scikit-learn pandas numpy matplotlib seaborn tqdm google-cloud-storage scipy openai

# 5. Run Phase 0
export OPENAI_API_KEY="your-openai-api-key-here"

./run_on_gcp.sh
```

### Option 2: Manual Steps

```bash
# After setup on GCP instance:

# Extract activations
python src/extract_activations.py \
    --prompts data/sanity_check_prompts.json \
    --output activations/sanity_check.npz \
    --model gpt2-small

# Train probes
python src/train_probes.py \
    --input activations/sanity_check.npz \
    --output results/sanity_check_results.csv \
    --prefix sanity
```

## Expected Results

After running on GCP, you'll get:

1. **Activations**: `activations/sanity_check.npz` (~50-100 MB)
   - Shape per layer: (100, 768) for GPT-2-small
   - 12 layers total
   - Automatically synced to GCS

2. **Probe Results**: `results/sanity_check_results.csv`
   - Accuracy per layer
   - Cross-validation scores
   - Recommendation to proceed or not

3. **Trained Probes**: `probes/sanity_layer_*_probe.pkl` (12 files)

## Success Criteria

- **STOP if** best layer accuracy < 55% (signal too weak)
- **PROCEED WITH CAUTION if** accuracy 55-70% (weak signal)
- **PROCEED CONFIDENTLY if** accuracy > 70% (strong signal!)

## Time & Cost Estimates

- **Setup**: 5-10 minutes
- **Activation extraction**: 5-8 minutes (with GPU)
- **Probe training**: 1-2 minutes
- **Total**: ~15 minutes

**Cost**: ~$0.10 (preemptible T4 GPU instance)

## Download Results

```bash
# From local machine
gsutil -m rsync -r ${GCS_BUCKET:-gs://temporal-grounding-gpt2-82feb}/results/ results/
gsutil -m rsync -r ${GCS_BUCKET:-gs://temporal-grounding-gpt2-82feb}/probes/ probes/
gsutil -m rsync -r ${GCS_BUCKET:-gs://temporal-grounding-gpt2-82feb}/activations/ activations/
```

## Cleanup

**IMPORTANT**: Delete instance when done to avoid charges!

```bash
gcloud compute instances delete temporal-gpt2-experiment \
    --project=new-one-82feb \
    --zone=us-central1-a
```

## Files Ready for GCP

- ✅ `data/sanity_check_prompts.json` - 50 prompt pairs
- ✅ `src/dataset.py` - Prompt generation
- ✅ `src/extract_activations.py` - Activation extraction
- ✅ `src/train_probes.py` - Probe training
- ✅ `src/utils.py` - GCS sync utilities
- ✅ `run_on_gcp.sh` - Automated run script
- ✅ `requirements.txt` - Dependencies
- ✅ `GCP_SETUP.md` - Detailed instructions

All files are at: **`${GCS_BUCKET:-gs://temporal-grounding-gpt2-82feb}/code/`**

---

**Date**: 2025-10-25
**Status**: Ready for GCP execution
**Next Phase**: Phase 0 - Activation Extraction & Probe Training
