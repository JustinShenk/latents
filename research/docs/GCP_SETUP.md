# GCP Setup Instructions

## Prerequisites

Before running any commands, set up your environment variables:

```bash
# Copy .env.example to .env and configure
cp .env.example .env

# Edit .env to set your GCS bucket
# GCS_BUCKET=gs://your-bucket-name

# Load environment variables
export GCS_BUCKET=gs://your-bucket-name  # Replace with your actual bucket

# Or source from .env file
source <(grep -v '^#' .env | xargs -I {} echo export {})
```

## Quick Start

### 1. Upload Project to GCS Bucket

```bash
# From local machine in the temporal-grounding-gpt2 directory
gsutil -m rsync -r . ${GCS_BUCKET:-gs://temporal-grounding-gpt2-82feb}/code/

# Verify upload
gsutil ls ${GCS_BUCKET:-gs://temporal-grounding-gpt2-82feb}/code/
```

### 2. Create GCP Compute Instance

```bash
# Create instance with T4 GPU (preemptible for cost savings)
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
    --scopes=cloud-platform \
    --metadata=startup-script='#!/bin/bash
export HOME=/root
cd /root
gsutil -m rsync -r ${GCS_BUCKET:-gs://temporal-grounding-gpt2-82feb}/code/ temporal-grounding-gpt2/
'
```

### 3. SSH into Instance

```bash
gcloud compute ssh temporal-gpt2-experiment --project=new-one-82feb --zone=us-central1-a
```

### 4. Setup on Instance

```bash
# Navigate to project
cd ~/temporal-grounding-gpt2

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformer-lens transformers scikit-learn pandas numpy matplotlib seaborn tqdm google-cloud-storage scipy openai

# Verify GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Set environment variables (replace with your actual values)
export GCS_BUCKET=gs://your-bucket-name
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 5. Run Phase 0

```bash
# Download the generated prompts from bucket
gsutil cp ${GCS_BUCKET:-gs://temporal-grounding-gpt2-82feb}/code/data/sanity_check_prompts.json data/

# Extract activations (this will use GPU)
python src/extract_activations.py \
    --prompts data/sanity_check_prompts.json \
    --output activations/sanity_check.npz \
    --model gpt2-small

# Train probes
python src/train_probes.py \
    --input activations/sanity_check.npz \
    --output results/sanity_check_results.csv \
    --prefix sanity

# Results are automatically synced to GCS bucket!
```

### 6. Download Results Locally

```bash
# From your local machine
gsutil -m rsync -r ${GCS_BUCKET:-gs://temporal-grounding-gpt2-82feb}/results/ temporal-grounding-gpt2/results/
gsutil -m rsync -r ${GCS_BUCKET:-gs://temporal-grounding-gpt2-82feb}/probes/ temporal-grounding-gpt2/probes/
gsutil -m rsync -r ${GCS_BUCKET:-gs://temporal-grounding-gpt2-82feb}/activations/ temporal-grounding-gpt2/activations/
```

### 7. Cleanup (IMPORTANT)

```bash
# Delete instance when done to stop charges
gcloud compute instances delete temporal-gpt2-experiment \
    --project=new-one-82feb \
    --zone=us-central1-a
```

## Cost Estimates

- **T4 GPU (preemptible)**: ~$0.35/hour
- **n1-standard-4**: ~$0.05/hour
- **Total**: ~$0.40/hour

**Phase 0 estimated time**: 10-15 minutes = ~$0.10
**Full experiment**: ~7 hours = ~$3.00

## Automated Sync

All scripts automatically sync results to GCS after completion:
- `${GCS_BUCKET:-gs://temporal-grounding-gpt2-82feb}/results/`
- `${GCS_BUCKET:-gs://temporal-grounding-gpt2-82feb}/activations/`
- `${GCS_BUCKET:-gs://temporal-grounding-gpt2-82feb}/probes/`

Even if instance is preempted, your results are safe!

## Quick Commands

```bash
# Check if instance is still running
gcloud compute instances list --project=new-one-82feb

# View logs
gcloud compute ssh temporal-gpt2-experiment --project=new-one-82feb --zone=us-central1-a --command="tail -f ~/temporal-grounding-gpt2/run.log"

# Download all results
gsutil -m rsync -r ${GCS_BUCKET:-gs://temporal-grounding-gpt2-82feb}/ temporal-grounding-gpt2/
```
