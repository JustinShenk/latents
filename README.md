# Temporal Steering with GPT-2

Steer GPT-2's temporal scope from immediate/tactical thinking to long-term/strategic thinking using Contrastive Activation Addition (CAA).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/justinshenk/temporal-steering/blob/main/temporal_steering_demo.ipynb)

## 🚀 Installation

### Option 1: Install from GitHub (Recommended)

```bash
pip install git+https://github.com/justinshenk/temporal-steering.git
```

### Option 2: Local Development

```bash
git clone https://github.com/justinshenk/temporal-steering.git
cd temporal-steering
pip install -e .
```

### Option 3: Try in Colab (No Installation)

Click the Colab badge above for an interactive demo!

## ⚡ Quick Start

### Python API

```python
from temporal_steering import TemporalSteering
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import numpy as np

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load steering vectors (download pre-computed or extract your own)
with open('steering_vectors/temporal_steering.json') as f:
    data = json.load(f)
    steering_vectors = {int(k): np.array(v) for k, v in data['layer_vectors'].items()}

# Initialize steering
steering = TemporalSteering(model, tokenizer, steering_vectors)

# Generate with long-term steering
result = steering.generate(
    prompt="What should policymakers prioritize to address climate change?",
    steering_strength=0.8,  # +1.0 = long-term, -1.0 = immediate
    temperature=0.7,
    max_length=100
)
print(result)
```

### Command Line

Extract steering vectors:
```bash
temporal-steering extract \
  --pairs data_download/train_prompts.json \
  --output steering_vectors/my_steering.json \
  --max-pairs 20
```

Launch interactive web demo:
```bash
temporal-steering demo \
  --steering steering_vectors/temporal_steering.json \
  --port 8080
```

Then open http://localhost:8080

## 🎯 What is Temporal Steering?

This project uses **Contrastive Activation Addition (CAA)** to control language model outputs along a temporal dimension:

1. **Extract activations** from prompt pairs with different temporal horizons
2. **Compute steering vectors**: `long_term_activations - immediate_activations`
3. **Apply during generation** by adding vectors to model activations

### Example Results

**Prompt**: "What should policymakers prioritize to address climate change?"

- **Immediate Steering (-1.0)**: Focuses on concrete actions, short-term fixes
- **Long-term Steering (+1.0)**: Discusses systemic change, future generations, strategic planning

## 📊 Pre-trained Steering Vectors

Download pre-computed steering vectors:
- GPT-2 (124M): [temporal_steering.json](steering_vectors/temporal_steering.json)

Or extract your own from custom prompt pairs!

## Setup (Research/Development)

For the full research pipeline:

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional)
export OPENAI_API_KEY="your-key-here"
export GOOGLE_APPLICATION_CREDENTIALS="path-to-gcp-credentials.json"

# Verify GPU
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
```

## Quick Start - Phase 0 (Sanity Check)

Run the complete Phase 0 pipeline:

```bash
cd temporal-grounding-gpt2

# Step 1: Generate 50 prompt pairs
python src/dataset.py --mode sanity

# Step 2: Extract activations
python src/extract_activations.py \
    --prompts data/sanity_check_prompts.json \
    --output activations/sanity_check.npz

# Step 3: Train probes
python src/train_probes.py \
    --input activations/sanity_check.npz \
    --output results/sanity_check_results.csv
```

Or use the master script:

```bash
python run_phase0.py
```

## Project Structure

```
temporal-grounding-gpt2/
├── data/               # Generated prompts
├── activations/        # Extracted model activations
├── probes/            # Trained linear probes
├── results/           # Results and figures
│   └── figures/       # Visualizations
├── notebooks/         # Analysis notebooks
├── src/               # Source code
│   ├── dataset.py              # Prompt generation
│   ├── extract_activations.py  # Activation extraction
│   ├── train_probes.py         # Probe training
│   └── utils.py                # GCS sync utilities
└── requirements.txt
```

## GCS Bucket

All results are automatically synced to: `gs://temporal-grounding-gpt2-82feb/`

## Phases

- **Phase 0**: Sanity check (50 prompts)
- **Phase 1**: Full dataset (300 prompts)
- **Phase 2**: Full activation extraction
- **Phase 3**: Probe training & optimization
- **Phase 4**: Control experiments
- **Phase 5**: Circuit analysis
- **Phase 6**: Visualization
- **Phase 7**: Statistical analysis
- **Phase 8**: Final report

## Success Criteria

- **Minimum**: >60% probe accuracy
- **Target**: >75% accuracy + robust controls
- **Stretch**: Cross-model validation

## Estimated Costs

- GCP Compute: $3-5 (T4 GPU, ~7 hours)
- OpenAI API: $5-10 (prompt generation)
- **Total**: ~$10-15
