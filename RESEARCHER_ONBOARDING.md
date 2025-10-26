# Researcher Onboarding Guide

Welcome! This guide will get you set up to run experiments on temporal steering.

## Quick Start (10 minutes)

```bash
# 1. Clone and setup environment
git clone https://github.com/JustinShenk/latents.git
cd latents
python3 -m venv venv_steering
source venv_steering/bin/activate  # On Windows: venv_steering\Scripts\activate
pip install -r requirements.txt

# 2. Verify installation
python -c "from latents import SteeringFramework; print('✓ Package imported')"

# 3. Run confound analysis experiment
python research/experiments/confound_analysis.py

# 4. View results
open research/results/pca_temporal_style_interactive.html  # macOS
# Or: firefox research/results/pca_temporal_style_interactive.html  # Linux
```

**Expected output**:
- PCA plots (static + interactive)
- Deconfounded steering vectors
- Quantitative separation metrics

## Project Structure

```
latents/
├── latents/          # Main package (don't edit for experiments)
│   ├── core/                   # Plugin architecture
│   ├── dimensions/             # Built-in steering dimensions
│   ├── extract_steering_vectors.py
│   └── model_adapter.py        # Multi-model support
│
├── research/                   # YOUR WORKSPACE
│   ├── experiments/            # Run experiments here
│   │   └── confound_analysis.py
│   ├── datasets/               # Experiment prompts
│   │   └── confound_experiment.json
│   ├── results/                # Outputs go here
│   │   ├── pca_temporal_style.png
│   │   └── pca_temporal_style_interactive.html
│   ├── docs/                   # Research documentation
│   ├── tools/                  # Analysis utilities
│   └── ISSUES.md               # Open research questions
│
├── steering_vectors/           # Pre-trained vectors
│   ├── temporal_scope.json
│   └── temporal_scope_deconfounded.json
│
├── examples/                   # Usage examples
│   ├── quick_demo.py
│   └── plugin_examples.py
│
└── data_download/              # Prompt datasets
```

## Running Experiments

### 1. Confound Analysis (Current Focus)

**Goal**: Verify temporal steering is not just detecting style

```bash
# Full experiment (takes ~5 min on CPU)
python research/experiments/confound_analysis.py

# Outputs:
# - research/datasets/confound_experiment.json (40 prompts)
# - research/results/pca_temporal_style.png (static plot)
# - research/results/pca_temporal_style_interactive.html (hover to see prompts!)
# - steering_vectors/temporal_scope_deconfounded.json (vectors)
```

**How to inspect results:**
1. Open `pca_temporal_style_interactive.html` in browser
2. Hover over points to see which prompts cluster together
3. Check if red (immediate) and blue (long-term) are separated
4. Check if circles (casual) and squares (formal) are separated

**What to look for:**
- ✅ Good: Clean separation along PC1 (temporal) and PC2 (style)
- ⚠️ Concern: Prompts cluster by style instead of temporal scope
- ❌ Bad: No clear separation (random scatter)

### 2. Behavioral Validation (Next Step)

**Goal**: Test if deconfounded vectors actually shift temporal scope

**Your task**: Implement human evaluation

```python
# Pseudocode for experiment
from latents import SteeringFramework

# Load deconfounded vectors
framework = SteeringFramework.load(
    model, tokenizer,
    'temporal_scope',
    vectors_file='steering_vectors/temporal_scope_deconfounded.json'
)

# Generate with different strengths
for strength in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    generated = framework.generate(
        prompt="How should we address climate change?",
        steerings=[('temporal_scope', strength)],
        temperature=0.7
    )
    # Save for human rating
```

**Human evaluation protocol**:
1. Generate 10 samples per condition (50 total)
2. Recruit raters (or rate yourself, blind)
3. Raters judge:
   - Temporal horizon: immediate / short / medium / long
   - Formality: casual / neutral / formal
4. Analyze: Does deconfounded steering shift temporal WITHOUT shifting formality?

### 3. Dataset Expansion (High Priority - See `research/ISSUES.md`)

**Goal**: Expand from 40 to 400 prompts

**Your task**: Generate more prompts with controls

**Template** for new prompts:
```json
{
  "topic": "education",
  "immediate_casual": "Kids are failing math tests. Quick fix?",
  "immediate_formal": "Student mathematics performance requires intervention. Solutions?",
  "longterm_casual": "How do we prep kids for jobs in 2075?",
  "longterm_formal": "What pedagogical frameworks prepare students for future economies?"
}
```

**Controls to maintain:**
- Word count: ±2 words across conditions
- Same semantic content (education, transportation, etc.)
- Avoid temporal keywords in style (e.g., don't use "long-term" in formal)
- Avoid style keywords in temporal (e.g., don't use "yo" in immediate)

**Validation**:
- Get 3 human raters (can be you + colleagues)
- Blind rating: Does immediate_casual feel more immediate than longterm_formal?
- Inter-rater reliability > 0.8 (use Cohen's kappa)

## Common Tasks

### Generate Steering Vectors

```bash
# For a new dimension (e.g., optimism)
python research/tools/extract_optimism_vectors.py \
  --pairs data/optimism_pairs.json \
  --output steering_vectors/optimism.json \
  --model gpt2 \
  --layers 7,8,9,10,11
```

### Test Steering Qualitatively

```python
from latents import SteeringFramework
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

framework = SteeringFramework.load(model, tokenizer, 'temporal_scope')

# Test immediate steering
result_immediate = framework.generate(
    prompt="Climate change policy?",
    steerings=[('temporal_scope', -0.8)],
    temperature=0.7,
    max_length=100
)

# Test long-term steering
result_longterm = framework.generate(
    prompt="Climate change policy?",
    steerings=[('temporal_scope', 0.8)],
    temperature=0.7,
    max_length=100
)

print("IMMEDIATE:", result_immediate)
print("\nLONG-TERM:", result_longterm)
```

### Run PCA on Custom Dataset

```python
# In your experiment script
from research.experiments.confound_analysis import (
    extract_activations_for_dataset,
    run_pca_analysis,
    visualize_pca_interactive
)

# Your custom dataset
my_dataset = {
    'condition_a': ["prompt 1", "prompt 2", ...],
    'condition_b': ["prompt 3", "prompt 4", ...]
}

# Extract activations
activations = extract_activations_for_dataset(model, tokenizer, my_dataset)

# PCA
pca, transformed, acts_by_cond = run_pca_analysis(activations, layer=10)

# Visualize
visualize_pca_interactive(acts_by_cond, my_dataset,
                          output_file='my_results.html')
```

## Troubleshooting

### Import Errors

```bash
# If: ModuleNotFoundError: No module named 'temporal_steering'
pip install -e .  # Install package in editable mode

# If: ModuleNotFoundError: No module named 'plotly'
pip install -r requirements.txt
```

### CUDA/GPU Issues

```python
# CPU-only mode (slower but works everywhere)
import torch
device = torch.device('cpu')
model = model.to(device)

# GPU mode (if available)
if torch.cuda.is_available():
    device = torch.device('cuda')
    model = model.to(device)
```

### Out of Memory

```python
# Use smaller batch sizes or smaller model
model = GPT2LMHeadModel.from_pretrained('gpt2')  # 124M params (smallest)
# Instead of:
# model = GPT2LMHeadModel.from_pretrained('gpt2-xl')  # 1.5B params
```

## Code Style & Contribution Guidelines

### Running Experiments

**DO**:
- ✅ Create new files in `research/experiments/`
- ✅ Save outputs to `research/results/`
- ✅ Document experiments in `research/docs/`
- ✅ Add your experiment to `research/ISSUES.md` if it addresses an issue
- ✅ Use descriptive filenames: `confound_analysis_v2.py`, not `test.py`

**DON'T**:
- ❌ Edit files in `latents/` (core package)
- ❌ Commit large binary files (>10MB) without asking
- ❌ Overwrite others' results files

### Naming Conventions

```python
# Experiments
confound_analysis.py          # Good
test.py                       # Bad
exp1.py                       # Bad

# Results
pca_temporal_style.png        # Good
result.png                    # Bad
fig1.png                      # Bad

# Datasets
confound_experiment.json      # Good
data.json                     # Bad
```

### Git Workflow

```bash
# Pull latest changes
git pull origin main

# Create feature branch for your experiment
git checkout -b experiment/your-name-confound-validation

# Make changes, commit frequently
git add research/experiments/my_experiment.py
git commit -m "Add confound validation with larger dataset"

# Push your branch
git push origin experiment/your-name-confound-validation

# Create pull request on GitHub
```

### Code Review Checklist

Before requesting review:
- [ ] Code runs without errors
- [ ] Results files are in `research/results/`
- [ ] Experiment documented in docstring or separate `.md` file
- [ ] Added any new dependencies to `requirements.txt`
- [ ] Large files (>10MB) added to `.gitignore` if not needed
- [ ] Commit messages are descriptive

## Research Questions & Issues

**Before starting work**, check `research/ISSUES.md` for:
1. Open research questions
2. Assigned tasks
3. Acceptance criteria

**Current high-priority issues**:
- Issue #1: Verify temporal steering not confounded by style
  - Expand dataset to 400 prompts
  - Behavioral validation
  - Multi-model testing

**Communication**:
- Add comments to issues with findings
- Tag others with @username for questions
- Update issue status when making progress

## Useful Resources

### Documentation
- **README.md**: Package overview
- **ARCHITECTURE.md**: Plugin system design
- **PLUGIN_GUIDE.md**: Create new steering dimensions
- **SCALING_AND_USAGE.md**: Multi-model support
- **research/ISSUES.md**: Current research questions
- **research/results/CONFOUND_ANALYSIS_RESULTS.md**: Current findings

### Papers & Background
- [Contrastive Activation Addition (CAA)](https://www.alignmentforum.org/posts/v7f8ayBxLhmMFRzpa/steering-llama-2-with-contrastive-activation-additions)
- nrimsky/CAA repository: https://github.com/nrimsky/CAA

### Slack/Discord (if set up)
- #general: General discussion
- #experiments: Share findings, ask for help
- #code-review: Request reviews

## Getting Help

1. **Check documentation** (this file, README.md, docstrings)
2. **Check `research/ISSUES.md`** for known issues
3. **Ask in Slack/Discord** (if set up)
4. **Open GitHub issue** for bugs or unclear documentation
5. **Tag @justinshenk** for urgent questions

## First Week Goals

**Day 1**: Setup + understand codebase
- [ ] Install dependencies
- [ ] Run confound_analysis.py successfully
- [ ] Open interactive PCA plot, understand what it shows
- [ ] Read CONFOUND_ANALYSIS_RESULTS.md

**Day 2-3**: Replicate existing work
- [ ] Generate your own prompts (5-10 per condition)
- [ ] Run PCA on your prompts
- [ ] Compare to existing results

**Day 4-5**: Contribute
- [ ] Pick task from `research/ISSUES.md`
- [ ] Implement and document
- [ ] Create PR for review

**Week 2+**: Independent research
- [ ] Design new experiment
- [ ] Discuss with team
- [ ] Execute and analyze

---

Welcome to the team! If anything is unclear, please improve this document for future researchers.

**Last updated**: 2025-10-26
**Questions?** Open a GitHub issue or tag @justinshenk
