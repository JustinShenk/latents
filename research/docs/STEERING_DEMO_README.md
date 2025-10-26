# Temporal Steering Demo

Interactive demonstration of **Contrastive Activation Addition (CAA)** for steering GPT-2's temporal scope in real-time.

## Overview

This demo allows you to control the temporal horizon of GPT-2's responses using activation steering:

- **Slider Interface**: Move from immediate/urgent thinking to long-term/transformative planning
- **Real-time Generation**: See how steering affects model outputs
- **Validation Tasks**: Test on questions where temporal scope affects the correct answer
- **Metrics**: Automatic analysis of temporal characteristics in generated text

## Quick Start

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run setup script (extracts steering vectors and launches demo)
./setup_steering_demo.sh
```

Then open http://localhost:5000 in your browser.

## How It Works

### Contrastive Activation Addition (CAA)

1. **Extract Contrastive Vectors**: Compute difference between long-term and immediate prompt activations
   ```
   steering_vector = avg(long_term_activations) - avg(immediate_activations)
   ```

2. **Apply During Generation**: Add steering vector to model activations at middle-to-late layers
   ```
   hidden_states_modified = hidden_states + (strength * steering_vector)
   ```

3. **Control Strength**: Slider from -1.0 (immediate) to +1.0 (long-term)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Extract Steering Vectors                  │
│                                                               │
│  Prompt Pairs:                                               │
│  • Immediate: "What can we do TODAY to stop wildfire?"      │
│  • Long-term: "How can we build LEGACY for future gens?"   │
│                           ↓                                  │
│  Extract activations at all layers from GPT-2               │
│                           ↓                                  │
│  Compute contrastive vectors: long - immediate              │
│                           ↓                                  │
│  Average across all pairs → steering vectors                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Real-time Steering Demo                    │
│                                                               │
│  User Input: "What should we do about climate change?"      │
│              [Slider: -1.0 (immediate) ←→ +1.0 (long-term)] │
│                           ↓                                  │
│  Forward pass through GPT-2 with hooks:                     │
│    - At layers 4-11 (middle-to-late):                       │
│      hidden[layer] += strength * steering_vector[layer]     │
│                           ↓                                  │
│  Generated text (steered toward temporal scope)             │
│                           ↓                                  │
│  Metrics: planning horizon, intervention type, etc.         │
└─────────────────────────────────────────────────────────────┘
```

## Manual Usage

### Step 1: Extract Steering Vectors

```bash
python3 src/extract_steering_vectors.py \
  --pairs data_download/test_prompts.json \
  --output steering_vectors/temporal_steering.json \
  --model gpt2 \
  --max-pairs 20
```

**Options**:
- `--pairs`: JSON file with prompt pairs (immediate_prompt, long_term_prompt)
- `--output`: Where to save steering vectors
- `--model`: GPT-2 variant (gpt2, gpt2-medium, gpt2-large)
- `--max-pairs`: Limit number of pairs (for faster extraction)
- `--layers`: Specific layers to extract (e.g., "6,7,8,9")

### Step 2: Launch Demo Server

```bash
python3 src/temporal_steering_demo.py \
  --steering steering_vectors/temporal_steering.json \
  --model gpt2 \
  --port 5000
```

**Options**:
- `--steering`: Path to steering vectors JSON
- `--model`: Must match model used for extraction
- `--port`: Server port (default: 5000)
- `--layers`: Layers to apply steering (default: auto-select middle-to-late)

## Validation Framework

The demo includes validation tasks where temporal scope should affect the correct answer:

### Task 1: Investment Decision
**Question**: Interest rates at 1%, will rise to 5% next year. What to do with $10,000?

- **Immediate focus**: Wait for higher rates (better short-term returns)
- **Long-term focus**: Invest now (time in market > timing market)

### Task 2: Climate Policy
**Question**: Coal plant can shut down now (job losses) or phase out over 20 years. What should policymakers do?

- **Immediate focus**: Gradual phase-out (protect workers)
- **Long-term focus**: Immediate shutdown + renewable investment (future generations)

### Task 3: Software Engineering
**Question**: Fix bugs in current product or build new infrastructure?

- **Immediate focus**: Fix bugs (users experiencing issues)
- **Long-term focus**: Build infrastructure (technical debt compounds)

## Expected Behavior

### Steering Strength: -1.0 (Strong Immediate)
- Keywords: "now", "today", "urgent", "quick", "immediate"
- Planning horizon: Hours, days, weeks
- Intervention type: Tactical, reactive
- Stakeholder scope: Individual, organizational

### Steering Strength: 0.0 (Neutral)
- Balanced temporal perspective
- Mix of short and long-term considerations

### Steering Strength: +1.0 (Strong Long-term)
- Keywords: "future", "generations", "legacy", "sustainable", "transformative"
- Planning horizon: Years, decades, centuries
- Intervention type: Fundamental, systemic
- Stakeholder scope: Societal, global

## Metrics Explained

The demo automatically analyzes generated text for temporal characteristics:

1. **Planning Horizon**: Detected timeframe (immediate, short-term, medium-term, long-term)
2. **Intervention Type**: Tactical vs transformative approach
3. **Stakeholder Scope**: Individual, organizational, or societal focus
4. **Temporal Balance**: Difference between long-term and immediate markers

## Advanced Usage

### Custom Prompt Pairs

Create your own steering vectors:

```json
[
  {
    "immediate_prompt": "What can we do RIGHT NOW to improve team morale?",
    "long_term_prompt": "How can we build a culture that lasts for generations?",
    "topic": "organizational culture"
  }
]
```

Then extract steering vectors using your custom pairs.

### Layer Selection

Experiment with different layers for steering:

```bash
# Early layers (may affect low-level features)
python3 src/temporal_steering_demo.py \
  --steering steering_vectors/temporal_steering.json \
  --layers "2,3,4,5"

# Middle layers (balanced)
python3 src/temporal_steering_demo.py \
  --steering steering_vectors/temporal_steering.json \
  --layers "5,6,7,8"

# Late layers (high-level semantics)
python3 src/temporal_steering_demo.py \
  --steering steering_vectors/temporal_steering.json \
  --layers "8,9,10,11"
```

### Batch Generation

Test steering systematically:

```python
from latents_demo import TemporalSteering, load_steering_vectors
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
steering_vectors = load_steering_vectors('steering_vectors/temporal_steering.json')

steering = TemporalSteering(model, tokenizer, steering_vectors)

# Generate at different strengths
prompt = "What should we do about climate change?"
for strength in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    output = steering.generate_with_steering(prompt, steering_strength=strength)
    print(f"\nStrength {strength:+.1f}:")
    print(output)
```

## Validation Protocol

### Stage 1: Human Ranking
1. Generate 10 responses at different slider positions
2. Ask humans to rank temporal scope (blind to slider position)
3. **Success**: Human rankings correlate with slider position (r > 0.7)

### Stage 2: Behavioral Validation
1. Use validation tasks where correct answer depends on temporal scope
2. Generate responses at -1.0 and +1.0
3. **Success**: Answers differ systematically (>60% show expected shift)

### Stage 3: Confound Analysis
1. Generate with constant style prompt but vary temporal implications
2. Test if steering affects temporal reasoning independent of style
3. **Success**: Temporal shift without style changes

## Troubleshooting

### Issue: Steering has no visible effect

**Solutions**:
- Increase steering strength (try ±2.0 or ±3.0)
- Use more prompt pairs for extraction (50+ recommended)
- Try different layers (late layers often better for semantic steering)
- Check that model matches between extraction and demo

### Issue: Model generates nonsense

**Solutions**:
- Reduce steering strength (try ±0.5)
- Lower temperature (try 0.5 instead of 0.8)
- Use fewer layers for steering
- Extract steering vectors from higher-quality prompt pairs

### Issue: Slow generation

**Solutions**:
- Use smaller model (gpt2 instead of gpt2-large)
- Reduce max_length
- Limit number of steering layers

## Files

```
src/
  extract_steering_vectors.py   - Extract CAA vectors from prompt pairs
  temporal_steering_demo.py      - Flask server with steering generation

templates/
  temporal_steering.html         - Interactive web UI

steering_vectors/
  temporal_steering.json         - Extracted steering vectors

data_download/
  test_prompts.json              - Prompt pairs for extraction
  train_prompts.json             - Additional prompt pairs
```

## References

- **Contrastive Activation Addition**: Li et al. (2024) - "Inference-Time Intervention"
- **Activation Steering**: Turner et al. (2023) - "Activation Addition: Steering Language Models Without Optimization"
- **Mechanistic Interpretability**: Elhage et al. (2021) - "A Mathematical Framework for Transformer Circuits"

## Next Steps

1. **Quantitative Validation**: Run systematic tests on validation tasks
2. **Circuit Analysis**: Identify which attention heads are most affected by steering
3. **Cross-model**: Test if steering vectors transfer across GPT-2 variants
4. **Implicit Temporal**: Extract steering from implicit temporal prompts (no keywords)
5. **Multi-dimensional**: Extend to other dimensions (risk tolerance, innovation vs tradition, etc.)

---

**Ready to explore temporal steering!** 🚀

Try the demo and experiment with different prompts and slider positions.
