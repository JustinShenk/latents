# Temporal Steering Demo - Quick Start Guide

## TL;DR

```bash
# Install dependencies
pip install -r requirements_steering.txt

# Run setup (extracts vectors and launches demo)
./setup_steering_demo.sh
```

Then open http://localhost:5000 in your browser.

---

## What This Does

Move a slider to control GPT-2's temporal thinking:

- **Left (-1.0)**: Immediate, urgent, short-term focus
- **Right (+1.0)**: Long-term, transformative, generational focus

Try prompts like:
- "What should we do about climate change?"
- "How can we improve education?"
- "What's the best approach to solving the housing crisis?"

Watch how the model's responses shift from tactical quick-fixes to strategic long-term planning.

---

## Prerequisites

1. Python 3.8+
2. Virtual environment activated:
   ```bash
   source venv/bin/activate
   ```

---

## Option 1: Automated Setup (Recommended)

```bash
./setup_steering_demo.sh
```

This script will:
1. Check dependencies (install if needed)
2. Extract steering vectors from 20 prompt pairs (~2-3 minutes)
3. Launch Flask server on port 5000
4. Open http://localhost:5000

---

## Option 2: Manual Setup

### Step 1: Install Dependencies

```bash
pip install torch transformers flask numpy
```

Or:

```bash
pip install -r requirements_steering.txt
```

### Step 2: Extract Steering Vectors

```bash
python3 src/extract_steering_vectors.py \
  --pairs data_download/test_prompts.json \
  --output steering_vectors/temporal_steering.json \
  --model gpt2 \
  --max-pairs 20
```

**Time**: ~2-3 minutes on CPU
**Output**: `steering_vectors/temporal_steering.json`

### Step 3: Launch Demo

```bash
python3 src/temporal_steering_demo.py \
  --steering steering_vectors/temporal_steering.json \
  --model gpt2 \
  --port 5000
```

**Server runs at**: http://localhost:5000

---

## Using the Demo

### 1. Enter a Prompt
Type or paste a question in the text area. Examples:

- "What should policymakers prioritize to address climate change?"
- "How can we solve the affordable housing crisis?"
- "What's the best approach to education reform?"

### 2. Adjust the Slider

- **Strong Immediate (-1.0 to -0.6)**: Focus on urgent, immediate actions
- **Moderate Immediate (-0.5 to -0.2)**: Near-term priorities
- **Neutral (-0.1 to 0.1)**: Balanced perspective
- **Moderate Long-term (0.2 to 0.5)**: Strategic planning
- **Strong Long-term (0.6 to 1.0)**: Transformative, generational thinking

### 3. Click Generate

Wait 2-5 seconds for generation.

### 4. Observe Results

- **Generated text**: Model's response with temporal steering applied
- **Metrics cards**:
  - Planning horizon (immediate, short-term, medium-term, long-term)
  - Intervention type (tactical, moderate, transformative)
  - Stakeholder scope (individual, organizational, societal)
  - Temporal balance (long-term - immediate marker count)

### 5. Try Validation Tasks

Scroll down to see pre-loaded tasks where temporal scope affects the correct answer:

1. **Investment decision**: Should you invest now or wait?
2. **Climate policy**: Shut down coal plant now or phase out?
3. **Software engineering**: Fix bugs or build infrastructure?

Generate responses at different slider positions to see if model's advice changes.

---

## Troubleshooting

### Dependencies Missing

```bash
# Make sure venv is activated
source venv/bin/activate

# Install requirements
pip install -r requirements_steering.txt
```

### Steering Vectors Not Found

```bash
# Extract vectors manually
python3 src/extract_steering_vectors.py \
  --pairs data_download/test_prompts.json \
  --output steering_vectors/temporal_steering.json \
  --max-pairs 20
```

### Port Already in Use

```bash
# Use different port
python3 src/temporal_steering_demo.py \
  --steering steering_vectors/temporal_steering.json \
  --port 5001
```

Then open http://localhost:5001

### Generation Too Slow

```bash
# Reduce max length in UI (try 50-75 tokens)
# Or reduce temperature to 0.5
```

### Steering Has No Effect

**Solutions**:
1. Increase steering strength (try -2.0 or +2.0 in slider)
2. Extract from more prompt pairs:
   ```bash
   python3 src/extract_steering_vectors.py \
     --pairs data_download/train_prompts.json \
     --max-pairs 50 \
     --output steering_vectors/temporal_steering_robust.json
   ```
3. Try different layers:
   ```bash
   python3 src/temporal_steering_demo.py \
     --steering steering_vectors/temporal_steering.json \
     --layers "8,9,10,11"
   ```

---

## Advanced Usage

### Use More Prompt Pairs (Better Quality)

```bash
python3 src/extract_steering_vectors.py \
  --pairs data_download/train_prompts.json \
  --output steering_vectors/temporal_steering_200pairs.json \
  --max-pairs 200
```

**Note**: This takes ~20-30 minutes but produces stronger steering.

### Specify Steering Layers

```bash
python3 src/temporal_steering_demo.py \
  --steering steering_vectors/temporal_steering.json \
  --layers "6,7,8,9,10,11"
```

Try:
- Early: `"2,3,4,5"` (may affect style)
- Middle: `"5,6,7,8"` (balanced)
- Late: `"8,9,10,11"` (semantic steering)

### Use Larger Model

```bash
# Extract with GPT-2 medium
python3 src/extract_steering_vectors.py \
  --pairs data_download/test_prompts.json \
  --model gpt2-medium \
  --output steering_vectors/temporal_steering_medium.json

# Run demo
python3 src/temporal_steering_demo.py \
  --steering steering_vectors/temporal_steering_medium.json \
  --model gpt2-medium
```

**Note**: Requires more memory and is slower.

---

## What to Try

### Experiment 1: Slider Sweep
1. Enter prompt: "What should we do about climate change?"
2. Generate at slider positions: -1.0, -0.5, 0.0, 0.5, 1.0
3. Compare responses: Do they shift from immediate to long-term?

### Experiment 2: Validation Tasks
1. Pick a validation task (e.g., investment decision)
2. Generate at -1.0 (immediate focus)
3. Generate at +1.0 (long-term focus)
4. Compare: Does the model give different advice?

### Experiment 3: Custom Prompts
Try prompts about:
- Career decisions
- Business strategy
- Health and wellness
- Social policy
- Technology adoption

Observe: Which prompts show strongest temporal shift?

---

## Expected Results

### Good Signs
- ✅ Responses at -1.0 use immediate keywords ("now", "today", "urgent")
- ✅ Responses at +1.0 use long-term keywords ("future", "legacy", "generations")
- ✅ Planning horizon metric changes with slider
- ✅ Validation task answers differ based on slider position

### Red Flags
- ⚠️ No visible difference between -1.0 and +1.0
- ⚠️ Metrics always show same values
- ⚠️ Generated text is nonsensical

**If red flags occur**: See troubleshooting section above.

---

## Next Steps

After trying the demo:

1. **Document observations**: Which prompts work best? Which show clear temporal shift?
2. **Validate systematically**: Run validation tasks and record whether answers change
3. **Explore confounds**: Does steering affect only temporal scope, or also style/length/topic?

For detailed analysis, see:
- `STEERING_DEMO_README.md` - Complete documentation
- `TEMPORAL_STEERING_COMPLETE.md` - Technical details
- `RIGOROUS_EXPERIMENTAL_DESIGN.md` - Validation protocol

---

## Summary

**What you get**:
- Real-time temporal steering of GPT-2
- Interactive slider interface
- Validation tasks for testing
- Metrics for analysis

**What it costs**:
- ~5 minutes setup time
- ~2GB RAM for GPT-2 small
- ~5 seconds per generation

**What it proves**:
- Temporal scope can be steered via activation addition
- Steering is interpretable (slider maps to temporal focus)
- No retraining required (inference-time only)

**What's next**:
- Systematic validation
- Circuit analysis
- Cross-model testing

---

**Ready to explore temporal steering!** 🚀

Run `./setup_steering_demo.sh` and start experimenting.
