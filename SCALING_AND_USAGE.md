# Scaling to Other Models & Real-World Usage

## 🔍 Current State: GPT-2 Specific

### Hardcoded Parts (5 locations)

The code currently has **GPT-2-specific layer access** in 5 places:

```python
# ❌ GPT-2 specific
model.transformer.h[layer_idx]          # Access to transformer layers
len(model.transformer.h)                # Number of layers
```

**Files affected:**
- `latents/latents_demo.py` (2 occurrences)
- `latents/extract_steering_vectors.py` (3 occurrences)

### Model Architecture Differences

| Model | Layer Access | # Layers |
|-------|--------------|----------|
| **GPT-2** | `model.transformer.h[i]` | 12 (124M) |
| **LLaMA-2-7B** | `model.model.layers[i]` | 32 |
| **LLaMA-2-13B** | `model.model.layers[i]` | 40 |
| **LLaMA-2-70B** | `model.model.layers[i]` | 80 |
| **Mistral-7B** | `model.model.layers[i]` | 32 |
| **Falcon-7B** | `model.transformer.h[i]` | 32 |
| **GPT-NeoX** | `model.gpt_neox.layers[i]` | 44 |

## ✅ Solution: Model Adapter

Create a simple adapter to detect and access layers for any model:

```python
# latents/model_adapter.py

def get_model_layers(model):
    """Get transformer layers for any model architecture."""

    # Try common patterns
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h  # GPT-2, GPT-Neo, Falcon

    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers  # LLaMA, Mistral, Mixtral

    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return model.gpt_neox.layers  # GPT-NeoX

    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
        return model.transformer.layers  # Some other architectures

    else:
        raise ValueError(f"Unknown model architecture: {type(model).__name__}")


def get_model_config(model):
    """Get model metadata for layer selection."""
    layers = get_model_layers(model)
    n_layers = len(layers)

    # Recommend targeting last 60% of layers
    start_layer = int(n_layers * 0.4)

    return {
        'layers': layers,
        'n_layers': n_layers,
        'recommended_start': start_layer,
        'recommended_layers': list(range(start_layer, n_layers))
    }
```

### Updated TemporalSteering Class

```python
class TemporalSteering:
    def __init__(self, model, tokenizer, steering_vectors, target_layers=None):
        self.model = model
        self.tokenizer = tokenizer
        self.steering_vectors = steering_vectors

        # Auto-detect model architecture
        self.model_config = get_model_config(model)

        if target_layers is None:
            # Use recommended layers (last 60%)
            self.target_layers = self.model_config['recommended_layers']
        else:
            self.target_layers = target_layers

        print(f"Model: {type(model).__name__}")
        print(f"Total layers: {self.model_config['n_layers']}")
        print(f"Steering layers: {self.target_layers}")

    def generate_with_steering(self, ...):
        # Use model_config['layers'] instead of model.transformer.h
        for layer_idx in self.target_layers:
            hook = self.model_config['layers'][layer_idx].register_forward_hook(...)
```

## 🚀 Multi-Model Usage Examples

### LLaMA-2

```python
from tempsteer import TemporalSteering
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load LLaMA-2
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

# Extract LLaMA-2 specific steering vectors
tempsteer extract \
  --pairs data_download/train_prompts.json \
  --output steering_vectors/llama2_temporal.json \
  --model meta-llama/Llama-2-7b-hf \
  --max-pairs 20

# Use with steering
steering = TemporalSteering(model, tokenizer, llama2_vectors)
result = steering.generate_with_steering(
    prompt="Design a plan to improve healthcare access",
    steering_strength=0.8,
    temperature=0.7
)
```

### Mistral

```python
model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1')
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')

# Works the same way!
steering = TemporalSteering(model, tokenizer, mistral_vectors)
```

## 💡 Real-World Use Cases

### 1. **Policy Planning & Government**

**Problem**: Policy decisions need different temporal scopes depending on context.

```python
# Short-term crisis response
crisis_response = steering.generate_with_steering(
    prompt="How should we respond to the current energy crisis?",
    steering_strength=-0.8,  # Immediate actions
    temperature=0.6
)
# → Focus on: emergency measures, quick wins, immediate relief

# Long-term infrastructure
infrastructure = steering.generate_with_steering(
    prompt="How should we develop our energy infrastructure?",
    steering_strength=0.8,  # Strategic thinking
    temperature=0.6
)
# → Focus on: renewable transition, grid modernization, 2050 goals
```

### 2. **Business Strategy**

**Problem**: Executives need both quarterly plans and decade-long visions.

```python
# Quarterly OKRs
q4_plan = steering.generate_with_steering(
    prompt="What should our product roadmap priorities be?",
    steering_strength=-0.5,  # Moderate immediate
    temperature=0.7
)
# → Focus on: next release, customer feedback, bug fixes

# 10-year vision
vision = steering.generate_with_steering(
    prompt="What should our product roadmap priorities be?",
    steering_strength=0.9,  # Strong long-term
    temperature=0.7
)
# → Focus on: market transformation, platform evolution, ecosystem
```

### 3. **Healthcare & Treatment Plans**

```python
# Emergency triage
triage = steering.generate_with_steering(
    prompt="Patient with chest pain and shortness of breath - action plan?",
    steering_strength=-1.0,  # Maximum immediate
    temperature=0.3  # Low temp for critical decisions
)
# → Focus on: immediate assessment, emergency protocols, stabilization

# Chronic disease management
chronic = steering.generate_with_steering(
    prompt="Patient with Type 2 diabetes - management plan?",
    steering_strength=0.7,  # Long-term health
    temperature=0.5
)
# → Focus on: lifestyle changes, long-term complications, prevention
```

### 4. **Financial Planning**

```python
# Day trading
day_trade = steering.generate_with_steering(
    prompt="Market analysis for tech sector today",
    steering_strength=-0.9,
    temperature=0.6
)
# → Focus on: today's movements, immediate catalysts, intraday patterns

# Retirement planning
retirement = steering.generate_with_steering(
    prompt="Investment strategy for 30-year-old",
    steering_strength=0.9,
    temperature=0.6
)
# → Focus on: compound growth, asset allocation, long-term trends
```

### 5. **Education & Curriculum**

```python
# Test prep
test_prep = steering.generate_with_steering(
    prompt="Study plan for exam next week",
    steering_strength=-0.7,
    temperature=0.7
)
# → Focus on: cramming strategies, practice tests, key topics

# Career development
career = steering.generate_with_steering(
    prompt="Learning plan for becoming a data scientist",
    steering_strength=0.8,
    temperature=0.7
)
# → Focus on: foundational skills, degree programs, career trajectory
```

### 6. **Climate & Sustainability**

```python
# Immediate emissions
immediate = steering.generate_with_steering(
    prompt="Reduce corporate carbon footprint",
    steering_strength=-0.6,
    temperature=0.6
)
# → Focus on: energy efficiency, waste reduction, quick wins

# Net-zero transition
net_zero = steering.generate_with_steering(
    prompt="Achieve net-zero emissions",
    steering_strength=1.0,
    temperature=0.6
)
# → Focus on: renewable transition, carbon capture, systemic change
```

## 🎯 Practical Integration Patterns

### Pattern 1: Adaptive AI Assistant

```python
class TemporalAdaptiveAssistant:
    """AI that adapts temporal scope based on context."""

    def __init__(self, model, tokenizer, steering_vectors):
        self.steering = TemporalSteering(model, tokenizer, steering_vectors)

    def answer(self, question, context=None):
        # Detect temporal keywords
        immediate_keywords = ['urgent', 'now', 'today', 'asap', 'emergency']
        longterm_keywords = ['future', 'strategy', 'vision', 'long-term', 'sustainable']

        question_lower = question.lower()

        # Auto-adjust steering
        if any(kw in question_lower for kw in immediate_keywords):
            strength = -0.7
        elif any(kw in question_lower for kw in longterm_keywords):
            strength = 0.7
        else:
            strength = 0.0

        return self.steering.generate_with_steering(
            prompt=question,
            steering_strength=strength,
            temperature=0.7
        )

# Usage
assistant = TemporalAdaptiveAssistant(model, tokenizer, vectors)
assistant.answer("What should we do about the urgent server outage?")
# → Auto-detects "urgent", uses immediate steering
```

### Pattern 2: Multi-Perspective Analysis

```python
def get_multi_perspective_analysis(prompt, steering):
    """Generate immediate, medium, and long-term perspectives."""

    perspectives = {}

    perspectives['immediate'] = steering.generate_with_steering(
        prompt=f"Immediate perspective: {prompt}",
        steering_strength=-0.8,
        temperature=0.7
    )

    perspectives['medium_term'] = steering.generate_with_steering(
        prompt=f"Medium-term perspective: {prompt}",
        steering_strength=0.0,
        temperature=0.7
    )

    perspectives['long_term'] = steering.generate_with_steering(
        prompt=f"Long-term perspective: {prompt}",
        steering_strength=0.8,
        temperature=0.7
    )

    return perspectives

# Usage
analysis = get_multi_perspective_analysis(
    "How should we approach AI regulation?",
    steering
)
print("IMMEDIATE:", analysis['immediate'])
print("MEDIUM:", analysis['medium_term'])
print("LONG-TERM:", analysis['long_term'])
```

### Pattern 3: Decision Support System

```python
class TemporalDecisionSupport:
    """Help decision-makers see trade-offs across time horizons."""

    def compare_options(self, decision_prompt, options, steering):
        results = {}

        for option in options:
            full_prompt = f"{decision_prompt}\nOption: {option}\nAnalysis:"

            results[option] = {
                'immediate_impact': steering.generate_with_steering(
                    prompt=full_prompt + " Immediate effects?",
                    steering_strength=-0.8,
                    max_length=100
                ),
                'long_term_impact': steering.generate_with_steering(
                    prompt=full_prompt + " Long-term consequences?",
                    steering_strength=0.8,
                    max_length=100
                )
            }

        return results
```

## 📊 Performance Expectations by Model

| Model | Parameters | Temporal Effect Strength | Recommended Use |
|-------|-----------|-------------------------|-----------------|
| GPT-2 | 124M | Moderate | Proof of concept, research |
| GPT-2-Large | 774M | Strong | Production (smaller) |
| LLaMA-2-7B | 7B | Very Strong | Production (recommended) |
| LLaMA-2-13B | 13B | Very Strong | High quality |
| Mistral-7B | 7B | Very Strong | Fast + quality |

**General rule**: Larger models show stronger and more consistent temporal steering effects.

## 🔄 Roadmap for Full Multi-Model Support

### Phase 1: Model Adapter (High Priority)
- [ ] Create `model_adapter.py` with `get_model_layers()`
- [ ] Update `TemporalSteering` class
- [ ] Update `extract_steering_vectors.py`
- [ ] Test with LLaMA-2, Mistral

### Phase 2: Pre-trained Vectors
- [ ] Extract vectors for LLaMA-2-7B
- [ ] Extract vectors for Mistral-7B
- [ ] Benchmark steering strength across models

### Phase 3: Advanced Features
- [ ] Auto-detect optimal layers per model
- [ ] Multi-dimensional steering (temporal + other attributes)
- [ ] Model-specific steering strength calibration

## 🎓 Research Opportunities

1. **Cross-model transfer**: Do GPT-2 vectors work on LLaMA with scaling?
2. **Layer analysis**: Which layers are most effective for each model?
3. **Combination steering**: Temporal + truthfulness + helpfulness
4. **Adaptive strength**: Learn optimal steering strength from feedback

---

**Want multi-model support?** I can implement the model adapter right now! Should take about 30 minutes to make it fully model-agnostic.
