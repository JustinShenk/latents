# Plugin Creation Guide

Create your own steering dimensions for the Horizon framework.

## Quick Start

### 1. Define Your Dimension

```python
# my_plugin.py
from temporal_steering import SteeringVector, register_steering

@register_steering("optimism")
class OptimismSteering(SteeringVector):
    """Steer between pessimistic and optimistic perspectives."""

    def get_dimension_name(self) -> str:
        return "optimism"

    def get_strength_range(self) -> tuple[float, float]:
        return (-1.0, 1.0)  # -1.0 = pessimistic, +1.0 = optimistic

    def interpret_strength(self, strength: float) -> str:
        if strength < -0.5:
            return "pessimistic/risk-focused"
        elif strength > 0.5:
            return "optimistic/opportunity-focused"
        else:
            return "balanced"
```

### 2. Create Contrastive Prompt Pairs

```python
# optimism_pairs.json
[
  {
    "negative_prompt": "What could go wrong with this new technology?",
    "positive_prompt": "What opportunities does this new technology create?"
  },
  {
    "negative_prompt": "Why might this startup fail?",
    "positive_prompt": "How could this startup succeed dramatically?"
  },
  {
    "negative_prompt": "What are the risks of expanding internationally?",
    "positive_prompt": "What are the opportunities from global expansion?"
  }
]
```

### 3. Extract Steering Vectors

```python
from temporal_steering import extract_steering_vectors
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load your pairs
pairs = load_prompt_pairs('optimism_pairs.json')

# Extract vectors
steering_vectors = compute_steering_vectors(model, tokenizer, pairs)

# Save
optimism_steering = OptimismSteering(steering_vectors, metadata={
    'extraction_date': '2024-10-26',
    'model': 'gpt2',
    'n_pairs': len(pairs)
})
optimism_steering.save_vectors('steering_vectors/optimism.json')
```

### 4. Use Your Plugin

```python
from temporal_steering import SteeringFramework

framework = SteeringFramework.load(
    model, tokenizer,
    'optimism',
    vectors_file='steering_vectors/optimism.json'
)

result = framework.generate(
    prompt="What's the future of renewable energy?",
    steerings=[('optimism', 0.7)],
    temperature=0.7
)
```

## Plugin Design Patterns

### Bipolar Dimensions

Steering between two opposite behaviors:

```python
@register_steering("formality")
class FormalitySteering(SteeringVector):
    def get_strength_range(self):
        return (-1.0, 1.0)  # -1.0 = casual, +1.0 = formal

    def interpret_strength(self, strength):
        if strength < -0.5:
            return "very casual/conversational"
        elif strength > 0.5:
            return "very formal/professional"
        else:
            return "neutral"
```

### Unipolar Dimensions

Steering from none to maximum:

```python
@register_steering("technical_detail")
class TechnicalDetailSteering(SteeringVector):
    def get_strength_range(self):
        return (0.0, 1.0)  # 0.0 = high-level, 1.0 = maximum detail

    def interpret_strength(self, strength):
        if strength < 0.3:
            return "executive summary"
        elif strength < 0.7:
            return "moderate detail"
        else:
            return "highly technical"
```

### Multi-Level Dimensions

More granular control:

```python
@register_steering("audience")
class AudienceSteering(SteeringVector):
    def get_strength_range(self):
        return (0.0, 1.0)

    def interpret_strength(self, strength):
        if strength < 0.2:
            return "expert/specialist"
        elif strength < 0.4:
            return "professional"
        elif strength < 0.6:
            return "educated general public"
        elif strength < 0.8:
            return "high school level"
        else:
            return "elementary level"
```

## Prompt Pair Design

### Good Contrastive Pairs

✅ **Clear contrast:**
```json
{
  "negative": "Explain the immediate business impact.",
  "positive": "Explain the long-term strategic value."
}
```

✅ **Same topic, different perspective:**
```json
{
  "negative": "What are the cybersecurity risks?",
  "positive": "What are the innovation opportunities?"
}
```

✅ **Natural language:**
```json
{
  "negative": "Write a casual email to a friend.",
  "positive": "Write a formal letter to a board member."
}
```

### Poor Contrastive Pairs

❌ **Different topics:**
```json
{
  "negative": "Explain quantum computing.",
  "positive": "Explain climate change."
}
```

❌ **Minimal contrast:**
```json
{
  "negative": "What should we do?",
  "positive": "What could we do?"
}
```

❌ **Multiple dimensions:**
```json
{
  "negative": "Brief technical explanation.",
  "positive": "Long simple explanation."
}
// Mixes length AND technicality
```

## Advanced Features

### Metadata and Versioning

```python
class MyS teering(SteeringVector):
    def __init__(self, layer_vectors, metadata=None):
        metadata = metadata or {}
        metadata.update({
            'version': '1.0',
            'author': 'Your Name',
            'description': 'Controls X dimension',
            'recommended_models': ['gpt2', 'llama-2-7b'],
            'extraction_method': 'CAA with 50 prompt pairs'
        })
        super().__init__(layer_vectors, metadata)
```

### Custom Extraction Logic

```python
class MyPluginSteering(SteeringVector):
    @classmethod
    def extract_from_dataset(cls, model, tokenizer, dataset_path):
        """Custom extraction logic."""
        # Load your specific dataset format
        # Apply custom preprocessing
        # Extract vectors
        # Return instance
        pass
```

### Composition Recommendations

```python
class MyPluginSteering(SteeringVector):
    def get_recommended_combinations(self):
        """Suggest compatible dimensions."""
        return {
            'complementary': ['temporal_scope', 'optimism'],
            'avoid': ['technical_detail'],  # May conflict
            'suggested_strengths': {
                'temporal_scope': 0.5,
                'optimism': 0.3
            }
        }
```

## Example Plugins

### 1. Optimism/Pessimism

```python
@register_steering("optimism")
class OptimismSteering(SteeringVector):
    """Risk-focused ↔ Opportunity-focused"""

    def get_dimension_name(self):
        return "optimism"

    def get_strength_range(self):
        return (-1.0, 1.0)

    def interpret_strength(self, strength):
        if strength < -0.5:
            return "pessimistic/risk-focused"
        elif strength > 0.5:
            return "optimistic/opportunity-focused"
        else:
            return "balanced"

    @classmethod
    def get_default_pairs(cls):
        return [
            {
                "negative": "What could go wrong?",
                "positive": "What opportunities exist?"
            },
            # More pairs...
        ]
```

### 2. Technical Detail

```python
@register_steering("technical_detail")
class TechnicalDetailSteering(SteeringVector):
    """High-level ↔ Technical/Detailed"""

    def get_dimension_name(self):
        return "technical_detail"

    def get_strength_range(self):
        return (-1.0, 1.0)

    def interpret_strength(self, strength):
        if strength < -0.5:
            return "executive summary/high-level"
        elif strength > 0.5:
            return "technical/implementation details"
        else:
            return "balanced detail"
```

### 3. Abstractness

```python
@register_steering("abstractness")
class AbstractnessSteering(SteeringVector):
    """Concrete examples ↔ Abstract principles"""

    def get_dimension_name(self):
        return "abstractness"

    def get_strength_range(self):
        return (-1.0, 1.0)

    def interpret_strength(self, strength):
        if strength < -0.5:
            return "concrete/example-based"
        elif strength > 0.5:
            return "abstract/principle-based"
        else:
            return "balanced"
```

## Testing Your Plugin

```python
# test_my_plugin.py
def test_plugin():
    from temporal_steering import SteeringFramework, STEERING_REGISTRY
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    # Check registration
    assert 'my_dimension' in STEERING_REGISTRY

    # Load model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Test loading
    framework = SteeringFramework.load(
        model, tokenizer, 'my_dimension',
        vectors_file='steering_vectors/my_dimension.json'
    )

    # Test generation at different strengths
    for strength in [-0.8, 0.0, 0.8]:
        result = framework.generate(
            prompt="Test prompt",
            steerings=[('my_dimension', strength)],
            max_length=50
        )
        print(f"Strength {strength}: {result[:100]}...")

    print("✓ All tests passed!")
```

## Publishing Your Plugin

### Package Structure

```
temporal-steering-plugin-optimism/
├── temporal_steering_plugins/
│   └── optimism.py
├── steering_vectors/
│   └── optimism.json
├── prompt_pairs/
│   └── optimism_pairs.json
├── tests/
│   └── test_optimism.py
├── pyproject.toml
├── README.md
└── LICENSE
```

### pyproject.toml

```toml
[project]
name = "temporal-steering-plugin-optimism"
version = "0.1.0"
description = "Optimism/pessimism steering dimension"
dependencies = [
    "temporal-steering>=0.1.0",
]

[project.entry-points."temporal_steering.plugins"]
optimism = "temporal_steering_plugins.optimism:OptimismSteering"
```

### Installation

```bash
pip install temporal-steering-plugin-optimism
```

Auto-discovered by the framework:

```python
from temporal_steering import SteeringFramework

framework.list_available_dimensions()
# ['temporal_scope', 'optimism', ...]
```

## Best Practices

1. **Clear Semantics**: Make your dimension's meaning obvious
2. **Consistent Scaling**: Use standard ranges (-1 to 1, or 0 to 1)
3. **Good Pairs**: Create 20-100 high-quality contrastive pairs
4. **Test Thoroughly**: Verify steering effect across different prompts
5. **Document Well**: Explain use cases and interpretation
6. **Version Your Vectors**: Track extraction method and model version

## Community

- Share your plugins in the [Plugin Registry](https://github.com/justinshenk/temporal-steering/wiki/Plugins)
- Discuss ideas in [Discussions](https://github.com/justinshenk/temporal-steering/discussions)
- Report issues with the plugin system: [Issues](https://github.com/justinshenk/temporal-steering/issues)

## Useful Dimension Ideas

- **Formality**: casual ↔ formal
- **Certainty**: hedging ↔ confident assertions
- **Brevity**: concise ↔ comprehensive
- **Creativity**: conventional ↔ novel
- **Risk tolerance**: conservative ↔ aggressive
- **Collaboration**: competitive ↔ cooperative
- **Focus**: broad/holistic ↔ narrow/specialized

---

Ready to build? Start with the quick start guide above!
