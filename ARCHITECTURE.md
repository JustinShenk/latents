# Horizon: Plugin Architecture for Multi-Dimensional LLM Steering

## Vision

**Horizon** is a framework for steering large language models along multiple behavioral dimensions using Contrastive Activation Addition (CAA). While temporal scope is the flagship feature, the architecture supports any steering dimension through a plugin system.

## Package Name Evolution

- **Old**: `temporal-steering` (specific, limited)
- **New**: `horizon` (abstract, extensible)
  - PyPI: `horizon` or `horizon-llm`
  - Import: `from horizon import TemporalSteering, SteeringFramework`
  - Tagline: "Steer LLMs along multiple dimensions"

## Core Architecture

### 1. Base Classes

```python
# horizon/core/steering_vector.py

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class SteeringVector(ABC):
    """Base class for all steering vectors."""

    def __init__(self, layer_vectors: Dict[int, np.ndarray], metadata: Dict[str, Any]):
        self.layer_vectors = layer_vectors
        self.metadata = metadata

    @abstractmethod
    def get_dimension_name(self) -> str:
        """Return human-readable name of this dimension (e.g., 'temporal_scope')."""
        pass

    @abstractmethod
    def get_strength_range(self) -> tuple[float, float]:
        """Return valid strength range (e.g., (-1.0, 1.0))."""
        pass

    @abstractmethod
    def interpret_strength(self, strength: float) -> str:
        """Convert strength value to human-readable description."""
        pass

    def apply_to_activations(self, activations, strength: float) -> torch.Tensor:
        """Apply steering vector to activations."""
        # Default implementation
        return activations + strength * self.get_vector()
```

### 2. Steering Framework

```python
# horizon/core/framework.py

class SteeringFramework:
    """Main interface for multi-dimensional steering."""

    def __init__(self, model, tokenizer, steerings: Dict[str, SteeringVector] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.steerings = steerings or {}
        self.model_config = get_model_config(model)

    def generate(
        self,
        prompt: str,
        steerings: list[tuple[str, float]] = None,
        max_length: int = 100,
        temperature: float = 0.7
    ):
        """
        Generate with multiple steering dimensions.

        Args:
            prompt: Input text
            steerings: List of (dimension_name, strength) tuples
                      e.g., [("temporal_scope", 0.8), ("optimism", 0.3)]
            max_length: Maximum tokens
            temperature: Sampling temperature
        """
        # Compose multiple steering vectors
        combined_vectors = self._compose_steerings(steerings)

        # Apply during generation
        return self._generate_with_hooks(prompt, combined_vectors, max_length, temperature)

    def _compose_steerings(self, steerings):
        """Combine multiple steering vectors."""
        if not steerings:
            return {}

        combined = {}
        for dimension_name, strength in steerings:
            if dimension_name not in self.steerings:
                raise ValueError(f"Unknown steering dimension: {dimension_name}")

            steering_vec = self.steerings[dimension_name]

            # Add weighted vectors for each layer
            for layer, vec in steering_vec.layer_vectors.items():
                if layer not in combined:
                    combined[layer] = np.zeros_like(vec)
                combined[layer] += strength * vec

        return combined

    @classmethod
    def load(cls, model, tokenizer, dimension: str, vectors_file: str = None):
        """
        Load a specific steering dimension.

        Args:
            dimension: "temporal", "optimism", "technical_detail", etc.
            vectors_file: Optional custom vectors file
        """
        # Auto-discover from registry
        steering_class = STEERING_REGISTRY.get(dimension)
        if steering_class is None:
            raise ValueError(f"Unknown dimension: {dimension}")

        # Load vectors
        if vectors_file is None:
            vectors_file = f"steering_vectors/{dimension}.json"

        vectors = steering_class.load_vectors(vectors_file)

        return cls(model, tokenizer, {dimension: vectors})
```

### 3. Registry System

```python
# horizon/core/registry.py

STEERING_REGISTRY = {}

def register_steering(dimension_name: str):
    """Decorator to register a steering dimension."""
    def decorator(cls):
        STEERING_REGISTRY[dimension_name] = cls
        return cls
    return decorator

# Usage:
@register_steering("temporal_scope")
class TemporalSteering(SteeringVector):
    def get_dimension_name(self):
        return "temporal_scope"

    def get_strength_range(self):
        return (-1.0, 1.0)

    def interpret_strength(self, strength):
        if strength < -0.5:
            return "immediate/tactical"
        elif strength > 0.5:
            return "long-term/strategic"
        else:
            return "balanced"
```

## Plugin Examples

### Built-in: Temporal Scope

```python
from horizon import SteeringFramework, TemporalSteering

# Primary use case (backward compatible)
framework = SteeringFramework.load(model, tokenizer, "temporal_scope")
result = framework.generate(
    prompt="Climate change policy?",
    steerings=[("temporal_scope", 0.8)],
    temperature=0.7
)
```

### Community Plugin: Optimism

```python
# horizon_plugins/optimism.py
from horizon import SteeringVector, register_steering

@register_steering("optimism")
class OptimismSteering(SteeringVector):
    """Steer between pessimistic and optimistic perspectives."""

    def get_dimension_name(self):
        return "optimism"

    def get_strength_range(self):
        return (-1.0, 1.0)  # -1.0 = pessimistic, +1.0 = optimistic

    def interpret_strength(self, strength):
        if strength < -0.5:
            return "pessimistic/risk-focused"
        elif strength > 0.5:
            return "optimistic/opportunity-focused"
        else:
            return "balanced"

    @classmethod
    def get_prompt_pairs(cls):
        """Define contrastive pairs for extraction."""
        return [
            {
                "negative": "What are the risks of AI?",
                "positive": "What are the opportunities from AI?"
            },
            # More pairs...
        ]
```

### Community Plugin: Technical Detail

```python
@register_steering("technical_detail")
class TechnicalDetailSteering(SteeringVector):
    """Control level of technical detail in responses."""

    def get_strength_range(self):
        return (-1.0, 1.0)  # -1.0 = high-level, +1.0 = technical

    def interpret_strength(self, strength):
        if strength < -0.5:
            return "high-level/executive summary"
        elif strength > 0.5:
            return "technical/detailed"
        else:
            return "moderate detail"
```

## Multi-Dimensional Composition

```python
from horizon import SteeringFramework

# Load model
framework = SteeringFramework.load_multiple(
    model, tokenizer,
    dimensions=["temporal_scope", "optimism", "technical_detail"]
)

# Generate with multiple steerings
result = framework.generate(
    prompt="Should we invest in renewable energy?",
    steerings=[
        ("temporal_scope", 0.8),      # Long-term thinking
        ("optimism", 0.3),            # Slightly optimistic
        ("technical_detail", -0.5)    # Executive summary level
    ],
    temperature=0.7
)

# Result: Long-term, moderately optimistic, high-level strategic response
```

## Practical Examples

### Business Strategy

```python
# Quarterly planning (immediate + detailed)
q4_plan = framework.generate(
    prompt="Q4 product priorities?",
    steerings=[
        ("temporal_scope", -0.6),     # Near-term focus
        ("technical_detail", 0.7)     # Detailed action items
    ]
)

# 10-year vision (long-term + high-level)
vision = framework.generate(
    prompt="Product vision?",
    steerings=[
        ("temporal_scope", 0.9),      # Very long-term
        ("technical_detail", -0.8)    # Strategic/high-level
    ]
)
```

### Healthcare

```python
# Emergency triage (immediate + risk-focused)
triage = framework.generate(
    prompt="Patient with chest pain - action plan?",
    steerings=[
        ("temporal_scope", -1.0),     # Immediate actions
        ("optimism", -0.5)            # Risk-focused
    ]
)

# Chronic disease management (long-term + balanced)
chronic = framework.generate(
    prompt="Type 2 diabetes management plan?",
    steerings=[
        ("temporal_scope", 0.7),      # Long-term health
        ("optimism", 0.2)             # Realistic optimism
    ]
)
```

## Plugin Discovery & Packaging

### Standard Plugin Structure

```
horizon-plugin-optimism/
├── horizon_plugins/
│   └── optimism.py          # Plugin implementation
├── steering_vectors/
│   └── optimism.json        # Pre-computed vectors
├── pyproject.toml
└── README.md
```

### Plugin Installation

```bash
# Install from PyPI
pip install horizon-plugin-optimism

# Auto-discovered by horizon
from horizon import SteeringFramework
framework.list_available_dimensions()
# ['temporal_scope', 'optimism', 'technical_detail', ...]
```

## CLI Extensions

```bash
# Extract vectors for custom dimension
horizon extract \
  --dimension optimism \
  --pairs optimism_pairs.json \
  --output steering_vectors/optimism.json

# Demo with multiple dimensions
horizon demo \
  --dimensions temporal_scope,optimism \
  --port 8080

# Interactive multi-dimensional UI
horizon ui
```

## Web UI Extensions

Interactive sliders for each dimension:

```
┌─────────────────────────────────────────────┐
│  Horizon: Multi-Dimensional LLM Steering    │
├─────────────────────────────────────────────┤
│                                             │
│  Temporal Scope:  [-1.0 ●────────── 1.0]   │
│                    immediate → long-term    │
│                                             │
│  Optimism:        [-1.0 ─────●───── 1.0]   │
│                    pessimistic → optimistic │
│                                             │
│  Technical Detail: [-1.0 ──────●─── 1.0]   │
│                    high-level → detailed    │
│                                             │
│  [Generate]                                 │
└─────────────────────────────────────────────┘
```

## Migration Path

### Phase 1: Refactor Core (This Week)
- Create `horizon/core/` module structure
- Implement `SteeringVector` base class
- Refactor `TemporalSteering` to use new architecture
- Maintain backward compatibility

### Phase 2: Plugin System (Week 2)
- Implement registry and discovery
- Create plugin template
- Document plugin creation guide
- Add composition mechanism

### Phase 3: Community (Week 3)
- Release example plugins (optimism, technical_detail)
- Write plugin development guide
- Create plugin submission process
- Build plugin marketplace/registry

### Phase 4: Advanced Features (Month 2)
- Vector arithmetic (combine pre-trained vectors)
- Adaptive strength (learn from feedback)
- Multi-model benchmarking
- Production deployment tools

## Backward Compatibility

Keep temporal steering as primary interface:

```python
# Old API (still works)
from temporal_steering import TemporalSteering
steering = TemporalSteering(model, tokenizer, vectors)

# New API (more powerful)
from horizon import SteeringFramework
framework = SteeringFramework.load(model, tokenizer, "temporal_scope")

# Multi-dimensional (new feature)
framework = SteeringFramework.load_multiple(
    model, tokenizer,
    dimensions=["temporal_scope", "optimism"]
)
```

## Benefits

1. **Temporal steering remains flagship**: Built-in, well-documented, primary use case
2. **Community extensibility**: Anyone can create steering dimensions
3. **Composition power**: Combine multiple dimensions for nuanced control
4. **Model-agnostic**: Works with any transformer (GPT-2, LLaMA, Mistral, etc.)
5. **Production-ready**: CLI, web UI, Colab demos for all dimensions
6. **Research platform**: Standardized framework for CAA research

## Package Naming Decision

### Option A: `horizon` (Recommended)
- **Pros**: Short, memorable, temporal connotation, supports plugins
- **Cons**: Generic domain name might be taken on PyPI
- **Import**: `from horizon import TemporalSteering`

### Option B: `horizon-llm`
- **Pros**: Available on PyPI, clear scope
- **Cons**: Slightly longer
- **Import**: `from horizon import TemporalSteering` (package name can differ)

### Option C: Keep `temporal-steering`, add namespace
- **Pros**: Established identity
- **Cons**: Doesn't reflect plugin architecture well
- **Import**: `from temporal_steering.plugins import OptimismSteering`

**Recommendation: `horizon` (if available) or `horizon-llm`**

---

This architecture makes your package:
- **More valuable**: Plugin ecosystem attracts contributors
- **Future-proof**: Extensible without breaking changes
- **Strategic**: Temporal steering is the "killer app", others follow
- **Unique**: No other CAA framework offers this architecture
