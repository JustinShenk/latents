# Latents: Multi-Dimensional LLM Steering

A flexible framework for steering language models along multiple behavioral dimensions using Contrastive Activation Addition (CAA).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JustinShenk/latents/blob/main/temporal_steering_demo.ipynb)

## Features

- **Plugin Architecture**: Extensible framework for custom steering dimensions
- **Multi-Model Support**: Works with GPT-2, LLaMA, Mistral, Falcon, and more
- **Temporal Scope Steering**: Flagship dimension (immediate ↔ long-term thinking)
- **Multi-Dimensional Composition**: Combine multiple steering dimensions simultaneously
- **Research Tools**: PCA analysis, confound testing, human evaluation utilities
- **Interactive Demos**: Web UI and Jupyter notebooks

## 🚀 Quick Start

### Installation

```bash
# From GitHub
pip install git+https://github.com/JustinShenk/latents.git

# Or clone for development
git clone https://github.com/JustinShenk/latents.git
cd latents
pip install -e .
```

### Basic Usage

```python
from temporal_steering import SteeringFramework
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Load steering framework with temporal dimension
framework = SteeringFramework.load(model, tokenizer, 'temporal_scope')

# Generate with long-term steering
result = framework.generate(
    prompt="How should we address climate change?",
    steerings=[('temporal_scope', 0.8)],  # +1.0 = long-term, -1.0 = immediate
    temperature=0.7,
    max_length=100
)
print(result)
```

### Multi-Dimensional Steering

```python
# Combine multiple dimensions
result = framework.generate(
    prompt="How should we address climate change?",
    steerings=[
        ('temporal_scope', 0.8),    # Long-term thinking
        ('formality', 0.5),         # Moderately formal (if available)
    ],
    temperature=0.7
)
```

## 📦 What's Inside

### Core Components

- **`temporal_steering/core/`**: Plugin architecture and framework
- **`temporal_steering/dimensions/`**: Built-in steering dimensions
- **`temporal_steering/model_adapter.py`**: Multi-model support layer
- **`temporal_steering/extract_steering_vectors.py`**: Vector extraction utilities

### Pre-trained Steering Vectors

Located in `steering_vectors/`:
- `temporal_scope.json` - Original temporal steering
- `temporal_scope_gpt2.json` - GPT-2 specific vectors
- `temporal_scope_deconfounded.json` - Style-controlled vectors

### Research & Experiments

Located in `research/`:
- **`experiments/`**: Confound analysis and validation studies
- **`datasets/`**: Experimental prompts and control datasets
- **`results/`**: Analysis outputs, PCA plots, evaluation metrics
- **`tools/`**: Analysis utilities (activation extraction, probe training)

## 🔌 Plugin System

Create custom steering dimensions:

```python
from temporal_steering.core import SteeringVector, register_steering
from typing import Tuple

@register_steering("optimism")
class OptimismSteering(SteeringVector):
    def get_dimension_name(self) -> str:
        return "optimism"

    def get_strength_range(self) -> Tuple[float, float]:
        return (-1.0, 1.0)

    def interpret_strength(self, strength: float) -> str:
        if strength <= -0.75:
            return "very pessimistic/defensive"
        elif strength <= -0.25:
            return "cautiously pessimistic"
        elif strength <= 0.25:
            return "balanced/realistic"
        elif strength <= 0.75:
            return "cautiously optimistic"
        else:
            return "very optimistic/ambitious"
```

See [PLUGIN_GUIDE.md](PLUGIN_GUIDE.md) for detailed instructions.

## 🧪 Extracting Steering Vectors

Extract vectors from your own prompt pairs:

```python
from temporal_steering.extract_steering_vectors import (
    compute_steering_vectors,
    save_steering_vectors
)
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Prompt pairs: [{"positive": "...", "negative": "..."}]
prompt_pairs = [
    {
        "positive": "Considering long-term consequences over the next 50 years...",
        "negative": "We need immediate action within the next few days..."
    },
    # ... more pairs
]

# Extract vectors
steering_vectors = compute_steering_vectors(
    model,
    tokenizer,
    prompt_pairs,
    layers=None  # Extract from all layers
)

# Save for later use
save_steering_vectors(steering_vectors, 'my_steering.json')
```

## 🎯 How It Works

**Contrastive Activation Addition (CAA)**:

1. Extract activations from contrastive prompt pairs (e.g., immediate vs. long-term)
2. Compute steering vectors: `positive_activations - negative_activations`
3. During generation, add scaled steering vectors to model activations

The plugin architecture allows you to:
- Define custom behavioral dimensions
- Combine multiple dimensions
- Control interpretation of steering strengths
- Support any transformer architecture

## 🔬 For Researchers

### Getting Started

See [RESEARCHER_ONBOARDING.md](RESEARCHER_ONBOARDING.md) for:
- 10-minute quick start guide
- Running confound analysis experiments
- Contributing new experiments
- Dataset expansion guidelines
- Code review checklist

### Current Research Focus

**Issue #1**: Verify temporal steering not confounded by style ([research/ISSUES.md](research/ISSUES.md))
- Preliminary PCA results show separation
- Needs: larger dataset, behavioral validation, multi-model testing

### Running Experiments

```bash
# Confound analysis (2×2: temporal × style)
python research/experiments/confound_analysis.py

# View interactive results
open research/results/pca_temporal_style_interactive.html
```

## 📚 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Plugin system design
- **[PLUGIN_GUIDE.md](PLUGIN_GUIDE.md)**: Create custom dimensions
- **[SCALING_AND_USAGE.md](SCALING_AND_USAGE.md)**: Multi-model support
- **[PACKAGE_NAMING_STRATEGY.md](PACKAGE_NAMING_STRATEGY.md)**: Naming decisions
- **[research/ISSUES.md](research/ISSUES.md)**: Open research questions

## 🚀 Multi-Model Support

Works with any transformer architecture:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# LLaMA example
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

framework = SteeringFramework.load(model, tokenizer, 'temporal_scope')
# Works the same way!
```

Supported architectures:
- GPT-2, GPT-Neo, GPT-J
- LLaMA, Llama-2, Llama-3
- Mistral, Mixtral
- Falcon, BLOOM, OPT
- And more...

## 🌐 Interactive Demo

Launch the web interface:

```bash
python temporal_steering/temporal_steering_demo.py \
  --steering steering_vectors/temporal_steering.json \
  --port 8080
```

Or try the [Colab notebook](https://colab.research.google.com/github/JustinShenk/latents/blob/main/temporal_steering_demo.ipynb).

## 📊 Example Results

**Prompt**: "What should policymakers prioritize to address climate change?"

**Immediate Steering (-0.8)**:
> "Implement immediate carbon pricing, ban single-use plastics now, emergency subsidies for renewable energy this quarter..."

**Long-term Steering (+0.8)**:
> "Establish institutional frameworks for intergenerational equity, invest in fundamental research for breakthrough technologies, create policy mechanisms that adapt over decades..."

## 🛣️ Roadmap

- [ ] PyPI package release
- [ ] Additional pre-trained dimensions (formality, technicality, abstractness)
- [ ] Multi-model benchmark suite
- [ ] Automated confound detection
- [ ] Integration with steering-vectors ecosystem

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@software{latents2025,
  author = {Shenk, Justin},
  title = {Latents: Multi-Dimensional LLM Steering},
  year = {2025},
  url = {https://github.com/JustinShenk/latents}
}
```

## 🙏 Acknowledgments

- Based on [Contrastive Activation Addition](https://www.alignmentforum.org/posts/v7f8ayBxLhmMFRzpa/steering-llama-2-with-contrastive-activation-additions) by Nina Rimsky
- Inspired by [nrimsky/CAA](https://github.com/nrimsky/CAA)

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

## 🤝 Contributing

We welcome contributions! See [RESEARCHER_ONBOARDING.md](RESEARCHER_ONBOARDING.md) for:
- Setting up your development environment
- Running experiments
- Code style guidelines
- Pull request process

For research questions, see [research/ISSUES.md](research/ISSUES.md).

## 📬 Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/JustinShenk/latents/issues)
- **Maintainer**: [@JustinShenk](https://github.com/JustinShenk)
