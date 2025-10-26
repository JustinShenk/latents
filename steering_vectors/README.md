# Pre-computed Steering Vectors

This directory contains pre-trained steering vectors for controlling temporal scope in language models.

## 📁 Files

- `temporal_steering.json` - GPT-2 temporal steering vectors (250KB)

## 📊 Vector Details

| Property | Value |
|----------|-------|
| **Model** | GPT-2 (124M parameters) |
| **Generated** | 2024-10-26 |
| **Training Data** | 20 prompt pairs from `data_download/test_prompts.json` |
| **Method** | Contrastive Activation Addition (CAA) |
| **Direction** | `long_term_activations - immediate_activations` |
| **Layers** | All 12 GPT-2 layers |
| **Recommended Layers** | 4-11 (strongest effects in layers 9-11) |

## 🔄 How to Reproduce

The vectors in this directory were generated using code **included in this package**.

### Method 1: Using CLI (Easiest)

```bash
# Install package
pip install -e .

# Extract vectors (recreates temporal_steering.json)
temporal-steering extract \
  --pairs data_download/test_prompts.json \
  --output steering_vectors/temporal_steering_new.json \
  --model gpt2 \
  --max-pairs 20
```

### Method 2: Using Python Script

```bash
python latents/extract_steering_vectors.py \
  --pairs data_download/test_prompts.json \
  --output steering_vectors/temporal_steering_new.json \
  --model gpt2 \
  --max-pairs 20
```

### Method 3: Reproduction Script

```bash
# Run the exact command that generated these vectors
bash scripts/reproduce_vectors.sh
```

## 🧪 Extract Your Own Vectors

You can create custom steering vectors from your own prompt pairs:

```bash
# Create your prompt pairs JSON (format below)
# Then extract vectors:
temporal-steering extract \
  --pairs your_prompts.json \
  --output your_vectors.json \
  --model gpt2 \
  --max-pairs 50
```

**Prompt pair format**:
```json
[
  {
    "immediate_prompt": "Develop a 1 week plan to improve productivity.",
    "long_term_prompt": "Develop a 20 year plan to improve productivity."
  }
]
```

Or use the alternate format (auto-converted):
```json
[
  {
    "short_prompt": "Develop a 1 week plan to...",
    "long_prompt": "Develop a 20 year plan to..."
  }
]
```

## 📈 Vector Strength by Layer

The steering effect varies by layer. Here are the norms (strength) of each layer's vector:

| Layer | Norm | Effect |
|-------|------|--------|
| 11 | 16.74 | **Strongest** |
| 10 | 11.83 | Very Strong |
| 9 | 8.60 | Strong |
| 8 | 6.29 | Moderate |
| 7 | 4.31 | Moderate |
| 6 | 2.86 | Weak |
| 5 | 1.50 | Weak |
| 0-4 | <1.5 | Very Weak |

**Recommendation**: Use layers 4-11 for steering (automatically selected by `TemporalSteering` class).

## 🔍 Inspect Vector Metadata

```python
import json

with open('steering_vectors/temporal_steering.json') as f:
    data = json.load(f)

print(data['metadata'])
# Shows: generation date, model, layers, extraction command, etc.
```

## 🧬 Code Used for Generation

The exact code used to generate these vectors is in this repository:

- **File**: `latents/extract_steering_vectors.py`
- **Commit**: Initial release (check git log for exact hash)
- **Command**: See `metadata.generation_info.command` in the JSON file

## ✅ Verification

To verify vectors match the expected format:

```python
import json
import numpy as np

with open('steering_vectors/temporal_steering.json') as f:
    data = json.load(f)

# Check structure
assert 'layer_vectors' in data
assert 'metadata' in data
assert len(data['layer_vectors']) == 12  # All GPT-2 layers

# Check each vector
for layer, vec in data['layer_vectors'].items():
    vec_array = np.array(vec)
    assert vec_array.shape == (768,)  # GPT-2 hidden dim
    print(f"Layer {layer}: shape {vec_array.shape}, norm {np.linalg.norm(vec_array):.3f}")

print("✓ Vectors verified!")
```

## 📚 References

- **Paper**: [Representation Engineering (Zou et al., 2023)](https://arxiv.org/abs/2310.01405)
- **Method**: Contrastive Activation Addition (CAA)
- **Original Library**: [steering-vectors](https://github.com/steering-vectors/steering-vectors)

## 🤝 Contributing

To contribute new steering vectors:

1. Extract vectors using your own prompt pairs
2. Verify they work with the demo
3. Add metadata (model, date, method, source)
4. Submit PR with description of what temporal dimension they capture

---

**Questions?** Open an issue on GitHub or see the main README.md
