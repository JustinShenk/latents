"""
Extract contrastive activation vectors for temporal steering using CAA.

This implements Contrastive Activation Addition (CAA) for temporal scope steering:
1. Extract activations from immediate vs long-term prompt pairs
2. Compute contrastive vectors (long_term - immediate) for each layer
3. Save steering vectors that can be added to model activations during generation
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
from tqdm import tqdm

from .model_adapter import get_model_config


def extract_activations_with_hooks(model, tokenizer, prompt, layer_idx=None):
    """
    Extract activations from specified layer using forward hooks.

    Works with any transformer model architecture.

    If layer_idx is None, extracts from all layers.
    Returns: dict mapping layer_idx -> activations (batch_size, seq_len, hidden_dim)
    """
    activations = {}

    def hook_fn(layer_num):
        def hook(module, input, output):
            # output[0] is the hidden states
            activations[layer_num] = output[0].detach()
        return hook

    # Get model layers (model-agnostic)
    model_config = get_model_config(model)
    layers = model_config['layers']

    # Register hooks
    hooks = []
    if layer_idx is not None:
        hook = layers[layer_idx].register_forward_hook(hook_fn(layer_idx))
        hooks.append(hook)
    else:
        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(hook_fn(i))
            hooks.append(hook)

    # Forward pass
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        model(**inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations


def compute_steering_vectors(model, tokenizer, prompt_pairs, layers=None):
    """
    Compute contrastive steering vectors from prompt pairs.

    Works with any transformer model (GPT-2, LLaMA, Mistral, etc.)

    Args:
        model: HuggingFace transformer model
        tokenizer: Corresponding tokenizer
        prompt_pairs: List of dicts with 'immediate_prompt' and 'long_term_prompt'
        layers: List of layer indices to extract (default: all layers)

    Returns:
        dict mapping layer_idx -> steering_vector (hidden_dim,)
    """
    # Auto-detect model architecture
    model_config = get_model_config(model)

    if layers is None:
        layers = list(range(model_config['n_layers']))

    # Accumulate contrastive activations for each layer
    layer_contrasts = {layer: [] for layer in layers}

    print(f"Computing steering vectors from {len(prompt_pairs)} prompt pairs...")

    for pair in tqdm(prompt_pairs):
        immediate = pair['immediate_prompt']
        long_term = pair['long_term_prompt']

        # Extract activations for both prompts
        immediate_acts = extract_activations_with_hooks(model, tokenizer, immediate)
        long_term_acts = extract_activations_with_hooks(model, tokenizer, long_term)

        # Compute contrastive vectors at each layer
        for layer in layers:
            # Take final token position for both
            imm_vec = immediate_acts[layer][0, -1, :]  # (hidden_dim,)
            long_vec = long_term_acts[layer][0, -1, :]  # (hidden_dim,)

            # Contrastive vector: long_term - immediate
            contrast = (long_vec - imm_vec).cpu().numpy()
            layer_contrasts[layer].append(contrast)

    # Average contrastive vectors across all pairs
    steering_vectors = {}
    for layer in layers:
        contrasts = np.stack(layer_contrasts[layer])  # (n_pairs, hidden_dim)
        steering_vectors[layer] = contrasts.mean(axis=0)  # (hidden_dim,)

    return steering_vectors


def load_prompt_pairs(pairs_file):
    """Load prompt pairs from JSON file."""
    with open(pairs_file, 'r') as f:
        data = json.load(f)

    # Convert to consistent format
    pairs = []
    for item in data:
        if 'immediate_prompt' in item and 'long_term_prompt' in item:
            pairs.append(item)
        elif 'short_prompt' in item and 'long_prompt' in item:
            # Convert from dataset format
            pairs.append({
                'immediate_prompt': item['short_prompt'],
                'long_term_prompt': item['long_prompt'],
                'topic': item.get('domain', item.get('topic', 'unknown'))
            })

    return pairs


def save_steering_vectors(steering_vectors, output_file):
    """Save steering vectors to file."""
    # Convert to serializable format
    data = {
        'layer_vectors': {
            str(layer): vec.tolist()
            for layer, vec in steering_vectors.items()
        },
        'metadata': {
            'n_layers': len(steering_vectors),
            'hidden_dim': len(next(iter(steering_vectors.values()))),
            'direction': 'long_term - immediate'
        }
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Saved steering vectors to {output_file}")


def analyze_steering_vectors(steering_vectors):
    """Analyze properties of steering vectors."""
    print("\n" + "="*70)
    print("STEERING VECTOR ANALYSIS")
    print("="*70)

    for layer, vec in sorted(steering_vectors.items()):
        norm = np.linalg.norm(vec)
        mean = vec.mean()
        std = vec.std()

        print(f"Layer {layer:2d}: norm={norm:.3f}, mean={mean:.6f}, std={std:.3f}")

    # Find layers with strongest steering effect
    norms = {layer: np.linalg.norm(vec) for layer, vec in steering_vectors.items()}
    sorted_layers = sorted(norms.items(), key=lambda x: x[1], reverse=True)

    print("\nStrongest steering layers (by norm):")
    for layer, norm in sorted_layers[:5]:
        print(f"  Layer {layer}: {norm:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract temporal steering vectors using CAA")
    parser.add_argument('--pairs', type=str, required=True, help='Path to prompt pairs JSON')
    parser.add_argument('--output', type=str, default='steering_vectors/temporal_steering.json',
                       help='Output file for steering vectors')
    parser.add_argument('--model', type=str, default='gpt2', help='Model name (gpt2, gpt2-medium, etc.)')
    parser.add_argument('--layers', type=str, default=None,
                       help='Comma-separated layer indices (default: all layers)')
    parser.add_argument('--max-pairs', type=int, default=None,
                       help='Maximum number of pairs to use (default: all)')

    args = parser.parse_args()

    print("="*70)
    print("TEMPORAL STEERING VECTOR EXTRACTION")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Prompt pairs: {args.pairs}")
    print(f"Output: {args.output}")
    print("="*70)
    print()

    # Load model and tokenizer
    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained(args.model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Parse layer indices
    if args.layers:
        layers = [int(x) for x in args.layers.split(',')]
    else:
        layers = None  # Use all layers

    # Load prompt pairs
    print(f"Loading prompt pairs from {args.pairs}...")
    pairs = load_prompt_pairs(args.pairs)

    if args.max_pairs:
        pairs = pairs[:args.max_pairs]

    print(f"Loaded {len(pairs)} prompt pairs")

    # Compute steering vectors
    steering_vectors = compute_steering_vectors(model, tokenizer, pairs, layers)

    # Analyze
    analyze_steering_vectors(steering_vectors)

    # Save
    save_steering_vectors(steering_vectors, args.output)

    print("\n✓ Complete!")
