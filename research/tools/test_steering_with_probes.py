"""
Test temporal steering against trained probes.

This script validates that temporal steering actually changes activations
in a way that trained temporal probes can detect. It's an "adversarial" test
to ensure steering works on the same temporal features probes were trained on.

Workflow:
1. Load trained temporal probes (from Phase 0)
2. Extract steering vectors from prompt pairs
3. Generate text at different steering strengths (-1.0 to +1.0)
4. Extract activations from steered generations
5. Run probes on those activations
6. Analyze correlation: Does probe prediction follow steering strength?

Success criteria:
- Strong negative steering → Probe predicts short-term (class 0)
- Strong positive steering → Probe predicts long-term (class 1)
- Correlation coefficient r > 0.7
"""

import torch
import pickle
import numpy as np
import json
import os
import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_probes(probes_dir='probes', prefix='proper'):
    """Load all trained temporal probes."""
    probes = {}
    for layer in range(12):
        probe_path = os.path.join(probes_dir, f'{prefix}_layer_{layer}_probe.pkl')
        if os.path.exists(probe_path):
            with open(probe_path, 'rb') as f:
                probes[layer] = pickle.load(f)
    return probes


def extract_steering_vectors_simple(model, tokenizer, prompt_pairs, max_pairs=10):
    """
    Extract steering vectors from contrastive prompt pairs.

    Returns: dict of {layer_idx: steering_vector}
    """
    steering_vectors = {i: [] for i in range(12)}

    for i, pair in enumerate(prompt_pairs[:max_pairs]):
        short_prompt = pair['short_prompt']
        long_prompt = pair['long_prompt']

        # Extract activations for both prompts
        short_acts = extract_activations(model, tokenizer, short_prompt)
        long_acts = extract_activations(model, tokenizer, long_prompt)

        # Compute contrastive vector at each layer
        for layer in range(12):
            contrast = long_acts[layer] - short_acts[layer]
            steering_vectors[layer].append(contrast)

    # Average across pairs
    for layer in range(12):
        steering_vectors[layer] = np.mean(steering_vectors[layer], axis=0)

    return steering_vectors


def extract_activations(model, tokenizer, prompt):
    """Extract activations from all layers for a given prompt."""
    inputs = tokenizer(prompt, return_tensors='pt')

    activations = {}

    def hook_fn(layer_idx):
        def hook(module, input, output):
            # output[0] is hidden_states: (batch, seq_len, hidden_dim)
            # Take last token activation
            activations[layer_idx] = output[0][0, -1, :].detach().cpu().numpy()
        return hook

    # Register hooks
    hooks = []
    for i, layer in enumerate(model.transformer.h):
        hook = layer.register_forward_hook(hook_fn(i))
        hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        model(**inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations


def generate_with_steering(model, tokenizer, prompt, steering_vectors,
                           strength=0.0, target_layers=None, max_length=50):
    """
    Generate text with temporal steering applied.

    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        prompt: Input prompt
        steering_vectors: Dict of {layer_idx: vector}
        strength: Steering strength (-1.0 to +1.0)
        target_layers: Which layers to apply steering (default: middle-late)
        max_length: Max tokens to generate

    Returns:
        generated_text: String
    """
    if target_layers is None:
        # Default: middle-to-late layers (4-11)
        target_layers = list(range(4, 12))

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors='pt')

    # Store activations during generation
    final_activations = {}

    def make_steering_hook(layer_idx):
        def hook(module, input, output):
            hidden_states = output[0]

            # Apply steering
            if layer_idx in target_layers:
                steering_vec = torch.tensor(
                    steering_vectors[layer_idx],
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )
                hidden_states = hidden_states + strength * steering_vec

            # Store final token activation for probe testing
            final_activations[layer_idx] = hidden_states[0, -1, :].detach().cpu().numpy()

            return (hidden_states,) + output[1:]
        return hook

    # Register hooks
    hooks = []
    for i, layer in enumerate(model.transformer.h):
        hook = layer.register_forward_hook(make_steering_hook(i))
        hooks.append(hook)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + max_length,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )

    # Remove hooks
    for hook in hooks:
        hook.remove()

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_text, final_activations


def run_probe_test(test_prompts, steering_strengths=None):
    """
    Main test function.

    Tests whether temporal steering changes activations in a way that
    trained temporal probes can detect.
    """
    if steering_strengths is None:
        steering_strengths = [-1.0, -0.5, 0.0, 0.5, 1.0]

    print("="*70)
    print("ADVERSARIAL PROBE TEST: Temporal Steering vs Trained Probes")
    print("="*70)
    print(f"\nObjective: Validate that temporal steering changes activations")
    print(f"in ways that trained temporal probes (78% acc) can detect.\n")
    print(f"Testing {len(test_prompts)} prompts at {len(steering_strengths)} strengths")
    print(f"Strengths: {steering_strengths}\n")

    # Load model and tokenizer
    print("Loading GPT-2 model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    print("✓ Model loaded\n")

    # Load trained probes
    print("Loading trained temporal probes...")
    probes = load_probes()
    print(f"✓ Loaded {len(probes)} probes (one per layer)\n")

    # Load or extract steering vectors
    steering_file = 'steering_vectors/temporal_steering.json'

    if os.path.exists(steering_file):
        print(f"Loading steering vectors from {steering_file}...")
        with open(steering_file, 'r') as f:
            steering_data = json.load(f)
            steering_vectors = {int(k): np.array(v) for k, v in steering_data['vectors'].items()}
        print("✓ Steering vectors loaded\n")
    else:
        print("Steering vectors not found. Extracting from prompt pairs...")
        # Load prompt pairs
        with open('data_download/test_prompts.json', 'r') as f:
            prompt_pairs = json.load(f)

        steering_vectors = extract_steering_vectors_simple(
            model, tokenizer, prompt_pairs, max_pairs=20
        )

        # Save for future use
        os.makedirs('steering_vectors', exist_ok=True)
        with open(steering_file, 'w') as f:
            json.dump({
                'vectors': {k: v.tolist() for k, v in steering_vectors.items()},
                'metadata': {'num_pairs': 20, 'model': 'gpt2'}
            }, f)
        print(f"✓ Steering vectors extracted and saved to {steering_file}\n")

    # Run test
    print("="*70)
    print("RUNNING TEST")
    print("="*70)

    results = []

    for prompt_idx, prompt in enumerate(test_prompts):
        print(f"\nPrompt {prompt_idx + 1}/{len(test_prompts)}: {prompt[:60]}...")

        for strength in steering_strengths:
            # Generate with steering
            generated_text, activations = generate_with_steering(
                model, tokenizer, prompt, steering_vectors,
                strength=strength, max_length=30
            )

            # Run probes on activations
            probe_predictions = {}
            probe_probabilities = {}

            for layer, probe in probes.items():
                X = activations[layer].reshape(1, -1)
                pred = probe.predict(X)[0]
                prob = probe.predict_proba(X)[0, 1]  # P(long-term)

                probe_predictions[layer] = pred
                probe_probabilities[layer] = prob

            # Store results
            results.append({
                'prompt': prompt,
                'strength': strength,
                'generated_text': generated_text,
                'probe_predictions': probe_predictions,
                'probe_probabilities': probe_probabilities
            })

            # Best layer (Layer 8 from Phase 0)
            layer_8_prob = probe_probabilities[8]
            layer_8_pred = "LONG" if probe_predictions[8] == 1 else "SHORT"

            print(f"  Strength {strength:+.1f} → Layer 8 probe: {layer_8_pred} (P={layer_8_prob:.2f})")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    # For each layer, compute correlation between steering strength and probe probability
    layer_correlations = {}

    for layer in probes.keys():
        strengths = [r['strength'] for r in results]
        probs = [r['probe_probabilities'][layer] for r in results]

        corr, p_value = spearmanr(strengths, probs)
        layer_correlations[layer] = {'correlation': corr, 'p_value': p_value}

    print("\nCorrelation between steering strength and probe predictions:")
    print(f"{'Layer':<8} {'Correlation':<15} {'P-value':<15} {'Result':<10}")
    print("-"*70)

    for layer in sorted(layer_correlations.keys()):
        corr = layer_correlations[layer]['correlation']
        p_val = layer_correlations[layer]['p_value']

        if corr > 0.7:
            result = "✓ STRONG"
        elif corr > 0.5:
            result = "○ MODERATE"
        else:
            result = "✗ WEAK"

        print(f"{layer:<8} {corr:>+.3f}{'':>10} {p_val:>.4f}{'':>10} {result}")

    # Best performing layer
    best_layer = max(layer_correlations.keys(),
                     key=lambda k: layer_correlations[k]['correlation'])
    best_corr = layer_correlations[best_layer]['correlation']

    print(f"\n{'='*70}")
    print(f"BEST LAYER: {best_layer}")
    print(f"{'='*70}")
    print(f"Correlation: {best_corr:+.3f}")
    print(f"P-value: {layer_correlations[best_layer]['p_value']:.4f}")

    # Overall conclusion
    print(f"\n{'='*70}")
    print(f"CONCLUSION")
    print(f"{'='*70}")

    if best_corr > 0.7:
        print("✓ SUCCESS: Strong correlation detected!")
        print("  Temporal steering effectively changes activations in ways that")
        print("  trained temporal probes can detect. The steering is working on")
        print("  the same temporal features that probes were trained to recognize.")
    elif best_corr > 0.5:
        print("○ PARTIAL SUCCESS: Moderate correlation detected.")
        print("  Steering has some effect on temporal features, but not as strong")
        print("  as expected. Consider:")
        print("    - Using more diverse steering prompt pairs")
        print("    - Targeting specific layers (e.g., Layer 8)")
        print("    - Increasing steering strength")
    else:
        print("✗ FAILURE: Weak or no correlation detected.")
        print("  Steering may not be changing temporal features effectively.")
        print("  Possible issues:")
        print("    - Steering vectors may be capturing non-temporal features")
        print("    - Wrong layers being targeted")
        print("    - Insufficient steering strength")

    print(f"{'='*70}\n")

    # Save results
    results_file = 'results/adversarial_probe_test.json'
    os.makedirs('results', exist_ok=True)

    with open(results_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = []
        for r in results:
            serializable_results.append({
                'prompt': r['prompt'],
                'strength': float(r['strength']),
                'generated_text': r['generated_text'],
                'probe_predictions': {int(k): int(v) for k, v in r['probe_predictions'].items()},
                'probe_probabilities': {int(k): float(v) for k, v in r['probe_probabilities'].items()}
            })

        json.dump({
            'results': serializable_results,
            'correlations': {int(k): {'correlation': float(v['correlation']),
                                      'p_value': float(v['p_value'])}
                            for k, v in layer_correlations.items()},
            'best_layer': int(best_layer),
            'best_correlation': float(best_corr)
        }, f, indent=2)

    print(f"✓ Results saved to {results_file}")

    return results, layer_correlations


if __name__ == "__main__":
    # Test prompts (neutral prompts about policy/planning/decisions)
    test_prompts = [
        "What should we do about climate change?",
        "How should we approach renewable energy?",
        "What's the best strategy for economic growth?",
        "How should we invest in infrastructure?",
        "What should we prioritize in education reform?",
        "How should we address healthcare costs?",
        "What's the right approach to urban development?",
        "How should we manage natural resources?",
    ]

    results, correlations = run_probe_test(test_prompts)
