#!/usr/bin/env python3
"""
Extract steering vectors for all built-in dimensions.

This script extracts activation vectors for formality, technicality, and abstractness
using prompt pairs from research/datasets/.

Usage:
    python scripts/extract_all_dimensions.py --model gpt2 --layers 6-11
    python scripts/extract_all_dimensions.py --model meta-llama/Llama-2-7b-hf --layers 10-20
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from latents.extract_steering_vectors import compute_steering_vectors, save_steering_vectors


def parse_layer_range(layer_str):
    """Parse layer string like '6-11' into list [6,7,8,9,10,11]."""
    if '-' in layer_str:
        start, end = map(int, layer_str.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(layer_str)]


def load_pairs_from_dataset(dataset_path):
    """Load contrastive pairs from dataset JSON file."""
    with open(dataset_path) as f:
        data = json.load(f)

    pairs = []
    for pair in data['pairs']:
        # Convert to format expected by compute_steering_vectors
        # (uses 'immediate_prompt' and 'long_term_prompt' keys internally)
        if 'casual' in pair and 'formal' in pair:
            # Formality: casual=negative, formal=positive
            pairs.append({
                'immediate_prompt': pair['casual'],
                'long_term_prompt': pair['formal']
            })
        elif 'simple' in pair and 'technical' in pair:
            # Technicality: simple=negative, technical=positive
            pairs.append({
                'immediate_prompt': pair['simple'],
                'long_term_prompt': pair['technical']
            })
        elif 'concrete' in pair and 'abstract' in pair:
            # Abstractness: concrete=negative, abstract=positive
            pairs.append({
                'immediate_prompt': pair['concrete'],
                'long_term_prompt': pair['abstract']
            })
        else:
            raise ValueError(f"Unknown pair format: {pair.keys()}")

    return pairs, data['metadata']


def extract_dimension(model, tokenizer, dimension_name, dataset_path, output_path, layers):
    """Extract vectors for a single dimension."""
    print(f"\n{'='*60}")
    print(f"Extracting: {dimension_name}")
    print(f"{'='*60}")

    # Load pairs from dataset
    pairs, metadata = load_pairs_from_dataset(dataset_path)
    print(f"Dataset: {dataset_path}")
    print(f"Pairs: {len(pairs)}")
    print(f"Contrast: {metadata.get('contrast', 'N/A')}")

    # Extract vectors
    steering_vectors = compute_steering_vectors(
        model=model,
        tokenizer=tokenizer,
        prompt_pairs=pairs,
        layers=layers
    )

    # Save
    save_steering_vectors(steering_vectors, output_path)
    print(f"✓ Saved to: {output_path}")

    return steering_vectors


def main():
    parser = argparse.ArgumentParser(description="Extract steering vectors for all dimensions")
    parser.add_argument(
        '--model',
        type=str,
        default='gpt2',
        help='HuggingFace model name (default: gpt2)'
    )
    parser.add_argument(
        '--layers',
        type=str,
        default='6-11',
        help='Layer range to extract, e.g., "6-11" or "10" (default: 6-11 for GPT-2)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='steering_vectors',
        help='Output directory for steering vectors (default: steering_vectors/)'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='research/datasets',
        help='Directory containing dataset files (default: research/datasets/)'
    )
    parser.add_argument(
        '--dimensions',
        type=str,
        nargs='+',
        default=['formality', 'technicality', 'abstractness'],
        help='Dimensions to extract (default: all three new dimensions)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use: cpu, cuda, mps (default: cpu)'
    )

    args = parser.parse_args()

    # Parse layers
    layers = parse_layer_range(args.layers)
    print(f"Extracting from layers: {layers}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    dataset_dir = Path(args.dataset_dir)

    # Load model
    print(f"\nLoading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Move to device
    if args.device != 'cpu':
        model = model.to(args.device)

    model.eval()

    print(f"Model loaded: {model.config.model_type}")
    print(f"Total layers: {model.config.n_layer if hasattr(model.config, 'n_layer') else model.config.num_hidden_layers}")

    # Extract vectors for each dimension
    results = {}
    for dim_name in args.dimensions:
        dataset_path = dataset_dir / f"{dim_name}_pairs.json"

        if not dataset_path.exists():
            print(f"⚠️  Dataset not found: {dataset_path}, skipping")
            continue

        output_path = output_dir / f"{dim_name}.json"

        try:
            vectors = extract_dimension(
                model=model,
                tokenizer=tokenizer,
                dimension_name=dim_name,
                dataset_path=str(dataset_path),
                output_path=str(output_path),
                layers=layers
            )
            results[dim_name] = output_path
        except Exception as e:
            print(f"✗ Failed to extract {dim_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Extracted {len(results)}/{len(args.dimensions)} dimensions:")
    for dim_name, path in results.items():
        print(f"  ✓ {dim_name}: {path}")

    print("\nUsage example:")
    print(f"""
from latents import SteeringFramework

framework = SteeringFramework.load(model, tokenizer, 'formality')
result = framework.generate(
    prompt="How does machine learning work?",
    steerings=[('formality', 0.8)],  # Formal language
    max_length=100
)
    """)


if __name__ == '__main__':
    main()
