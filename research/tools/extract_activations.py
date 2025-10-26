"""Extract activations from GPT-2 for temporal grounding analysis."""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from tqdm import tqdm
import json
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.tools.utils import sync_to_bucket


def extract_activations_for_prompts(prompts_file, output_file, model_name="gpt2-small"):
    """
    Extract activations from all layers for given prompts.

    Args:
        prompts_file: JSON file with prompt pairs
        output_file: Where to save activations (.npz)
        model_name: Model to use
    """

    print(f"{'='*60}")
    print(f"ACTIVATION EXTRACTION")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Input: {prompts_file}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")

    # Load model
    print(f"Loading {model_name}...")
    model = HookedTransformer.from_pretrained(model_name)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("✓ Using GPU")
    else:
        print("⚠ Using CPU (this will be slow)")

    # Load prompts
    with open(prompts_file, 'r') as f:
        prompt_pairs = json.load(f)

    print(f"✓ Loaded {len(prompt_pairs)} prompt pairs\n")

    # Storage
    all_activations = []
    all_labels = []
    all_metadata = []

    print(f"Extracting activations...")

    for pair in tqdm(prompt_pairs, desc="Processing pairs"):
        for temporal_type in ['short', 'long']:
            prompt = pair[f'{temporal_type}_prompt']

            # Tokenize
            tokens = model.to_tokens(prompt)

            # Run with cache
            with torch.no_grad():
                logits, cache = model.run_with_cache(tokens)

            # Extract residual stream at last token position for each layer
            layer_activations = {}

            for layer in range(model.cfg.n_layers):
                # Get residual stream after this layer
                resid = cache[f"blocks.{layer}.hook_resid_post"]
                # Take last token position
                layer_activations[f"layer_{layer}"] = resid[0, -1, :].cpu().numpy()

            all_activations.append(layer_activations)
            all_labels.append(0 if temporal_type == 'short' else 1)
            all_metadata.append({
                "id": pair['id'],
                "domain": pair['domain'],
                "temporal_type": temporal_type,
                "horizon": pair[f'{temporal_type}_horizon']
            })

    print("\n✓ Extraction complete")

    # Convert to arrays by layer
    print("Organizing data by layer...")
    data_to_save = {
        'labels': np.array(all_labels),
        'metadata': all_metadata
    }

    # Organize by layer
    for layer in range(model.cfg.n_layers):
        layer_data = np.array([act[f"layer_{layer}"] for act in all_activations])
        data_to_save[f'layer_{layer}'] = layer_data

    print(f"  Shape per layer: {layer_data.shape}")
    print(f"  Total samples: {len(all_labels)}")
    labels_array = np.array(all_labels)
    print(f"  Short-term: {(labels_array == 0).sum()}, Long-term: {(labels_array == 1).sum()}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save
    print(f"\nSaving to {output_file}...")
    np.savez_compressed(output_file, **data_to_save)
    print(f"✓ Saved activations")

    # Sync to bucket
    print("\nSyncing to GCS bucket...")
    sync_to_bucket(output_file)

    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")

    return data_to_save


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', type=str, required=True,
                       help='Path to prompts JSON file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output NPZ file')
    parser.add_argument('--model', type=str, default='gpt2-small',
                       help='Model name (default: gpt2-small)')

    args = parser.parse_args()

    # Change to project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    extract_activations_for_prompts(args.prompts, args.output, args.model)
