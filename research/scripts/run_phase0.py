#!/usr/bin/env python
"""Master script to run Phase 0: Sanity Check."""

import os
import sys
import subprocess


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        print(f"\n⚠ ERROR: {description} failed!")
        sys.exit(1)

    print(f"\n✓ {description} complete\n")


def main():
    """Run Phase 0 pipeline."""

    print(f"\n{'='*60}")
    print("PHASE 0: SANITY CHECK")
    print("='*60}")
    print("This will:")
    print("  1. Generate 50 prompt pairs (~5 min, ~$0.50)")
    print("  2. Extract GPT-2 activations (~10 min)")
    print("  3. Train linear probes (~2 min)")
    print("  4. Sync all results to GCS bucket")
    print(f"{'='*60}\n")

    # Check API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set!")
        print("Run: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    print(f"✓ OpenAI API key found")

    # Check GPU
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        print(f"✓ GPU available: {gpu_available}")
        if not gpu_available:
            print("  ⚠ Warning: No GPU detected, this will be slow")
    except ImportError:
        print("⚠ PyTorch not installed yet, will check after pip install")

    response = input("\nProceed with Phase 0? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)

    # Step 1: Generate prompts
    run_command(
        ['python', 'temporal_steering/dataset.py', '--mode', 'sanity'],
        "Step 1/3: Generating prompt pairs"
    )

    # Step 2: Extract activations
    run_command(
        ['python', 'research/tools/extract_activations.py',
         '--prompts', 'data/sanity_check_prompts.json',
         '--output', 'activations/sanity_check.npz'],
        "Step 2/3: Extracting activations"
    )

    # Step 3: Train probes
    run_command(
        ['python', 'research/tools/train_probes.py',
         '--input', 'activations/sanity_check.npz',
         '--output', 'results/sanity_check_results.csv',
         '--prefix', 'sanity'],
        "Step 3/3: Training probes"
    )

    print(f"\n{'='*60}")
    print("PHASE 0 COMPLETE!")
    print(f"{'='*60}")
    print("\nResults saved to:")
    print("  - data/sanity_check_prompts.json")
    print("  - activations/sanity_check.npz")
    print("  - probes/sanity_layer_*_probe.pkl")
    print("  - results/sanity_check_results.csv")
    print("\nAll files synced to: gs://temporal-grounding-gpt2-82feb/")
    print("\nNext steps:")
    print("  1. Review results/sanity_check_results.csv")
    print("  2. Check best layer accuracy")
    print("  3. Decide whether to proceed to Phase 1 (full dataset)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
