"""Train linear probes for temporal horizon classification."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import pickle
import pandas as pd
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.utils import sync_to_bucket


def train_probes_all_layers(activations_file, results_file, prefix='sanity'):
    """
    Train a linear probe for each layer.
    Report accuracy for each.

    Args:
        activations_file: NPZ file with activations
        results_file: Where to save results CSV
        prefix: Prefix for saved probe files
    """

    print(f"{'='*60}")
    print(f"PROBE TRAINING")
    print(f"{'='*60}")
    print(f"Input: {activations_file}")
    print(f"Output: {results_file}")
    print(f"{'='*60}\n")

    # Load activations
    data = np.load(activations_file, allow_pickle=True)
    labels = data['labels']

    print(f"Dataset statistics:")
    print(f"  Total samples: {len(labels)}")
    print(f"  Short-term (0): {sum(labels == 0)}")
    print(f"  Long-term (1): {sum(labels == 1)}")
    print(f"  Balance: {sum(labels == 0) / len(labels):.1%} / {sum(labels == 1) / len(labels):.1%}\n")

    results = []

    # Determine number of layers
    n_layers = sum(1 for key in data.keys() if key.startswith('layer_'))

    print(f"Training probes for {n_layers} layers...\n")

    for layer in range(n_layers):
        print(f"{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        X = data[f'layer_{layer}']

        print(f"  Activation shape: {X.shape}")

        # Train probe with cross-validation (parallelized)
        probe = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)

        # 5-fold CV (parallelized across folds)
        print(f"  Running 5-fold cross-validation (parallelized)...")
        cv_scores = cross_val_score(probe, X, labels, cv=5, scoring='accuracy', n_jobs=-1)

        print(f"  CV Scores: {cv_scores}")
        print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        # Train on full data for later use
        probe.fit(X, labels)

        # Ensure probes directory exists
        os.makedirs('probes', exist_ok=True)

        # Save probe
        probe_file = f'probes/{prefix}_layer_{layer}_probe.pkl'
        with open(probe_file, 'wb') as f:
            pickle.dump(probe, f)

        print(f"  ✓ Saved probe to {probe_file}\n")

        results.append({
            'layer': layer,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'n_samples': len(labels),
            'n_features': X.shape[1]
        })

    # Ensure results directory exists
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)

    print(f"{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))
    print()

    # Find best layer
    best_layer = results_df.loc[results_df['cv_accuracy_mean'].idxmax()]
    print(f"{'='*60}")
    print(f"BEST LAYER")
    print(f"{'='*60}")
    print(f"  Layer: {int(best_layer['layer'])}")
    print(f"  Accuracy: {best_layer['cv_accuracy_mean']:.3f} (+/- {best_layer['cv_accuracy_std']:.3f})")

    # Recommendation
    print(f"\n{'='*60}")
    print(f"RECOMMENDATION")
    print(f"{'='*60}")

    best_acc = best_layer['cv_accuracy_mean']

    if best_acc < 0.55:
        print("  ⚠ STOP: Accuracy < 55%")
        print("  Signal is too weak. Investigate before proceeding.")
        print("  Possible issues:")
        print("    - Prompts not diverse enough")
        print("    - Model doesn't encode temporal information")
        print("    - Data preprocessing issues")
    elif best_acc < 0.70:
        print("  ⚠ PROCEED WITH CAUTION: Accuracy 55-70%")
        print("  Signal is weak but detectable.")
        print("  Consider:")
        print("    - Generating more diverse prompts")
        print("    - Checking prompt quality")
        print("    - Running control experiments carefully")
    else:
        print("  ✓ PROCEED CONFIDENTLY: Accuracy > 70%")
        print("  Strong signal detected!")
        print("  Model clearly encodes temporal information.")

    print(f"{'='*60}\n")

    # Sync to bucket
    print("Syncing results to GCS bucket...")
    sync_to_bucket(results_file)
    sync_to_bucket('probes/')

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                       help='Path to activations NPZ file')
    parser.add_argument('--output', type=str,
                       default='results/sanity_check_results.csv',
                       help='Path to output results CSV')
    parser.add_argument('--prefix', type=str, default='sanity',
                       help='Prefix for saved probe files')

    args = parser.parse_args()

    # Change to project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    train_probes_all_layers(args.input, args.output, args.prefix)
