"""
Proper probe training with train→test evaluation (NO data leakage).

Key differences from original:
1. Train probes ONLY on training set
2. Evaluate on HELD-OUT test/val sets
3. Report both CV (train) and true held-out performance
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import pickle
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.tools.utils import sync_to_bucket


def train_and_evaluate_probes(train_file, val_file, test_file, results_dir, prefix='proper'):
    """
    Train probes on training set, evaluate on held-out val/test sets.

    This is the CORRECT way to avoid data leakage.
    """

    print(f"{'='*70}")
    print(f"PROPER PROBE TRAINING & EVALUATION (No Data Leakage)")
    print(f"{'='*70}")
    print(f"Train: {train_file}")
    print(f"Val:   {val_file}")
    print(f"Test:  {test_file}")
    print(f"{'='*70}\n")

    # Load all datasets
    train_data = np.load(train_file, allow_pickle=True)
    val_data = np.load(val_file, allow_pickle=True)
    test_data = np.load(test_file, allow_pickle=True)

    train_labels = train_data['labels']
    val_labels = val_data['labels']
    test_labels = test_data['labels']

    print(f"Dataset sizes:")
    print(f"  Train: {len(train_labels)} samples")
    print(f"  Val:   {len(val_labels)} samples")
    print(f"  Test:  {len(test_labels)} samples\n")

    # Determine number of layers
    n_layers = sum(1 for key in train_data.keys() if key.startswith('layer_'))

    results = []

    os.makedirs('probes', exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    for layer in range(n_layers):
        print(f"{'='*70}")
        print(f"Layer {layer}")
        print(f"{'='*70}")

        # Get activations
        X_train = train_data[f'layer_{layer}']
        X_val = val_data[f'layer_{layer}']
        X_test = test_data[f'layer_{layer}']

        print(f"  Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

        # Train probe with cross-validation ON TRAINING SET ONLY
        probe = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)

        print(f"  Running 5-fold CV on TRAINING set only...")
        cv_scores = cross_val_score(probe, X_train, train_labels, cv=5,
                                   scoring='accuracy', n_jobs=-1)

        print(f"    CV scores: {cv_scores}")
        print(f"    CV mean: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

        # Train on FULL training set
        print(f"  Training probe on full training set...")
        probe.fit(X_train, train_labels)

        # Evaluate on HELD-OUT validation set
        val_pred = probe.predict(X_val)
        val_accuracy = accuracy_score(val_labels, val_pred)
        print(f"  Validation accuracy: {val_accuracy:.3f}")

        # Evaluate on HELD-OUT test set (probe has NEVER seen this!)
        test_pred = probe.predict(X_test)
        test_accuracy = accuracy_score(test_labels, test_pred)
        print(f"  Test accuracy: {test_accuracy:.3f}")

        # Confusion matrix for test set
        cm = confusion_matrix(test_labels, test_pred)
        print(f"  Test confusion matrix:")
        print(f"    {cm}")

        # Save probe
        probe_file = f'probes/{prefix}_layer_{layer}_probe.pkl'
        with open(probe_file, 'wb') as f:
            pickle.dump(probe, f)
        print(f"  ✓ Saved to {probe_file}\n")

        results.append({
            'layer': layer,
            'train_cv_mean': cv_scores.mean(),
            'train_cv_std': cv_scores.std(),
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'n_train': len(train_labels),
            'n_val': len(val_labels),
            'n_test': len(test_labels),
            'n_features': X_train.shape[1]
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_file = os.path.join(results_dir, f'{prefix}_results.csv')
    results_df.to_csv(results_file, index=False)

    print(f"{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))
    print()

    # Best layers
    best_train = results_df.loc[results_df['train_cv_mean'].idxmax()]
    best_test = results_df.loc[results_df['test_accuracy'].idxmax()]

    print(f"{'='*70}")
    print(f"BEST PERFORMANCE")
    print(f"{'='*70}")
    print(f"Best Training (CV): Layer {int(best_train['layer'])} - {best_train['train_cv_mean']:.3f}")
    print(f"Best Test:          Layer {int(best_test['layer'])} - {best_test['test_accuracy']:.3f}")
    print(f"{'='*70}\n")

    # Sync
    sync_to_bucket(results_file)
    sync_to_bucket('probes/')

    return results_df


def evaluate_control_with_trained_probes(control_file, train_file, results_dir, prefix='control'):
    """
    Evaluate control dataset using probes trained on ORIGINAL training set.

    This tests if probes generalize to keyword-ablated versions.
    """

    print(f"{'='*70}")
    print(f"CONTROL EVALUATION (Using Pre-Trained Probes)")
    print(f"{'='*70}")
    print(f"Control data: {control_file}")
    print(f"Training set (for probe loading): {train_file}")
    print(f"{'='*70}\n")

    # Load control data
    control_data = np.load(control_file, allow_pickle=True)
    control_labels = control_data['labels']

    print(f"Control dataset: {len(control_labels)} samples\n")

    n_layers = sum(1 for key in control_data.keys() if key.startswith('layer_'))

    results = []

    for layer in range(n_layers):
        print(f"Layer {layer}:")

        # Load probe trained on ORIGINAL training set
        probe_file = f'probes/proper_layer_{layer}_probe.pkl'

        if not os.path.exists(probe_file):
            print(f"  ⚠ Probe not found: {probe_file}")
            continue

        with open(probe_file, 'rb') as f:
            probe = pickle.load(f)

        # Get control activations
        X_control = control_data[f'layer_{layer}']

        # Evaluate
        control_pred = probe.predict(X_control)
        control_accuracy = accuracy_score(control_labels, control_pred)

        print(f"  Control accuracy: {control_accuracy:.3f}")

        # Confusion matrix
        cm = confusion_matrix(control_labels, control_pred)
        print(f"  Confusion: {cm.tolist()}\n")

        results.append({
            'layer': layer,
            'control_accuracy': control_accuracy,
            'n_samples': len(control_labels)
        })

    # Save
    results_df = pd.DataFrame(results)
    results_file = os.path.join(results_dir, f'{prefix}_results.csv')
    results_df.to_csv(results_file, index=False)

    print(f"{'='*70}")
    print(f"CONTROL RESULTS")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))
    print(f"{'='*70}\n")

    sync_to_bucket(results_file)

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'control'], required=True)
    parser.add_argument('--train', type=str, help='Training activations')
    parser.add_argument('--val', type=str, help='Validation activations')
    parser.add_argument('--test', type=str, help='Test activations')
    parser.add_argument('--control', type=str, help='Control activations')
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--prefix', type=str, default='proper')

    args = parser.parse_args()

    # Change to project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if args.mode == 'train':
        if not all([args.train, args.val, args.test]):
            parser.error("--train, --val, and --test required for train mode")
        train_and_evaluate_probes(args.train, args.val, args.test,
                                 args.results_dir, args.prefix)

    elif args.mode == 'control':
        if not all([args.control, args.train]):
            parser.error("--control and --train required for control mode")
        evaluate_control_with_trained_probes(args.control, args.train,
                                            args.results_dir, args.prefix)
