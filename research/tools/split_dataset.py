"""Split dataset into train/val/test sets."""

import json
import os
import sys
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.tools.utils import save_checkpoint


def split_dataset(prompts_file, train_size=200, val_size=50, test_size=50):
    """
    Split prompts into train/val/test sets, stratified by domain.

    Args:
        prompts_file: JSON file with all prompt pairs
        train_size: Number of pairs for training
        val_size: Number of pairs for validation
        test_size: Number of pairs for test
    """

    print(f"{'='*60}")
    print(f"DATASET SPLITTING")
    print(f"{'='*60}")
    print(f"Input: {prompts_file}")
    print(f"Target: {train_size} train / {val_size} val / {test_size} test")
    print(f"{'='*60}\n")

    # Load prompts
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)

    print(f"Loaded {len(prompts)} prompt pairs\n")

    # Group by domain
    domains = {}
    for p in prompts:
        domain = p['domain']
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(p)

    print("By domain:")
    for domain, items in domains.items():
        print(f"  {domain}: {len(items)}")
    print()

    # Split each domain proportionally
    train_prompts = []
    val_prompts = []
    test_prompts = []

    total = train_size + val_size + test_size

    for domain, domain_prompts in domains.items():
        n_domain = len(domain_prompts)

        # Calculate proportional splits
        domain_train = int(n_domain * train_size / total)
        domain_val = int(n_domain * val_size / total)
        domain_test = n_domain - domain_train - domain_val

        # First split: separate test set
        train_val, test = train_test_split(
            domain_prompts,
            test_size=domain_test,
            random_state=42
        )

        # Second split: separate train and val
        train, val = train_test_split(
            train_val,
            test_size=domain_val,
            random_state=42
        )

        train_prompts.extend(train)
        val_prompts.extend(val)
        test_prompts.extend(test)

        print(f"{domain}:")
        print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    print()
    print(f"Total splits:")
    print(f"  Train: {len(train_prompts)}")
    print(f"  Val: {len(val_prompts)}")
    print(f"  Test: {len(test_prompts)}")
    print()

    # Save splits
    for split_name, split_data in [
        ('train', train_prompts),
        ('val', val_prompts),
        ('test', test_prompts)
    ]:
        output_file = f'data/{split_name}_prompts.json'
        save_checkpoint(split_data, output_file, also_sync=True)
        print(f"✓ Saved {split_name}: {output_file}")

    print(f"\n{'='*60}")
    print(f"DATASET SPLIT COMPLETE")
    print(f"{'='*60}\n")

    return train_prompts, val_prompts, test_prompts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/raw_prompts.json',
                       help='Path to raw prompts JSON file')
    parser.add_argument('--train-size', type=int, default=200,
                       help='Number of training pairs')
    parser.add_argument('--val-size', type=int, default=50,
                       help='Number of validation pairs')
    parser.add_argument('--test-size', type=int, default=50,
                       help='Number of test pairs')

    args = parser.parse_args()

    # Change to project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    split_dataset(args.input, args.train_size, args.val_size, args.test_size)
