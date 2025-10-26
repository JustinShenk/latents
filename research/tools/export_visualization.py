"""
Utility to export visualizations with metadata for Vercel deployment.

Usage in your experiment scripts:
    from research.tools.export_visualization import export_visualization

    export_visualization(
        html_file='research/results/my_plot.html',
        metadata={
            'id': 'my-experiment',
            'title': 'My Experiment Title',
            'description': 'Detailed description of what this experiment shows',
            'date': '2025-10-26',
            'status': 'preliminary',  # or 'validated', 'in_progress'
            'category': 'research',  # or 'demo'
            'badges': ['Interactive Plotly', 'Issue #2'],
            'metadata': {
                'dataset_size': 100,
                'model': 'GPT-2',
                'layers_analyzed': [8, 9, 10],
                # ... other experiment-specific metadata
            }
        }
    )
"""

import json
import shutil
from pathlib import Path
from datetime import datetime


def export_visualization(html_file, metadata, update_experiments_json=True):
    """
    Export a visualization to the public/ directory and update experiments.json.

    Args:
        html_file: Path to the HTML file to export
        metadata: Dictionary with experiment metadata (see docstring for structure)
        update_experiments_json: Whether to update public/experiments.json
    """
    html_path = Path(html_file)
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_file}")

    # Determine destination based on category
    category = metadata.get('category', 'research')
    dest_dir = Path('public') / category
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Convert filename to URL-friendly format
    dest_name = metadata['id'] + '.html'
    dest_path = dest_dir / dest_name

    # Copy HTML file
    shutil.copy(html_path, dest_path)
    print(f"✓ Exported {html_path.name} → {dest_path}")

    # Update metadata with file path
    metadata['file'] = f'/{category}/{dest_name}'

    if update_experiments_json:
        _update_experiments_json(metadata)


def _update_experiments_json(new_metadata):
    """Update or add experiment to experiments.json."""
    experiments_file = Path('public/experiments.json')

    # Load existing data
    if experiments_file.exists():
        with open(experiments_file, 'r') as f:
            data = json.load(f)
    else:
        data = {'experiments': [], 'examples': []}

    # Determine which list to update
    category = new_metadata.get('category', 'research')
    list_key = 'experiments' if category == 'research' else 'examples'

    # Find and update existing entry, or append new one
    existing_idx = None
    for idx, item in enumerate(data[list_key]):
        if item['id'] == new_metadata['id']:
            existing_idx = idx
            break

    if existing_idx is not None:
        data[list_key][existing_idx] = new_metadata
        print(f"✓ Updated existing entry: {new_metadata['id']}")
    else:
        data[list_key].append(new_metadata)
        print(f"✓ Added new entry: {new_metadata['id']}")

    # Save updated data
    with open(experiments_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Updated {experiments_file}")


def generate_metadata_template(experiment_id, title, description):
    """Generate a metadata template for a new experiment."""
    return {
        'id': experiment_id,
        'title': title,
        'description': description,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'status': 'in_progress',  # or 'preliminary', 'validated'
        'category': 'research',
        'badges': ['Interactive Plotly'],
        'metadata': {
            'dataset_size': 0,
            'model': 'GPT-2',
            'layers_analyzed': [],
            # Add more fields as needed
        },
        'related_files': {
            # 'dataset': '/path/to/dataset.json',
            # 'issue': 'https://github.com/...',
        }
    }


if __name__ == '__main__':
    # Example usage
    print("Example usage:")
    print("""
from research.tools.export_visualization import export_visualization

export_visualization(
    html_file='research/results/my_experiment.html',
    metadata={
        'id': 'my-experiment',
        'title': 'My Experiment',
        'description': 'Description of experiment',
        'date': '2025-10-26',
        'status': 'preliminary',
        'category': 'research',
        'badges': ['Interactive Plotly'],
        'metadata': {
            'dataset_size': 100,
            'model': 'GPT-2',
        }
    }
)
    """)
