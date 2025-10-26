"""
Confound Analysis Experiment: Temporal Scope vs Stylistic Features

⚠️  PRELIMINARY EXPERIMENT - RESULTS REQUIRE VERIFICATION

Research Question: Does temporal steering capture true temporal scope,
or is it confounded by correlated stylistic features (formality, tone)?

Experimental Design:
- 2×2 crossed design: Temporal (immediate vs long-term) × Style (casual vs formal)
- 40 prompts total (10 per cell)
- Extract activations from GPT-2 layers 7-11
- PCA analysis to check for separation
- Test if steering is truly temporal vs stylistic

Expected Results:
- Good: PCA separates temporal dimension cleanly (PC1)
- Confounded: PCA shows style dominates (temporal is secondary)
- Ideal: Two orthogonal PCs (temporal + style are independent)

Limitations:
- Small sample size (10 prompts/cell)
- Single model (GPT-2 only)
- No behavioral validation yet
- Manual prompt generation

Next Steps for Validation:
- Human evaluation of deconfounded steering
- Scale to larger sample sizes
- Test on multiple models (LLaMA-2, GPT-2-large)
- Quantitative behavioral metrics
"""

import torch
import numpy as np
import json
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, List, Tuple

# For interactive plots
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  plotly not available - will only generate static plots")

# Use new plugin architecture
from latents import SteeringFramework, TemporalSteeringVector
from latents.extract_steering_vectors import extract_activations_with_hooks


# Dataset: 2×2 Design (Temporal × Style)
EXPERIMENTAL_DATASET = {
    "immediate_casual": [
        "Yo, housing crisis is crazy right now, what should mayor do asap?",
        "Dude, these traffic jams are killing me. Quick fixes?",
        "Man, our servers are down! What do we do RIGHT NOW?",
        "This bug is breaking prod. How do we patch it today?",
        "Energy prices going nuts this week. Immediate relief?",
        "Our cash flow is terrible this month. Emergency moves?",
        "Students are failing these tests. What works now?",
        "Climate protests happening tomorrow. How do we respond?",
        "Supply chain is broken. Fast solutions?",
        "Investors are freaking out. How do we calm them down this quarter?"
    ],

    "immediate_formal": [
        "The housing shortage requires urgent municipal intervention. What immediate measures should be implemented?",
        "Traffic congestion has reached critical levels. What short-term interventions are recommended?",
        "We are experiencing a critical system outage. What are the immediate remediation steps?",
        "A production bug has been identified. What is the expedited resolution protocol?",
        "Energy costs have increased substantially. What emergency relief measures should be deployed?",
        "Cash flow has deteriorated significantly. What immediate financial actions are required?",
        "Student performance metrics are declining. What interventions should be implemented this semester?",
        "Climate protests are scheduled. What is the appropriate immediate response?",
        "Supply chain disruption requires urgent attention. What near-term solutions exist?",
        "Investor confidence has decreased. What quarterly stabilization measures are needed?"
    ],

    "longterm_casual": [
        "How do we build cities our grandkids will actually want to live in?",
        "What kind of transportation will people use in 50 years?",
        "How should we design systems that last for decades?",
        "What codebase architecture will serve us for years to come?",
        "How do we create an energy future that's sustainable forever?",
        "What business model works for the next generation?",
        "How do we educate kids for jobs that don't exist yet?",
        "What's the long-term plan for climate that actually works?",
        "How do we build resilient supply chains for the future?",
        "What company culture will thrive in 2050?"
    ],

    "longterm_formal": [
        "What institutional frameworks ensure sustainable urban development for future generations?",
        "What transportation infrastructure investments will serve society for the next half-century?",
        "What architectural principles ensure long-term system resilience and adaptability?",
        "What software engineering practices optimize for decadal maintainability?",
        "What energy infrastructure enables sustainable transition across generations?",
        "What organizational structures support multi-generational value creation?",
        "What pedagogical frameworks prepare students for an uncertain future?",
        "What policy mechanisms ensure climate resilience for future populations?",
        "What supply chain designs optimize for long-term sustainability and adaptation?",
        "What governance structures foster organizational longevity and evolution?"
    ]
}


def create_experimental_dataset(output_file: str):
    """Save experimental dataset to JSON."""
    data = {
        "metadata": {
            "design": "2x2 crossed (Temporal × Style)",
            "factors": {
                "temporal": ["immediate", "longterm"],
                "style": ["casual", "formal"]
            },
            "n_prompts_per_cell": 10,
            "total_prompts": 40,
            "purpose": "Test if temporal steering is confounded by style"
        },
        "prompts": EXPERIMENTAL_DATASET
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Saved experimental dataset to {output_file}")
    print(f"  Total prompts: {sum(len(v) for v in EXPERIMENTAL_DATASET.values())}")


def extract_activations_for_dataset(
    model,
    tokenizer,
    dataset: Dict[str, List[str]],
    layers: List[int] = [7, 8, 9, 10, 11]
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Extract activations for all prompts in dataset.

    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        dataset: Dict mapping condition -> list of prompts
        layers: Which layers to extract (default: 7-11 for GPT-2)

    Returns:
        Dict mapping condition -> layer -> activations array
        Shape: (n_prompts, hidden_dim) per layer
    """
    print("Extracting activations for experimental dataset...")
    print(f"Layers: {layers}")
    print()

    activations_by_condition = {}

    for condition, prompts in dataset.items():
        print(f"Condition: {condition} ({len(prompts)} prompts)")
        condition_acts = {layer: [] for layer in layers}

        for prompt in prompts:
            # Extract activations for this prompt
            acts = extract_activations_with_hooks(model, tokenizer, prompt)

            # Collect last token activations for each layer
            for layer in layers:
                last_token_act = acts[layer][0, -1, :].cpu().numpy()
                condition_acts[layer].append(last_token_act)

        # Stack into arrays
        for layer in layers:
            condition_acts[layer] = np.stack(condition_acts[layer])

        activations_by_condition[condition] = condition_acts

    return activations_by_condition


def run_pca_analysis(
    activations: Dict[str, Dict[int, np.ndarray]],
    layer: int = 10
) -> Tuple[PCA, np.ndarray, Dict[str, np.ndarray]]:
    """
    Run PCA on activations to visualize factor structure.

    Args:
        activations: Dict mapping condition -> layer -> activations
        layer: Which layer to analyze

    Returns:
        (pca_model, all_activations, activations_by_condition)
    """
    print(f"\nPCA Analysis on Layer {layer}")
    print("=" * 70)

    # Collect all activations for this layer
    all_acts = []
    labels = []
    condition_order = []

    for condition, layer_acts in activations.items():
        acts = layer_acts[layer]  # (n_prompts, hidden_dim)
        all_acts.append(acts)
        labels.extend([condition] * len(acts))
        condition_order.append(condition)

    all_acts = np.vstack(all_acts)  # (40, hidden_dim)

    print(f"Total samples: {len(all_acts)}")
    print(f"Hidden dim: {all_acts.shape[1]}")

    # Standardize
    scaler = StandardScaler()
    all_acts_scaled = scaler.fit_transform(all_acts)

    # PCA
    pca = PCA(n_components=4)
    transformed = pca.fit_transform(all_acts_scaled)

    # Print variance explained
    print("\nVariance Explained:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var:.3f} ({var*100:.1f}%)")

    print(f"  Total (PC1-4): {pca.explained_variance_ratio_.sum():.3f}")

    # Separate by condition
    acts_by_cond = {}
    start_idx = 0
    for condition in activations.keys():
        n = len(activations[condition][layer])
        acts_by_cond[condition] = transformed[start_idx:start_idx+n]
        start_idx += n

    return pca, transformed, acts_by_cond


def visualize_pca(
    acts_by_condition: Dict[str, np.ndarray],
    output_file: str = "research/results/pca_temporal_style.png"
):
    """
    Visualize PCA results to check factor separation.

    Expected patterns:
    - PC1 separates temporal (immediate vs long-term)
    - PC2 separates style (casual vs formal)
    - If both clear: orthogonal factors ✓
    - If PC1 is style: confound ✗
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color/marker scheme
    colors = {
        'immediate_casual': 'red',
        'immediate_formal': 'darkred',
        'longterm_casual': 'blue',
        'longterm_formal': 'darkblue'
    }

    markers = {
        'immediate_casual': 'o',
        'immediate_formal': 's',
        'longterm_casual': 'o',
        'longterm_formal': 's'
    }

    labels_pretty = {
        'immediate_casual': 'Immediate + Casual',
        'immediate_formal': 'Immediate + Formal',
        'longterm_casual': 'Long-term + Casual',
        'longterm_formal': 'Long-term + Formal'
    }

    # PC1 vs PC2
    ax = axes[0]
    for condition, acts in acts_by_condition.items():
        ax.scatter(
            acts[:, 0], acts[:, 1],
            c=colors[condition],
            marker=markers[condition],
            s=100,
            alpha=0.7,
            label=labels_pretty[condition]
        )

    ax.set_xlabel('PC1 (Temporal?)', fontsize=12)
    ax.set_ylabel('PC2 (Style?)', fontsize=12)
    ax.set_title('PCA: PC1 vs PC2', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)

    # PC1 vs PC3
    ax = axes[1]
    for condition, acts in acts_by_condition.items():
        ax.scatter(
            acts[:, 0], acts[:, 2],
            c=colors[condition],
            marker=markers[condition],
            s=100,
            alpha=0.7,
            label=labels_pretty[condition]
        )

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC3', fontsize=12)
    ax.set_title('PCA: PC1 vs PC3', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)

    plt.tight_layout()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved PCA visualization to {output_file}")


def visualize_pca_interactive(
    acts_by_condition: Dict[str, np.ndarray],
    prompts_by_condition: Dict[str, List[str]],
    output_file: str = "research/results/pca_temporal_style_interactive.html"
):
    """
    Create interactive PCA visualization with hover text showing prompts.

    Hovering over points reveals the actual prompt text for inspection.
    """
    if not PLOTLY_AVAILABLE:
        print("⚠️  Skipping interactive visualization (plotly not installed)")
        return

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('PC1 vs PC2', 'PC1 vs PC3'),
        horizontal_spacing=0.12
    )

    # Color/symbol scheme
    colors = {
        'immediate_casual': '#FF4444',    # red
        'immediate_formal': '#AA0000',    # dark red
        'longterm_casual': '#4444FF',     # blue
        'longterm_formal': '#0000AA'      # dark blue
    }

    symbols = {
        'immediate_casual': 'circle',
        'immediate_formal': 'square',
        'longterm_casual': 'circle',
        'longterm_formal': 'square'
    }

    labels_pretty = {
        'immediate_casual': 'Immediate + Casual',
        'immediate_formal': 'Immediate + Formal',
        'longterm_casual': 'Long-term + Casual',
        'longterm_formal': 'Long-term + Formal'
    }

    # Add traces for PC1 vs PC2
    for condition, acts in acts_by_condition.items():
        prompts = prompts_by_condition[condition]

        fig.add_trace(
            go.Scatter(
                x=acts[:, 0],
                y=acts[:, 1],
                mode='markers',
                name=labels_pretty[condition],
                marker=dict(
                    color=colors[condition],
                    symbol=symbols[condition],
                    size=12,
                    line=dict(width=1, color='white')
                ),
                text=prompts,
                hovertemplate='<b>%{text}</b><br>' +
                             'PC1: %{x:.2f}<br>' +
                             'PC2: %{y:.2f}<br>' +
                             '<extra></extra>',
                legendgroup=condition,
                showlegend=True
            ),
            row=1, col=1
        )

    # Add traces for PC1 vs PC3
    for condition, acts in acts_by_condition.items():
        prompts = prompts_by_condition[condition]

        fig.add_trace(
            go.Scatter(
                x=acts[:, 0],
                y=acts[:, 2],
                mode='markers',
                name=labels_pretty[condition],
                marker=dict(
                    color=colors[condition],
                    symbol=symbols[condition],
                    size=12,
                    line=dict(width=1, color='white')
                ),
                text=prompts,
                hovertemplate='<b>%{text}</b><br>' +
                             'PC1: %{x:.2f}<br>' +
                             'PC3: %{y:.2f}<br>' +
                             '<extra></extra>',
                legendgroup=condition,
                showlegend=False
            ),
            row=1, col=2
        )

    # Update layout
    fig.update_xaxes(title_text="PC1 (Temporal?)", row=1, col=1, zeroline=True, zerolinewidth=1, zerolinecolor='gray')
    fig.update_yaxes(title_text="PC2 (Style?)", row=1, col=1, zeroline=True, zerolinewidth=1, zerolinecolor='gray')
    fig.update_xaxes(title_text="PC1", row=1, col=2, zeroline=True, zerolinewidth=1, zerolinecolor='gray')
    fig.update_yaxes(title_text="PC3", row=1, col=2, zeroline=True, zerolinewidth=1, zerolinecolor='gray')

    fig.update_layout(
        title={
            'text': 'PCA: Temporal Scope vs Style Confound Analysis<br><sub>Hover over points to see prompts</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=600,
        width=1400,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template='plotly_white'
    )

    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_file)
    print(f"✓ Saved interactive PCA visualization to {output_file}")
    print(f"  Open in browser to explore: file://{Path(output_file).absolute()}")


def compute_mean_separation(acts_by_condition: Dict[str, np.ndarray]):
    """
    Quantify how well temporal vs style factors separate.

    Metrics:
    - Temporal separation: distance(immediate, longterm) along PC1
    - Style separation: distance(casual, formal) along PC2
    - Ratio tells us which dimension dominates
    """
    print("\nMean Separation Analysis")
    print("=" * 70)

    # Compute means for each condition
    means = {cond: acts.mean(axis=0) for cond, acts in acts_by_condition.items()}

    # Temporal separation (across PC1)
    immediate_mean = (means['immediate_casual'] + means['immediate_formal']) / 2
    longterm_mean = (means['longterm_casual'] + means['longterm_formal']) / 2

    temporal_sep_pc1 = abs(immediate_mean[0] - longterm_mean[0])
    temporal_sep_pc2 = abs(immediate_mean[1] - longterm_mean[1])

    # Style separation (across PC2)
    casual_mean = (means['immediate_casual'] + means['longterm_casual']) / 2
    formal_mean = (means['immediate_formal'] + means['longterm_formal']) / 2

    style_sep_pc1 = abs(casual_mean[0] - formal_mean[0])
    style_sep_pc2 = abs(casual_mean[1] - formal_mean[1])

    print(f"Temporal separation:")
    print(f"  PC1: {temporal_sep_pc1:.3f}")
    print(f"  PC2: {temporal_sep_pc2:.3f}")
    print(f"  Ratio (PC1/PC2): {temporal_sep_pc1/temporal_sep_pc2:.2f}x")

    print(f"\nStyle separation:")
    print(f"  PC1: {style_sep_pc1:.3f}")
    print(f"  PC2: {style_sep_pc2:.3f}")
    print(f"  Ratio (PC2/PC1): {style_sep_pc2/style_sep_pc1:.2f}x")

    print(f"\nInterpretation:")
    if temporal_sep_pc1 > 2 * style_sep_pc1:
        print("  ✓ PC1 primarily captures TEMPORAL dimension")
    else:
        print("  ⚠ PC1 is confounded by style")

    if style_sep_pc2 > 2 * temporal_sep_pc2:
        print("  ✓ PC2 primarily captures STYLE dimension")
    else:
        print("  ⚠ PC2 is confounded by temporal")

    return {
        'temporal_pc1': temporal_sep_pc1,
        'temporal_pc2': temporal_sep_pc2,
        'style_pc1': style_sep_pc1,
        'style_pc2': style_sep_pc2
    }


def extract_deconfounded_vectors(
    activations: Dict[str, Dict[int, np.ndarray]],
    layers: List[int] = [7, 8, 9, 10, 11]
) -> Dict[int, np.ndarray]:
    """
    Extract temporal steering vectors controlling for style.

    Method: Average across style conditions
    - immediate_vector = mean(immediate_casual, immediate_formal)
    - longterm_vector = mean(longterm_casual, longterm_formal)
    - temporal_vector = longterm_vector - immediate_vector

    This averages out style confounds.
    """
    print("\nExtracting Deconfounded Temporal Vectors")
    print("=" * 70)

    deconfounded_vectors = {}

    for layer in layers:
        # Get activations for this layer
        imm_casual = activations['immediate_casual'][layer]
        imm_formal = activations['immediate_formal'][layer]
        long_casual = activations['longterm_casual'][layer]
        long_formal = activations['longterm_formal'][layer]

        # Average across style to get pure temporal effect
        immediate_avg = (imm_casual.mean(axis=0) + imm_formal.mean(axis=0)) / 2
        longterm_avg = (long_casual.mean(axis=0) + long_formal.mean(axis=0)) / 2

        # Contrastive vector
        temporal_vec = longterm_avg - immediate_avg

        deconfounded_vectors[layer] = temporal_vec

        print(f"Layer {layer:2d}: norm={np.linalg.norm(temporal_vec):.3f}")

    return deconfounded_vectors


if __name__ == "__main__":
    print("=" * 70)
    print("CONFOUND ANALYSIS: TEMPORAL SCOPE vs STYLE")
    print("=" * 70)
    print()

    # Create dataset
    dataset_file = "research/datasets/confound_experiment.json"
    create_experimental_dataset(dataset_file)
    print()

    # Load model
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print()

    # Extract activations
    activations = extract_activations_for_dataset(
        model, tokenizer,
        EXPERIMENTAL_DATASET,
        layers=[7, 8, 9, 10, 11]
    )

    # Save activations
    acts_file = "research/results/confound_activations.npz"
    Path(acts_file).parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        acts_file,
        **{f"{cond}_layer{layer}": acts
           for cond, layer_acts in activations.items()
           for layer, acts in layer_acts.items()}
    )
    print(f"\n✓ Saved activations to {acts_file}")

    # PCA analysis on layer 10 (strongest effect layer from original extraction)
    pca, all_transformed, acts_by_cond = run_pca_analysis(activations, layer=10)

    # Visualize (static)
    visualize_pca(acts_by_cond)

    # Visualize (interactive with hover text)
    visualize_pca_interactive(acts_by_cond, EXPERIMENTAL_DATASET)

    # Quantify separation
    separations = compute_mean_separation(acts_by_cond)

    # Extract deconfounded vectors
    deconfounded = extract_deconfounded_vectors(activations)

    # Save deconfounded vectors
    deconf_file = "steering_vectors/temporal_scope_deconfounded.json"
    data = {
        'layer_vectors': {
            str(layer): vec.tolist()
            for layer, vec in deconfounded.items()
        },
        'metadata': {
            'extraction_method': 'Deconfounded CAA (controlled for style)',
            'design': '2x2 crossed (Temporal × Style)',
            'model': 'gpt2',
            'layers': [7, 8, 9, 10, 11],
            'n_prompts': 40,
            'separations': {k: float(v) for k, v in separations.items()}  # Convert numpy floats
        }
    }

    with open(deconf_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Saved deconfounded vectors to {deconf_file}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Check PCA plot: research/results/pca_temporal_style.png")
    print("  2. Compare original vs deconfounded steering vectors")
    print("  3. Human eval: Does deconfounded steering change temporal scope?")
