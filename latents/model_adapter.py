"""
Model adapter for universal transformer layer access.

Supports GPT-2, LLaMA, Mistral, Falcon, GPT-NeoX, and other architectures.
"""

from typing import List, Dict, Any
import torch.nn as nn


def get_model_layers(model) -> nn.ModuleList:
    """
    Get transformer layers for any model architecture.

    Args:
        model: A HuggingFace transformer model

    Returns:
        nn.ModuleList: The transformer layers

    Raises:
        ValueError: If model architecture is not supported

    Examples:
        >>> from transformers import GPT2LMHeadModel, AutoModelForCausalLM
        >>>
        >>> # GPT-2
        >>> model = GPT2LMHeadModel.from_pretrained('gpt2')
        >>> layers = get_model_layers(model)
        >>>
        >>> # LLaMA-2
        >>> model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
        >>> layers = get_model_layers(model)
    """

    # GPT-2, GPT-Neo, GPT-J, Falcon (some variants)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h

    # LLaMA, LLaMA-2, Mistral, Mixtral, Phi
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers

    # GPT-NeoX, Pythia
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return model.gpt_neox.layers

    # Some Falcon variants
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
        return model.transformer.layers

    # BLOOM
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h

    # OPT
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        return model.model.decoder.layers

    else:
        # Try to provide helpful error message
        model_type = type(model).__name__
        available_attrs = dir(model)

        raise ValueError(
            f"Unsupported model architecture: {model_type}\n"
            f"Could not find transformer layers. Available top-level attributes:\n"
            f"{[attr for attr in available_attrs if not attr.startswith('_')]}\n\n"
            f"Please open an issue at: https://github.com/justinshenk/temporal-steering/issues"
        )


def get_model_config(model) -> Dict[str, Any]:
    """
    Get model metadata and recommendations for layer selection.

    Args:
        model: A HuggingFace transformer model

    Returns:
        dict: Configuration with keys:
            - 'layers': nn.ModuleList of transformer layers
            - 'n_layers': int, number of layers
            - 'model_type': str, detected model type
            - 'recommended_start': int, suggested starting layer for steering
            - 'recommended_layers': list[int], suggested layers to apply steering
            - 'hidden_size': int, hidden dimension size (if available)

    Examples:
        >>> config = get_model_config(model)
        >>> print(f"Model has {config['n_layers']} layers")
        >>> print(f"Recommended steering layers: {config['recommended_layers']}")
    """

    layers = get_model_layers(model)
    n_layers = len(layers)

    # Detect model type
    model_type = type(model).__name__

    # Recommend targeting last 60% of layers (where temporal effects are strongest)
    # Based on empirical findings that later layers show stronger steering effects
    start_layer = max(0, int(n_layers * 0.4))
    recommended_layers = list(range(start_layer, n_layers))

    # Try to get hidden size
    hidden_size = None
    if hasattr(model.config, 'hidden_size'):
        hidden_size = model.config.hidden_size
    elif hasattr(model.config, 'n_embd'):
        hidden_size = model.config.n_embd

    config = {
        'layers': layers,
        'n_layers': n_layers,
        'model_type': model_type,
        'recommended_start': start_layer,
        'recommended_layers': recommended_layers,
        'hidden_size': hidden_size,
        'model_name': getattr(model.config, 'model_type', 'unknown')
    }

    return config


def print_model_info(model):
    """
    Print diagnostic information about model architecture.

    Useful for debugging and understanding model structure.

    Args:
        model: A HuggingFace transformer model
    """
    try:
        config = get_model_config(model)

        print("="*70)
        print("MODEL INFORMATION")
        print("="*70)
        print(f"Model Type: {config['model_type']}")
        print(f"Model Family: {config['model_name']}")
        print(f"Total Layers: {config['n_layers']}")
        print(f"Hidden Size: {config['hidden_size']}")
        print(f"Recommended Steering Layers: {config['recommended_start']}-{config['n_layers']-1}")
        print(f"  → {len(config['recommended_layers'])} layers: {config['recommended_layers']}")
        print("="*70)

        # Performance estimates based on model size
        if config['n_layers'] <= 12:
            speed = "Fast"
            quality = "Good for research/prototyping"
        elif config['n_layers'] <= 32:
            speed = "Moderate"
            quality = "Production quality"
        else:
            speed = "Slow (requires GPU)"
            quality = "Highest quality"

        print(f"Performance: {speed}")
        print(f"Quality: {quality}")
        print("="*70)

    except ValueError as e:
        print(f"Error: {e}")


# Supported model architectures
SUPPORTED_ARCHITECTURES = {
    'GPT-2 family': ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
    'GPT-Neo/J': ['EleutherAI/gpt-neo-*', 'EleutherAI/gpt-j-*'],
    'LLaMA family': ['meta-llama/Llama-2-*', 'meta-llama/Meta-Llama-3-*'],
    'Mistral': ['mistralai/Mistral-7B-*', 'mistralai/Mixtral-*'],
    'Falcon': ['tiiuae/falcon-*'],
    'GPT-NeoX': ['EleutherAI/pythia-*', 'EleutherAI/gpt-neox-*'],
    'BLOOM': ['bigscience/bloom-*'],
    'OPT': ['facebook/opt-*'],
    'Phi': ['microsoft/phi-*'],
}


def is_model_supported(model) -> bool:
    """
    Check if a model architecture is supported.

    Args:
        model: A HuggingFace transformer model

    Returns:
        bool: True if supported, False otherwise
    """
    try:
        get_model_layers(model)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    # Quick test
    print("Testing model adapter...")
    print("\nSupported architectures:")
    for family, models in SUPPORTED_ARCHITECTURES.items():
        print(f"  {family}: {', '.join(models)}")
