"""
Horizon: Multi-dimensional LLM steering using Contrastive Activation Addition.

Primary focus: Temporal scope control (immediate ↔ long-term)
Extensible: Supports plugins for other behavioral dimensions

Works with any transformer model: GPT-2, LLaMA, Mistral, Falcon, and more.
"""

__version__ = "0.1.0"

# Core framework
from .core import SteeringVector, SteeringFramework, register_steering, STEERING_REGISTRY

# Built-in dimensions
from .dimensions import TemporalSteeringVector

# Model adapter utilities
from .model_adapter import get_model_config, get_model_layers, print_model_info

# Backward compatibility: keep old TemporalSteering class
from .temporal_steering_demo import TemporalSteering

__all__ = [
    # New plugin architecture
    "SteeringVector",
    "SteeringFramework",
    "register_steering",
    "STEERING_REGISTRY",
    "TemporalSteeringVector",

    # Backward compatibility
    "TemporalSteering",

    # Model utilities
    "get_model_config",
    "get_model_layers",
    "print_model_info",

    "__version__"
]
