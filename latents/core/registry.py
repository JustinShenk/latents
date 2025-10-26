"""
Registry system for steering vector plugins.
"""

from typing import Dict, Type
from .steering_vector import SteeringVector


# Global registry of steering dimensions
STEERING_REGISTRY: Dict[str, Type[SteeringVector]] = {}


def register_steering(dimension_name: str):
    """
    Decorator to register a steering dimension.

    Usage:
        @register_steering("temporal_scope")
        class TemporalSteering(SteeringVector):
            ...

    Args:
        dimension_name: Unique identifier for this dimension
    """
    def decorator(cls: Type[SteeringVector]):
        if dimension_name in STEERING_REGISTRY:
            raise ValueError(
                f"Dimension '{dimension_name}' already registered by "
                f"{STEERING_REGISTRY[dimension_name].__name__}"
            )

        if not issubclass(cls, SteeringVector):
            raise TypeError(
                f"{cls.__name__} must inherit from SteeringVector"
            )

        STEERING_REGISTRY[dimension_name] = cls
        return cls

    return decorator


def list_available_dimensions() -> list[str]:
    """List all registered steering dimensions."""
    return sorted(STEERING_REGISTRY.keys())


def get_steering_class(dimension_name: str) -> Type[SteeringVector]:
    """
    Get steering class by dimension name.

    Args:
        dimension_name: Name of the dimension

    Returns:
        SteeringVector subclass

    Raises:
        ValueError: If dimension not found
    """
    if dimension_name not in STEERING_REGISTRY:
        available = list_available_dimensions()
        raise ValueError(
            f"Unknown dimension: '{dimension_name}'. "
            f"Available dimensions: {available}"
        )

    return STEERING_REGISTRY[dimension_name]


def print_registry():
    """Print all registered steering dimensions."""
    print("=" * 70)
    print("REGISTERED STEERING DIMENSIONS")
    print("=" * 70)

    if not STEERING_REGISTRY:
        print("No dimensions registered.")
        return

    for name, cls in sorted(STEERING_REGISTRY.items()):
        instance = cls({}, {})  # Temporary instance for metadata
        strength_range = instance.get_strength_range()

        print(f"  {name}")
        print(f"    Class: {cls.__name__}")
        print(f"    Strength Range: {strength_range}")
        print(f"    Example: {instance.interpret_strength(strength_range[1])}")
        print()
