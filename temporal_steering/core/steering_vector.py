"""
Base class for steering vectors.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
import json
from pathlib import Path


class SteeringVector(ABC):
    """
    Base class for all steering vector dimensions.

    A steering vector represents a behavioral direction in activation space,
    computed using Contrastive Activation Addition (CAA) from prompt pairs.
    """

    def __init__(self, layer_vectors: Dict[int, np.ndarray], metadata: Dict[str, Any] = None):
        """
        Initialize steering vector.

        Args:
            layer_vectors: Dict mapping layer_idx -> steering vector (hidden_dim,)
            metadata: Optional metadata about vector extraction
        """
        self.layer_vectors = layer_vectors
        self.metadata = metadata or {}

    @abstractmethod
    def get_dimension_name(self) -> str:
        """
        Return human-readable name of this steering dimension.

        Examples: 'temporal_scope', 'optimism', 'technical_detail'
        """
        pass

    @abstractmethod
    def get_strength_range(self) -> Tuple[float, float]:
        """
        Return valid strength range for this dimension.

        Returns:
            (min_strength, max_strength) tuple

        Examples:
            (-1.0, 1.0) for bipolar dimensions (immediate ↔ long-term)
            (0.0, 1.0) for unipolar dimensions (none → maximum)
        """
        pass

    @abstractmethod
    def interpret_strength(self, strength: float) -> str:
        """
        Convert strength value to human-readable description.

        Args:
            strength: Steering strength value

        Returns:
            Human-readable interpretation

        Examples:
            0.8 → "long-term/strategic"
            -0.6 → "immediate/tactical"
        """
        pass

    def get_vector_for_layer(self, layer_idx: int) -> np.ndarray:
        """Get steering vector for specific layer."""
        return self.layer_vectors.get(layer_idx)

    def get_all_layers(self) -> list[int]:
        """Get list of all available layers."""
        return sorted(self.layer_vectors.keys())

    @classmethod
    def load_vectors(cls, vectors_file: str):
        """
        Load steering vectors from JSON file.

        Args:
            vectors_file: Path to steering vectors JSON

        Returns:
            Instance of the steering vector class
        """
        with open(vectors_file, 'r') as f:
            data = json.load(f)

        # Convert to numpy arrays
        layer_vectors = {
            int(layer): np.array(vec)
            for layer, vec in data['layer_vectors'].items()
        }

        metadata = data.get('metadata', {})

        return cls(layer_vectors, metadata)

    def save_vectors(self, output_file: str):
        """
        Save steering vectors to JSON file.

        Args:
            output_file: Path to save vectors
        """
        data = {
            'layer_vectors': {
                str(layer): vec.tolist()
                for layer, vec in self.layer_vectors.items()
            },
            'metadata': {
                **self.metadata,
                'dimension': self.get_dimension_name(),
                'strength_range': self.get_strength_range(),
            }
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved {self.get_dimension_name()} steering vectors to {output_file}")

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"dimension={self.get_dimension_name()}, "
                f"n_layers={len(self.layer_vectors)})")
