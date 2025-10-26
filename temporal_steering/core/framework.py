"""
Main framework for multi-dimensional steering.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .steering_vector import SteeringVector
from .registry import get_steering_class, list_available_dimensions
from ..model_adapter import get_model_config


class SteeringFramework:
    """
    Framework for multi-dimensional LLM steering using Contrastive Activation Addition.

    Supports composing multiple steering dimensions simultaneously.
    """

    def __init__(
        self,
        model,
        tokenizer,
        steerings: Optional[Dict[str, SteeringVector]] = None,
        target_layers: Optional[List[int]] = None
    ):
        """
        Initialize steering framework.

        Args:
            model: HuggingFace transformer model
            tokenizer: Corresponding tokenizer
            steerings: Dict mapping dimension_name -> SteeringVector
            target_layers: Layers to apply steering (default: auto-detect)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.steerings = steerings or {}

        # Auto-detect model architecture
        self.model_config = get_model_config(model)

        # Default to recommended layers (last 60% of model)
        if target_layers is None:
            self.target_layers = self.model_config['recommended_layers']
        else:
            self.target_layers = target_layers

    @classmethod
    def load(
        cls,
        model,
        tokenizer,
        dimension: str,
        vectors_file: Optional[str] = None
    ):
        """
        Load a single steering dimension.

        Args:
            model: HuggingFace model
            tokenizer: Tokenizer
            dimension: Name of dimension (e.g., 'temporal_scope')
            vectors_file: Optional custom vectors file

        Returns:
            SteeringFramework instance with single dimension

        Example:
            >>> framework = SteeringFramework.load(model, tokenizer, "temporal_scope")
            >>> result = framework.generate(
            ...     prompt="Climate policy?",
            ...     steerings=[("temporal_scope", 0.8)]
            ... )
        """
        # Get steering class from registry
        steering_class = get_steering_class(dimension)

        # Auto-find vectors file if not specified
        if vectors_file is None:
            vectors_file = f"steering_vectors/{dimension}.json"

        # Load vectors
        steering_vec = steering_class.load_vectors(vectors_file)

        return cls(model, tokenizer, {dimension: steering_vec})

    @classmethod
    def load_multiple(
        cls,
        model,
        tokenizer,
        dimensions: List[str],
        vectors_dir: str = "steering_vectors"
    ):
        """
        Load multiple steering dimensions.

        Args:
            model: HuggingFace model
            tokenizer: Tokenizer
            dimensions: List of dimension names
            vectors_dir: Directory containing vectors

        Returns:
            SteeringFramework with multiple dimensions

        Example:
            >>> framework = SteeringFramework.load_multiple(
            ...     model, tokenizer,
            ...     dimensions=["temporal_scope", "optimism"]
            ... )
            >>> result = framework.generate(
            ...     prompt="Investment strategy?",
            ...     steerings=[("temporal_scope", 0.8), ("optimism", 0.3)]
            ... )
        """
        steerings = {}

        for dimension in dimensions:
            steering_class = get_steering_class(dimension)
            vectors_file = Path(vectors_dir) / f"{dimension}.json"

            steering_vec = steering_class.load_vectors(str(vectors_file))
            steerings[dimension] = steering_vec

        return cls(model, tokenizer, steerings)

    def generate(
        self,
        prompt: str,
        steerings: Optional[List[Tuple[str, float]]] = None,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text with multi-dimensional steering.

        Args:
            prompt: Input text
            steerings: List of (dimension_name, strength) tuples
                      e.g., [("temporal_scope", 0.8), ("optimism", 0.3)]
                      If None, generates without steering
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text

        Example:
            >>> result = framework.generate(
            ...     prompt="How should we address climate change?",
            ...     steerings=[
            ...         ("temporal_scope", 0.8),
            ...         ("optimism", 0.3),
            ...         ("technical_detail", -0.5)
            ...     ],
            ...     temperature=0.7
            ... )
        """
        # Encode input
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids']

        # Compose steering vectors
        combined_vectors = self._compose_steerings(steerings)

        # Register hooks to add steering
        hooks = self._register_steering_hooks(combined_vectors)

        # Generate
        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        return generated_text

    def _compose_steerings(
        self,
        steerings: Optional[List[Tuple[str, float]]]
    ) -> Dict[int, np.ndarray]:
        """
        Compose multiple steering vectors into combined vectors per layer.

        Args:
            steerings: List of (dimension_name, strength) tuples

        Returns:
            Dict mapping layer_idx -> combined steering vector
        """
        if not steerings:
            return {}

        combined = {}

        for dimension_name, strength in steerings:
            if dimension_name not in self.steerings:
                available = list(self.steerings.keys())
                raise ValueError(
                    f"Unknown dimension: '{dimension_name}'. "
                    f"Loaded dimensions: {available}"
                )

            steering_vec = self.steerings[dimension_name]

            # Add weighted vectors for each layer
            for layer_idx in self.target_layers:
                vec = steering_vec.get_vector_for_layer(layer_idx)

                if vec is None:
                    continue

                if layer_idx not in combined:
                    combined[layer_idx] = np.zeros_like(vec)

                combined[layer_idx] += strength * vec

        return combined

    def _register_steering_hooks(
        self,
        combined_vectors: Dict[int, np.ndarray]
    ) -> List:
        """
        Register forward hooks to apply steering during generation.

        Args:
            combined_vectors: Dict mapping layer_idx -> steering vector

        Returns:
            List of registered hooks
        """
        hooks = []

        if not combined_vectors:
            return hooks

        def make_steering_hook(layer_idx: int, steering_vec: np.ndarray):
            def hook(module, input, output):
                # output is (hidden_states, ...)
                hidden_states = output[0]

                # Convert steering vector to tensor
                steering_tensor = torch.tensor(
                    steering_vec,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )

                # Apply steering
                hidden_states = hidden_states + steering_tensor

                return (hidden_states,) + output[1:]

            return hook

        # Register hooks for layers with steering
        layers = self.model_config['layers']

        for layer_idx, steering_vec in combined_vectors.items():
            hook = layers[layer_idx].register_forward_hook(
                make_steering_hook(layer_idx, steering_vec)
            )
            hooks.append(hook)

        return hooks

    def list_loaded_dimensions(self) -> List[str]:
        """List all currently loaded dimensions."""
        return sorted(self.steerings.keys())

    def get_dimension_info(self, dimension_name: str) -> Dict:
        """
        Get information about a loaded dimension.

        Args:
            dimension_name: Name of dimension

        Returns:
            Dict with dimension metadata
        """
        if dimension_name not in self.steerings:
            raise ValueError(f"Dimension not loaded: {dimension_name}")

        steering = self.steerings[dimension_name]

        return {
            'dimension': steering.get_dimension_name(),
            'strength_range': steering.get_strength_range(),
            'n_layers': len(steering.layer_vectors),
            'available_layers': steering.get_all_layers(),
            'metadata': steering.metadata
        }

    @staticmethod
    def list_available_dimensions() -> List[str]:
        """List all registered dimensions (not just loaded)."""
        return list_available_dimensions()

    def __repr__(self):
        loaded = list(self.steerings.keys())
        return (f"SteeringFramework("
                f"model={self.model_config['model_type']}, "
                f"dimensions={loaded})")
