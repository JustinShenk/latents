"""
Technicality steering dimension.

Controls the technical depth of model outputs from simple/layman
to technical/specialized language.
"""

from typing import Tuple
from ..core import SteeringVector, register_steering


@register_steering("technicality")
class TechnicalitySteeringVector(SteeringVector):
    """
    Technicality steering: simple/layman ↔ technical/specialized.

    Strength interpretation:
        -1.0: Maximum simple/layman language
         0.0: Balanced/moderate technical level
        +1.0: Maximum technical/specialized language
    """

    def get_dimension_name(self) -> str:
        return "technicality"

    def get_strength_range(self) -> Tuple[float, float]:
        return (-1.0, 1.0)

    def interpret_strength(self, strength: float) -> str:
        """
        Interpret steering strength as technicality level.

        Args:
            strength: Value between -1.0 and 1.0

        Returns:
            Human-readable interpretation
        """
        if strength <= -0.75:
            return "very simple/elementary"
        elif strength <= -0.25:
            return "accessible/general audience"
        elif strength <= 0.25:
            return "moderate/informed audience"
        elif strength <= 0.75:
            return "technical/specialist"
        else:
            return "highly technical/expert"

    @classmethod
    def get_default_prompt_pairs(cls):
        """
        Return default contrastive prompt pairs for technicality extraction.

        These pairs contrast simple explanations vs technical descriptions.
        """
        return [
            {
                "negative": "Machine learning is when computers learn from examples instead of being told exactly what to do.",
                "positive": "Machine learning employs statistical algorithms to optimize objective functions by iteratively adjusting model parameters based on gradient descent over training data."
            },
            {
                "negative": "The computer stores information in memory so it can use it later.",
                "positive": "The system utilizes volatile DRAM for runtime state persistence and non-volatile storage for long-term data retention across power cycles."
            },
            {
                "negative": "The medicine helps your body fight infections.",
                "positive": "The antibiotic agent inhibits bacterial cell wall synthesis by binding to penicillin-binding proteins, resulting in compromised structural integrity and cellular lysis."
            },
            {
                "negative": "The website loads faster because it caches data.",
                "positive": "The application implements a multi-tier caching strategy utilizing CDN edge nodes, Redis for session state, and browser-level HTTP caching directives to minimize latency."
            },
            {
                "negative": "DNA contains the instructions for making proteins in cells.",
                "positive": "The nucleotide sequence encodes genetic information through codon triplets that specify amino acid residues during ribosomal translation of mRNA transcripts."
            },
            {
                "negative": "The car's engine turns chemical energy into motion.",
                "positive": "The internal combustion engine converts hydrocarbon oxidation enthalpy into mechanical work through thermodynamic expansion of gases in the cylinder chamber."
            },
            {
                "negative": "Encryption scrambles your data so others can't read it.",
                "positive": "The cryptographic algorithm applies a symmetric cipher with a 256-bit key to transform plaintext through multiple rounds of substitution and permutation operations."
            },
            {
                "negative": "The neural network recognizes patterns in the data.",
                "positive": "The deep convolutional architecture extracts hierarchical feature representations through successive layers of learnable filters with ReLU activations and batch normalization."
            }
        ]
