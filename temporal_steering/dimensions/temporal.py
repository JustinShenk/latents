"""
Temporal scope steering dimension.

Controls the temporal horizon of model outputs from immediate/tactical
to long-term/strategic thinking.
"""

from typing import Tuple
from ..core import SteeringVector, register_steering


@register_steering("temporal_scope")
class TemporalSteeringVector(SteeringVector):
    """
    Temporal scope steering: immediate/tactical ↔ long-term/strategic.

    Strength interpretation:
        -1.0: Maximum immediate/tactical focus
         0.0: Balanced/neutral
        +1.0: Maximum long-term/strategic focus
    """

    def get_dimension_name(self) -> str:
        return "temporal_scope"

    def get_strength_range(self) -> Tuple[float, float]:
        return (-1.0, 1.0)

    def interpret_strength(self, strength: float) -> str:
        """
        Interpret steering strength as temporal horizon.

        Args:
            strength: Value between -1.0 and 1.0

        Returns:
            Human-readable interpretation
        """
        if strength <= -0.75:
            return "immediate/urgent (hours-days)"
        elif strength <= -0.25:
            return "near-term/tactical (weeks-months)"
        elif strength <= 0.25:
            return "balanced/moderate (months-years)"
        elif strength <= 0.75:
            return "long-term/strategic (years-decades)"
        else:
            return "very long-term/civilizational (decades-centuries)"

    @classmethod
    def get_default_prompt_pairs(cls):
        """
        Return default contrastive prompt pairs for temporal extraction.

        These pairs contrast immediate vs long-term perspectives.
        """
        return [
            {
                "immediate_prompt": "What should we do about the current economic downturn?",
                "long_term_prompt": "How should we build a resilient economy for future generations?"
            },
            {
                "immediate_prompt": "Fix the bug causing user complaints today.",
                "long_term_prompt": "Design infrastructure to prevent this class of bugs systematically."
            },
            {
                "immediate_prompt": "Respond to the PR crisis right now.",
                "long_term_prompt": "Build organizational culture and values that prevent crises."
            },
            {
                "immediate_prompt": "What diet should I try this month?",
                "long_term_prompt": "How should I develop sustainable health habits for life?"
            },
            {
                "immediate_prompt": "Get short-term funding to survive this quarter.",
                "long_term_prompt": "Build a sustainable business model for decades."
            }
        ]
