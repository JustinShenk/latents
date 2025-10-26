"""
Abstractness steering dimension.

Controls the level of abstraction in model outputs from concrete/specific
to abstract/general concepts.
"""

from typing import Tuple
from ..core import SteeringVector, register_steering


@register_steering("abstractness")
class AbstractnessSteeringVector(SteeringVector):
    """
    Abstractness steering: concrete/specific ↔ abstract/general.

    Strength interpretation:
        -1.0: Maximum concrete/specific (examples, details)
         0.0: Balanced (mix of concrete and abstract)
        +1.0: Maximum abstract/general (principles, concepts)
    """

    def get_dimension_name(self) -> str:
        return "abstractness"

    def get_strength_range(self) -> Tuple[float, float]:
        return (-1.0, 1.0)

    def interpret_strength(self, strength: float) -> str:
        """
        Interpret steering strength as abstractness level.

        Args:
            strength: Value between -1.0 and 1.0

        Returns:
            Human-readable interpretation
        """
        if strength <= -0.75:
            return "very concrete/detailed examples"
        elif strength <= -0.25:
            return "specific/practical focus"
        elif strength <= 0.25:
            return "balanced concrete-abstract"
        elif strength <= 0.75:
            return "conceptual/general principles"
        else:
            return "highly abstract/theoretical"

    @classmethod
    def get_default_prompt_pairs(cls):
        """
        Return default contrastive prompt pairs for abstractness extraction.

        These pairs contrast concrete examples vs abstract concepts.
        """
        return [
            {
                "negative": "Apple released the iPhone 15 on September 22, 2023, featuring a 6.1-inch display and 48MP camera.",
                "positive": "Product launches involve strategic timing, competitive positioning, and value proposition communication."
            },
            {
                "negative": "Sara went to the grocery store at 3pm to buy milk, eggs, and bread for dinner.",
                "positive": "Daily routines involve planning, resource acquisition, and fulfillment of needs."
            },
            {
                "negative": "The Python script crashed on line 47 with a KeyError when accessing the 'user_id' dictionary key.",
                "positive": "Software failures arise from incorrect assumptions about state and inadequate error handling."
            },
            {
                "negative": "Tesla's Model 3 costs $40,000, has 272 miles of range, and accelerates 0-60 in 5.8 seconds.",
                "positive": "Transportation technologies balance cost, performance, and sustainability tradeoffs."
            },
            {
                "negative": "On Monday, the team held a standup meeting at 9am where each member shared their progress.",
                "positive": "Organizational coordination mechanisms facilitate information sharing and alignment."
            },
            {
                "negative": "The recipe calls for 2 cups flour, 1 cup sugar, 3 eggs, and baking at 350°F for 45 minutes.",
                "positive": "Culinary processes transform raw ingredients through controlled chemical reactions."
            },
            {
                "negative": "Amazon Prime increased its annual fee from $119 to $139 in February 2022.",
                "positive": "Pricing strategies reflect value perception, competitive dynamics, and cost structures."
            },
            {
                "negative": "The patient received 500mg of amoxicillin three times daily for a sinus infection.",
                "positive": "Medical interventions target pathophysiological mechanisms to restore homeostasis."
            }
        ]
