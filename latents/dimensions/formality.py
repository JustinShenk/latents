"""
Formality steering dimension.

Controls the formality level of model outputs from casual/colloquial
to formal/professional language.
"""

from typing import Tuple
from ..core import SteeringVector, register_steering


@register_steering("formality")
class FormalitySteeringVector(SteeringVector):
    """
    Formality steering: casual/colloquial ↔ formal/professional.

    Strength interpretation:
        -1.0: Maximum casual/colloquial style
         0.0: Balanced/neutral
        +1.0: Maximum formal/professional style
    """

    def get_dimension_name(self) -> str:
        return "formality"

    def get_strength_range(self) -> Tuple[float, float]:
        return (-1.0, 1.0)

    def interpret_strength(self, strength: float) -> str:
        """
        Interpret steering strength as formality level.

        Args:
            strength: Value between -1.0 and 1.0

        Returns:
            Human-readable interpretation
        """
        if strength <= -0.75:
            return "very casual/slang"
        elif strength <= -0.25:
            return "conversational/informal"
        elif strength <= 0.25:
            return "balanced/neutral"
        elif strength <= 0.75:
            return "professional/polished"
        else:
            return "highly formal/academic"

    @classmethod
    def get_default_prompt_pairs(cls):
        """
        Return default contrastive prompt pairs for formality extraction.

        These pairs contrast casual vs formal language styles.
        """
        return [
            {
                "negative": "Hey, what's up with the weather today?",
                "positive": "Could you please provide information about current meteorological conditions?"
            },
            {
                "negative": "That's a pretty cool idea, we should totally do it!",
                "positive": "That is an excellent proposal which merits serious consideration for implementation."
            },
            {
                "negative": "Sorry, can't make it to the meeting, got stuff to do.",
                "positive": "I regret to inform you that I am unable to attend the meeting due to prior commitments."
            },
            {
                "negative": "The project's going great, we're making good progress!",
                "positive": "The project is proceeding satisfactorily, with substantial advancement toward our objectives."
            },
            {
                "negative": "Just wanted to check in and see how things are going.",
                "positive": "I am writing to inquire about the current status of the matter."
            },
            {
                "negative": "Thanks a bunch for your help with this!",
                "positive": "I would like to express my sincere gratitude for your valuable assistance."
            },
            {
                "negative": "We need to figure out what went wrong here.",
                "positive": "It is necessary to conduct a thorough analysis to identify the root cause of this issue."
            },
            {
                "negative": "Can you give me the lowdown on what happened?",
                "positive": "Could you please provide a comprehensive summary of the events?"
            }
        ]
