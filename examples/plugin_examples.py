"""
Example steering dimension plugins.

These demonstrate how to create custom steering dimensions
using the plugin architecture.
"""

from temporal_steering import SteeringVector, register_steering
from typing import Tuple


@register_steering("optimism")
class OptimismSteering(SteeringVector):
    """
    Optimism/Pessimism dimension: risk-focused ↔ opportunity-focused.

    Strength interpretation:
        -1.0: Maximum pessimism/risk focus
         0.0: Balanced perspective
        +1.0: Maximum optimism/opportunity focus

    Use cases:
        - Financial analysis (bear vs bull perspectives)
        - Strategic planning (risks vs opportunities)
        - Product roadmaps (challenges vs potential)
    """

    def get_dimension_name(self) -> str:
        return "optimism"

    def get_strength_range(self) -> Tuple[float, float]:
        return (-1.0, 1.0)

    def interpret_strength(self, strength: float) -> str:
        if strength <= -0.75:
            return "very pessimistic/defensive"
        elif strength <= -0.25:
            return "cautiously pessimistic"
        elif strength <= 0.25:
            return "balanced/realistic"
        elif strength <= 0.75:
            return "cautiously optimistic"
        else:
            return "very optimistic/ambitious"

    @classmethod
    def get_default_prompt_pairs(cls):
        """Contrastive pairs for extracting optimism vectors."""
        return [
            {
                "negative_prompt": "What could go wrong with this new technology?",
                "positive_prompt": "What opportunities does this new technology create?"
            },
            {
                "negative_prompt": "Why might this startup fail?",
                "positive_prompt": "How could this startup succeed dramatically?"
            },
            {
                "negative_prompt": "What are the risks of expanding internationally?",
                "positive_prompt": "What are the growth opportunities from global expansion?"
            },
            {
                "negative_prompt": "What challenges will we face in the market?",
                "positive_prompt": "What market opportunities can we capture?"
            },
            {
                "negative_prompt": "Why might this product not succeed?",
                "positive_prompt": "How could this product transform the industry?"
            },
        ]


@register_steering("technical_detail")
class TechnicalDetailSteering(SteeringVector):
    """
    Technical detail dimension: high-level ↔ implementation details.

    Strength interpretation:
        -1.0: Maximum high-level/executive summary
         0.0: Moderate detail
        +1.0: Maximum technical/implementation detail

    Use cases:
        - Documentation for different audiences
        - Technical vs executive communication
        - Teaching at different depth levels
    """

    def get_dimension_name(self) -> str:
        return "technical_detail"

    def get_strength_range(self) -> Tuple[float, float]:
        return (-1.0, 1.0)

    def interpret_strength(self, strength: float) -> str:
        if strength <= -0.75:
            return "executive summary/very high-level"
        elif strength <= -0.25:
            return "overview with key concepts"
        elif strength <= 0.25:
            return "balanced technical depth"
        elif strength <= 0.75:
            return "detailed technical explanation"
        else:
            return "implementation-level detail"

    @classmethod
    def get_default_prompt_pairs(cls):
        """Contrastive pairs for technical detail vectors."""
        return [
            {
                "negative_prompt": "Explain cloud computing to a CEO in one paragraph.",
                "positive_prompt": "Explain the technical architecture of cloud computing with implementation details."
            },
            {
                "negative_prompt": "What's the business value of machine learning?",
                "positive_prompt": "Explain how machine learning algorithms work mathematically."
            },
            {
                "negative_prompt": "Give a high-level overview of our system architecture.",
                "positive_prompt": "Describe our system architecture with specific technologies and implementation patterns."
            },
            {
                "negative_prompt": "Summarize the key strategic insights.",
                "positive_prompt": "Provide detailed analysis with specific metrics and data points."
            },
            {
                "negative_prompt": "What's the big picture?",
                "positive_prompt": "What are the specific technical steps to implement this?"
            },
        ]


@register_steering("abstractness")
class AbstractnessSteering(SteeringVector):
    """
    Abstractness dimension: concrete examples ↔ abstract principles.

    Strength interpretation:
        -1.0: Maximum concrete/example-based
         0.0: Mix of examples and principles
        +1.0: Maximum abstract/theoretical

    Use cases:
        - Teaching (examples for beginners, theory for advanced)
        - Communication styles (concrete vs conceptual thinkers)
        - Documentation (tutorials vs reference)
    """

    def get_dimension_name(self) -> str:
        return "abstractness"

    def get_strength_range(self) -> Tuple[float, float]:
        return (-1.0, 1.0)

    def interpret_strength(self, strength: float) -> str:
        if strength <= -0.75:
            return "very concrete/example-heavy"
        elif strength <= -0.25:
            return "mostly concrete with some principles"
        elif strength <= 0.25:
            return "balanced examples and theory"
        elif strength <= 0.75:
            return "mostly abstract with some examples"
        else:
            return "very abstract/theoretical"

    @classmethod
    def get_default_prompt_pairs(cls):
        """Contrastive pairs for abstractness vectors."""
        return [
            {
                "negative_prompt": "Give me a concrete example of leadership.",
                "positive_prompt": "Explain the abstract principles of leadership."
            },
            {
                "negative_prompt": "Show me a specific case study of innovation.",
                "positive_prompt": "Describe the general theory of innovation."
            },
            {
                "negative_prompt": "Tell me a story that illustrates this concept.",
                "positive_prompt": "Define this concept in theoretical terms."
            },
            {
                "negative_prompt": "What's a real-world example of this?",
                "positive_prompt": "What's the underlying pattern or principle?"
            },
            {
                "negative_prompt": "How did this work in practice?",
                "positive_prompt": "What's the conceptual framework?"
            },
        ]


@register_steering("formality")
class FormalitySteering(SteeringVector):
    """
    Formality dimension: casual ↔ formal.

    Strength interpretation:
        -1.0: Maximum casual/conversational
         0.0: Neutral/professional
        +1.0: Maximum formal/ceremonial

    Use cases:
        - Adapting tone for different audiences
        - Business vs casual communication
        - Academic vs popular writing
    """

    def get_dimension_name(self) -> str:
        return "formality"

    def get_strength_range(self) -> Tuple[float, float]:
        return (-1.0, 1.0)

    def interpret_strength(self, strength: float) -> str:
        if strength <= -0.75:
            return "very casual/slang"
        elif strength <= -0.25:
            return "conversational/friendly"
        elif strength <= 0.25:
            return "professional/neutral"
        elif strength <= 0.75:
            return "formal/polished"
        else:
            return "very formal/ceremonial"

    @classmethod
    def get_default_prompt_pairs(cls):
        """Contrastive pairs for formality vectors."""
        return [
            {
                "negative_prompt": "Hey, what's up with the project?",
                "positive_prompt": "I would like to inquire about the status of the project."
            },
            {
                "negative_prompt": "Write a casual email to a coworker.",
                "positive_prompt": "Draft a formal memorandum to senior management."
            },
            {
                "negative_prompt": "Explain this like you're talking to a friend.",
                "positive_prompt": "Present this in a professional business context."
            },
            {
                "negative_prompt": "Give me the quick version.",
                "positive_prompt": "Provide a comprehensive formal analysis."
            },
            {
                "negative_prompt": "What do you think about this idea?",
                "positive_prompt": "Please present your professional assessment of this proposal."
            },
        ]


# Example usage and testing
if __name__ == "__main__":
    from temporal_steering import SteeringFramework, STEERING_REGISTRY
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("=" * 70)
    print("EXAMPLE STEERING PLUGINS")
    print("=" * 70)
    print()

    print("Registered dimensions:")
    for dim in sorted(STEERING_REGISTRY.keys()):
        cls = STEERING_REGISTRY[dim]
        instance = cls({}, {})
        print(f"  {dim:20s} → {instance.interpret_strength(1.0)}")

    print()
    print("=" * 70)
    print()

    # Note: To actually use these plugins, you need to:
    # 1. Extract steering vectors from prompt pairs
    # 2. Save vectors to steering_vectors/{dimension}.json
    # 3. Load with SteeringFramework.load()

    print("To extract vectors for a plugin:")
    print()
    print("  from temporal_steering.extract_steering_vectors import compute_steering_vectors")
    print("  from examples.plugin_examples import OptimismSteering")
    print()
    print("  # Load model")
    print("  model = GPT2LMHeadModel.from_pretrained('gpt2')")
    print("  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')")
    print()
    print("  # Get prompt pairs")
    print("  pairs = OptimismSteering.get_default_prompt_pairs()")
    print()
    print("  # Extract vectors")
    print("  vectors = compute_steering_vectors(model, tokenizer, pairs)")
    print()
    print("  # Save")
    print("  optimism = OptimismSteering(vectors, {'extraction_date': '2024-10-26'})")
    print("  optimism.save_vectors('steering_vectors/optimism.json')")
    print()
