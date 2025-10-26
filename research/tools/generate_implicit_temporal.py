"""
Generate prompts with IMPLICIT temporal scope (no explicit time mentions).

This tests whether the model understands temporal semantics beyond keyword matching.
"""

import json
import os
from openai import OpenAI

# Implicit temporal markers for different scales
IMMEDIATE_MARKERS = [
    "right now", "immediately", "urgent", "crisis", "today",
    "this moment", "instant", "emergency", "pressing"
]

SHORT_TERM_MARKERS = [
    "this week", "soon", "upcoming", "next steps", "quick wins",
    "rapid", "sprint", "near-term", "current"
]

LONG_TERM_MARKERS = [
    "legacy", "future generations", "sustainable", "transformative",
    "fundamental change", "grandchildren", "lasting impact", "era-defining",
    "civilization", "centuries", "posterity"
]

def generate_implicit_pairs(api_key, n_pairs=50):
    """
    Generate prompt pairs that imply temporal scope without explicit time mentions.

    Strategy:
    - Use contextual cues (urgency, scope, generational impact)
    - Avoid explicit numbers with time units
    - Test semantic temporal understanding
    """

    client = OpenAI(api_key=api_key)

    prompt_template = """Generate a pair of questions/prompts about the SAME topic but with DIFFERENT implied temporal scopes.

Requirements:
1. DO NOT use explicit time periods (no "5 years", "1 week", etc.)
2. Use contextual cues to imply temporal scope:
   - Immediate/urgent: words like "now", "crisis", "urgent", "today"
   - Long-term: words like "legacy", "future generations", "fundamental", "lasting"
3. Keep the core topic IDENTICAL
4. Make prompts natural and realistic

Examples:

Topic: Climate action
Immediate: "What emergency measures can stop the wildfire from spreading to nearby homes?"
Long-term: "How can we redesign cities to be resilient against climate change for future generations?"

Topic: Career development
Immediate: "What skills should I focus on to get promoted this quarter?"
Long-term: "What foundational expertise will define successful careers in the coming decades?"

Topic: Technology adoption
Immediate: "Which tools can boost our team's productivity starting today?"
Long-term: "What technological capabilities will transform how humanity works and lives?"

Now generate ONE pair for the following topic:
Topic: {topic}

Format as JSON:
{{
    "topic": "...",
    "immediate_prompt": "...",
    "long_term_prompt": "..."
}}
"""

    topics = [
        # Personal development
        "personal finance", "health and wellness", "learning new skills",
        "relationships", "work-life balance",

        # Professional
        "career advancement", "business strategy", "team management",
        "product development", "customer satisfaction",

        # Societal
        "education reform", "healthcare policy", "urban planning",
        "economic inequality", "social justice",

        # Global
        "climate change", "technological progress", "space exploration",
        "global cooperation", "sustainable development",

        # Scientific
        "medical research", "artificial intelligence", "renewable energy",
        "biodiversity conservation", "ocean health",

        # Creative
        "artistic movements", "cultural preservation", "storytelling",
        "musical innovation", "architectural design",

        # Governance
        "policy reform", "democratic participation", "international relations",
        "legal frameworks", "institutional change"
    ]

    pairs = []

    for i, topic in enumerate(topics[:n_pairs]):
        print(f"Generating pair {i+1}/{n_pairs}: {topic}")

        try:
            response = client.chat.completions.create(
                model="gpt-4",  # Use GPT-4 for better quality
                messages=[
                    {"role": "system", "content": "You are an expert at creating subtle, contextual prompts that imply temporal scope without explicit time mentions."},
                    {"role": "user", "content": prompt_template.format(topic=topic)}
                ],
                temperature=0.8,
                max_tokens=300
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON (handle potential markdown formatting)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            pair_data = json.loads(content)

            # Verify no explicit time periods
            immediate = pair_data['immediate_prompt'].lower()
            long_term = pair_data['long_term_prompt'].lower()

            # Simple check for explicit time periods
            time_words = ['year', 'month', 'week', 'day', 'hour', 'minute', 'decade', 'century']
            has_time = any(f"{num} {word}" in immediate or f"{num} {word}" in long_term
                          for num in ['1', '2', '3', '5', '10', '20', '50', '100']
                          for word in time_words)

            if has_time:
                print(f"  ⚠ Skipping - contains explicit time period")
                continue

            pairs.append({
                'id': len(pairs),
                'topic': topic,
                'immediate_prompt': pair_data['immediate_prompt'],
                'long_term_prompt': pair_data['long_term_prompt']
            })

            print(f"  ✓ Generated")
            print(f"    Immediate: {pair_data['immediate_prompt'][:80]}...")
            print(f"    Long-term: {pair_data['long_term_prompt'][:80]}...")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    return pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', type=str, required=True)
    parser.add_argument('--n-pairs', type=int, default=50)
    parser.add_argument('--output', type=str, default='data/implicit_temporal_prompts.json')

    args = parser.parse_args()

    print("="*70)
    print("IMPLICIT TEMPORAL SCOPE PROMPT GENERATION")
    print("="*70)
    print(f"Target: {args.n_pairs} pairs")
    print(f"Output: {args.output}")
    print("="*70)
    print()

    pairs = generate_implicit_pairs(args.api_key, args.n_pairs)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(pairs, f, indent=2)

    print()
    print("="*70)
    print(f"✓ Generated {len(pairs)} pairs")
    print(f"✓ Saved to {args.output}")
    print("="*70)
