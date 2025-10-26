"""
Generate adversarial temporal prompts to test if probes do keyword detection or semantic understanding.

Adversarial strategy:
1. Semantic-keyword mismatch: Long-term keywords with immediate context
2. Contradictory markers: Both immediate and long-term cues
3. Irrelevant temporal framing: Temporal markers that don't affect the core question

If probes follow keywords blindly → keyword detection only
If probes get confused/use context → some semantic understanding
"""

import json

def generate_adversarial_examples():
    """Create adversarial temporal prompts with keyword-semantic mismatches."""

    adversarial_pairs = []

    # Type 1: Long-term KEYWORD + Immediate CONTEXT (should confuse keyword detector)
    type1_immediate_context_long_keyword = [
        {
            "type": "keyword_mismatch",
            "semantic_scope": "immediate",
            "keyword_scope": "long_term",
            "prompt": "What legacy should we build for future generations to stop this forest fire from spreading in the next hour?",
            "ground_truth": "immediate",  # Context is urgent/immediate
            "keyword_signal": "long_term",  # "legacy", "future generations"
            "test": "Does probe follow keyword or context?"
        },
        {
            "type": "keyword_mismatch",
            "semantic_scope": "immediate",
            "keyword_scope": "long_term",
            "prompt": "How can we create sustainable change for our grandchildren by fixing this server outage happening right now?",
            "ground_truth": "immediate",
            "keyword_signal": "long_term",
            "test": "Server outage is immediate, but keywords suggest long-term"
        },
        {
            "type": "keyword_mismatch",
            "semantic_scope": "immediate",
            "keyword_scope": "long_term",
            "prompt": "What foundational transformation for coming decades will prevent this patient from dying in the next five minutes?",
            "ground_truth": "immediate",
            "keyword_signal": "long_term",
            "test": "Life-threatening emergency vs long-term keywords"
        },
        {
            "type": "keyword_mismatch",
            "semantic_scope": "immediate",
            "keyword_scope": "long_term",
            "prompt": "What institutional reforms for posterity should we implement to stop the building evacuation alarm sounding now?",
            "ground_truth": "immediate",
            "keyword_signal": "long_term",
            "test": "Evacuation = immediate, posterity = long-term"
        },
        {
            "type": "keyword_mismatch",
            "semantic_scope": "immediate",
            "keyword_scope": "long_term",
            "prompt": "How should we reshape civilization for future eras to prevent today's stock market crash?",
            "ground_truth": "immediate",
            "keyword_signal": "long_term",
            "test": "Stock crash = immediate, civilization/eras = long-term"
        },
    ]

    # Type 2: Short-term KEYWORD + Long-term CONTEXT
    type2_longterm_context_short_keyword = [
        {
            "type": "keyword_mismatch",
            "semantic_scope": "long_term",
            "keyword_scope": "immediate",
            "prompt": "What emergency actions right now will fundamentally transform how future generations address climate change?",
            "ground_truth": "long_term",
            "keyword_signal": "immediate",
            "test": "Fundamental transformation = long, emergency/now = immediate"
        },
        {
            "type": "keyword_mismatch",
            "semantic_scope": "long_term",
            "keyword_scope": "immediate",
            "prompt": "What urgent interventions today will create lasting institutional change across centuries?",
            "ground_truth": "long_term",
            "keyword_signal": "immediate",
            "test": "Centuries = long, urgent/today = immediate"
        },
        {
            "type": "keyword_mismatch",
            "semantic_scope": "long_term",
            "keyword_scope": "immediate",
            "prompt": "What quick fixes this morning will ensure sustainable prosperity for our grandchildren's grandchildren?",
            "ground_truth": "long_term",
            "keyword_signal": "immediate",
            "test": "Grandchildren's grandchildren = long, quick/morning = immediate"
        },
        {
            "type": "keyword_mismatch",
            "semantic_scope": "long_term",
            "keyword_scope": "immediate",
            "prompt": "What immediate steps will build the foundational infrastructure defining the next era of human civilization?",
            "ground_truth": "long_term",
            "keyword_signal": "immediate",
            "test": "Next era/civilization = long, immediate = immediate"
        },
        {
            "type": "keyword_mismatch",
            "semantic_scope": "long_term",
            "keyword_scope": "immediate",
            "prompt": "What can we do in the next ten minutes to create generational wealth for posterity?",
            "ground_truth": "long_term",
            "keyword_signal": "immediate",
            "test": "Generational/posterity = long, ten minutes = immediate"
        },
    ]

    # Type 3: Both temporal markers present (contradictory)
    type3_contradictory = [
        {
            "type": "contradictory",
            "semantic_scope": "ambiguous",
            "keyword_scope": "both",
            "prompt": "In the long run, what should we do immediately about climate change?",
            "ground_truth": "ambiguous",
            "keyword_signal": "both",
            "test": "Both 'long run' and 'immediately' present"
        },
        {
            "type": "contradictory",
            "semantic_scope": "ambiguous",
            "keyword_scope": "both",
            "prompt": "For future generations, what urgent actions do we need right now?",
            "ground_truth": "ambiguous",
            "keyword_signal": "both",
            "test": "Future generations + urgent + right now"
        },
        {
            "type": "contradictory",
            "semantic_scope": "ambiguous",
            "keyword_scope": "both",
            "prompt": "What are the immediate implications of decisions that will affect our grandchildren?",
            "ground_truth": "ambiguous",
            "keyword_signal": "both",
            "test": "Immediate + grandchildren"
        },
        {
            "type": "contradictory",
            "semantic_scope": "ambiguous",
            "keyword_scope": "both",
            "prompt": "How can today's emergency measures create sustainable change for coming decades?",
            "ground_truth": "ambiguous",
            "keyword_signal": "both",
            "test": "Today/emergency + sustainable/decades"
        },
        {
            "type": "contradictory",
            "semantic_scope": "ambiguous",
            "keyword_scope": "both",
            "prompt": "What quick wins this week will shape the next era of human development?",
            "ground_truth": "ambiguous",
            "keyword_signal": "both",
            "test": "Quick/this week + next era"
        },
    ]

    # Type 4: Semantically nonsensical (should be rejected by good understanding)
    type4_nonsensical = [
        {
            "type": "nonsensical",
            "semantic_scope": "nonsensical",
            "keyword_scope": "long_term",
            "prompt": "What legacy for future generations should we build by sending this email before lunch?",
            "ground_truth": "nonsensical",
            "keyword_signal": "long_term",
            "test": "Email before lunch cannot create generational legacy"
        },
        {
            "type": "nonsensical",
            "semantic_scope": "nonsensical",
            "keyword_scope": "immediate",
            "prompt": "What should we do in the next thirty seconds to fundamentally reshape civilization?",
            "ground_truth": "nonsensical",
            "keyword_signal": "immediate",
            "test": "30 seconds cannot reshape civilization"
        },
        {
            "type": "nonsensical",
            "semantic_scope": "nonsensical",
            "keyword_scope": "long_term",
            "prompt": "How can making coffee this morning create sustainable transformation for posterity?",
            "ground_truth": "nonsensical",
            "keyword_signal": "both",
            "test": "Making coffee won't affect posterity"
        },
        {
            "type": "nonsensical",
            "semantic_scope": "nonsensical",
            "keyword_scope": "immediate",
            "prompt": "What urgent crisis response is needed for the long-term project of choosing a lunch restaurant?",
            "ground_truth": "nonsensical",
            "keyword_signal": "immediate",
            "test": "Lunch choice doesn't need crisis response"
        },
        {
            "type": "nonsensical",
            "semantic_scope": "nonsensical",
            "keyword_scope": "long_term",
            "prompt": "How will fixing this typo create a legacy for future generations?",
            "ground_truth": "nonsensical",
            "keyword_signal": "long_term",
            "test": "Typo fix won't create legacy"
        },
    ]

    # Combine all adversarial examples
    all_adversarial = (
        type1_immediate_context_long_keyword +
        type2_longterm_context_short_keyword +
        type3_contradictory +
        type4_nonsensical
    )

    # Add IDs
    for i, item in enumerate(all_adversarial):
        item['id'] = i

    return all_adversarial


def create_adversarial_test_pairs():
    """
    Create matched adversarial pairs for probe testing.

    Each adversarial prompt gets a 'natural' counterpart for comparison.
    """

    pairs = []

    # Adversarial + Natural matches
    test_cases = [
        {
            "adversarial": "What legacy should we build for future generations to stop this forest fire from spreading in the next hour?",
            "adversarial_label": "immediate (context)",
            "natural_immediate": "What emergency actions will stop this forest fire from spreading in the next hour?",
            "natural_longterm": "What legacy should we build for future generations regarding wildfire prevention?",
            "test": "Does probe follow keywords (predicts long-term for adversarial) or context (predicts immediate)?"
        },
        {
            "adversarial": "What urgent interventions today will create lasting institutional change across centuries?",
            "adversarial_label": "long_term (context)",
            "natural_immediate": "What urgent interventions today will prevent the current crisis?",
            "natural_longterm": "What institutional changes will create lasting impact across centuries?",
            "test": "Keywords say immediate, context says long-term"
        },
        {
            "adversarial": "In the long run, what should we do immediately about climate change?",
            "adversarial_label": "ambiguous (both)",
            "natural_immediate": "What should we do immediately about climate change?",
            "natural_longterm": "In the long run, what should we do about climate change?",
            "test": "Both markers present - probe should be confused"
        },
    ]

    for i, case in enumerate(test_cases):
        pairs.append({
            'id': i,
            'adversarial_prompt': case['adversarial'],
            'natural_immediate': case['natural_immediate'],
            'natural_longterm': case['natural_longterm'],
            'ground_truth': case['adversarial_label'],
            'test_objective': case['test']
        })

    return pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='data/adversarial_test.json')
    parser.add_argument('--format', choices=['all', 'pairs'], default='all')

    args = parser.parse_args()

    if args.format == 'all':
        adversarial = generate_adversarial_examples()
        output_data = adversarial
    else:
        pairs = create_adversarial_test_pairs()
        output_data = pairs

    # Save
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Generated {len(output_data)} adversarial examples")
    print(f"✓ Saved to {args.output}")

    # Print summary
    if args.format == 'all':
        types = {}
        for item in output_data:
            t = item['type']
            types[t] = types.get(t, 0) + 1
        print("\nBreakdown:")
        for t, count in types.items():
            print(f"  {t}: {count}")
