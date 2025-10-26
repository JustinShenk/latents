"""Generate control datasets for robustness testing."""

import json
import re
import os
import sys
import random
from openai import OpenAI

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.tools.utils import save_checkpoint


def ablate_temporal_keywords(prompt):
    """
    Replace temporal markers with generic placeholders.

    Examples:
        "3 days" -> "timeframe A"
        "5 years" -> "timeframe B"
        "short-term" -> "scope X"
    """

    # Specific time periods
    prompt = re.sub(r'\b\d+\s*(day|days)\b', 'period A', prompt, flags=re.IGNORECASE)
    prompt = re.sub(r'\b\d+\s*(week|weeks)\b', 'period B', prompt, flags=re.IGNORECASE)
    prompt = re.sub(r'\b\d+\s*(month|months)\b', 'period C', prompt, flags=re.IGNORECASE)
    prompt = re.sub(r'\b\d+\s*(year|years)\b', 'period D', prompt, flags=re.IGNORECASE)

    # Qualitative terms
    prompt = re.sub(r'\b(short-term|near-term|immediate)\b', 'scope X', prompt, flags=re.IGNORECASE)
    prompt = re.sub(r'\b(long-term|far-term)\b', 'scope Y', prompt, flags=re.IGNORECASE)

    # Other temporal words
    prompt = re.sub(r'\b(decade|century|quarterly|annual)\b', 'duration Z', prompt, flags=re.IGNORECASE)

    return prompt


def generate_keyword_ablation(base_prompts_file, n_samples=50):
    """
    Generate keyword-ablated versions of prompts.

    Tests whether probe relies on temporal keywords or semantic understanding.
    """

    print(f"{'='*60}")
    print("CONTROL 1: Keyword Ablation")
    print(f"{'='*60}\n")

    with open(base_prompts_file, 'r') as f:
        prompts = json.load(f)

    # Use test set prompts for ablation
    test_prompts = prompts[:n_samples]

    ablated_prompts = []

    for pair in test_prompts:
        ablated_pair = {
            'id': pair['id'],
            'domain': pair['domain'],
            'core_task': pair['core_task'],
            'short_prompt': ablate_temporal_keywords(pair['short_prompt']),
            'long_prompt': ablate_temporal_keywords(pair['long_prompt']),
            'short_horizon': 'ablated',
            'long_horizon': 'ablated',
            'original_short': pair['short_prompt'],
            'original_long': pair['long_prompt']
        }
        ablated_prompts.append(ablated_pair)

    output_file = 'data/control_ablated.json'
    save_checkpoint(ablated_prompts, output_file, also_sync=True)

    print(f"Generated {len(ablated_prompts)} ablated prompt pairs")
    print(f"✓ Saved to {output_file}\n")

    # Show example
    print("Example:")
    print(f"  Original: {test_prompts[0]['short_prompt']}")
    print(f"  Ablated:  {ablated_prompts[0]['short_prompt']}")
    print()

    return ablated_prompts


def generate_trap_prompts(api_key, n_samples=50):
    """
    Generate "trap" prompts: long-term keywords but short-term content.

    Tests whether probe is fooled by misleading keywords.
    Good probe should classify by true content, not keywords.
    """

    print(f"{'='*60}")
    print("CONTROL 2: Trap Prompts")
    print(f"{'='*60}\n")

    client = OpenAI(api_key=api_key)

    template = """Generate a planning task that is inherently SHORT-TERM (can only realistically be done in days or weeks), but phrase it using LONG-TERM temporal keywords (decades, years).

The mismatch should be obvious - the task itself doesn't make sense over a long timeframe.

Examples:
- "Develop a 20-year strategic roadmap for making breakfast tomorrow morning"
- "Create a decade-long plan to choose what to wear today"
- "Build a century-spanning framework for sending a single email"

Generate 1 trap prompt (long-term keywords, short-term task):"""

    trap_prompts = []

    for i in range(n_samples):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                max_tokens=100,
                messages=[{"role": "user", "content": template}]
            )

            trap_prompt = response.choices[0].message.content.strip().strip('"').strip("'")

            trap_prompts.append({
                'id': i,
                'prompt': trap_prompt,
                'true_label': 'short',  # Actually short-term despite keywords
                'stated_label': 'long',  # Keywords suggest long-term
                'type': 'trap_long_keywords_short_content'
            })

            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{n_samples} trap prompts")

        except Exception as e:
            print(f"  ⚠ Error on trap prompt {i}: {e}")
            continue

    output_file = 'data/control_traps.json'
    save_checkpoint(trap_prompts, output_file, also_sync=True)

    print(f"\n✓ Generated {len(trap_prompts)} trap prompts")
    print(f"✓ Saved to {output_file}\n")

    # Show examples
    print("Examples:")
    for i in range(min(3, len(trap_prompts))):
        print(f"  {i+1}. {trap_prompts[i]['prompt']}")
    print()

    return trap_prompts


def generate_nonplanning_prompts(api_key, n_samples=50):
    """
    Generate prompts with temporal language but NOT about planning.

    Tests whether temporal representations are specific to planning contexts.

    Examples:
        - "Summarize historical events from the 1960s"
        - "Describe geological changes over millions of years"
    """

    print(f"{'='*60}")
    print("CONTROL 3: Non-Planning Temporal Contexts")
    print(f"{'='*60}\n")

    client = OpenAI(api_key=api_key)

    contexts = [
        "historical analysis",
        "geological processes",
        "biological evolution",
        "cultural trends",
        "technological development"
    ]

    template = """Generate a prompt about {context} that involves different timescales but is NOT about planning or future actions.

Requirements:
- Should be descriptive or analytical, not prescriptive
- Should mention temporal scales naturally
- Should NOT involve planning, strategy, or future actions

Generate one {context} prompt:"""

    nonplanning_prompts = []

    for i in range(n_samples):
        context = random.choice(contexts)

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                max_tokens=100,
                messages=[{"role": "user", "content": template.format(context=context)}]
            )

            prompt_text = response.choices[0].message.content.strip().strip('"').strip("'")

            # Create short and long versions
            short_markers = ["recent", "the last few years", "current decade"]
            long_markers = ["over centuries", "throughout history", "across millennia"]

            short_prompt = f"Analyze {prompt_text} focusing on {random.choice(short_markers)}"
            long_prompt = f"Analyze {prompt_text} focusing on {random.choice(long_markers)}"

            nonplanning_prompts.append({
                'id': i,
                'context': context,
                'short_prompt': short_prompt,
                'long_prompt': long_prompt,
                'type': 'nonplanning_temporal'
            })

            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{n_samples} non-planning prompts")

        except Exception as e:
            print(f"  ⚠ Error on non-planning prompt {i}: {e}")
            continue

    output_file = 'data/control_nonplanning.json'
    save_checkpoint(nonplanning_prompts, output_file, also_sync=True)

    print(f"\n✓ Generated {len(nonplanning_prompts)} non-planning prompts")
    print(f"✓ Saved to {output_file}\n")

    return nonplanning_prompts


def generate_all_controls(base_prompts_file, api_key):
    """Generate all control datasets."""

    print(f"\n{'='*60}")
    print("GENERATING ALL CONTROL DATASETS")
    print(f"{'='*60}\n")

    # Control 1: Keyword ablation (fast, no API calls)
    ablated = generate_keyword_ablation(base_prompts_file, n_samples=50)

    # Control 2: Trap prompts (requires API)
    traps = generate_trap_prompts(api_key, n_samples=50)

    # Control 3: Non-planning (requires API)
    nonplanning = generate_nonplanning_prompts(api_key, n_samples=50)

    print(f"{'='*60}")
    print("ALL CONTROLS GENERATED")
    print(f"{'='*60}")
    print(f"  Keyword ablated: {len(ablated)}")
    print(f"  Trap prompts: {len(traps)}")
    print(f"  Non-planning: {len(nonplanning)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--base-prompts', type=str, default='data/test_prompts.json',
                       help='Base prompts file for ablation')
    parser.add_argument('--api-key', type=str,
                       default=os.environ.get('OPENAI_API_KEY'),
                       help='OpenAI API key')

    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: OpenAI API key not provided!")
        sys.exit(1)

    # Change to project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    generate_all_controls(args.base_prompts, args.api_key)
