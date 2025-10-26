"""Dataset generation for temporal grounding experiments."""

import json
import random
import os
import sys
from openai import OpenAI

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import save_checkpoint


def generate_sanity_check_prompts(api_key):
    """
    Generate 50 prompt pairs across 5 domains.
    Each pair: identical task, different temporal horizons.
    """

    client = OpenAI(api_key=api_key)

    domains = [
        "business planning",
        "scientific research",
        "personal projects",
        "technical/engineering",
        "creative/artistic"
    ]

    template = """Generate a concrete, specific task in the domain of {domain} that could meaningfully be planned over different time horizons.

Requirements:
- Task should be realistic and specific
- Should be complex enough that time horizon matters
- Provide ONLY the core task description with NO temporal markers
- One task per response

Example good task: "Launch a new product line for sustainable home goods"
Example bad task: "Do planning" (too vague)

Task:"""

    prompts = []

    for domain in domains:
        print(f"\n{'='*60}")
        print(f"Generating {domain} tasks...")
        print(f"{'='*60}")

        for i in range(10):  # 10 tasks per domain
            try:
                # Generate core task
                response = client.chat.completions.create(
                    model="gpt-4",
                    max_tokens=150,
                    messages=[{
                        "role": "user",
                        "content": template.format(domain=domain)
                    }]
                )

                core_task = response.choices[0].message.content.strip()

                # Remove quotes if present
                core_task = core_task.strip('"').strip("'")

                # Create temporal variants
                short_horizons = ["3 days", "1 week", "2 weeks", "1 month"]
                long_horizons = ["3 years", "5 years", "10 years", "20 years"]

                # Randomly sample one short and one long
                short_h = random.choice(short_horizons)
                long_h = random.choice(long_horizons)

                prompt_pair = {
                    "id": len(prompts),
                    "domain": domain,
                    "core_task": core_task,
                    "short_prompt": f"Develop a {short_h} plan to {core_task.lower()}",
                    "long_prompt": f"Develop a {long_h} plan to {core_task.lower()}",
                    "short_horizon": short_h,
                    "long_horizon": long_h
                }

                prompts.append(prompt_pair)

                print(f"  [{i+1}/10] Generated: {core_task[:60]}...")

            except Exception as e:
                print(f"  ⚠ Error generating task {i+1}: {e}")
                continue

    # Save
    output_file = 'data/sanity_check_prompts.json'
    save_checkpoint(prompts, output_file, also_sync=True)

    print(f"\n{'='*60}")
    print(f"✓ Generated {len(prompts)} prompt pairs")
    print(f"✓ Saved to {output_file}")
    print(f"{'='*60}")

    # Show samples
    print("\n--- Sample Prompts ---")
    for i in range(min(3, len(prompts))):
        p = prompts[i]
        print(f"\nPair {i}:")
        print(f"  Domain: {p['domain']}")
        print(f"  Short: {p['short_prompt']}")
        print(f"  Long: {p['long_prompt']}")

    return prompts


def generate_full_dataset(api_key, target_pairs=300):
    """
    Generate full dataset with 300 prompt pairs.
    Includes length matching and quality checks.
    """

    client = OpenAI(api_key=api_key)

    domains = [
        "business planning",
        "scientific research",
        "personal projects",
        "technical/engineering",
        "creative/artistic"
    ]

    pairs_per_domain = target_pairs // len(domains)

    template = """Generate a concrete, specific task in the domain of {domain} that could meaningfully be planned over different time horizons.

Requirements:
- Task should be realistic and specific
- Should be complex enough that time horizon matters
- Provide ONLY the core task description with NO temporal markers
- One task per response

Task:"""

    def match_prompt_length(short_prompt, long_prompt, tolerance=15):
        """Ensure prompts are within tolerance tokens of each other."""
        short_tokens = len(short_prompt.split())
        long_tokens = len(long_prompt.split())
        return abs(short_tokens - long_tokens) <= tolerance

    prompts = []

    for domain in domains:
        print(f"\n{'='*60}")
        print(f"Generating {domain} tasks (target: {pairs_per_domain})...")
        print(f"{'='*60}")

        domain_prompts = 0
        attempts = 0
        max_attempts = pairs_per_domain * 3

        while domain_prompts < pairs_per_domain and attempts < max_attempts:
            attempts += 1

            try:
                # Generate core task
                response = client.chat.completions.create(
                    model="gpt-4",
                    max_tokens=150,
                    messages=[{
                        "role": "user",
                        "content": template.format(domain=domain)
                    }]
                )

                core_task = response.choices[0].message.content.strip().strip('"').strip("'")

                # Create temporal variants
                short_horizons = ["3 days", "1 week", "2 weeks", "1 month", "3 months"]
                long_horizons = ["2 years", "3 years", "5 years", "10 years", "20 years"]

                short_h = random.choice(short_horizons)
                long_h = random.choice(long_horizons)

                short_prompt = f"Develop a {short_h} plan to {core_task.lower()}"
                long_prompt = f"Develop a {long_h} plan to {core_task.lower()}"

                # Quality check: length matching
                if not match_prompt_length(short_prompt, long_prompt):
                    continue

                prompt_pair = {
                    "id": len(prompts),
                    "domain": domain,
                    "core_task": core_task,
                    "short_prompt": short_prompt,
                    "long_prompt": long_prompt,
                    "short_horizon": short_h,
                    "long_horizon": long_h
                }

                prompts.append(prompt_pair)
                domain_prompts += 1

                if domain_prompts % 10 == 0:
                    print(f"  Progress: {domain_prompts}/{pairs_per_domain}")

            except Exception as e:
                print(f"  ⚠ Error on attempt {attempts}: {e}")
                continue

        print(f"  ✓ Completed: {domain_prompts} pairs (attempts: {attempts})")

    # Save
    output_file = 'data/raw_prompts.json'
    save_checkpoint(prompts, output_file, also_sync=True)

    print(f"\n{'='*60}")
    print(f"✓ Generated {len(prompts)} prompt pairs")
    print(f"✓ Saved to {output_file}")
    print(f"{'='*60}")

    return prompts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                       choices=['sanity', 'full'],
                       help='Generation mode: sanity check or full dataset')
    parser.add_argument('--api-key', type=str,
                       default=os.environ.get('OPENAI_API_KEY'),
                       help='OpenAI API key')

    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: OpenAI API key not provided!")
        print("Set OPENAI_API_KEY environment variable or pass --api-key")
        sys.exit(1)

    # Change to project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if args.mode == 'sanity':
        generate_sanity_check_prompts(args.api_key)
    elif args.mode == 'full':
        generate_full_dataset(args.api_key)
