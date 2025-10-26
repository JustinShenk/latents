"""Quick demo of loading and using pre-trained GPT-2 temporal steering vectors."""

import json
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from latents import TemporalSteering


def load_steering_vectors(path):
    """Load steering vectors from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)

    # Convert back to numpy arrays with integer keys
    steering_vectors = {
        int(layer): np.array(vec)
        for layer, vec in data['layer_vectors'].items()
    }

    print(f"✓ Loaded vectors for {len(steering_vectors)} layers")
    print(f"  Vector dimension: {len(steering_vectors[0])}")
    print(f"  Metadata: {data['metadata']}")

    return steering_vectors


def main():
    # Load model
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Load pre-trained steering vectors
    print("\nLoading steering vectors...")
    steering_vectors = load_steering_vectors('steering_vectors/temporal_steering.json')

    # Initialize steering system
    print("\nInitializing steering system...")
    steering = TemporalSteering(model, tokenizer, steering_vectors)
    print(f"  Active layers: {steering.target_layers}")

    # Test prompt
    prompt = "What should policymakers prioritize to address climate change?"

    print("\n" + "="*80)
    print(f"PROMPT: {prompt}")
    print("="*80)

    # Generate with immediate steering
    print("\n🔥 IMMEDIATE STEERING (-1.0)")
    print("-"*80)
    immediate = steering.generate_with_steering(
        prompt=prompt,
        steering_strength=-1.0,
        temperature=0.7,
        max_length=100
    )
    print(immediate)

    # Generate with neutral
    print("\n⚖️  NEUTRAL STEERING (0.0)")
    print("-"*80)
    neutral = steering.generate_with_steering(
        prompt=prompt,
        steering_strength=0.0,
        temperature=0.7,
        max_length=100
    )
    print(neutral)

    # Generate with long-term steering
    print("\n🌱 LONG-TERM STEERING (+1.0)")
    print("-"*80)
    long_term = steering.generate_with_steering(
        prompt=prompt,
        steering_strength=1.0,
        temperature=0.7,
        max_length=100
    )
    print(long_term)

    print("\n" + "="*80)
    print("✓ Demo complete!")


if __name__ == "__main__":
    main()
