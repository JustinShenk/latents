"""
Interactive temporal steering demo using Contrastive Activation Addition (CAA).

Provides a web UI with temporal slider to demonstrate real-time steering of GPT-2's
temporal scope from immediate to long-term.
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import json

from .model_adapter import get_model_config


class TemporalSteering:
    """Applies temporal steering vectors during generation.

    Works with any transformer model (GPT-2, LLaMA, Mistral, etc.)
    """

    def __init__(self, model, tokenizer, steering_vectors, target_layers=None):
        self.model = model
        self.tokenizer = tokenizer
        self.steering_vectors = steering_vectors

        # Auto-detect model architecture
        self.model_config = get_model_config(model)

        # Default to recommended layers (last 60% of model)
        if target_layers is None:
            self.target_layers = self.model_config['recommended_layers']
        else:
            self.target_layers = target_layers

        print(f"Model: {self.model_config['model_type']} ({self.model_config['n_layers']} layers)")
        print(f"Steering will be applied to layers: {self.target_layers}")

    def generate_with_steering(self, prompt, steering_strength=0.0, max_length=100, temperature=0.7):
        """
        Generate text with temporal steering applied.

        Args:
            prompt: Input text
            steering_strength: -1.0 (immediate) to +1.0 (long-term)
            max_length: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        # Encode input
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids']

        # Register hooks to add steering vectors
        hooks = []

        def make_steering_hook(layer_idx, strength):
            def hook(module, input, output):
                # output is (hidden_states, ...)
                hidden_states = output[0]

                # Get steering vector for this layer
                if layer_idx in self.steering_vectors:
                    steering_vec = torch.tensor(
                        self.steering_vectors[layer_idx],
                        dtype=hidden_states.dtype,
                        device=hidden_states.device
                    )

                    # Apply steering with strength
                    # Positive strength = toward long-term
                    # Negative strength = toward immediate
                    hidden_states = hidden_states + strength * steering_vec

                return (hidden_states,) + output[1:]

            return hook

        # Register hooks for target layers (model-agnostic)
        layers = self.model_config['layers']
        for layer_idx in self.target_layers:
            hook = layers[layer_idx].register_forward_hook(
                make_steering_hook(layer_idx, steering_strength)
            )
            hooks.append(hook)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Decode
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return generated_text


# Flask app
app = Flask(__name__, template_folder='../templates')
steering_system = None


@app.route('/')
def index():
    """Serve the main UI."""
    return render_template('temporal_steering.html')


@app.route('/generate', methods=['POST'])
def generate():
    """Generate text with temporal steering."""
    data = request.json

    prompt = data.get('prompt', '')
    steering_strength = float(data.get('steering', 0.0))
    temperature = float(data.get('temperature', 0.7))
    max_length = int(data.get('max_length', 100))

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    # Generate
    generated = steering_system.generate_with_steering(
        prompt=prompt,
        steering_strength=steering_strength,
        max_length=max_length,
        temperature=temperature
    )

    # Analyze temporal characteristics (heuristic)
    metrics = analyze_temporal_characteristics(generated)

    return jsonify({
        'generated': generated,
        'metrics': metrics
    })


@app.route('/validation_tasks', methods=['GET'])
def get_validation_tasks():
    """Return validation tasks where temporal scope affects correct answer."""
    tasks = [
        {
            'id': 1,
            'question': 'You have $10,000 to invest. Interest rates are at 1% but will rise to 5% next year. What should you do?',
            'context': 'Financial decision with temporal implications',
            'immediate_answer': 'Wait to invest when rates are higher (better returns)',
            'long_term_answer': 'Invest now and compound over decades (time in market > timing market)'
        },
        {
            'id': 2,
            'question': 'A coal plant can be shut down immediately (job losses) or phased out over 20 years. What should policymakers do?',
            'context': 'Climate policy with temporal trade-offs',
            'immediate_answer': 'Phase out gradually to protect workers and communities',
            'long_term_answer': 'Immediate shutdown with massive renewable investment for future generations'
        },
        {
            'id': 3,
            'question': 'Should we prioritize fixing bugs in the current product or building new infrastructure?',
            'context': 'Software engineering trade-offs',
            'immediate_answer': 'Fix bugs - users are experiencing issues now',
            'long_term_answer': 'Build infrastructure - technical debt will compound'
        }
    ]

    return jsonify({'tasks': tasks})


def analyze_temporal_characteristics(text):
    """
    Heuristic analysis of temporal characteristics in generated text.

    Returns metrics that might correlate with temporal scope.
    """
    text_lower = text.lower()

    # Count temporal markers
    immediate_markers = ['now', 'today', 'immediate', 'urgent', 'quick', 'current', 'short-term']
    long_term_markers = ['future', 'generation', 'legacy', 'sustainable', 'long-term', 'decade', 'century']

    immediate_count = sum(1 for marker in immediate_markers if marker in text_lower)
    long_term_count = sum(1 for marker in long_term_markers if marker in text_lower)

    # Planning horizon (heuristic based on keywords)
    if 'hour' in text_lower or 'minute' in text_lower:
        horizon = 'immediate'
    elif 'day' in text_lower or 'week' in text_lower:
        horizon = 'short-term'
    elif 'month' in text_lower or 'quarter' in text_lower:
        horizon = 'medium-term'
    elif 'year' in text_lower or 'decade' in text_lower:
        horizon = 'long-term'
    else:
        horizon = 'unclear'

    # Intervention type
    if any(word in text_lower for word in ['fix', 'patch', 'quick', 'temporary']):
        intervention = 'tactical'
    elif any(word in text_lower for word in ['transform', 'redesign', 'fundamental', 'systematic']):
        intervention = 'transformative'
    else:
        intervention = 'moderate'

    # Stakeholder scope
    if any(word in text_lower for word in ['i', 'me', 'my', 'personal']):
        stakeholders = 'individual'
    elif any(word in text_lower for word in ['team', 'company', 'organization']):
        stakeholders = 'organizational'
    elif any(word in text_lower for word in ['society', 'world', 'humanity', 'generation']):
        stakeholders = 'societal'
    else:
        stakeholders = 'unclear'

    return {
        'immediate_markers': immediate_count,
        'long_term_markers': long_term_count,
        'planning_horizon': horizon,
        'intervention_type': intervention,
        'stakeholder_scope': stakeholders,
        'temporal_balance': long_term_count - immediate_count
    }


def load_steering_vectors(steering_file):
    """Load steering vectors from JSON."""
    with open(steering_file, 'r') as f:
        data = json.load(f)

    # Convert back to numpy arrays
    steering_vectors = {
        int(layer): np.array(vec)
        for layer, vec in data['layer_vectors'].items()
    }

    return steering_vectors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Temporal steering demo server")
    parser.add_argument('--steering', type=str, required=True,
                       help='Path to steering vectors JSON')
    parser.add_argument('--model', type=str, default='gpt2',
                       help='Model name (gpt2, gpt2-medium, etc.)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Server port')
    parser.add_argument('--layers', type=str, default=None,
                       help='Comma-separated layer indices for steering (default: auto)')

    args = parser.parse_args()

    print("="*70)
    print("TEMPORAL STEERING DEMO")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Steering vectors: {args.steering}")
    print(f"Port: {args.port}")
    print("="*70)
    print()

    # Load model
    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained(args.model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Load steering vectors
    print(f"Loading steering vectors from {args.steering}...")
    steering_vectors = load_steering_vectors(args.steering)
    print(f"Loaded vectors for {len(steering_vectors)} layers")

    # Parse target layers
    target_layers = None
    if args.layers:
        target_layers = [int(x) for x in args.layers.split(',')]

    # Initialize steering system
    steering_system = TemporalSteering(model, tokenizer, steering_vectors, target_layers)

    # Start server
    print(f"\n✓ Server starting on http://localhost:{args.port}")
    print("Press Ctrl+C to stop")
    print()

    app.run(host='0.0.0.0', port=args.port, debug=True)
