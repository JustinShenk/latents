"""Command-line interface for temporal steering."""

import argparse
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Temporal Steering: Control GPT-2's temporal scope"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract steering vectors command
    extract_parser = subparsers.add_parser("extract", help="Extract steering vectors")
    extract_parser.add_argument("--pairs", required=True, help="Path to prompt pairs JSON")
    extract_parser.add_argument("--output", default="steering_vectors/temporal_steering.json",
                               help="Output file for steering vectors")
    extract_parser.add_argument("--model", default="gpt2", help="Model name")
    extract_parser.add_argument("--layers", help="Comma-separated layer indices")
    extract_parser.add_argument("--max-pairs", type=int, help="Maximum pairs to use")

    # Launch demo server command
    demo_parser = subparsers.add_parser("demo", help="Launch interactive web demo")
    demo_parser.add_argument("--steering", required=True, help="Path to steering vectors")
    demo_parser.add_argument("--model", default="gpt2", help="Model name")
    demo_parser.add_argument("--port", type=int, default=8080, help="Server port")
    demo_parser.add_argument("--layers", help="Comma-separated layer indices")

    args = parser.parse_args()

    if args.command == "extract":
        from latents.extract_steering_vectors import main as extract_main
        import sys
        sys.argv = [
            "extract_steering_vectors.py",
            "--pairs", args.pairs,
            "--output", args.output,
            "--model", args.model
        ]
        if args.layers:
            sys.argv.extend(["--layers", args.layers])
        if args.max_pairs:
            sys.argv.extend(["--max-pairs", str(args.max_pairs)])
        extract_main()

    elif args.command == "demo":
        from latents.temporal_steering_demo import app, model, tokenizer, steering_system
        import json
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        import numpy as np
        from latents import TemporalSteering

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
        model_obj = GPT2LMHeadModel.from_pretrained(args.model)
        tokenizer_obj = GPT2Tokenizer.from_pretrained(args.model)
        tokenizer_obj.pad_token = tokenizer_obj.eos_token
        model_obj.eval()

        # Load steering vectors
        print(f"Loading steering vectors from {args.steering}...")
        with open(args.steering, 'r') as f:
            data = json.load(f)
        steering_vectors = {
            int(layer): np.array(vec)
            for layer, vec in data['layer_vectors'].items()
        }
        print(f"Loaded vectors for {len(steering_vectors)} layers")

        # Parse target layers
        target_layers = None
        if args.layers:
            target_layers = [int(x) for x in args.layers.split(',')]

        # Initialize steering system
        global steering_system
        steering_system = TemporalSteering(model_obj, tokenizer_obj, steering_vectors, target_layers)

        print(f"\n✓ Server starting on http://localhost:{args.port}")
        print("Press Ctrl+C to stop")
        print()

        app.run(host='0.0.0.0', port=args.port, debug=True)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
