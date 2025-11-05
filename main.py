# main.py

import argparse
import torch
from pathlib import Path
import sys

from core.embedder import NeuralEmbedder
from core.extractor import NeuralExtractor
from data.cifar10 import get_cifar10_loaders

def main():
    parser = argparse.ArgumentParser(
        description='Neural Network Steganography Framework'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Embed secret into model')
    embed_parser.add_argument('--model', default='resnet18', help='Model architecture')
    embed_parser.add_argument('--secret', required=True, help='Path to secret file')
    embed_parser.add_argument('--output', required=True, help='Path to save watermarked model')
    embed_parser.add_argument('--metadata', required=True, help='Path to save metadata')

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract secret from model')
    extract_parser.add_argument('--model', required=True, help='Path to watermarked model')
    extract_parser.add_argument('--metadata', required=True, help='Path to metadata file')
    extract_parser.add_argument('--output', required=True, help='Path to save recovered secret')

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run comprehensive benchmarks')
    benchmark_parser.add_argument('--output-dir', default='./results', help='Results directory')

    args = parser.parse_args()

    if args.command == 'embed':
        # Load secret
        with open(args.secret, 'rb') as f:
            payload = f.read()

        # Embed
        embedder = NeuralEmbedder(model_name=args.model)
        _, test_loader = get_cifar10_loaders()

        watermarked_model, metadata = embedder.embed(payload, test_loader)

        # Save
        torch.save(watermarked_model.state_dict(), args.output)
        import json
        import pickle

        # Save embedding_map and original_weights separately as pickle since they contain tuples/dicts
        embedding_map = metadata.pop('embedding_map')
        original_weights = metadata.pop('original_weights', {})

        with open(args.metadata + '.map', 'wb') as f:
            pickle.dump({'embedding_map': embedding_map, 'original_weights': original_weights}, f)

        # Save rest as JSON
        with open(args.metadata, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nWatermarked model saved to: {args.output}")
        print(f"Metadata saved to: {args.metadata}")

    elif args.command == 'extract':
        # Load model and metadata
        import json
        import pickle
        import torchvision.models as models

        # Load metadata
        with open(args.metadata, 'r') as f:
            metadata = json.load(f)

        # Load embedding map and original weights
        with open(args.metadata + '.map', 'rb') as f:
            pickled_data = pickle.load(f)
            metadata['embedding_map'] = pickled_data['embedding_map']
            metadata['original_weights'] = pickled_data.get('original_weights', {})

        # Load model
        model = models.resnet18()
        model.load_state_dict(torch.load(args.model))

        # Extract
        extractor = NeuralExtractor()
        recovered_payload, stats = extractor.extract(model, metadata)

        # Save
        with open(args.output, 'wb') as f:
            f.write(recovered_payload)

        print(f"\nRecovered secret saved to: {args.output}")
        print(f"Mean confidence: {stats['mean_confidence']:.4f}")
        print(f"Extraction success: {stats['success']}")

    elif args.command == 'benchmark':
        from experiments.run_benchmarks import BenchmarkOrchestrator
        orchestrator = BenchmarkOrchestrator(output_dir=args.output_dir)
        orchestrator.run_comprehensive_benchmarks()

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
