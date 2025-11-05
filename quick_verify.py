#!/usr/bin/env python3
"""
Quick verification that the fixes work correctly.
Tests basic embed/extract cycle without attacks.
"""

import torch
import numpy as np
from core.embedder import NeuralEmbedder
from core.extractor import NeuralExtractor
from data.cifar10 import get_cifar10_loaders

print("=" * 80)
print("QUICK VERIFICATION TEST - Fixed Algorithm")
print("=" * 80)
print()

# Test payload
test_message = b"Hello, Neural Steganography! This is a test message."
print(f"Test payload: {test_message.decode()}")
print(f"Size: {len(test_message)} bytes ({len(test_message) * 8} bits)")
print()

# Initialize
print("Initializing components...")
embedder = NeuralEmbedder(device='cpu')  # Use CPU for quick test
extractor = NeuralExtractor(device='cpu')

# Load data (just a small subset for quick test)
print("Loading CIFAR-10 (this may take a moment)...")
_, test_loader = get_cifar10_loaders(batch_size=128, num_workers=0)

# Embed
print("\n" + "-" * 80)
print("STEP 1: EMBEDDING")
print("-" * 80)
watermarked_model, metadata = embedder.embed(test_message, test_loader)

print(f"\n✓ Embedding complete")
print(f"  Capacity used: {metadata['capacity_percent']:.2f}% of model")
print(f"  Baseline accuracy: {metadata['baseline_accuracy']:.4f}")
print(f"  Embedded accuracy: {metadata['embedded_accuracy']:.4f}")
print(f"  Accuracy drop: {metadata['accuracy_drop']:.4f}")
print(f"  Original weights stored: {len(metadata['original_weights'])}")

# Extract (no attack)
print("\n" + "-" * 80)
print("STEP 2: EXTRACTION (No Attack)")
print("-" * 80)
recovered_payload, stats = extractor.extract(watermarked_model, metadata)

print(f"\n✓ Extraction complete")
print(f"  Mean confidence: {stats['mean_confidence']:.4f}")
print(f"  Min confidence: {stats['min_confidence']:.4f}")
print(f"  Extraction success: {stats['success']}")

# Verify
survival_rate = extractor.compute_survival_rate(test_message, recovered_payload)
exact_match = (test_message == recovered_payload)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Original:  {test_message.decode(errors='replace')[:60]}")
print(f"Recovered: {recovered_payload.decode(errors='replace')[:60]}")
print()
print(f"Survival Rate:  {survival_rate * 100:.2f}%")
print(f"Exact Match:    {exact_match}")
print(f"Mean Confidence: {stats['mean_confidence']:.4f}")

if exact_match and survival_rate > 0.95:
    print("\n✅ SUCCESS! The fixes are working correctly!")
    print("   - Exact match achieved")
    print("   - Survival rate > 95%")
    print("   - High confidence scores")
    print("\nYou can now run: python test_runner.py")
elif survival_rate > 0.90:
    print("\n⚠️  PARTIAL SUCCESS")
    print("   - High survival rate but not exact match")
    print("   - May need minor adjustments")
else:
    print("\n❌ FAILED - Survival rate too low")
    print("   - Something is still wrong")

print("=" * 80)
