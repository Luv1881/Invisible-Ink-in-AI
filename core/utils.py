# core/utils.py

import torch
import numpy as np
from typing import Dict, Any
import json


def whiten_bits(bits: np.ndarray, seed: int) -> np.ndarray:
    """
    XOR payload bits with a pseudo-random keystream to balance statistics.

    Args:
        bits: Binary numpy array
        seed: Seed for deterministic RNG

    Returns:
        Whitened binary numpy array
    """
    rng = np.random.default_rng(seed)
    mask = rng.integers(0, 2, size=bits.shape, dtype=np.uint8)
    return np.bitwise_xor(bits.astype(np.uint8), mask)


def dewhiten_bits(bits: np.ndarray, seed: int) -> np.ndarray:
    """
    Reverse whitening by regenerating the same keystream.

    Args:
        bits: Whitened binary numpy array
        seed: Seed used during whitening

    Returns:
        Original binary numpy array
    """
    return whiten_bits(bits, seed)  # XOR with same mask reverts

def print_model_info(model: torch.nn.Module) -> None:
    """
    Print model architecture and parameter statistics.

    Args:
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (float32)")


def bytes_to_bits(data: bytes) -> np.ndarray:
    """
    Convert bytes to binary array.

    Args:
        data: Byte string

    Returns:
        Binary numpy array
    """
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """
    Convert binary array to bytes.

    Args:
        bits: Binary numpy array

    Returns:
        Byte string
    """
    return np.packbits(bits).tobytes()


def calculate_capacity(model: torch.nn.Module, threshold: float = 0.015) -> Dict[str, Any]:
    """
    Calculate theoretical embedding capacity of a model.

    Args:
        model: Neural network model
        threshold: Accuracy threshold

    Returns:
        Dictionary with capacity statistics
    """
    total_weights = 0
    embeddable_weights = 0

    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) >= 2:
            layer_size = param.numel()
            total_weights += layer_size
            embeddable_weights += layer_size

    # Theoretical capacity (conservative estimate: 2-12% of parameters)
    min_capacity_bits = int(embeddable_weights * 0.02)
    max_capacity_bits = int(embeddable_weights * 0.12)
    optimal_capacity_bits = int(embeddable_weights * 0.05)

    return {
        'total_weights': total_weights,
        'embeddable_weights': embeddable_weights,
        'min_capacity_bits': min_capacity_bits,
        'max_capacity_bits': max_capacity_bits,
        'optimal_capacity_bits': optimal_capacity_bits,
        'min_capacity_bytes': min_capacity_bits // 8,
        'max_capacity_bytes': max_capacity_bits // 8,
        'optimal_capacity_bytes': optimal_capacity_bits // 8
    }


def serialize_metadata(metadata: Dict, output_path: str) -> None:
    """
    Serialize embedding metadata to JSON and pickle.

    Args:
        metadata: Embedding metadata dictionary
        output_path: Base path for output files
    """
    import pickle

    # Extract embedding_map (contains non-JSON-serializable tuples)
    embedding_map = metadata.pop('embedding_map', None)

    # Save JSON-serializable metadata
    with open(output_path + '.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save embedding_map separately
    if embedding_map is not None:
        with open(output_path + '.map', 'wb') as f:
            pickle.dump(embedding_map, f)


def load_metadata(base_path: str) -> Dict:
    """
    Load embedding metadata from JSON and pickle files.

    Args:
        base_path: Base path to metadata files

    Returns:
        Complete metadata dictionary
    """
    import pickle

    # Load JSON metadata
    with open(base_path + '.json', 'r') as f:
        metadata = json.load(f)

    # Load embedding_map
    try:
        with open(base_path + '.map', 'rb') as f:
            metadata['embedding_map'] = pickle.load(f)
    except FileNotFoundError:
        print("Warning: No embedding map found")

    return metadata


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

    # Make PyTorch deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
