# core/extractor.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from core import utils

class NeuralExtractor:
    """
    Extracts embedded secrets from watermarked neural network weights.
    Supports partial corruption recovery and confidence scoring.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def extract(self,
                watermarked_model: nn.Module,
                metadata: Dict) -> Tuple[bytes, Dict]:
        """
        Extract payload from watermarked model.

        Workflow:
        1. Locate perturbed weights using metadata
        2. Decode bit values from weight perturbations
        3. Apply error correction decoding
        4. Compute per-bit confidence scores
        5. Return recovered payload and statistics

        Args:
            watermarked_model: Model with embedded payload
            metadata: Embedding metadata from embedder

        Returns:
            (recovered_payload_bytes, extraction_statistics)
        """
        print("=" * 80)
        print("NEURAL NETWORK EXTRACTION ENGINE")
        print("=" * 80)

        embedding_map = metadata['embedding_map']
        original_weights = metadata.get('original_weights', {})
        payload_size = metadata['payload_size_bits']
        redundancy_factor = int(metadata.get('redundancy_factor', 1))
        qim_step_multiplier = metadata.get('qim_step_multiplier', 0.04)
        embedding_seed = metadata.get('embedding_seed')
        whitening_seed = metadata.get('whitening_seed')
        layer_stats = metadata.get('layer_stats', {})

        if embedding_seed is None:
            raise ValueError("Metadata missing 'embedding_seed'; cannot reconstruct dithers.")
        if whitening_seed is None:
            raise ValueError("Metadata missing 'whitening_seed'; cannot reverse whitening.")

        # Handle adaptive redundancy if present
        bit_redundancies = metadata.get('bit_redundancies', None)
        if bit_redundancies is not None:
            expected_votes = sum(bit_redundancies[:payload_size])
            print(f"Using adaptive redundancy: {len(bit_redundancies)} bits with variable redundancy")
        else:
            expected_votes = payload_size * redundancy_factor
            bit_redundancies = [redundancy_factor] * payload_size

        if expected_votes == 0:
            return b"", {
                'extracted_bits': 0,
                'mean_confidence': 0.0,
                'min_confidence': 0.0,
                'success': True,
                'bit_confidences': []
            }

        if expected_votes > len(embedding_map):
            raise ValueError(
                f"Embedding map has insufficient entries ({len(embedding_map)}) "
                f"for payload size {payload_size} with redundancy x{redundancy_factor}."
            )

        usable_embedding_map = embedding_map[:expected_votes]
        rng = np.random.default_rng(int(embedding_seed))

        print(f"Extracting {payload_size} bits from {len(usable_embedding_map)} weight locations")
        print(f"Using original weights: {len(original_weights) > 0}")

        # Extract raw bits
        extracted_bits = []
        bit_confidences = []

        watermarked_model.to(self.device)
        watermarked_model.eval()

        current_bit_votes = []
        current_confidences = []
        current_bit_idx = 0

        named_params = dict(watermarked_model.named_parameters())

        for layer_name, weight_idx, original_sensitivity in usable_embedding_map:
            if layer_name not in named_params:
                continue

            layer_param = named_params[layer_name]
            flat_tensor = layer_param.data.view(-1)
            current_weight = float(flat_tensor[weight_idx].item())

            layer_std = layer_stats.get(layer_name)
            if layer_std is None:
                layer_std = float(layer_param.data.std().item() + 1e-12)

            step = max(qim_step_multiplier * layer_std, 1e-8)
            dither = rng.uniform(-0.5 * step, 0.5 * step)

            scaled = (current_weight + dither) / step
            quantized = int(np.floor(scaled + 0.5))
            bit_value = quantized & 1

            distance = abs(scaled - quantized)
            confidence = float(max(0.0, 1.0 - min(distance, 0.5) / 0.5))

            current_bit_votes.append(bit_value)
            current_confidences.append(confidence)

            if original_weights:
                original_weight = original_weights.get((layer_name, weight_idx))
                if original_weight is not None:
                    weight_diff = current_weight - original_weight
                    heuristic_conf = abs(weight_diff) / (abs(original_weight) + 1e-10)
                    heuristic_conf = min(heuristic_conf * 100, 1.0)
                    current_confidences[-1] = max(current_confidences[-1], heuristic_conf)

            # Use adaptive redundancy if available
            current_redundancy = bit_redundancies[current_bit_idx] if current_bit_idx < len(bit_redundancies) else redundancy_factor

            if len(current_bit_votes) == current_redundancy:
                # Soft-decision decoding: confidence-weighted majority voting (Phase 3)
                # Instead of simple majority, weight each vote by its confidence
                weighted_sum = sum(vote * conf for vote, conf in zip(current_bit_votes, current_confidences))
                total_confidence = sum(current_confidences)

                if total_confidence > 0:
                    # Weighted average: > 0.5 means more 1s than 0s (weighted by confidence)
                    final_bit = 1 if (weighted_sum / total_confidence) > 0.5 else 0
                else:
                    # Fallback to simple majority if no confidence
                    final_bit = 1 if sum(current_bit_votes) > current_redundancy / 2 else 0

                avg_confidence = float(np.mean(current_confidences))

                extracted_bits.append(final_bit)
                bit_confidences.append(avg_confidence)

                current_bit_votes = []
                current_confidences = []
                current_bit_idx += 1

                if current_bit_idx >= payload_size:
                    break

        if len(extracted_bits) < payload_size:
            padding = [0] * (payload_size - len(extracted_bits))
            extracted_bits.extend(padding)

        extracted_bits_array = np.array(extracted_bits[:payload_size], dtype=np.uint8)
        dewhitened_bits = utils.dewhiten_bits(extracted_bits_array, int(whitening_seed))
        recovered_bytes = np.packbits(dewhitened_bits).tobytes()

        # Compute statistics
        if bit_confidences:
            mean_confidence = float(np.mean(bit_confidences))
            min_confidence = float(np.min(bit_confidences))
        else:
            mean_confidence = 0.0
            min_confidence = 0.0

        stats = {
            'extracted_bits': len(extracted_bits),
            'mean_confidence': mean_confidence,
            'min_confidence': min_confidence,
            'success': mean_confidence >= 0.90,
            'bit_confidences': bit_confidences
        }

        print(f"\nExtraction complete:")
        print(f"  Bits recovered: {len(extracted_bits)}/{payload_size}")
        print(f"  Mean confidence: {mean_confidence:.4f}")
        print(f"  Min confidence: {min_confidence:.4f}")
        print(f"  Success: {stats['success']}")
        print("=" * 80)

        return recovered_bytes, stats

    def compute_survival_rate(self,
                              original_payload: bytes,
                              extracted_payload: bytes) -> float:
        """
        Compute bit-level survival rate.

        Args:
            original_payload: Original embedded payload
            extracted_payload: Recovered payload after attack

        Returns:
            Survival rate (0.0 to 1.0)
        """
        original_bits = np.unpackbits(np.frombuffer(original_payload, dtype=np.uint8))
        extracted_bits = np.unpackbits(np.frombuffer(extracted_payload, dtype=np.uint8))

        # Handle length mismatch
        min_len = min(len(original_bits), len(extracted_bits))
        original_bits = original_bits[:min_len]
        extracted_bits = extracted_bits[:min_len]

        matches = np.sum(original_bits == extracted_bits)
        survival_rate = matches / len(original_bits)

        return survival_rate
