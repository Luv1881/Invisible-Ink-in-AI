# core/embedder.py

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import copy
from typing import Dict, List, Tuple, Optional
from core import utils

class NeuralEmbedder:
    """
    Embeds binary secrets into ResNet-18 weight parameters using
    adaptive capacity optimization and layer-wise sensitivity analysis.
    """

    def __init__(self,
                 model_name: str = 'resnet18',
                 device: str = 'cuda',
                 accuracy_threshold: float = 0.02,  # Increased from 0.015 to allow stronger embedding
                 desired_redundancy: int = 25,  # Increased from 15 to 25 for critical resilience
                 qim_step_multiplier: float = 0.25,  # Increased from 0.15 to 0.25 for 4-bit quantization
                 embedding_seed: Optional[int] = None,
                 whitening_seed: Optional[int] = None,
                 use_ecc: bool = True,
                 adaptive_redundancy: bool = True,  # Enable adaptive redundancy
                 quantization_simulation: bool = True):  # Simulate quantization during embedding
        """
        Initialize the embedding engine with maximum robustness.

        Args:
            model_name: Model architecture ('resnet18', 'resnet50')
            device: 'cuda' or 'cpu'
            accuracy_threshold: Maximum allowed accuracy drop (0.02 = 2%)
            desired_redundancy: Base repetition factor (default 25 for extreme robustness)
            qim_step_multiplier: QIM step size multiplier (0.25 for 4-bit quantization resistance)
            use_ecc: Enable error correction codes
            adaptive_redundancy: Use higher redundancy for vulnerable weights
            quantization_simulation: Test quantization resistance during embedding
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.accuracy_threshold = accuracy_threshold
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.model.to(self.device)
        self.embedding_map = {}
        self.desired_redundancy = max(1, int(desired_redundancy))
        self.qim_step_multiplier = max(qim_step_multiplier, 1e-6)
        self.use_ecc = use_ecc
        self.adaptive_redundancy = adaptive_redundancy
        self.quantization_simulation = quantization_simulation

        # Seeds (resolved per-embed to allow randomness when not provided)
        self._base_embedding_seed = embedding_seed
        self._base_whitening_seed = whitening_seed
        self.embedding_seed: Optional[int] = None
        self.whitening_seed: Optional[int] = None

        # Cache for layer statistics
        self.layer_stats: Dict[str, float] = {}
        self.pruning_scores: Dict[str, np.ndarray] = {}  # Track pruning importance

        print(f"[EMBEDDER] Initialized with:")
        print(f"  - Redundancy: {desired_redundancy}x (adaptive={adaptive_redundancy})")
        print(f"  - Step multiplier: {qim_step_multiplier}x")
        print(f"  - ECC: {use_ecc}")
        print(f"  - Quantization simulation: {quantization_simulation}")

    def _load_model(self, model_name: str) -> nn.Module:
        """
        Load pre-trained model and adapt for CIFAR-10.

        ImageNet models have 1000 output classes, but CIFAR-10 has only 10.
        We replace the final FC layer with a CIFAR-10 compatible layer.
        """
        if model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Adapt final layer for CIFAR-10 (10 classes instead of 1000)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)

        # Initialize the new FC layer with reasonable weights
        nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(model.fc.bias, 0)

        print(f"[MODEL] Adapted {model_name} for CIFAR-10 (10 classes)")
        print(f"[WARNING] Model needs fine-tuning on CIFAR-10 for optimal accuracy")

        model.eval()
        return model

    def finetune_model_for_cifar10(self,
                                    train_loader: torch.utils.data.DataLoader,
                                    test_loader: torch.utils.data.DataLoader,
                                    num_epochs: int = 10,
                                    learning_rate: float = 0.001):
        """
        Fine-tune the adapted model on CIFAR-10 to achieve reasonable accuracy.

        This fixes the model/dataset mismatch by training the new FC layer
        and fine-tuning the backbone on CIFAR-10.

        Args:
            train_loader: CIFAR-10 training data
            test_loader: CIFAR-10 test data
            num_epochs: Number of training epochs
            learning_rate: Learning rate
        """
        print(f"\n[FINE-TUNING] Training model on CIFAR-10 for {num_epochs} epochs...")
        print("This fixes the ImageNet→CIFAR-10 domain gap\n")

        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if batch_idx % 100 == 99:
                    print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}: "
                          f"Loss={running_loss/100:.3f}, Acc={100.*correct/total:.2f}%")
                    running_loss = 0.0

            # Evaluate on test set
            test_acc = self._evaluate_accuracy(self.model, test_loader, num_batches=100)
            print(f"  Epoch {epoch+1}/{num_epochs} - Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

            scheduler.step()

        self.model.eval()
        final_acc = self._evaluate_accuracy(self.model, test_loader, num_batches=100)
        print(f"\n[FINE-TUNING COMPLETE] Final accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
        return final_acc

    @staticmethod
    def _generate_seed() -> int:
        """Generate a high-entropy 64-bit seed."""
        return int(np.random.SeedSequence().entropy)

    def _resolve_seeds(self) -> None:
        """Resolve embedding and whitening seeds for the current session."""
        if self._base_embedding_seed is not None:
            self.embedding_seed = int(self._base_embedding_seed)
        else:
            self.embedding_seed = self._generate_seed()

        if self._base_whitening_seed is not None:
            self.whitening_seed = int(self._base_whitening_seed)
        else:
            self.whitening_seed = self._generate_seed()

    def _compute_layer_stats(self) -> Dict[str, float]:
        """
        Pre-compute layer statistics (standard deviation) used for QIM embedding.

        Returns:
            Mapping from parameter name to standard deviation.
        """
        stats = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:
                stats[name] = float(param.data.std().item() + 1e-12)
        return stats

    def compute_layer_sensitivity(self,
                                   dataloader: torch.utils.data.DataLoader,
                                   num_batches: int = 10) -> Dict[str, np.ndarray]:
        """
        Compute gradient-based sensitivity for each layer.

        Algorithm:
        1. Forward pass on validation data
        2. Compute loss gradients w.r.t. all weights
        3. Take absolute values: sensitivity = |∂Loss/∂W|
        4. Return per-layer sensitivity maps

        Args:
            dataloader: CIFAR-10 validation loader
            num_batches: Number of batches to average over

        Returns:
            Dict mapping layer names to sensitivity arrays
        """
        self.model.train()  # Enable gradient computation
        criterion = nn.CrossEntropyLoss()
        sensitivity_maps = {}

        # Initialize sensitivity accumulators
        for name, param in self.model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:  # Only conv/fc layers
                sensitivity_maps[name] = torch.zeros_like(param.data)

        # Accumulate gradients over batches
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            self.model.zero_grad()
            loss.backward()

            # Accumulate absolute gradients
            for name, param in self.model.named_parameters():
                if name in sensitivity_maps and param.grad is not None:
                    sensitivity_maps[name] += param.grad.abs()

        # Average and convert to numpy
        for name in sensitivity_maps:
            sensitivity_maps[name] = (sensitivity_maps[name] / num_batches).cpu().numpy()

        self.model.eval()
        return sensitivity_maps

    def compute_quantization_robustness(self) -> Dict[str, np.ndarray]:
        """
        Compute quantization robustness score for each weight.

        Weights with larger absolute values are more robust to quantization
        because the relative error from quantization is smaller.

        Robustness score = |weight_value|

        Returns:
            Dict mapping layer names to robustness arrays
        """
        robustness_maps = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:
                # Robustness = absolute value (larger weights survive quantization better)
                robustness_maps[name] = np.abs(param.data.cpu().numpy())

        return robustness_maps

    def compute_pruning_importance(self) -> Dict[str, np.ndarray]:
        """
        Compute pruning importance scores for weights.

        Weights with high importance scores are more likely to be kept during pruning.
        We want to embed in weights that are UNLIKELY to be pruned (high importance).

        For structured pruning (channel-level):
        - Importance = L2 norm of filter/channel

        For unstructured pruning (weight-level):
        - Importance = |weight_value| × gradient magnitude

        Returns:
            Dict mapping layer names to importance scores (higher = more important)
        """
        importance_maps = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad or len(param.shape) < 2:
                continue

            weights = param.data.cpu().numpy()

            # For convolutional layers: compute channel-level importance
            if len(param.shape) == 4:  # Conv2D: [out_channels, in_channels, H, W]
                # L2 norm per output channel
                channel_norms = np.sqrt(np.sum(weights ** 2, axis=(1, 2, 3), keepdims=True))
                # Broadcast to all weights in each channel
                importance = np.broadcast_to(channel_norms, weights.shape)

            # For fully connected layers: use absolute value as importance
            else:
                importance = np.abs(weights)

            importance_maps[name] = importance

        self.pruning_scores = importance_maps
        return importance_maps

    def rank_weights_by_sensitivity(self,
                                     sensitivity_maps: Dict[str, np.ndarray],
                                     use_quantization_aware: bool = True,
                                     use_pruning_aware: bool = True) -> List[Tuple]:
        """
        Rank all weights by multi-factor robustness score.

        Strategy:
        - Low sensitivity: Less impact on model accuracy
        - High quantization robustness: Survives quantization
        - High pruning importance: Won't be pruned

        Combined score = sensitivity / (quantization_robustness × pruning_importance + eps)
        Lower score = better candidate

        Args:
            sensitivity_maps: Layer-wise gradient sensitivity
            use_quantization_aware: Factor in quantization robustness
            use_pruning_aware: Factor in pruning resistance

        Returns:
            List of tuples: (layer_name, weight_flat_index, combined_score)
            Sorted by ascending score (best candidates first)
        """
        all_weights = []

        # Compute robustness maps
        if use_quantization_aware:
            robustness_maps = self.compute_quantization_robustness()
            print("[WEIGHT SELECTION] Quantization-aware: ON")
        else:
            robustness_maps = None
            print("[WEIGHT SELECTION] Quantization-aware: OFF")

        if use_pruning_aware:
            importance_maps = self.compute_pruning_importance()
            print("[WEIGHT SELECTION] Pruning-aware: ON")
        else:
            importance_maps = None
            print("[WEIGHT SELECTION] Pruning-aware: OFF")

        for layer_name, sens_map in sensitivity_maps.items():
            flat_sens = sens_map.flatten()

            # Get quantization robustness
            if robustness_maps and layer_name in robustness_maps:
                flat_robust = robustness_maps[layer_name].flatten()
            else:
                flat_robust = np.ones_like(flat_sens)

            # Get pruning importance
            if importance_maps and layer_name in importance_maps:
                flat_importance = importance_maps[layer_name].flatten()
            else:
                flat_importance = np.ones_like(flat_sens)

            for idx in range(len(flat_sens)):
                sens_val = flat_sens[idx]
                robust_val = flat_robust[idx]
                importance_val = flat_importance[idx]

                # Combined score: sensitivity / (robustness × importance)
                # Lower = better (low sensitivity, high robustness, high importance)
                eps = 1e-8
                combined_score = sens_val / (robust_val * importance_val + eps)

                all_weights.append((layer_name, idx, combined_score))

        # Sort by combined score (ascending - best candidates first)
        all_weights.sort(key=lambda x: x[2])

        print(f"[WEIGHT SELECTION] Ranked {len(all_weights)} weights")
        print(f"                   Best score: {all_weights[0][2]:.2e}")
        print(f"                   Worst score: {all_weights[-1][2]:.2e}")

        return all_weights

    def adaptive_capacity_search(self,
                                  ranked_weights: List[Tuple],
                                  payload_bits: np.ndarray,
                                  dataloader: torch.utils.data.DataLoader,
                                  baseline_accuracy: float) -> Tuple[int, int]:
        """
        Binary search to find maximum capacity within accuracy threshold.

        Algorithm (from Section 4.4):
        1. Initialize capacity bounds [low, high]
        2. While low <= high:
            a. mid = (low + high) / 2
            b. Select top mid fraction of least-sensitive weights
            c. Embed payload with error correction
            d. Evaluate accuracy on validation set
            e. If accuracy >= baseline - threshold: low = mid + 1
            f. Else: high = mid - 1
        3. Return optimal capacity at high

        Args:
            ranked_weights: Weights sorted by sensitivity
            payload_bits: Binary payload to embed
            dataloader: Validation dataloader
            baseline_accuracy: Clean model accuracy

        Returns:
            Tuple of (optimal_capacity, effective_redundancy)
        """
        total_weights = len(ranked_weights)
        payload_size = len(payload_bits)

        if payload_size == 0:
            raise ValueError("Payload is empty - nothing to embed.")

        # Binary search bounds (in number of weights)
        min_required_weights = payload_size  # Minimum: one bit per weight
        low = min_required_weights
        high = min(int(total_weights * 0.12), payload_size * max(5, self.desired_redundancy * 2))
        high = max(high, min_required_weights)

        optimal_capacity = min_required_weights
        optimal_redundancy = 1

        print(f"Binary search for optimal capacity: [{low}, {high}] weights")
        print(f"Payload size: {payload_size} bits")

        while low <= high:
            mid = (low + high) // 2
            print(f"  Testing capacity: {mid} weights ({mid/total_weights*100:.2f}% of model)")

            # Embed at this capacity (don't store originals during search)
            temp_model, _, redundancy, used_weights, _ = self._embed_at_capacity(
                ranked_weights[:mid],
                payload_bits,
                store_originals=False
            )

            # Evaluate accuracy
            accuracy = self._evaluate_accuracy(temp_model, dataloader)
            accuracy_drop = baseline_accuracy - accuracy

            print(f"    Accuracy: {accuracy:.4f} (drop: {accuracy_drop:.4f})")
            print(f"    Effective redundancy: x{redundancy}, weights used: {used_weights}")

            if accuracy_drop <= self.accuracy_threshold:
                optimal_capacity = mid
                optimal_redundancy = redundancy
                low = mid + 1  # Try larger capacity
            else:
                high = mid - 1  # Reduce capacity

        print(f"Optimal capacity: {optimal_capacity} weights ({optimal_capacity/total_weights*100:.2f}%)")
        print(f"Effective redundancy at optimal capacity: x{optimal_redundancy}")
        return optimal_capacity, optimal_redundancy

    def _compute_adaptive_redundancy(self,
                                      selected_weights: List[Tuple],
                                      payload_length: int) -> List[int]:
        """
        Compute adaptive redundancy for each bit based on weight vulnerability.

        Strategy:
        - Weights are pre-sorted by quality (best first)
        - Bits in best 33% of weights: base redundancy (e.g., 20x)
        - Bits in middle 33%: 1.5x base (e.g., 30x)
        - Bits in worst 33%: 2x base (e.g., 40x)

        This ensures vulnerable positions get extra protection.

        Args:
            selected_weights: Sorted weights (best first)
            payload_length: Number of bits to embed

        Returns:
            List of redundancy values for each bit
        """
        if not self.adaptive_redundancy:
            # Uniform redundancy
            return [self.desired_redundancy] * payload_length

        # Calculate vulnerability tiers based on score distribution
        scores = [score for _, _, score in selected_weights]

        # Sort weights into 3 tiers
        tier_size = len(scores) // 3
        tier1_threshold = scores[min(tier_size, len(scores)-1)] if tier_size > 0 else float('inf')
        tier2_threshold = scores[min(tier_size * 2, len(scores)-1)] if tier_size > 0 else float('inf')

        # Assign redundancy per bit based on which tier its weights fall into
        redundancies = []
        base_redundancy = max(15, self.desired_redundancy // 2)  # Lower base for efficiency

        weights_per_bit_estimate = len(selected_weights) // payload_length if payload_length > 0 else 1

        for bit_idx in range(payload_length):
            # Find which weights will be used for this bit
            weight_start = bit_idx * weights_per_bit_estimate
            weight_end = min(weight_start + weights_per_bit_estimate, len(selected_weights))

            if weight_end > len(selected_weights):
                # Use maximum redundancy for bits that might run out of weights
                redundancies.append(self.desired_redundancy * 2)
                continue

            # Average score for this bit's weights
            bit_weights = selected_weights[weight_start:weight_end]
            if not bit_weights:
                redundancies.append(base_redundancy)
                continue

            avg_score = np.mean([score for _, _, score in bit_weights])

            # Assign redundancy based on tier
            if avg_score <= tier1_threshold:
                # Best tier - use base redundancy
                redundancies.append(base_redundancy)
            elif avg_score <= tier2_threshold:
                # Middle tier - use 1.5x redundancy
                redundancies.append(int(base_redundancy * 1.5))
            else:
                # Worst tier - use 2x redundancy
                redundancies.append(base_redundancy * 2)

        avg_redundancy = np.mean(redundancies)
        print(f"[ADAPTIVE REDUNDANCY] Per-bit redundancy: min={min(redundancies)}, "
              f"max={max(redundancies)}, avg={avg_redundancy:.1f}")

        return redundancies

    def _embed_at_capacity(self,
                           selected_weights: List[Tuple],
                           payload_bits: np.ndarray,
                           store_originals: bool = False) -> Tuple[nn.Module, Optional[Dict], int, int]:
        """
        Embed payload into selected weights with adaptive redundancy.

        Embedding scheme:
        - For each bit in payload:
          - Select least-sensitive weight
          - Apply QIM encoding with dithering
          - bit=0: quantized value is even
          - bit=1: quantized value is odd
        - Uses adaptive redundancy: vulnerable positions get higher redundancy

        Args:
            selected_weights: List of (layer_name, weight_idx, sensitivity)
            payload_bits: Binary array to embed
            store_originals: Whether to store original weight values

        Returns:
            (model_copy, original_weights_dict, redundancy_factor, weights_used)
        """
        # Create a deep copy of the model to avoid modifying the original
        model_copy = copy.deepcopy(self.model)
        model_copy.to(self.device)

        payload_length = len(payload_bits)
        if payload_length == 0:
            return model_copy, {} if store_originals else None, 1, 0

        max_possible_redundancy = len(selected_weights) // payload_length
        if max_possible_redundancy == 0:
            raise ValueError(
                f"Not enough weights ({len(selected_weights)}) to embed payload of {payload_length} bits."
            )

        # Compute adaptive redundancy if enabled, otherwise use uniform
        if self.adaptive_redundancy and payload_length > 1:
            bit_redundancies = self._compute_adaptive_redundancy(selected_weights, payload_length)
            # Adjust if not enough weights available
            total_needed = sum(bit_redundancies)
            if total_needed > len(selected_weights):
                scale_factor = len(selected_weights) / total_needed
                bit_redundancies = [max(1, int(r * scale_factor)) for r in bit_redundancies]
            redundancy_factor = int(np.mean(bit_redundancies))
        else:
            redundancy_factor = min(self.desired_redundancy, max_possible_redundancy)
            bit_redundancies = [redundancy_factor] * payload_length

        # Calculate total weights needed
        usable_weight_count = sum(bit_redundancies)
        usable_weights = selected_weights[:usable_weight_count]

        # Store original weights if requested
        original_weights = {} if store_originals else None

        rng = np.random.default_rng(self.embedding_seed)
        weights_modified = 0

        named_params = dict(model_copy.named_parameters())

        # Build cumulative redundancy map for adaptive redundancy
        cumulative_weights = 0
        bit_position_map = []
        for bit_idx, redundancy in enumerate(bit_redundancies):
            for _ in range(redundancy):
                bit_position_map.append(bit_idx)
            cumulative_weights += redundancy

        for i, (layer_name, weight_idx, _) in enumerate(usable_weights):
            # Use adaptive redundancy mapping
            if i < len(bit_position_map):
                bit_position = bit_position_map[i]
            else:
                # Fallback to uniform if somehow we exceed the map
                bit_position = min(i // redundancy_factor, payload_length - 1)

            bit_value = int(payload_bits[bit_position])

            if layer_name not in named_params:
                continue

            layer_param = named_params[layer_name]
            original_shape = layer_param.shape
            flat_weights = layer_param.data.view(-1)

            # Store original weight value before modification
            if store_originals:
                original_weights[(layer_name, weight_idx)] = float(flat_weights[weight_idx].item())

            layer_std = self.layer_stats.get(layer_name, float(flat_weights.std().item() + 1e-12))
            step = max(layer_std * self.qim_step_multiplier, 1e-8)
            dither = rng.uniform(-0.5 * step, 0.5 * step)

            current_weight = float(flat_weights[weight_idx].item())
            scaled = (current_weight + dither) / step
            quantized = int(np.floor(scaled + 0.5))

            if (quantized & 1) != bit_value:
                direction = 1 if scaled >= quantized else -1
                if direction == 0:
                    direction = 1
                quantized += direction

            new_weight = (quantized * step) - dither
            flat_weights[weight_idx] = new_weight
            layer_param.data = flat_weights.view(original_shape)
            weights_modified += 1

        # Store bit_redundancies in metadata for extraction
        metadata_extra = {'bit_redundancies': bit_redundancies} if self.adaptive_redundancy else {}
        return model_copy, original_weights, redundancy_factor, weights_modified, metadata_extra

    def _simulate_quantization(self,
                                model: nn.Module,
                                bits: int = 8) -> nn.Module:
        """
        Simulate weight quantization attack.

        Args:
            model: Model to quantize
            bits: Number of bits (4 or 8)

        Returns:
            Quantized model
        """
        import copy
        quantized_model = copy.deepcopy(model)

        for param in quantized_model.parameters():
            if param.requires_grad:
                data = param.data
                min_val = data.min()
                max_val = data.max()

                # Quantize to N bits
                num_levels = 2 ** bits
                scale = (max_val - min_val) / (num_levels - 1)

                # Quantize
                quantized = torch.round((data - min_val) / scale)
                # Dequantize
                param.data = quantized * scale + min_val

        return quantized_model

    def _validate_quantization_resistance(self,
                                           model: nn.Module,
                                           payload_bits: np.ndarray,
                                           metadata: Dict,
                                           target_bits: int = 4,
                                           target_survival: float = 0.85) -> Tuple[bool, float]:
        """
        Validate that embedded payload survives quantization.

        This simulates quantization attack and tests extraction survival rate.
        If survival rate is too low, returns False to signal that step size
        should be increased.

        Args:
            model: Watermarked model
            payload_bits: Original payload
            metadata: Embedding metadata
            target_bits: Quantization bits to test (4 or 8)
            target_survival: Minimum acceptable survival rate

        Returns:
            (success, survival_rate)
        """
        if not self.quantization_simulation:
            return True, 1.0

        print(f"\n[Q-VALIDATION] Testing {target_bits}-bit quantization resistance...")

        # Simulate quantization
        quantized_model = self._simulate_quantization(model, bits=target_bits)

        # Try to extract
        try:
            from core.extractor import NeuralExtractor
            extractor = NeuralExtractor(device=str(self.device))

            recovered_bytes, _ = extractor.extract(quantized_model, metadata)
            recovered_bits = np.unpackbits(np.frombuffer(recovered_bytes, dtype=np.uint8))[:len(payload_bits)]

            # Compute survival rate
            matches = np.sum(recovered_bits == payload_bits)
            survival_rate = matches / len(payload_bits)

            success = survival_rate >= target_survival

            if success:
                print(f"[Q-VALIDATION] ✓ Survival rate: {survival_rate*100:.2f}% (target: {target_survival*100:.0f}%)")
            else:
                print(f"[Q-VALIDATION] ✗ Survival rate: {survival_rate*100:.2f}% (below target: {target_survival*100:.0f}%)")

            return success, survival_rate

        except Exception as e:
            print(f"[Q-VALIDATION] Error during validation: {e}")
            return False, 0.0

    def _evaluate_accuracy(self,
                           model: nn.Module,
                           dataloader: torch.utils.data.DataLoader,
                           num_batches: int = 50) -> float:
        """
        Evaluate model top-1 accuracy on CIFAR-10.

        Args:
            model: Model to evaluate
            dataloader: CIFAR-10 test dataloader
            num_batches: Number of batches to evaluate

        Returns:
            Top-1 accuracy (0.0 to 1.0)
        """
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = correct / total
        return accuracy

    def embed(self,
              payload: bytes,
              dataloader: torch.utils.data.DataLoader) -> Tuple[nn.Module, Dict]:
        """
        Main embedding function - coordinates the entire embedding process.

        Workflow:
        1. Convert payload bytes to bit array
        2. Compute layer-wise sensitivity
        3. Rank weights by sensitivity
        4. Evaluate baseline accuracy
        5. Binary search for optimal capacity
        6. Embed payload at optimal capacity
        7. Store embedding metadata
        8. Return watermarked model and metadata

        Args:
            payload: Binary secret to embed
            dataloader: CIFAR-10 validation dataloader

        Returns:
            (watermarked_model, embedding_metadata)
        """
        print("=" * 80)
        print("NEURAL NETWORK EMBEDDING ENGINE")
        print("=" * 80)

        # Convert payload to bits and prepare whitening
        payload_bits = utils.bytes_to_bits(payload)
        print(f"Payload: {len(payload)} bytes = {len(payload_bits)} bits")

        # Resolve seeds and pre-compute statistics for this embedding run
        self._resolve_seeds()
        self.layer_stats = self._compute_layer_stats()
        whitened_bits = utils.whiten_bits(payload_bits, self.whitening_seed)

        # Step 1: Compute sensitivity
        print("\n[1/5] Computing layer-wise sensitivity...")
        sensitivity_maps = self.compute_layer_sensitivity(dataloader)

        # Step 2: Rank weights
        print("[2/5] Ranking weights by sensitivity...")
        ranked_weights = self.rank_weights_by_sensitivity(sensitivity_maps)
        print(f"  Total embeddable weights: {len(ranked_weights)}")

        # Step 3: Baseline accuracy
        print("[3/5] Evaluating baseline accuracy...")
        baseline_acc = self._evaluate_accuracy(self.model, dataloader)
        print(f"  Baseline accuracy: {baseline_acc:.4f}")

        # Step 4: Adaptive capacity search
        print("[4/5] Searching for optimal capacity...")
        optimal_capacity, _ = self.adaptive_capacity_search(
            ranked_weights, whitened_bits, dataloader, baseline_acc
        )

        # Step 5: Final embedding with original weights storage
        print("[5/5] Embedding payload...")
        watermarked_model, original_weights, redundancy_factor, used_weights, metadata_extra = self._embed_at_capacity(
            ranked_weights[:optimal_capacity],
            whitened_bits,
            store_originals=True  # Store original weights for extraction
        )
        print(f"  Effective redundancy used: x{redundancy_factor}")
        print(f"  Weights modified: {used_weights}")

        # Verify embedding accuracy
        final_acc = self._evaluate_accuracy(watermarked_model, dataloader)
        print(f"  Watermarked accuracy: {final_acc:.4f} (drop: {baseline_acc - final_acc:.4f})")

        # Validate quantization resistance (Phase 1)
        quantization_survival_4bit = 0.0
        quantization_survival_8bit = 0.0

        if self.quantization_simulation:
            # Test 8-bit quantization
            _, survival_8bit = self._validate_quantization_resistance(
                watermarked_model, whitened_bits,
                {'embedding_map': ranked_weights[:used_weights],
                 'original_weights': original_weights,
                 'redundancy_factor': redundancy_factor,
                 'qim_step_multiplier': self.qim_step_multiplier,
                 'embedding_seed': self.embedding_seed,
                 'whitening_seed': self.whitening_seed,
                 'layer_stats': self.layer_stats,
                 'payload_size_bits': len(payload_bits),
                 'bit_redundancies': metadata_extra.get('bit_redundancies') if metadata_extra else None},
                target_bits=8,
                target_survival=0.90
            )
            quantization_survival_8bit = survival_8bit

            # Test 4-bit quantization
            _, survival_4bit = self._validate_quantization_resistance(
                watermarked_model, whitened_bits,
                {'embedding_map': ranked_weights[:used_weights],
                 'original_weights': original_weights,
                 'redundancy_factor': redundancy_factor,
                 'qim_step_multiplier': self.qim_step_multiplier,
                 'embedding_seed': self.embedding_seed,
                 'whitening_seed': self.whitening_seed,
                 'layer_stats': self.layer_stats,
                 'payload_size_bits': len(payload_bits),
                 'bit_redundancies': metadata_extra.get('bit_redundancies') if metadata_extra else None},
                target_bits=4,
                target_survival=0.85
            )
            quantization_survival_4bit = survival_4bit

        # Store metadata
        total_embeddable = max(1, len(ranked_weights))
        metadata = {
            'payload_size_bits': len(payload_bits),
            'whitened_payload_bits': len(whitened_bits),
            'capacity_weights': used_weights,
            'selected_capacity_weights': optimal_capacity,
            'capacity_percent': used_weights / total_embeddable * 100,
            'baseline_accuracy': baseline_acc,
            'embedded_accuracy': final_acc,
            'accuracy_drop': baseline_acc - final_acc,
            'embedding_map': ranked_weights[:used_weights],  # Store locations actually used
            'original_weights': original_weights,  # Store original weight values for extraction
            'redundancy_factor': redundancy_factor,
            'qim_step_multiplier': self.qim_step_multiplier,
            'embedding_seed': self.embedding_seed,
            'whitening_seed': self.whitening_seed,
            'layer_stats': self.layer_stats,
            'desired_redundancy': self.desired_redundancy,
            'adaptive_redundancy': self.adaptive_redundancy,
            'quantization_simulation': self.quantization_simulation
        }

        # Merge adaptive redundancy metadata
        if metadata_extra:
            metadata.update(metadata_extra)

        # Add quantization validation results
        metadata['quantization_survival_4bit'] = quantization_survival_4bit
        metadata['quantization_survival_8bit'] = quantization_survival_8bit

        print("\n" + "=" * 80)
        print(f"EMBEDDING COMPLETE")
        print(f"Capacity: {metadata['capacity_percent']:.2f}% of model parameters")
        print(f"Redundancy factor: x{redundancy_factor}")
        print(f"Accuracy preserved: {final_acc >= baseline_acc - self.accuracy_threshold}")
        if self.quantization_simulation:
            print(f"8-bit quant survival: {quantization_survival_8bit*100:.1f}%")
            print(f"4-bit quant survival: {quantization_survival_4bit*100:.1f}%")
        print("=" * 80)

        return watermarked_model, metadata
