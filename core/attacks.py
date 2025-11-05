# core/attacks.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict
import numpy as np

class TransformationSimulator:
    """
    Simulates model transformations and adversarial attacks:
    - Fine-tuning
    - Structured pruning
    - Quantization
    - FGSM adversarial attacks
    - PGD adversarial attacks
    """

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def fine_tune(self,
                  model: nn.Module,
                  train_loader: DataLoader,
                  num_epochs: int = 5,
                  learning_rate: float = 0.001) -> nn.Module:
        """
        Fine-tune model on CIFAR-10 for specified epochs.

        This simulates post-watermarking retraining attack.

        Args:
            model: Watermarked model
            train_loader: CIFAR-10 training data
            num_epochs: Number of fine-tuning epochs (default 5, max 10)
            learning_rate: Learning rate for fine-tuning

        Returns:
            Fine-tuned model
        """
        print(f"\n[ATTACK] Fine-tuning for {num_epochs} epochs (lr={learning_rate})...")

        model.to(self.device)
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
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

        model.eval()
        print(f"Fine-tuning complete. Final accuracy: {100.*correct/total:.2f}%")
        return model

    def structured_prune(self,
                         model: nn.Module,
                         sparsity: float = 0.5) -> nn.Module:
        """
        Apply structured pruning (remove entire channels/filters).

        Removes least important channels based on L1 norm.

        Args:
            model: Watermarked model
            sparsity: Fraction of channels to remove (0.0 to 0.7)

        Returns:
            Pruned model
        """
        print(f"\n[ATTACK] Structured pruning at {sparsity*100}% sparsity...")

        model.to(self.device)
        model.eval()

        # For each convolutional layer, prune channels
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                num_channels = module.out_channels
                num_to_prune = int(num_channels * sparsity)

                # Compute channel importance (L1 norm)
                weights = module.weight.data
                channel_norms = torch.sum(torch.abs(weights), dim=(1, 2, 3))

                # Get indices of least important channels
                _, prune_indices = torch.topk(channel_norms, num_to_prune, largest=False)

                # Zero out pruned channels
                module.weight.data[prune_indices, :, :, :] = 0
                if module.bias is not None:
                    module.bias.data[prune_indices] = 0

                print(f"  Pruned {num_to_prune}/{num_channels} channels in {name}")

        print("Structured pruning complete.")
        return model

    def quantize_8bit(self, model: nn.Module) -> nn.Module:
        """
        Apply uniform 8-bit quantization to all weights.

        Quantization: w_q = round((w - min) / scale) * scale + min
        where scale = (max - min) / 255

        Args:
            model: Watermarked model

        Returns:
            Quantized model
        """
        print("\n[ATTACK] Applying 8-bit quantization...")

        model.to(self.device)
        model.eval()

        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:
                weights = param.data

                # Compute quantization parameters
                w_min = weights.min()
                w_max = weights.max()
                scale = (w_max - w_min) / 255.0

                # Quantize
                quantized = torch.round((weights - w_min) / scale)
                dequantized = quantized * scale + w_min

                param.data = dequantized
                print(f"  Quantized {name}: range [{w_min:.6f}, {w_max:.6f}], scale={scale:.6f}")

        print("8-bit quantization complete.")
        return model

    def fgsm_attack(self,
                    model: nn.Module,
                    metadata: Dict,
                    epsilon: float = 0.01) -> nn.Module:
        """
        Fast Gradient Sign Method attack targeting embedded watermark.

        Perturb weights in gradient direction to maximize extraction error.

        Args:
            model: Watermarked model
            metadata: Embedding metadata (contains embedding locations)
            epsilon: Attack strength

        Returns:
            Attacked model
        """
        print(f"\n[ATTACK] FGSM attack with epsilon={epsilon}...")

        model.to(self.device)
        model.train()

        embedding_map = metadata['embedding_map']

        # Create dummy loss targeting embedded weights
        loss = 0.0
        for layer_name, weight_idx, _ in embedding_map[:1000]:  # Sample for efficiency
            param = dict(model.named_parameters())[layer_name]
            flat_weights = param.flatten()
            loss += flat_weights[weight_idx] ** 2

        # Compute gradient
        model.zero_grad()
        loss.backward()

        # Apply FGSM perturbation
        with torch.no_grad():
            for layer_name, weight_idx, _ in embedding_map:
                param = dict(model.named_parameters())[layer_name]
                if param.grad is not None:
                    original_shape = param.shape
                    flat_param = param.flatten()
                    flat_grad = param.grad.flatten()

                    # FGSM: w' = w + ε * sign(grad)
                    flat_param[weight_idx] += epsilon * torch.sign(flat_grad[weight_idx])

                    param.data = flat_param.reshape(original_shape)

        model.eval()
        print("FGSM attack complete.")
        return model

    def pgd_attack(self,
                   model: nn.Module,
                   metadata: Dict,
                   epsilon: float = 0.01,
                   alpha: float = 0.002,
                   num_steps: int = 7) -> nn.Module:
        """
        Projected Gradient Descent attack targeting watermark.

        Multi-step iterative attack with projection to epsilon ball.

        Args:
            model: Watermarked model
            metadata: Embedding metadata
            epsilon: Attack radius
            alpha: Step size
            num_steps: Number of PGD iterations

        Returns:
            Attacked model
        """
        print(f"\n[ATTACK] PGD attack: {num_steps} steps, eps={epsilon}, alpha={alpha}...")

        model.to(self.device)
        embedding_map = metadata['embedding_map']

        # Store original weights
        original_weights = {}
        for layer_name, weight_idx, _ in embedding_map:
            param = dict(model.named_parameters())[layer_name]
            original_weights[(layer_name, weight_idx)] = param.data.flatten()[weight_idx].clone()

        # Iterative PGD
        for step in range(num_steps):
            model.train()

            # Compute loss on embedded weights
            loss = 0.0
            for layer_name, weight_idx, _ in embedding_map[:1000]:
                param = dict(model.named_parameters())[layer_name]
                flat_weights = param.flatten()
                loss += flat_weights[weight_idx] ** 2

            # Gradient step
            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for layer_name, weight_idx, _ in embedding_map:
                    param = dict(model.named_parameters())[layer_name]
                    if param.grad is not None:
                        original_shape = param.shape
                        flat_param = param.flatten()
                        flat_grad = param.grad.flatten()

                        # PGD step
                        flat_param[weight_idx] += alpha * torch.sign(flat_grad[weight_idx])

                        # Project to epsilon ball around original
                        original_val = original_weights[(layer_name, weight_idx)]
                        flat_param[weight_idx] = torch.clamp(
                            flat_param[weight_idx],
                            original_val - epsilon,
                            original_val + epsilon
                        )

                        param.data = flat_param.reshape(original_shape)

            if step % 2 == 0:
                print(f"  PGD step {step+1}/{num_steps}, loss={loss.item():.6f}")

        model.eval()
        print("PGD attack complete.")
        return model

    def quantize_4bit(self, model: nn.Module) -> nn.Module:
        """
        Apply uniform 4-bit quantization to all weights.

        More aggressive than 8-bit, uses only 16 discrete levels.

        Args:
            model: Watermarked model

        Returns:
            Quantized model
        """
        print("\n[ATTACK] Applying 4-bit quantization...")

        model.to(self.device)
        model.eval()

        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:
                weights = param.data

                # Compute quantization parameters (16 levels = 4 bits)
                w_min = weights.min()
                w_max = weights.max()
                scale = (w_max - w_min) / 15.0

                # Quantize
                quantized = torch.round((weights - w_min) / scale)
                dequantized = quantized * scale + w_min

                param.data = dequantized
                print(f"  Quantized {name}: range [{w_min:.6f}, {w_max:.6f}], scale={scale:.6f}")

        print("4-bit quantization complete.")
        return model

    def add_gaussian_noise(self,
                           model: nn.Module,
                           noise_std: float = 0.001) -> nn.Module:
        """
        Add Gaussian noise to all weights.

        Simulates weight perturbations from various sources.

        Args:
            model: Watermarked model
            noise_std: Standard deviation of Gaussian noise

        Returns:
            Noisy model
        """
        print(f"\n[ATTACK] Adding Gaussian noise (std={noise_std})...")

        model.to(self.device)
        model.eval()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and len(param.shape) >= 2:
                    noise = torch.randn_like(param.data) * noise_std
                    param.data += noise
                    print(f"  Added noise to {name}: mean={noise.mean():.6f}, std={noise.std():.6f}")

        print("Gaussian noise addition complete.")
        return model

    def combined_attack(self,
                       model: nn.Module,
                       train_loader: DataLoader = None,
                       prune_sparsity: float = 0.3,
                       quantize_bits: int = 8,
                       noise_std: float = 0.001) -> nn.Module:
        """
        Apply multiple attacks in sequence.

        Simulates realistic deployment scenario with multiple transformations.

        Args:
            model: Watermarked model
            train_loader: Optional training data for fine-tuning
            prune_sparsity: Pruning sparsity level
            quantize_bits: Quantization bit width (4 or 8)
            noise_std: Gaussian noise standard deviation

        Returns:
            Attacked model
        """
        print(f"\n[COMBINED ATTACK] Prune({prune_sparsity*100}%) + Quantize({quantize_bits}bit) + Noise(std={noise_std})...")

        # Step 1: Prune
        if prune_sparsity > 0:
            model = self.structured_prune(model, sparsity=prune_sparsity)

        # Step 2: Quantize
        if quantize_bits == 8:
            model = self.quantize_8bit(model)
        elif quantize_bits == 4:
            model = self.quantize_4bit(model)

        # Step 3: Add noise
        if noise_std > 0:
            model = self.add_gaussian_noise(model, noise_std=noise_std)

        # Step 4: Optional fine-tuning
        if train_loader is not None:
            model = self.fine_tune(model, train_loader, num_epochs=1, learning_rate=0.0001)

        print("Combined attack complete.")
        return model
