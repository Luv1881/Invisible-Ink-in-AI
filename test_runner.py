#!/usr/bin/env python3
"""
Comprehensive Test Runner for Neural Network Steganography Framework

This script runs extensive benchmarks including:
- Identity (no attack)
- Fine-tuning attacks (varying epochs)
- Structured pruning (varying sparsity levels)
- Quantization (8-bit and 4-bit)
- Adversarial attacks (FGSM, PGD)
- Gaussian noise
- Combined attacks
- Statistical security analysis
"""

import torch
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import time
import warnings

from core.embedder import NeuralEmbedder
from core.extractor import NeuralExtractor
from core.attacks import TransformationSimulator
from core.security import SecurityEvaluator
from data.cifar10 import get_cifar10_loaders


class ComprehensiveTester:
    """Run comprehensive tests with validation and detailed analysis."""

    # Critical thresholds (adjusted for improved robustness)
    MIN_ACCEPTABLE_ACCURACY = 0.40  # 40% minimum for CIFAR-10 (after fine-tuning)
    MIN_ACCEPTABLE_SURVIVAL = 0.80  # 80% survival rate threshold
    CAPACITY_WARNING_THRESHOLD = 0.01  # Warn if using < 1% capacity

    def __init__(self, finetune_epochs: int = 10, skip_finetuning: bool = False):
        """
        Initialize comprehensive tester.

        Args:
            finetune_epochs: Number of epochs to fine-tune model on CIFAR-10
            skip_finetuning: If True, skip fine-tuning (faster but lower accuracy)
        """
        print("=" * 100)
        print("NEURAL NETWORK STEGANOGRAPHY - COMPREHENSIVE TEST RUNNER")
        print("=" * 100)
        print()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {self.device.upper()}")

        if self.device == 'cpu':
            print("WARNING: Running on CPU. Tests will be slower. Consider using a GPU for faster results.")
        print()

        # Initialize components with maximum robustness settings
        print("Loading components...")
        self.embedder = NeuralEmbedder(
            device=self.device,
            desired_redundancy=25,  # Dramatically increased for extreme robustness
            qim_step_multiplier=0.25,  # Maximum step size for 4-bit quantization resistance
            use_ecc=True,
            adaptive_redundancy=True,  # Use adaptive redundancy
            quantization_simulation=True  # Validate quantization resistance
        )
        self.extractor = NeuralExtractor(device=self.device)
        self.attacker = TransformationSimulator(device=self.device)
        self.security = SecurityEvaluator(device=self.device)

        # Load data
        print("Loading CIFAR-10 dataset...")
        self.train_loader, self.test_loader = get_cifar10_loaders(batch_size=128, num_workers=0)

        # Fine-tune model for CIFAR-10
        if not skip_finetuning:
            print(f"\n{'='*100}")
            print("FIXING MODEL/DATASET MISMATCH")
            print(f"{'='*100}")
            print("The ImageNet-pretrained model needs fine-tuning on CIFAR-10")
            print(f"This will take ~{finetune_epochs} epochs but significantly improves accuracy\n")

            self.embedder.finetune_model_for_cifar10(
                self.train_loader,
                self.test_loader,
                num_epochs=finetune_epochs,
                learning_rate=0.001
            )
        else:
            print("\n[WARNING] Skipping fine-tuning - model will have very low accuracy")

        self.results = []
        self.warnings_issued = []
        print("\nInitialization complete\n")

    def load_test_input(self, filepath: str = 'test_input.txt') -> bytes:
        """Load test input file."""
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            print(f"Loaded test input: {filepath}")
            print(f"  Size: {len(data)} bytes ({len(data) * 8} bits)")
            return data
        except FileNotFoundError:
            print(f"Test input file not found. Creating default payload...")
            data = b"Neural Network Steganography Test - " * 10  # ~370 bytes
            return data

    def validate_embedding_results(self, metadata: dict):
        """Validate embedding results and issue warnings for critical issues."""
        warnings_found = []

        # Check baseline accuracy
        baseline_acc = metadata.get('baseline_accuracy', 0)
        if baseline_acc < self.MIN_ACCEPTABLE_ACCURACY:
            warning = f"CRITICAL: Extremely low baseline accuracy ({baseline_acc:.4f}). Model may be untrained or broken."
            warnings_found.append(warning)
            print(f"\n  WARNING: {warning}")

        # Check embedded accuracy
        embedded_acc = metadata.get('embedded_accuracy', 0)
        if embedded_acc < self.MIN_ACCEPTABLE_ACCURACY:
            warning = f"CRITICAL: Extremely low embedded accuracy ({embedded_acc:.4f}). Watermark severely damaged model."
            warnings_found.append(warning)
            print(f"\n  WARNING: {warning}")

        # Check capacity utilization
        capacity_pct = metadata.get('capacity_percent', 0)
        if capacity_pct < self.CAPACITY_WARNING_THRESHOLD:
            warning = f"Low capacity utilization ({capacity_pct:.4f}%). Consider increasing payload size or redundancy."
            warnings_found.append(warning)
            print(f"\n  INFO: {warning}")

        self.warnings_issued.extend(warnings_found)
        return warnings_found

    def run_single_test(self, payload: bytes, attack_name: str, attack_func, attack_params: dict, needs_metadata: bool = False):
        """Run a single embedding-attack-extraction test."""
        print("\n" + "=" * 100)
        print(f"TEST: {attack_name}")
        print("=" * 100)

        test_result = {
            'attack_name': attack_name,
            'attack_params': str(attack_params),
            'payload_size_bytes': len(payload),
            'payload_size_bits': len(payload) * 8,
        }

        try:
            # Step 1: Embed
            print("\n[1/4] Embedding payload...")
            start_time = time.time()
            watermarked_model, metadata = self.embedder.embed(payload, self.test_loader)
            embed_time = time.time() - start_time

            test_result['embed_time'] = f"{embed_time:.1f}s"
            test_result['capacity_weights'] = metadata['capacity_weights']
            test_result['capacity_percent'] = f"{metadata['capacity_percent']:.4f}%"
            test_result['baseline_accuracy'] = f"{metadata['baseline_accuracy']:.4f}"
            test_result['embedded_accuracy'] = f"{metadata['embedded_accuracy']:.4f}"
            test_result['accuracy_drop_embed'] = f"{metadata['accuracy_drop']:.4f}"

            # Validate embedding results
            if attack_name == "No Attack (Identity)":
                self.validate_embedding_results(metadata)

            # Store clean model for security test
            clean_model = self.embedder.model

            # Step 2: Apply attack
            print(f"\n[2/4] Applying attack: {attack_name}...")
            start_time = time.time()
            if attack_func:
                # Add metadata if needed
                if needs_metadata:
                    attack_params['metadata'] = metadata
                attacked_model = attack_func(watermarked_model, **attack_params)
            else:
                attacked_model = watermarked_model
            attack_time = time.time() - start_time

            test_result['attack_time'] = f"{attack_time:.1f}s"

            # Evaluate attacked model
            attacked_acc = self.embedder._evaluate_accuracy(attacked_model, self.test_loader)
            test_result['attacked_accuracy'] = f"{attacked_acc:.4f}"
            test_result['accuracy_drop_attack'] = f"{metadata['embedded_accuracy'] - attacked_acc:.4f}"

            # Step 3: Extract
            print(f"\n[3/4] Extracting payload...")
            start_time = time.time()
            recovered_payload, extract_stats = self.extractor.extract(attacked_model, metadata)
            extract_time = time.time() - start_time

            test_result['extract_time'] = f"{extract_time:.1f}s"
            test_result['mean_confidence'] = f"{extract_stats['mean_confidence']:.4f}"
            test_result['min_confidence'] = f"{extract_stats['min_confidence']:.4f}"

            # Step 4: Compute survival rate
            survival_rate = self.extractor.compute_survival_rate(payload, recovered_payload)
            test_result['survival_rate'] = f"{survival_rate * 100:.2f}%"
            test_result['survival_rate_numeric'] = survival_rate

            # Check for low survival rate
            if survival_rate < self.MIN_ACCEPTABLE_SURVIVAL:
                warning = f"Low survival rate for {attack_name}: {survival_rate*100:.2f}%"
                self.warnings_issued.append(warning)
                print(f"\n  WARNING: {warning}")

            # Compare payloads
            original_preview = payload[:100].decode('utf-8', errors='replace')
            recovered_preview = recovered_payload[:100].decode('utf-8', errors='replace')

            test_result['original_preview'] = original_preview
            test_result['recovered_preview'] = recovered_preview
            test_result['exact_match'] = payload == recovered_payload

            print(f"\nTest complete: {attack_name}")
            print(f"  Survival Rate: {test_result['survival_rate']}")
            print(f"  Exact Match: {test_result['exact_match']}")

            # Security analysis for identity attack
            if attack_name == "No Attack (Identity)":
                print(f"\n[4/4] Running security analysis...")
                try:
                    sec_results = self.security.comprehensive_security_analysis(clean_model, watermarked_model)

                    # Handle potential nan values
                    ks_pvalue = sec_results['ks_test']['p_value']
                    mw_pvalue = sec_results['mw_test']['p_value']
                    chi2_pvalue = sec_results['chi2_test']['p_value']

                    test_result['ks_pvalue'] = f"{ks_pvalue:.6f}" if not np.isnan(ks_pvalue) else "NaN"
                    test_result['mw_pvalue'] = f"{mw_pvalue:.6f}" if not np.isnan(mw_pvalue) else "NaN"
                    test_result['chi2_pvalue'] = f"{chi2_pvalue:.6f}" if not np.isnan(chi2_pvalue) else "NaN"

                    # Compute mean p-value excluding nan
                    valid_pvalues = [p for p in [ks_pvalue, mw_pvalue, chi2_pvalue] if not np.isnan(p)]
                    if valid_pvalues:
                        mean_pvalue = np.mean(valid_pvalues)
                        test_result['mean_pvalue'] = f"{mean_pvalue:.6f}"
                        test_result['statistically_undetectable'] = all(p >= 0.05 for p in valid_pvalues)
                    else:
                        test_result['mean_pvalue'] = "NaN"
                        test_result['statistically_undetectable'] = False
                        warning = "All statistical tests returned NaN - data may be invalid"
                        self.warnings_issued.append(warning)
                        print(f"\n  WARNING: {warning}")

                    # Check for nan values
                    if np.isnan(chi2_pvalue):
                        warning = "Chi-square test returned NaN - may indicate identical distributions or sparse bins"
                        self.warnings_issued.append(warning)
                        print(f"\n  WARNING: {warning}")

                except Exception as e:
                    print(f"\n  ERROR in security analysis: {str(e)}")
                    test_result['security_error'] = str(e)

            test_result['status'] = 'SUCCESS'

        except Exception as e:
            print(f"\nTest failed: {str(e)}")
            import traceback
            traceback.print_exc()
            test_result['status'] = 'FAILED'
            test_result['error'] = str(e)

        self.results.append(test_result)
        return test_result

    def run_comprehensive_benchmark_suite(self, payload: bytes):
        """Run comprehensive benchmark suite with all attack types."""
        print("\n" + "=" * 100)
        print("RUNNING COMPREHENSIVE BENCHMARK SUITE")
        print("=" * 100)

        # Define comprehensive test cases
        test_cases = [
            # Identity (baseline)
            {
                'name': 'No Attack (Identity)',
                'func': None,
                'params': {},
                'needs_metadata': False
            },

            # Fine-tuning attacks (varying epochs)
            {
                'name': 'Fine-tune (2 epochs)',
                'func': self.attacker.fine_tune,
                'params': {'train_loader': self.train_loader, 'num_epochs': 2, 'learning_rate': 0.001},
                'needs_metadata': False
            },
            {
                'name': 'Fine-tune (5 epochs)',
                'func': self.attacker.fine_tune,
                'params': {'train_loader': self.train_loader, 'num_epochs': 5, 'learning_rate': 0.001},
                'needs_metadata': False
            },
            {
                'name': 'Fine-tune (10 epochs)',
                'func': self.attacker.fine_tune,
                'params': {'train_loader': self.train_loader, 'num_epochs': 10, 'learning_rate': 0.001},
                'needs_metadata': False
            },

            # Structured pruning (varying sparsity)
            {
                'name': 'Structured Prune (30%)',
                'func': self.attacker.structured_prune,
                'params': {'sparsity': 0.3},
                'needs_metadata': False
            },
            {
                'name': 'Structured Prune (50%)',
                'func': self.attacker.structured_prune,
                'params': {'sparsity': 0.5},
                'needs_metadata': False
            },
            {
                'name': 'Structured Prune (60%)',
                'func': self.attacker.structured_prune,
                'params': {'sparsity': 0.6},
                'needs_metadata': False
            },
            {
                'name': 'Structured Prune (70%)',
                'func': self.attacker.structured_prune,
                'params': {'sparsity': 0.7},
                'needs_metadata': False
            },

            # Quantization attacks
            {
                'name': '8-bit Quantization',
                'func': self.attacker.quantize_8bit,
                'params': {},
                'needs_metadata': False
            },
            {
                'name': '4-bit Quantization',
                'func': self.attacker.quantize_4bit,
                'params': {},
                'needs_metadata': False
            },

            # Gaussian noise
            {
                'name': 'Gaussian Noise (std=0.001)',
                'func': self.attacker.add_gaussian_noise,
                'params': {'noise_std': 0.001},
                'needs_metadata': False
            },
            {
                'name': 'Gaussian Noise (std=0.005)',
                'func': self.attacker.add_gaussian_noise,
                'params': {'noise_std': 0.005},
                'needs_metadata': False
            },

            # Adversarial attacks
            {
                'name': 'FGSM Attack (epsilon=0.01)',
                'func': self.attacker.fgsm_attack,
                'params': {'epsilon': 0.01},
                'needs_metadata': True
            },
            {
                'name': 'PGD Attack (epsilon=0.01)',
                'func': self.attacker.pgd_attack,
                'params': {'epsilon': 0.01, 'alpha': 0.002, 'num_steps': 7},
                'needs_metadata': True
            },

            # Combined attacks (realistic deployment scenarios)
            {
                'name': 'Combined: Prune(30%) + Quantize(8bit)',
                'func': self.attacker.combined_attack,
                'params': {'prune_sparsity': 0.3, 'quantize_bits': 8, 'noise_std': 0.0},
                'needs_metadata': False
            },
            {
                'name': 'Combined: Prune(50%) + Quantize(4bit)',
                'func': self.attacker.combined_attack,
                'params': {'prune_sparsity': 0.5, 'quantize_bits': 4, 'noise_std': 0.0},
                'needs_metadata': False
            },
            {
                'name': 'Combined: Prune(30%) + Quantize(8bit) + Noise(0.001)',
                'func': self.attacker.combined_attack,
                'params': {'prune_sparsity': 0.3, 'quantize_bits': 8, 'noise_std': 0.001},
                'needs_metadata': False
            },
        ]

        for i, test_case in enumerate(test_cases):
            print(f"\n{'='*100}")
            print(f"Progress: {i+1}/{len(test_cases)}")
            print(f"{'='*100}")

            self.run_single_test(
                payload,
                test_case['name'],
                test_case['func'],
                test_case['params'].copy(),  # Copy to avoid modifying original
                test_case['needs_metadata']
            )

    def display_results_table(self):
        """Display results in a formatted table."""
        print("\n\n" + "=" * 100)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 100)

        if not self.results:
            print("No results to display.")
            return

        # Table header
        header_format = "{:<40} {:<12} {:<12} {:<12} {:<12} {:<10}"
        print(header_format.format(
            "Attack Type",
            "Survival",
            "Embed Acc",
            "Attack Acc",
            "Mean Conf",
            "Status"
        ))
        print("-" * 100)

        # Table rows
        for result in self.results:
            if result['status'] == 'SUCCESS':
                print(header_format.format(
                    result['attack_name'][:39],
                    result.get('survival_rate', 'N/A'),
                    result.get('embedded_accuracy', 'N/A'),
                    result.get('attacked_accuracy', 'N/A'),
                    result.get('mean_confidence', 'N/A'),
                    result['status']
                ))
            else:
                print(f"{result['attack_name']:<40} {'FAILED':<12} Error: {result.get('error', 'Unknown')}")

        # Security results (if available)
        identity_result = next((r for r in self.results if r['attack_name'] == "No Attack (Identity)" and r['status'] == 'SUCCESS'), None)
        if identity_result and 'mean_pvalue' in identity_result:
            print("\n" + "=" * 100)
            print("STATISTICAL SECURITY ANALYSIS")
            print("=" * 100)
            print(f"Kolmogorov-Smirnov p-value: {identity_result.get('ks_pvalue', 'N/A')}")
            print(f"Mann-Whitney p-value:       {identity_result.get('mw_pvalue', 'N/A')}")
            print(f"Chi-Square p-value:         {identity_result.get('chi2_pvalue', 'N/A')}")
            print(f"Mean p-value:               {identity_result.get('mean_pvalue', 'N/A')}")
            print(f"Statistically Undetectable: {identity_result.get('statistically_undetectable', 'N/A')}")
            print()
            if identity_result.get('statistically_undetectable'):
                print("PASS: All p-values > 0.05 indicate the watermark is statistically undetectable")
            else:
                print("FAIL: Some p-values < 0.05 indicate the watermark may be detectable")

    def analyze_vulnerabilities(self):
        """Analyze results and report critical vulnerabilities."""
        print("\n\n" + "=" * 100)
        print("VULNERABILITY ANALYSIS")
        print("=" * 100)

        vulnerabilities = []

        # Analyze each attack
        for result in self.results:
            if result['status'] != 'SUCCESS':
                continue

            attack_name = result['attack_name']
            survival_rate = result.get('survival_rate_numeric', 1.0)

            # Check for critical vulnerabilities
            if survival_rate < 0.50:
                vulnerabilities.append({
                    'severity': 'CRITICAL',
                    'attack': attack_name,
                    'survival': survival_rate,
                    'message': f"Severe data loss: {survival_rate*100:.1f}% survival"
                })
            elif survival_rate < 0.80:
                vulnerabilities.append({
                    'severity': 'HIGH',
                    'attack': attack_name,
                    'survival': survival_rate,
                    'message': f"Significant data loss: {survival_rate*100:.1f}% survival"
                })
            elif survival_rate < 0.95:
                vulnerabilities.append({
                    'severity': 'MEDIUM',
                    'attack': attack_name,
                    'survival': survival_rate,
                    'message': f"Moderate data loss: {survival_rate*100:.1f}% survival"
                })

        # Display vulnerabilities
        if vulnerabilities:
            print("\nIdentified Vulnerabilities:")
            print("-" * 100)
            for vuln in vulnerabilities:
                print(f"[{vuln['severity']}] {vuln['attack']}")
                print(f"  {vuln['message']}")
                print()

            # Recommendations
            print("\nRECOMMENDATIONS:")
            print("-" * 100)

            # Check for quantization issues
            quant_vulns = [v for v in vulnerabilities if 'Quantization' in v['attack']]
            if quant_vulns:
                print("1. QUANTIZATION VULNERABILITY DETECTED:")
                print("   - Implement quantization-aware embedding")
                print("   - Increase bit redundancy in encoding")
                print("   - Consider using error correction codes (ECC)")
                print()

            # Check for pruning issues
            prune_vulns = [v for v in vulnerabilities if 'Prune' in v['attack']]
            if prune_vulns:
                print("2. PRUNING VULNERABILITY DETECTED:")
                print("   - Embed in pruning-resistant weight locations")
                print("   - Use channel importance for weight selection")
                print("   - Increase embedding redundancy")
                print()

            # Check for fine-tuning issues
            finetune_vulns = [v for v in vulnerabilities if 'Fine-tune' in v['attack']]
            if finetune_vulns:
                print("3. FINE-TUNING VULNERABILITY DETECTED:")
                print("   - Use stronger embedding strength")
                print("   - Implement gradient-based robustness")
                print("   - Consider regularization during embedding")
                print()

            # Check for combined attacks
            combined_vulns = [v for v in vulnerabilities if 'Combined' in v['attack']]
            if combined_vulns:
                print("4. COMBINED ATTACK VULNERABILITY DETECTED:")
                print("   - This is the most realistic threat scenario")
                print("   - Implement multi-layered defense")
                print("   - Use adaptive redundancy based on weight importance")
                print()

        else:
            print("\nPASS: No critical vulnerabilities detected!")
            print("All attacks resulted in >95% survival rate.")

    def display_warnings(self):
        """Display all warnings issued during testing."""
        if self.warnings_issued:
            print("\n\n" + "=" * 100)
            print("WARNINGS AND ISSUES")
            print("=" * 100)
            for i, warning in enumerate(self.warnings_issued, 1):
                print(f"{i}. {warning}")
            print()

    def save_detailed_results(self, output_file: str = 'test_results.txt'):
        """Save detailed results to a text file."""
        with open(output_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("NEURAL NETWORK STEGANOGRAPHY - COMPREHENSIVE TEST RESULTS\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n\n")

            # Write warnings
            if self.warnings_issued:
                f.write("WARNINGS:\n")
                f.write("-" * 100 + "\n")
                for warning in self.warnings_issued:
                    f.write(f"  - {warning}\n")
                f.write("\n")

            # Write detailed results
            for result in self.results:
                f.write(f"\nTest: {result['attack_name']}\n")
                f.write("-" * 100 + "\n")
                for key, value in result.items():
                    if key not in ['original_preview', 'recovered_preview']:  # Skip large text
                        f.write(f"{key:30s}: {value}\n")
                f.write("\n")

        print(f"\nDetailed results saved to: {output_file}")


def main():
    """
    Main test runner.

    Usage:
        python test_runner.py              # Full test with 10 epochs fine-tuning
        python test_runner.py --quick      # Quick test (skip fine-tuning, faster but less accurate)
        python test_runner.py --epochs 5   # Custom fine-tuning epochs
    """
    import argparse

    parser = argparse.ArgumentParser(description='Neural Steganography Test Runner')
    parser.add_argument('--quick', action='store_true',
                        help='Skip model fine-tuning for faster testing (lower accuracy)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of fine-tuning epochs (default: 10)')
    args = parser.parse_args()

    # Initialize tester
    tester = ComprehensiveTester(
        finetune_epochs=args.epochs,
        skip_finetuning=args.quick
    )

    # Load test input
    payload = tester.load_test_input('test_input.txt')

    print(f"\nTest Input Preview (first 200 chars):")
    print(f"{payload[:200].decode('utf-8', errors='replace')}")
    print()

    # Run comprehensive benchmarks
    tester.run_comprehensive_benchmark_suite(payload)

    # Display results
    tester.display_results_table()
    tester.analyze_vulnerabilities()
    tester.display_warnings()

    # Save results
    tester.save_detailed_results('test_results.txt')

    print("\n" + "=" * 100)
    print("COMPREHENSIVE TEST SUITE COMPLETE")
    print("=" * 100)
    print()


if __name__ == '__main__':
    main()
