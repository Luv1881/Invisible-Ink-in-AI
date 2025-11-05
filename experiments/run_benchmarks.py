# experiments/run_benchmarks.py

import torch
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import json
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.embedder import NeuralEmbedder
from core.extractor import NeuralExtractor
from core.attacks import TransformationSimulator
from core.security import SecurityEvaluator
from data.cifar10 import get_cifar10_loaders

class BenchmarkOrchestrator:
    """
    Orchestrates comprehensive benchmarking experiments.
    """

    def __init__(self, output_dir: str = './results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.embedder = NeuralEmbedder()
        self.extractor = NeuralExtractor()
        self.attacker = TransformationSimulator()
        self.security = SecurityEvaluator()

        # Load CIFAR-10
        self.train_loader, self.test_loader = get_cifar10_loaders()

        self.results = []

    def run_single_experiment(self,
                              payload_size_bytes: int,
                              attack_config: Dict) -> Dict:
        """
        Run a single embedding-attack-extraction experiment.

        Args:
            payload_size_bytes: Size of secret payload
            attack_config: Attack parameters {type, params}

        Returns:
            Experiment results dict
        """
        experiment_id = len(self.results)
        timestamp = datetime.now().isoformat()

        print(f"\n{'='*80}")
        print(f"EXPERIMENT {experiment_id}")
        print(f"Payload: {payload_size_bytes} bytes, Attack: {attack_config['type']}")
        print(f"{'='*80}")

        # Generate random payload
        payload = np.random.bytes(payload_size_bytes)

        # Phase 1: Embedding
        print("\n--- PHASE 1: EMBEDDING ---")
        embed_start = time.time()
        watermarked_model, metadata = self.embedder.embed(payload, self.test_loader)
        embed_time = time.time() - embed_start

        # Store clean model for security tests
        clean_model = self.embedder.model

        # Phase 2: Attack
        print("\n--- PHASE 2: ATTACK ---")
        attack_start = time.time()

        if attack_config['type'] == 'identity':
            attacked_model = watermarked_model
        elif attack_config['type'] == 'fine_tune':
            attacked_model = self.attacker.fine_tune(
                watermarked_model,
                self.train_loader,
                num_epochs=attack_config.get('epochs', 5)
            )
        elif attack_config['type'] == 'structured_prune':
            attacked_model = self.attacker.structured_prune(
                watermarked_model,
                sparsity=attack_config.get('sparsity', 0.5)
            )
        elif attack_config['type'] == 'quantize_8bit':
            attacked_model = self.attacker.quantize_8bit(watermarked_model)
        elif attack_config['type'] == 'fgsm':
            attacked_model = self.attacker.fgsm_attack(
                watermarked_model,
                metadata,
                epsilon=attack_config.get('epsilon', 0.01)
            )
        elif attack_config['type'] == 'pgd':
            attacked_model = self.attacker.pgd_attack(
                watermarked_model,
                metadata,
                epsilon=attack_config.get('epsilon', 0.01),
                num_steps=attack_config.get('steps', 7)
            )
        else:
            raise ValueError(f"Unknown attack: {attack_config['type']}")

        attack_time = time.time() - attack_start

        # Evaluate attacked model accuracy
        attacked_accuracy = self.embedder._evaluate_accuracy(attacked_model, self.test_loader)

        # Phase 3: Extraction
        print("\n--- PHASE 3: EXTRACTION ---")
        extract_start = time.time()
        recovered_payload, extract_stats = self.extractor.extract(attacked_model, metadata)
        extract_time = time.time() - extract_start

        # Compute survival rate
        survival_rate = self.extractor.compute_survival_rate(payload, recovered_payload)

        # Phase 4: Security Analysis (only for identity attack)
        security_results = None
        if attack_config['type'] == 'identity':
            print("\n--- PHASE 4: SECURITY ANALYSIS ---")
            security_results = self.security.comprehensive_security_analysis(
                clean_model, watermarked_model
            )

        # Compile results
        result = {
            'run_id': experiment_id,
            'timestamp': timestamp,
            'model_name': 'resnet18',
            'payload_size_bytes': payload_size_bytes,
            'payload_size_bits': payload_size_bytes * 8,

            # Capacity metrics
            'capacity_weights': metadata['capacity_weights'],
            'capacity_percent': metadata['capacity_percent'],

            # Timing
            'embedding_time_sec': embed_time,
            'attack_time_sec': attack_time,
            'extraction_time_sec': extract_time,
            'total_time_sec': embed_time + attack_time + extract_time,

            # Accuracy metrics
            'baseline_accuracy': metadata['baseline_accuracy'],
            'embedded_accuracy': metadata['embedded_accuracy'],
            'accuracy_drop_embedding': metadata['accuracy_drop'],
            'attacked_accuracy': attacked_accuracy,
            'accuracy_drop_attack': metadata['embedded_accuracy'] - attacked_accuracy,

            # Attack configuration
            'attack_type': attack_config['type'],
            'attack_params': json.dumps(attack_config),

            # Extraction metrics
            'survival_rate': survival_rate,
            'mean_bit_confidence': extract_stats['mean_confidence'],
            'min_bit_confidence': extract_stats['min_confidence'],
            'extraction_success': extract_stats['success'],

            # Security metrics (only if available)
            'ks_pvalue': security_results['ks_test']['p_value'] if security_results else None,
            'mw_pvalue': security_results['mw_test']['p_value'] if security_results else None,
            'chi2_pvalue': security_results['chi2_test']['p_value'] if security_results else None,
            'mean_pvalue': security_results['summary']['mean_p_value'] if security_results else None,
            'all_undetectable': security_results['summary']['all_undetectable'] if security_results else None,

            # Hardware metrics
            'gpu_memory_mb': torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None,

            # Notes
            'notes': ''
        }

        self.results.append(result)

        print(f"\n{'='*80}")
        print(f"EXPERIMENT {experiment_id} COMPLETE")
        print(f"Survival Rate: {survival_rate*100:.2f}%")
        print(f"Mean Confidence: {extract_stats['mean_confidence']:.4f}")
        print(f"Attacked Accuracy: {attacked_accuracy:.4f}")
        print(f"{'='*80}\n")

        return result

    def run_comprehensive_benchmarks(self):
        """
        Run comprehensive benchmark suite as specified in project document.
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE BENCHMARK SUITE")
        print("Neural Network Steganography - ResNet-18")
        print("="*80 + "\n")

        # Test payload sizes
        payload_sizes = [128, 512, 1024, 4096]  # bytes

        # Attack configurations
        attacks = [
            {'type': 'identity', 'name': 'No Attack'},
            {'type': 'fine_tune', 'epochs': 5, 'name': 'Fine-tune 5 epochs'},
            {'type': 'fine_tune', 'epochs': 10, 'name': 'Fine-tune 10 epochs'},
            {'type': 'structured_prune', 'sparsity': 0.3, 'name': 'Prune 30%'},
            {'type': 'structured_prune', 'sparsity': 0.5, 'name': 'Prune 50%'},
            {'type': 'structured_prune', 'sparsity': 0.6, 'name': 'Prune 60%'},
            {'type': 'quantize_8bit', 'name': '8-bit Quantization'},
            {'type': 'fgsm', 'epsilon': 0.01, 'name': 'FGSM ε=0.01'},
            {'type': 'fgsm', 'epsilon': 0.03, 'name': 'FGSM ε=0.03'},
            {'type': 'pgd', 'epsilon': 0.01, 'steps': 7, 'name': 'PGD ε=0.01'},
        ]

        total_experiments = len(payload_sizes) * len(attacks)
        experiment_count = 0

        # Run all combinations
        for payload_size in payload_sizes:
            for attack_config in attacks:
                experiment_count += 1
                print(f"\nProgress: {experiment_count}/{total_experiments}")

                try:
                    self.run_single_experiment(payload_size, attack_config)
                except Exception as e:
                    print(f"ERROR in experiment: {e}")
                    import traceback
                    traceback.print_exc()
                    self.results.append({
                        'run_id': len(self.results),
                        'timestamp': datetime.now().isoformat(),
                        'payload_size_bytes': payload_size,
                        'attack_type': attack_config['type'],
                        'notes': f'FAILED: {str(e)}',
                        'survival_rate': 0.0
                    })

        # Save results
        self.save_results()

        # Generate visualizations
        self.generate_visualizations()

        print("\n" + "="*80)
        print("BENCHMARK SUITE COMPLETE")
        print(f"Total experiments: {len(self.results)}")
        print(f"Results saved to: {self.output_dir}")
        print("="*80)

    def save_results(self):
        """Save results to CSV and JSON."""
        # CSV
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / 'benchmarks.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")

        # JSON (with full metadata)
        json_path = self.output_dir / 'benchmarks.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Detailed results saved to: {json_path}")

    def generate_visualizations(self):
        """Generate plots and dashboards."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = pd.DataFrame(self.results)

        # Plot 1: Capacity vs Accuracy
        plt.figure(figsize=(10, 6))
        plt.scatter(df['capacity_percent'], df['embedded_accuracy'], alpha=0.6)
        plt.xlabel('Capacity (% of model parameters)')
        plt.ylabel('Model Accuracy')
        plt.title('Capacity vs Accuracy Trade-off')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'capacity_vs_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 2: Survival Rate Heatmap
        pivot = df.pivot_table(
            values='survival_rate',
            index='attack_type',
            columns='payload_size_bytes',
            aggfunc='mean'
        )
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1)
        plt.title('Survival Rate Heatmap (Attack Type vs Payload Size)')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'survival_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 3: Statistical Undetectability
        identity_df = df[df['attack_type'] == 'identity']
        if not identity_df.empty and 'mean_pvalue' in identity_df.columns:
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(identity_df)), identity_df['mean_pvalue'])
            plt.axhline(y=0.05, color='r', linestyle='--', label='Detection Threshold')
            plt.xlabel('Experiment')
            plt.ylabel('Mean p-value')
            plt.title('Statistical Undetectability (Higher is Better)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.output_dir / 'statistical_undetectability.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"\nVisualizations saved to: {self.output_dir}")


if __name__ == '__main__':
    orchestrator = BenchmarkOrchestrator(output_dir='./results')
    orchestrator.run_comprehensive_benchmarks()
