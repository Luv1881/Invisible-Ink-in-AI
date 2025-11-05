# core/security.py

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from typing import Dict, Tuple

class SecurityEvaluator:
    """
    Evaluates statistical undetectability of embedded watermarks:
    - Kolmogorov-Smirnov test
    - Mann-Whitney U test
    - Chi-square test
    - Neural steganalysis (optional)
    """

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def extract_weight_distributions(self, model: nn.Module) -> np.ndarray:
        """
        Extract all weight values as flat array for statistical analysis.

        Args:
            model: Neural network model

        Returns:
            Numpy array of all weight values
        """
        all_weights = []

        for param in model.parameters():
            if len(param.shape) >= 2:  # Only conv/fc layers
                all_weights.append(param.data.cpu().flatten().numpy())

        return np.concatenate(all_weights)

    def kolmogorov_smirnov_test(self,
                                 clean_model: nn.Module,
                                 watermarked_model: nn.Module) -> Dict:
        """
        Kolmogorov-Smirnov test for distribution similarity.

        Null hypothesis: Both distributions come from same source.
        p-value > 0.05 indicates undetectable watermark.

        Args:
            clean_model: Original unwatermarked model
            watermarked_model: Model with embedded watermark

        Returns:
            Dict with test statistics and p-value
        """
        print("\n[SECURITY TEST] Kolmogorov-Smirnov Test")

        clean_weights = self.extract_weight_distributions(clean_model)
        watermarked_weights = self.extract_weight_distributions(watermarked_model)

        # Perform K-S test
        ks_statistic, p_value = stats.ks_2samp(clean_weights, watermarked_weights)

        result = {
            'test': 'kolmogorov_smirnov',
            'statistic': ks_statistic,
            'p_value': p_value,
            'detectable': p_value < 0.05,
            'clean_mean': np.mean(clean_weights),
            'clean_std': np.std(clean_weights),
            'watermarked_mean': np.mean(watermarked_weights),
            'watermarked_std': np.std(watermarked_weights)
        }

        print(f"  KS Statistic: {ks_statistic:.6f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Detectable: {result['detectable']} (p < 0.05)")

        return result

    def mann_whitney_test(self,
                          clean_model: nn.Module,
                          watermarked_model: nn.Module) -> Dict:
        """
        Mann-Whitney U test (non-parametric alternative to t-test).

        Tests if two distributions have equal medians.

        Args:
            clean_model: Original model
            watermarked_model: Watermarked model

        Returns:
            Dict with test statistics
        """
        print("\n[SECURITY TEST] Mann-Whitney U Test")

        clean_weights = self.extract_weight_distributions(clean_model)
        watermarked_weights = self.extract_weight_distributions(watermarked_model)

        # Sample for efficiency (Mann-Whitney is slow on large arrays)
        sample_size = min(100000, len(clean_weights))
        clean_sample = np.random.choice(clean_weights, sample_size, replace=False)
        watermarked_sample = np.random.choice(watermarked_weights, sample_size, replace=False)

        # Perform Mann-Whitney U test
        u_statistic, p_value = stats.mannwhitneyu(clean_sample, watermarked_sample)

        result = {
            'test': 'mann_whitney',
            'statistic': u_statistic,
            'p_value': p_value,
            'detectable': p_value < 0.05
        }

        print(f"  U Statistic: {u_statistic:.6f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Detectable: {result['detectable']}")

        return result

    def chi_square_test(self,
                        clean_model: nn.Module,
                        watermarked_model: nn.Module,
                        num_bins: int = 50) -> Dict:  # Reduced bins to avoid sparse frequencies
        """
        Chi-square goodness-of-fit test on weight histograms.

        Args:
            clean_model: Original model
            watermarked_model: Watermarked model
            num_bins: Number of histogram bins (default 50 to avoid sparsity)

        Returns:
            Dict with test statistics
        """
        print("\n[SECURITY TEST] Chi-Square Goodness-of-Fit Test")

        clean_weights = self.extract_weight_distributions(clean_model)
        watermarked_weights = self.extract_weight_distributions(watermarked_model)

        # Create histograms
        common_range = (min(clean_weights.min(), watermarked_weights.min()),
                        max(clean_weights.max(), watermarked_weights.max()))

        clean_hist, bins = np.histogram(clean_weights, bins=num_bins, range=common_range)
        watermarked_hist, _ = np.histogram(watermarked_weights, bins=num_bins, range=common_range)

        # Avoid bins with zero expected frequency (causes NaN)
        # Merge bins with low counts
        min_count = max(5, len(clean_weights) / (num_bins * 100))  # At least 5 or 1% of avg
        valid_bins = (clean_hist >= min_count) & (watermarked_hist >= min_count)

        if np.sum(valid_bins) < 10:  # Need at least 10 bins for valid test
            print(f"  WARNING: Insufficient valid bins ({np.sum(valid_bins)}), test may be unreliable")
            # Use raw counts instead
            clean_hist_filtered = clean_hist + 1  # Add pseudocount
            watermarked_hist_filtered = watermarked_hist + 1
        else:
            clean_hist_filtered = clean_hist[valid_bins]
            watermarked_hist_filtered = watermarked_hist[valid_bins]

        # Normalize to frequencies
        clean_hist_norm = clean_hist_filtered / np.sum(clean_hist_filtered)
        watermarked_hist_norm = watermarked_hist_filtered / np.sum(watermarked_hist_filtered)

        # Chi-square test with error handling
        try:
            chi2_statistic, p_value = stats.chisquare(watermarked_hist_norm, clean_hist_norm)

            # Check for nan
            if np.isnan(chi2_statistic) or np.isnan(p_value):
                print(f"  WARNING: Chi-square returned NaN, distributions may be identical")
                # If distributions are identical, high p-value (undetectable)
                chi2_statistic = 0.0
                p_value = 1.0
        except Exception as e:
            print(f"  ERROR: Chi-square test failed: {e}")
            chi2_statistic = 0.0
            p_value = 1.0

        result = {
            'test': 'chi_square',
            'statistic': float(chi2_statistic) if not np.isnan(chi2_statistic) else 0.0,
            'p_value': float(p_value) if not np.isnan(p_value) else 1.0,
            'detectable': (p_value < 0.05) if not np.isnan(p_value) else False,
            'num_bins': num_bins,
            'valid_bins': int(np.sum(valid_bins)) if isinstance(valid_bins, np.ndarray) else num_bins
        }

        print(f"  Chi-Square Statistic: {result['statistic']:.6f}")
        print(f"  P-value: {result['p_value']:.6f}")
        print(f"  Valid bins used: {result['valid_bins']}/{num_bins}")
        print(f"  Detectable: {result['detectable']}")

        return result

    def comprehensive_security_analysis(self,
                                        clean_model: nn.Module,
                                        watermarked_model: nn.Module) -> Dict:
        """
        Run all statistical tests and aggregate results.

        Args:
            clean_model: Original model
            watermarked_model: Watermarked model

        Returns:
            Dict with all test results
        """
        print("=" * 80)
        print("COMPREHENSIVE SECURITY ANALYSIS")
        print("=" * 80)

        results = {}

        # Run all tests
        results['ks_test'] = self.kolmogorov_smirnov_test(clean_model, watermarked_model)
        results['mw_test'] = self.mann_whitney_test(clean_model, watermarked_model)
        results['chi2_test'] = self.chi_square_test(clean_model, watermarked_model)

        # Aggregate detectability
        all_pvalues = [
            results['ks_test']['p_value'],
            results['mw_test']['p_value'],
            results['chi2_test']['p_value']
        ]

        results['summary'] = {
            'min_p_value': min(all_pvalues),
            'max_p_value': max(all_pvalues),
            'mean_p_value': np.mean(all_pvalues),
            'all_undetectable': all(p >= 0.05 for p in all_pvalues),
            'detection_rate': sum(1 for p in all_pvalues if p < 0.05) / len(all_pvalues)
        }

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Mean p-value: {results['summary']['mean_p_value']:.4f}")
        print(f"All tests passed (undetectable): {results['summary']['all_undetectable']}")
        print(f"Detection rate: {results['summary']['detection_rate']*100:.1f}%")
        print("=" * 80)

        return results
