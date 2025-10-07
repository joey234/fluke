#!/usr/bin/env python
"""
Statistical Analysis Module for FLUKE
Handles per-sample analysis and statistical significance testing with weighted delta metric.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import glob
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """Performs statistical analysis on FLUKE results with proper significance testing."""
    
    def __init__(self, base_path: str = "../"):
        """Initialize the statistical analyzer."""
        self.base_path = Path(base_path)
        
    def analyze_modification_impact(self, 
                                   original_file: str, 
                                   modified_file: str,
                                   task: str = None) -> Dict:
        """
        Analyze the impact of a modification with statistical testing.
        
        Args:
            original_file: Path to CSV with original predictions
            modified_file: Path to CSV with modified predictions
            task: Task name (optional, for context)
            
        Returns:
            Dictionary with metrics and statistical test results
        """
        # Load data
        orig_df = pd.read_csv(original_file)
        mod_df = pd.read_csv(modified_file)
        
        # Extract predictions and labels
        orig_pred, orig_label = self._extract_predictions_labels(orig_df, 'original')
        mod_pred, mod_label = self._extract_predictions_labels(mod_df, 'modified')
        
        # Create binary accuracy arrays (1 = correct, 0 = incorrect)
        orig_correct = np.array([1 if p == l else 0 for p, l in zip(orig_pred, orig_label)])
        mod_correct = np.array([1 if p == l else 0 for p, l in zip(mod_pred, mod_label)])
        
        # Calculate accuracies
        orig_acc = orig_correct.mean() * 100
        mod_acc = mod_correct.mean() * 100
        
        # Calculate metrics
        absolute_drop = orig_acc - mod_acc
        relative_drop = ((orig_acc - mod_acc) / orig_acc * 100) if orig_acc > 0 else 0
        
        # Calculate weighted delta metric
        if orig_acc > 0:
            weighted_delta = (mod_acc - orig_acc) * np.log10(orig_acc) / np.log10(100)
        else:
            weighted_delta = 0
        
        # Statistical significance testing
        significance_results = self._perform_significance_tests(orig_correct, mod_correct)
        
        return {
            'n_samples': len(orig_correct),
            'original_accuracy': round(orig_acc, 3),
            'modified_accuracy': round(mod_acc, 3),
            'absolute_drop': round(absolute_drop, 3),
            'relative_drop_%': round(relative_drop, 3),
            'weighted_delta': round(weighted_delta, 3),
            **significance_results
        }
    
    def _extract_predictions_labels(self, df: pd.DataFrame, data_type: str) -> Tuple[List, List]:
        """Extract predictions and labels from dataframe."""
        # Try different column naming conventions
        pred_cols = ['pred', 'prediction', f'{data_type}_pred', 'modified_pred', 'original_pred']
        label_cols = ['label', f'{data_type}_label', 'modified_label', 'original_label']
        
        predictions = None
        labels = None
        
        for col in pred_cols:
            if col in df.columns:
                predictions = df[col].tolist()
                break
        
        for col in label_cols:
            if col in df.columns:
                labels = df[col].tolist()
                break
        
        if predictions is None or labels is None:
            raise ValueError(f"Could not find prediction or label columns in dataframe. Columns: {df.columns.tolist()}")
        
        return predictions, labels
    
    def _perform_significance_tests(self, orig_correct: np.ndarray, mod_correct: np.ndarray) -> Dict:
        """
        Perform statistical significance tests on paired samples.
        
        Args:
            orig_correct: Binary array of correct/incorrect for original
            mod_correct: Binary array of correct/incorrect for modified
            
        Returns:
            Dictionary with test results
        """
        results = {}
        
        # Check if arrays are identical (no change)
        if np.array_equal(orig_correct, mod_correct):
            results['wilcoxon_pvalue'] = 1.0
            results['mannwhitney_pvalue'] = 1.0
            results['mcnemar_pvalue'] = 1.0
        else:
            # Wilcoxon signed-rank test (for paired samples)
            try:
                _, wilcoxon_p = stats.wilcoxon(orig_correct, mod_correct)
                results['wilcoxon_pvalue'] = wilcoxon_p
            except:
                results['wilcoxon_pvalue'] = 1.0
            
            # Mann-Whitney U test (treats as independent samples)
            try:
                _, mannwhitney_p = stats.mannwhitneyu(orig_correct, mod_correct, alternative='two-sided')
                results['mannwhitney_pvalue'] = mannwhitney_p
            except:
                results['mannwhitney_pvalue'] = 1.0
            
            # McNemar's test (specifically for paired binary data)
            try:
                # Create contingency table
                both_correct = np.sum((orig_correct == 1) & (mod_correct == 1))
                orig_only = np.sum((orig_correct == 1) & (mod_correct == 0))
                mod_only = np.sum((orig_correct == 0) & (mod_correct == 1))
                both_wrong = np.sum((orig_correct == 0) & (mod_correct == 0))
                
                # McNemar's test focuses on the discordant pairs
                from statsmodels.stats.contingency_tables import mcnemar
                table = [[both_correct, orig_only], [mod_only, both_wrong]]
                result = mcnemar(table, exact=False, correction=True)
                results['mcnemar_pvalue'] = result.pvalue
            except:
                results['mcnemar_pvalue'] = 1.0
        
        # Take minimum p-value as the main result
        results['min_pvalue'] = min(results['wilcoxon_pvalue'], 
                                    results['mannwhitney_pvalue'],
                                    results['mcnemar_pvalue'])
        
        # Determine significance level
        results['significance'] = self._get_significance_level(results['min_pvalue'])
        results['is_significant'] = results['min_pvalue'] < 0.05
        
        return results
    
    def _get_significance_level(self, pvalue: float) -> str:
        """Get significance level symbol."""
        if pvalue < 0.001:
            return "***"  # Very highly significant
        elif pvalue < 0.01:
            return "**"   # Highly significant
        elif pvalue < 0.05:
            return "*"    # Significant
        elif pvalue < 0.1:
            return "."    # Marginally significant
        else:
            return "ns"   # Not significant
    
    def analyze_task_modifications(self, task: str, model: str) -> pd.DataFrame:
        """
        Analyze all modifications for a specific task and model.
        
        Args:
            task: Task name (coref, dialogue, ner, sa)
            model: Model name
            
        Returns:
            DataFrame with all modification results
        """
        results = []
        
        # Find result files for this model and task
        results_dir = self.base_path / f"LLM/results/{task}"
        
        # Get baseline file
        baseline_patterns = [
            f"{model}-*-{task}.csv",
            f"{model}-*-sst2.csv",  # For sentiment analysis
            f"{model}-*-ner.csv"
        ]
        
        baseline_file = None
        for pattern in baseline_patterns:
            files = list(results_dir.glob(pattern))
            if files:
                baseline_file = files[0]
                break
        
        if not baseline_file:
            print(f"Warning: No baseline file found for {model} on {task}")
            return pd.DataFrame()
        
        # Find modification files
        mod_patterns = [
            f"{model}-*_100.csv",
            f"{model}-*-capitalization.csv",
            f"{model}-*-punctuation.csv",
            f"{model}-*-typo*.csv"
        ]
        
        mod_files = []
        for pattern in mod_patterns:
            mod_files.extend(results_dir.glob(pattern))
        
        # Analyze each modification
        for mod_file in mod_files:
            # Extract modification name from filename
            mod_name = mod_file.stem.split('-')[-1].replace('_100', '')
            
            try:
                result = self.analyze_modification_impact(
                    str(baseline_file),
                    str(mod_file),
                    task=task
                )
                result['model'] = model
                result['task'] = task
                result['modification'] = mod_name
                results.append(result)
            except Exception as e:
                print(f"Error analyzing {mod_file}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def create_summary_table(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Create a summary table with key metrics."""
        if results_df.empty:
            return pd.DataFrame()
        
        # Pivot to get modifications as rows, metrics as columns
        summary = results_df.pivot_table(
            values=['weighted_delta', 'min_pvalue', 'significance'],
            index='modification',
            aggfunc={'weighted_delta': 'mean', 
                    'min_pvalue': 'mean',
                    'significance': lambda x: x.mode()[0] if len(x) > 0 else 'ns'}
        )
        
        # Sort by weighted delta (most negative = biggest drop)
        summary = summary.sort_values('weighted_delta')
        
        return summary


def analyze_with_proper_statistics(task: str, model: str):
    """
    Example function showing how to use the statistical analyzer.
    """
    analyzer = StatisticalAnalyzer()
    
    # Analyze all modifications for a model
    results = analyzer.analyze_task_modifications(task, model)
    
    if not results.empty:
        print(f"\n{'='*60}")
        print(f"Statistical Analysis: {model.upper()} on {task.upper()}")
        print(f"{'='*60}\n")
        
        # Display results
        for _, row in results.iterrows():
            print(f"\nModification: {row['modification']}")
            print(f"  Original Accuracy: {row['original_accuracy']:.2f}%")
            print(f"  Modified Accuracy: {row['modified_accuracy']:.2f}%")
            print(f"  Weighted Delta: {row['weighted_delta']:.3f}")
            print(f"  Statistical Significance: {row['significance']} (p={row['min_pvalue']:.4f})")
            print(f"  Tests: Wilcoxon p={row['wilcoxon_pvalue']:.4f}, "
                  f"Mann-Whitney p={row['mannwhitney_pvalue']:.4f}, "
                  f"McNemar p={row['mcnemar_pvalue']:.4f}")
        
        # Create summary
        summary = analyzer.create_summary_table(results)
        print(f"\n{'='*60}")
        print("Summary Table")
        print(f"{'='*60}")
        print(summary)
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Statistical Analysis Module for FLUKE")
    print("="*60)
    
    # Test with a specific model and task
    # results = analyze_with_proper_statistics('coref', 'gpt-5')
    
    print("\nKey points for proper statistical testing:")
    print("1. Tests are performed on binary arrays (correct/incorrect) per sample")
    print("2. Wilcoxon test: For paired samples (same examples, different conditions)")
    print("3. Mann-Whitney U: For independent samples comparison")
    print("4. McNemar's test: Specifically for paired binary data")
    print("5. Weighted delta is calculated from aggregated accuracies")
    print("\nSignificance levels:")
    print("  *** : p < 0.001 (very highly significant)")
    print("  **  : p < 0.01  (highly significant)")
    print("  *   : p < 0.05  (significant)")
    print("  .   : p < 0.1   (marginally significant)")
    print("  ns  : p ≥ 0.1   (not significant)")