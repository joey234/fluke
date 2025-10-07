#!/usr/bin/env python
"""
Consolidated FLUKE Analysis Framework
This script consolidates and cleans up the analysis from parse_coref_dialog, parse_ner, and parse_sa notebooks.
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import glob
import ast
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class FLUKEAnalyzer:
    """Main analyzer class for FLUKE experiment results."""
    
    def __init__(self, base_path: str = "../"):
        """Initialize the analyzer with base path to experiments."""
        self.base_path = Path(base_path)
        self.results = {}
        self.modification_types = [
            'active_to_passive', 'capitalization', 'casual', 'compound_word',
            'concept_replacement', 'coordinating_conjunction', 'derivation',
            'dialectal', 'discourse', 'geographical_bias', 'grammatical_role',
            'length_bias', 'negation', 'punctuation', 'sentiment', 
            'temporal_bias', 'typo_bias'
        ]
        self.models = {
            'plm': ['bert', 'gpt2', 't5'],
            'llm': ['gpt4o', 'claude-3-5-sonnet', 'llama', 'mixtral', 
                    'gpt-5-standard', 'gpt-5', 'deepseek-r1-deepseek', 
                    'deepseek-r1', 'deepseek', 'o1', 'o3']
        }
        
    def load_results(self, task: str, include_plm: bool = False) -> pd.DataFrame:
        """Load results for a specific task (coref, dialogue, ner, sa)."""
        results_data = []
        
        # Load PLM results (optional)
        if include_plm:
            plm_path = self.base_path / f"PLM/{self._get_plm_task_dir(task)}"
            if plm_path.exists():
                results_data.extend(self._load_plm_results(plm_path, task))
            
        # Load LLM results  
        llm_path = self.base_path / f"LLM/results/{self._get_llm_task_dir(task)}"
        if llm_path.exists():
            results_data.extend(self._load_llm_results(llm_path, task))
            
        return pd.DataFrame(results_data)
    
    def _get_plm_task_dir(self, task: str) -> str:
        """Map task name to PLM directory structure."""
        mapping = {
            'coref': 'coreference_resolution',
            'dialogue': 'dialogue_contradiction_detection',
            'ner': 'ner',
            'sa': 'sentiment_analysis'
        }
        return mapping.get(task, task)
    
    def _get_llm_task_dir(self, task: str) -> str:
        """Map task name to LLM directory structure."""
        mapping = {
            'coref': 'coref',
            'dialogue': 'dialogue', 
            'ner': 'ner',
            'sa': 'sa'
        }
        return mapping.get(task, task)
    
    def _get_combined_results_path(self, task: str) -> Path:
        """Get path to pre-processed combined results file (notebook style)."""
        task_mapping = {
            'coref': 'coreference',
            'dialogue': 'dialogue',
            'ner': 'ner', 
            'sa': 'sentiment'
        }
        
        plm_task_mapping = {
            'coref': 'coreference_resolution',
            'dialogue': 'dialogue_contradiction_detection',
            'ner': 'ner',
            'sa': 'sentiment_analysis'
        }
        
        task_name = task_mapping.get(task, task)
        plm_task_name = plm_task_mapping.get(task, task)
        
        # Try PLM directory first (where notebooks expect files)
        combined_file = self.base_path / "PLM" / plm_task_name / "tmp" / f"{task_name}_combined_results.csv"
        if combined_file.exists():
            return combined_file
            
        # Also try llm_results variant
        llm_file = self.base_path / "PLM" / plm_task_name / "tmp" / f"{task_name}_llm_results.csv"
        if llm_file.exists():
            return llm_file
            
        return None
    
    def _load_plm_results(self, path: Path, task: str) -> List[Dict]:
        """Load PLM results from CSV files."""
        results = []
        for model in self.models['plm']:
            model_results = self._process_plm_model_results(path, model, task)
            results.extend(model_results)
        return results
    
    def _load_llm_results(self, path: Path, task: str) -> List[Dict]:
        """Load LLM results from CSV files."""
        results = []
        
        # First, try to load pre-processed combined results file (notebook style)
        combined_file = self._get_combined_results_path(task)
        if combined_file and combined_file.exists():
            try:
                df = pd.read_csv(combined_file)
                for _, row in df.iterrows():
                    # Handle different boolean representations for significance
                    significant_val = row.get('significant', False)
                    if isinstance(significant_val, str):
                        significant_val = significant_val.lower() in ['yes', 'true', '1']
                    
                    result = {
                        'model': row['model'],
                        'modification': row['modification'],
                        'task': task,
                        'model_type': 'llm',
                        'accuracy': row.get('modified_acc', row.get('accuracy', 0)),
                        'original_accuracy': row.get('original_acc', None),
                        'weighted_delta': row.get('weighted_delta', None),
                        'p_value': row.get('p_value', None),
                        'significant': significant_val,
                        'significance': row.get('significance', 'ns')
                    }
                    results.append(result)
                return results
            except Exception as e:
                print(f"Error loading combined results {combined_file}: {e}")
        
        # Fallback to raw CSV processing (original approach)
        csv_files = glob.glob(str(path / "*.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                filename = Path(csv_file).stem
                
                # Parse model and modification from filename
                model, modification = self._parse_llm_filename(filename)
                if model and modification:
                    accuracy = self._calculate_accuracy(df)
                    # Only add if we have valid accuracy
                    if accuracy >= 0:
                        result = {
                            'model': model,
                            'modification': modification,
                            'task': task,
                            'model_type': 'llm',
                            'accuracy': accuracy,
                            'total_samples': len(df),
                            'filename': filename
                        }
                        results.append(result)
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue
                
        return results
    
    def _process_plm_model_results(self, path: Path, model: str, task: str) -> List[Dict]:
        """Process PLM model results."""
        results = []
        
        # Look for accuracy comparison CSV
        comparison_file = path / f"{model}_accuracy_comparison.csv"
        if comparison_file.exists():
            df = pd.read_csv(comparison_file)
            for _, row in df.iterrows():
                result = {
                    'model': model,
                    'modification': row.get('modification', 'original'),
                    'task': task,
                    'model_type': 'plm',
                    'accuracy': row.get('accuracy', 0),
                    'total_samples': row.get('total_samples', 0)
                }
                results.append(result)
                
        return results
    
    def _parse_llm_filename(self, filename: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse LLM result filename to extract model and modification."""
        # Handle special cases for new models
        if filename.startswith('gpt-5-standard'):
            model = 'gpt-5'
            remaining = filename.replace('gpt-5-standard-', '')
        elif filename.startswith('deepseek-r1-deepseek'):
            model = 'deepseek-r1'
            remaining = filename.replace('deepseek-r1-deepseek-', '')
        else:
            parts = filename.split('-')
            if len(parts) >= 3:
                model = parts[0]
                remaining = '-'.join(parts[1:])
            else:
                return None, None
        
        # Extract modification from remaining part
        # Handle patterns like "0shot-modification_100" or "0shot-modification"
        if '0shot-' in remaining:
            modification = remaining.split('0shot-')[-1].replace('_100', '')
        else:
            modification = remaining.replace('_100', '')
            
        return model, modification
    
    def _calculate_accuracy(self, df: pd.DataFrame) -> float:
        """Calculate accuracy from dataframe."""
        if 'correct' in df.columns:
            return df['correct'].mean() * 100
        elif 'accuracy' in df.columns:
            return df['accuracy'].mean() * 100
        elif 'label' in df.columns and 'pred' in df.columns:
            # Calculate accuracy from label and pred columns
            correct = (df['label'] == df['pred']).sum()
            total = len(df)
            return (correct / total * 100) if total > 0 else 0.0
        elif 'modified_label' in df.columns and 'modified_pred' in df.columns:
            # Calculate accuracy for modified/coref task data
            correct = (df['modified_label'] == df['modified_pred']).sum()
            total = len(df)
            return (correct / total * 100) if total > 0 else 0.0
        elif 'original_label' in df.columns and 'original_pred' in df.columns:
            # Fallback to original predictions if modified not available
            correct = (df['original_label'] == df['original_pred']).sum()
            total = len(df)
            return (correct / total * 100) if total > 0 else 0.0
        else:
            return 0.0
    
    def analyze_performance_drop(self, task: str, use_weighted_delta: bool = True) -> pd.DataFrame:
        """Analyze performance drop for each modification using weighted delta metric.
        
        The weighted delta metric is: (B - A) * log10(A) / log10(100)
        where A is original accuracy and B is modified accuracy.
        """
        df = self.load_results(task)
        
        if df.empty:
            return pd.DataFrame()
        
        # Calculate performance drops
        drops = []
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            
            # Get original/baseline performance
            baseline = model_df[model_df['modification'] == 'original']
            if baseline.empty:
                baseline = model_df[model_df['modification'] == task]
                
            if not baseline.empty:
                baseline_acc = baseline.iloc[0]['accuracy']
                
                for _, row in model_df.iterrows():
                    if row['modification'] != 'original' and row['modification'] != task:
                        modified_acc = row['accuracy']
                        
                        # Calculate different drop metrics
                        absolute_drop = baseline_acc - modified_acc
                        relative_drop_pct = (absolute_drop / baseline_acc) * 100 if baseline_acc > 0 else 0
                        
                        # Calculate weighted delta metric (FLUKE metric)
                        if use_weighted_delta and baseline_acc > 0:
                            weighted_delta = (modified_acc - baseline_acc) * np.log10(baseline_acc) / np.log10(100)
                        else:
                            weighted_delta = 0
                        
                        drops.append({
                            'model': model,
                            'model_type': row['model_type'],
                            'modification': row['modification'],
                            'baseline_accuracy': baseline_acc,
                            'modified_accuracy': modified_acc,
                            'absolute_drop': absolute_drop,
                            'relative_drop_%': relative_drop_pct,
                            'weighted_delta': round(weighted_delta, 3)
                        })
                        
        return pd.DataFrame(drops)
    
    def statistical_significance_test(self, original_file: str, modified_file: str) -> Dict:
        """
        Perform statistical significance test between original and modified predictions.
        
        NOTE: For proper statistical testing, we need per-sample predictions, not aggregated accuracies.
        This method expects CSV files with individual predictions and labels.
        
        Args:
            original_file: Path to CSV with original predictions
            modified_file: Path to CSV with modified predictions
            
        Returns:
            Dictionary with test results and significance levels
        """
        try:
            # Load prediction files
            orig_df = pd.read_csv(original_file)
            mod_df = pd.read_csv(modified_file)
            
            # Extract predictions and labels
            orig_pred, orig_label = self._extract_pred_label(orig_df)
            mod_pred, mod_label = self._extract_pred_label(mod_df)
            
            # Create binary correctness arrays
            orig_correct = np.array([1 if p == l else 0 for p, l in zip(orig_pred, orig_label)])
            mod_correct = np.array([1 if p == l else 0 for p, l in zip(mod_pred, mod_label)])
            
            # Calculate accuracies
            orig_acc = orig_correct.mean() * 100
            mod_acc = mod_correct.mean() * 100
            
            # Perform statistical tests
            results = {
                'original_accuracy': round(orig_acc, 3),
                'modified_accuracy': round(mod_acc, 3),
                'weighted_delta': round((mod_acc - orig_acc) * np.log10(orig_acc) / np.log10(100), 3) if orig_acc > 0 else 0
            }
            
            # Statistical tests
            if np.array_equal(orig_correct, mod_correct):
                results['wilcoxon_pvalue'] = 1.0
                results['mannwhitney_pvalue'] = 1.0
            else:
                # Wilcoxon for paired samples
                try:
                    _, results['wilcoxon_pvalue'] = stats.wilcoxon(orig_correct, mod_correct)
                except:
                    results['wilcoxon_pvalue'] = 1.0
                
                # Mann-Whitney for independent samples
                try:
                    _, results['mannwhitney_pvalue'] = stats.mannwhitneyu(orig_correct, mod_correct, alternative='two-sided')
                except:
                    results['mannwhitney_pvalue'] = 1.0
            
            # Determine significance
            min_pvalue = min(results['wilcoxon_pvalue'], results['mannwhitney_pvalue'])
            results['min_pvalue'] = min_pvalue
            results['significant'] = min_pvalue < 0.05
            
            if min_pvalue < 0.001:
                results['significance'] = '***'
            elif min_pvalue < 0.01:
                results['significance'] = '**'
            elif min_pvalue < 0.05:
                results['significance'] = '*'
            elif min_pvalue < 0.1:
                results['significance'] = '.'
            else:
                results['significance'] = 'ns'
                
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_pred_label(self, df: pd.DataFrame) -> Tuple[List, List]:
        """Extract predictions and labels from dataframe."""
        # Try common column names
        pred_cols = ['pred', 'prediction', 'modified_pred', 'original_pred']
        label_cols = ['label', 'modified_label', 'original_label']
        
        pred = None
        label = None
        
        for col in pred_cols:
            if col in df.columns:
                pred = df[col].tolist()
                break
        
        for col in label_cols:
            if col in df.columns:
                label = df[col].tolist()
                break
                
        if pred is None or label is None:
            raise ValueError(f"Could not find prediction or label columns. Available: {df.columns.tolist()}")
            
        return pred, label
    
    def generate_summary_report(self, output_path: str = "analysis_report.md"):
        """Generate a comprehensive summary report."""
        report = []
        report.append("# FLUKE Analysis Summary Report\n")
        
        tasks = ['coref', 'dialogue', 'ner', 'sa']
        
        for task in tasks:
            report.append(f"\n## {task.upper()} Task Analysis\n")
            
            # Load and analyze results
            df = self.load_results(task)
            if not df.empty:
                # Overall statistics
                report.append(f"### Overall Statistics\n")
                report.append(f"- Total models evaluated: {df['model'].nunique()}\n")
                report.append(f"- Total modifications tested: {df['modification'].nunique()}\n")
                report.append(f"- Average accuracy: {df['accuracy'].mean():.2f}%\n")
                
                # Top performing models
                top_models = df.groupby('model')['accuracy'].mean().sort_values(ascending=False).head(5)
                report.append(f"\n### Top 5 Performing Models\n")
                for model, acc in top_models.items():
                    report.append(f"- {model}: {acc:.2f}%\n")
                
                # Most challenging modifications
                drops_df = self.analyze_performance_drop(task)
                if not drops_df.empty:
                    challenging = drops_df.groupby('modification')['relative_drop_%'].mean().sort_values(ascending=False).head(5)
                    report.append(f"\n### Most Challenging Modifications (Avg. Drop)\n")
                    for mod, drop in challenging.items():
                        report.append(f"- {mod}: {drop:.2f}% drop\n")
                        
        # Save report
        with open(output_path, 'w') as f:
            f.writelines(report)
            
        print(f"Report saved to {output_path}")
        return ''.join(report)


class NERAnalyzer:
    """Specialized analyzer for NER task results."""
    
    def __init__(self, base_path: str = "../"):
        self.base_path = Path(base_path)
        self.entity_types = ['person', 'location', 'organization', 'misc']
        
    def analyze_entity_level_performance(self, results_path: str) -> pd.DataFrame:
        """Analyze performance at entity type level."""
        # Implementation for entity-level analysis
        pass
    
    def compute_f1_scores(self, predictions_file: str, ground_truth_file: str) -> Dict:
        """Compute precision, recall, and F1 scores."""
        # Implementation for F1 score computation
        pass


class DialogueAnalyzer:
    """Specialized analyzer for Dialogue task results."""
    
    def __init__(self, base_path: str = "../"):
        self.base_path = Path(base_path)
        
    def analyze_contradiction_patterns(self, results_path: str) -> pd.DataFrame:
        """Analyze patterns in contradiction detection."""
        # Implementation for contradiction pattern analysis
        pass


class SentimentAnalyzer:
    """Specialized analyzer for Sentiment Analysis task results."""
    
    def __init__(self, base_path: str = "../"):
        self.base_path = Path(base_path)
        self.sentiment_classes = ['positive', 'negative', 'neutral']
        
    def analyze_sentiment_shift(self, original_df: pd.DataFrame, modified_df: pd.DataFrame) -> Dict:
        """Analyze how modifications affect sentiment predictions."""
        shifts = {
            'positive_to_negative': 0,
            'positive_to_neutral': 0,
            'negative_to_positive': 0,
            'negative_to_neutral': 0,
            'neutral_to_positive': 0,
            'neutral_to_negative': 0,
            'no_change': 0
        }
        
        # Implementation for sentiment shift analysis
        return shifts


def main():
    """Main execution function."""
    print("Starting FLUKE Consolidated Analysis...")
    
    # Initialize analyzer
    analyzer = FLUKEAnalyzer()
    
    # Generate comprehensive report
    report = analyzer.generate_summary_report("fluke_analysis_report.md")
    
    # Analyze each task
    tasks = ['coref', 'dialogue', 'ner', 'sa']
    
    for task in tasks:
        print(f"\nAnalyzing {task.upper()} task...")
        
        # Load results
        df = analyzer.load_results(task)
        if not df.empty:
            # Save consolidated results
            output_file = f"consolidated_{task}_results.csv"
            df.to_csv(output_file, index=False)
            print(f"  - Saved consolidated results to {output_file}")
            
            # Analyze performance drops
            drops_df = analyzer.analyze_performance_drop(task)
            if not drops_df.empty:
                drops_file = f"performance_drops_{task}.csv"
                drops_df.to_csv(drops_file, index=False)
                print(f"  - Saved performance drop analysis to {drops_file}")
        else:
            print(f"  - No results found for {task}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
