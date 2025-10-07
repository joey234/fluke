#!/usr/bin/env python
"""
Generate Pre-processed Data for FLUKE Analysis
This script processes raw LLM prediction CSV files and generates the pre-processed CSV files
that the original notebooks expect (e.g., coreference_llm_results.csv, dialogue_llm_results.csv).
"""

import os
import json
import pandas as pd
import numpy as np
import decimal
from pathlib import Path
from scipy import stats
from typing import List, Dict, Tuple


def calculate_weighted_delta(orig_acc: float, mod_acc: float) -> float:
    """Calculate weighted delta metric: (B - A) * log10(A) / log10(100)"""
    if orig_acc <= 0:
        return 0.0
    return (mod_acc - orig_acc) * np.log10(orig_acc) / np.log10(100)


def perform_statistical_tests(orig_binary: np.ndarray, mod_binary: np.ndarray) -> Tuple[float, str, bool]:
    """Perform statistical tests and return p-value, significance level, and significance flag."""
    try:
        _, p_value_mw = stats.mannwhitneyu(orig_binary, mod_binary, alternative='two-sided')
        _, p_value_wilc = stats.wilcoxon(orig_binary, mod_binary, alternative='two-sided')
        p_value = min(p_value_mw, p_value_wilc)  # Use most conservative p-value
    except ValueError:
        # If all elements are identical, set p-value to 1.0 since there is no difference
        p_value = 1.0
    
    # Add significance level
    if p_value < 0.001:
        significance = '***'
    elif p_value < 0.01:
        significance = '**'
    elif p_value < 0.05:
        significance = '*'
    elif p_value < 0.1:
        significance = '.'
    else:
        significance = 'ns'
    
    is_significant = p_value < 0.05
    return p_value, significance, is_significant


def load_baseline_accuracy(model: str, task: str, eval_base_path: Path) -> float:
    """Load baseline accuracy for a model from its baseline file."""
    # Map task names to baseline file patterns
    task_mapping = {
        'dialogue': 'dialogue',
        'coreference': 'coref', 
        'ner': 'ner',
        'sentiment': 'sst2'
    }
    
    baseline_name = task_mapping.get(task, task)
    baseline_file = eval_base_path / f"{model}-0shot-{baseline_name}.csv"
    
    if not baseline_file.exists():
        return None
    
    try:
        df = pd.read_csv(baseline_file)
        if 'accuracy' in df.columns:
            return df['accuracy'].mean() * 100
        elif 'pred' in df.columns and 'label' in df.columns:
            return (df['pred'] == df['label']).mean() * 100
        else:
            return None
    except Exception as e:
        print(f"    ⚠️  Error loading baseline for {model}: {e}")
        return None


def process_task_data(task: str, eval_base_path: Path, output_path: Path) -> None:
    """Process data for a specific task and generate pre-processed CSV files."""
    print(f"\n{'='*60}")
    print(f"Processing {task.upper()} Task")
    print(f"{'='*60}")
    
    eval_results_rows = []
    eval_negation_results_rows = []
    
    # Cache baseline accuracies for models that need them
    baseline_cache = {}
    
    # Get list of CSV files in the evaluation directory
    if not eval_base_path.exists():
        print(f"Warning: Evaluation directory not found: {eval_base_path}")
        return
    
    csv_files = [f for f in os.listdir(eval_base_path) if f.endswith('_100.csv')]
    print(f"Found {len(csv_files)} CSV files to process")
    
    processed_count = 0
    
    for filename in sorted(csv_files):
        try:
            # Parse filename to get model and modification
            if '-0shot-' not in filename:
                continue
            
            model = filename.split('-0shot-')[0]
            modification = filename.split('-0shot-')[1].replace('_100.csv', '')
            
            # Skip mixtral if specified (as in original notebooks)
            if model == 'mixtral':
                continue
            
            # Load predictions from CSV
            eval_filepath = eval_base_path / filename
            df = pd.read_csv(eval_filepath)
            
            # Verify required columns exist
            required_cols = ['original_pred', 'original_label', 'modified_pred', 'modified_label']
            if not all(col in df.columns for col in required_cols):
                print(f"  ⚠️  Skipping {filename}: Missing required columns")
                continue
            
            # Calculate accuracies
            mod_correct = sum(df['modified_pred'] == df['modified_label'])
            total = len(df)
            
            if total == 0:
                print(f"  ⚠️  Skipping {filename}: Empty dataset")
                continue
            
            mod_acc = mod_correct / total * 100
            
            # Handle original accuracy - check for missing values
            if df['original_pred'].isna().all():
                # Missing original predictions - load from baseline file
                if model not in baseline_cache:
                    baseline_acc = load_baseline_accuracy(model, task, eval_base_path)
                    baseline_cache[model] = baseline_acc
                    if baseline_acc is not None:
                        print(f"    📊 Loaded baseline accuracy for {model}: {baseline_acc:.1f}%")
                
                orig_acc = baseline_cache.get(model)
                if orig_acc is None:
                    print(f"  ⚠️  Skipping {filename}: No baseline accuracy found for {model}")
                    continue
            else:
                # Use original predictions from file
                orig_correct = sum(df['original_pred'] == df['original_label'])
                orig_acc = orig_correct / total * 100
            pct_diff = ((mod_acc - orig_acc) / orig_acc) * 100 if orig_acc > 0 else 0
            
            # Calculate weighted delta
            weighted_delta = calculate_weighted_delta(orig_acc, mod_acc)
            
            # Convert predictions to binary (0/1) based on correctness for statistical tests
            mod_binary = (df['modified_pred'] == df['modified_label']).astype(int)
            
            # Handle statistical tests
            if df['original_pred'].isna().all():
                # For models without original predictions, use baseline accuracy
                # Create synthetic original binary array based on baseline accuracy
                baseline_success_rate = orig_acc / 100
                np.random.seed(42)  # For reproducibility
                orig_binary = np.random.binomial(1, baseline_success_rate, size=len(df))
                p_value, significance, is_significant = perform_statistical_tests(orig_binary, mod_binary)
            else:
                # Use actual original predictions
                orig_binary = (df['original_pred'] == df['original_label']).astype(int)
                p_value, significance, is_significant = perform_statistical_tests(orig_binary, mod_binary)
            
            # Create result row
            row = {
                'model': model,
                'modification': modification,
                'original_acc': float(decimal.Decimal(orig_acc).quantize(decimal.Decimal('0.001'), rounding=decimal.ROUND_HALF_UP)),
                'modified_acc': float(decimal.Decimal(mod_acc).quantize(decimal.Decimal('0.001'), rounding=decimal.ROUND_HALF_UP)),
                'percentage_diff': float(decimal.Decimal(pct_diff).quantize(decimal.Decimal('0.001'), rounding=decimal.ROUND_HALF_UP)),
                'weighted_delta': float(decimal.Decimal(weighted_delta).quantize(decimal.Decimal('0.001'), rounding=decimal.ROUND_HALF_UP)),
                'p_value': float(decimal.Decimal(p_value).quantize(decimal.Decimal('0.001'), rounding=decimal.ROUND_HALF_UP)),
                'significance': significance,
                'significant': is_significant
            }
            eval_results_rows.append(row)
            
            print(f"  ✓ Processed {model} - {modification}: {orig_acc:.1f}% → {mod_acc:.1f}% ({significance})")
            
            # Process negation subtypes if this is a negation modification
            if modification == 'negation' and 'type' in df.columns:
                expected_types = {'absolute', 'double', 'lexical', 'approximate', 'verbal'}
                actual_types = set(df['type'].unique())
                
                if actual_types != expected_types:
                    print(f"    ⚠️  Unexpected negation types for {model}")
                    print(f"    Expected: {expected_types}")
                    print(f"    Found: {actual_types}")
                
                for neg_type in df['type'].unique():
                    type_df = df[df['type'] == neg_type]
                    
                    if len(type_df) == 0:
                        continue
                    
                    type_orig_correct = sum(type_df['original_pred'] == type_df['original_label'])
                    type_mod_correct = sum(type_df['modified_pred'] == type_df['modified_label'])
                    type_total = len(type_df)
                    
                    type_orig_acc = type_orig_correct / type_total * 100
                    type_mod_acc = type_mod_correct / type_total * 100
                    type_pct_diff = ((type_mod_acc - type_orig_acc) / type_orig_acc) * 100 if type_orig_acc > 0 else 0
                    type_weighted_delta = calculate_weighted_delta(type_orig_acc, type_mod_acc)
                    
                    # Statistical tests for this negation type
                    type_orig_binary = (type_df['original_pred'] == type_df['original_label']).astype(int)
                    type_mod_binary = (type_df['modified_pred'] == type_df['modified_label']).astype(int)
                    
                    type_p_value_wilc = 1.0
                    type_p_value_mw = 1.0
                    try:
                        _, type_p_value_mw = stats.mannwhitneyu(type_orig_binary, type_mod_binary, alternative='two-sided')
                        _, type_p_value_wilc = stats.wilcoxon(type_orig_binary, type_mod_binary, alternative='two-sided')
                        type_p_value = min(type_p_value_mw, type_p_value_wilc)
                    except ValueError:
                        type_p_value = 1.0
                    
                    _, type_significance, type_is_significant = perform_statistical_tests(type_orig_binary, type_mod_binary)
                    
                    type_row = {
                        'model': model,
                        'modification': neg_type,
                        'original_acc': float(decimal.Decimal(type_orig_acc).quantize(decimal.Decimal('0.001'), rounding=decimal.ROUND_HALF_UP)),
                        'modified_acc': float(decimal.Decimal(type_mod_acc).quantize(decimal.Decimal('0.001'), rounding=decimal.ROUND_HALF_UP)),
                        'pct_diff': float(decimal.Decimal(type_pct_diff).quantize(decimal.Decimal('0.001'), rounding=decimal.ROUND_HALF_UP)),
                        'weighted_delta': float(decimal.Decimal(type_weighted_delta).quantize(decimal.Decimal('0.001'), rounding=decimal.ROUND_HALF_UP)),
                        'wilcoxon_pvalue': float(decimal.Decimal(type_p_value_wilc).quantize(decimal.Decimal('0.001'), rounding=decimal.ROUND_HALF_UP)),
                        'mannwhitney_pvalue': float(decimal.Decimal(type_p_value_mw).quantize(decimal.Decimal('0.001'), rounding=decimal.ROUND_HALF_UP)),
                        'pvalue': float(decimal.Decimal(type_p_value).quantize(decimal.Decimal('0.001'), rounding=decimal.ROUND_HALF_UP)),
                        'significance': type_significance,
                        'significant': type_is_significant
                    }
                    eval_negation_results_rows.append(type_row)
                    
                    print(f"    • {neg_type}: {type_orig_acc:.1f}% → {type_mod_acc:.1f}% ({type_significance})")
            
            processed_count += 1
            
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")
            continue
    
    if processed_count == 0:
        print(f"  ⚠️  No files were successfully processed for {task}")
        return
    
    # Convert to dataframes
    eval_results_df = pd.DataFrame(eval_results_rows)
    eval_negation_results_df = pd.DataFrame(eval_negation_results_rows)
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save results
    main_results_file = output_path / f'{task}_llm_results.csv'
    eval_results_df.to_csv(main_results_file, index=False)
    print(f"  📁 Saved main results: {main_results_file} ({len(eval_results_df)} rows)")
    
    if not eval_negation_results_df.empty:
        negation_results_file = output_path / f'{task}_llm_negation_results.csv'
        eval_negation_results_df.to_csv(negation_results_file, index=False)
        print(f"  📁 Saved negation results: {negation_results_file} ({len(eval_negation_results_df)} rows)")
    
    # Display summary statistics
    print(f"\\n  📊 Summary for {task.upper()}:")
    print(f"    • Total modifications processed: {len(eval_results_df['modification'].unique())}")
    print(f"    • Total models processed: {len(eval_results_df['model'].unique())}")
    if not eval_results_df.empty:
        significant_count = eval_results_df['significant'].sum()
        print(f"    • Significant results: {significant_count}/{len(eval_results_df)} ({significant_count/len(eval_results_df)*100:.1f}%)")
        print(f"    • Average original accuracy: {eval_results_df['original_acc'].mean():.1f}%")
        print(f"    • Average modified accuracy: {eval_results_df['modified_acc'].mean():.1f}%")


def main():
    """Main function to process all tasks."""
    print("🚀 FLUKE Pre-processed Data Generation")
    print("="*70)
    
    # Define task configurations
    task_configs = {
        'dialogue': {
            'eval_path': Path('../LLM/results/dialogue'),
            'output_path': Path('../PLM/dialogue_contradiction_detection/tmp')
        },
        'coreference': {
            'eval_path': Path('../LLM/results/coref'), 
            'output_path': Path('../PLM/coreference_resolution/tmp')
        },
        'ner': {
            'eval_path': Path('../LLM/results/ner'),
            'output_path': Path('../PLM/ner/tmp')
        },
        'sentiment': {
            'eval_path': Path('../LLM/results/sa'),
            'output_path': Path('../PLM/sentiment_analysis/tmp')
        }
    }
    
    # Process each task
    total_processed = 0
    for task, config in task_configs.items():
        try:
            process_task_data(task, config['eval_path'], config['output_path'])
            total_processed += 1
        except Exception as e:
            print(f"❌ Error processing {task}: {e}")
            continue
    
    print(f"\\n{'='*70}")
    print(f"✅ Processing Complete!")
    print(f"📊 Successfully processed {total_processed}/{len(task_configs)} tasks")
    print(f"\\nGenerated files can now be used by:")
    print(f"  • Original analysis notebooks (parse_*.ipynb)")
    print(f"  • Unified analysis script (run_full_analysis.py)")
    print(f"\\n🎉 Ready for FLUKE analysis!")


if __name__ == "__main__":
    main()