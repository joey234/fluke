#!/usr/bin/env python
"""
LLM Benchmark Analysis for FLUKE
Focuses on main benchmark results, separating them from modification tests.
"""

import sys
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from consolidated_analysis import FLUKEAnalyzer
from utils import FileOrganizer

def separate_benchmark_and_modifications(df):
    """Separate main benchmark results from modification tests."""
    # Main benchmark patterns
    benchmark_patterns = ['coref', 'dialogue', 'ner', 'sst2', 'sa']
    
    # Identify main benchmark results (no "_100" suffix, matches benchmark pattern)
    df['is_modification'] = df['modification'].str.contains('_100|capitalization|punctuation|typo|passive|compound|grammatical|geographical|temporal|discourse|negation|sentiment|dialectal|derivation|concept_replacement|coordinating|length_bias|singlish|casual', na=False)
    df['is_benchmark'] = df['modification'].isin(benchmark_patterns) | df['modification'].str.contains('sst2|cot-coref|cot-dialogue|cot-sst2', na=False)
    
    benchmark_df = df[df['is_benchmark']].copy()
    modification_df = df[df['is_modification']].copy()
    
    return benchmark_df, modification_df

def main():
    """Run focused LLM benchmark analysis."""
    print("\n" + "="*80)
    print("       FLUKE LLM Benchmark Analysis - Main Results Focus")
    print("="*80)
    
    # Initialize components
    analyzer = FLUKEAnalyzer("../")
    organizer = FileOrganizer("llm_benchmark_output")
    
    tasks = ['coref', 'dialogue', 'ner', 'sa']
    
    # Collect all results
    all_results = []
    
    print("\n📊 Loading LLM Benchmark Results...")
    print("-" * 60)
    
    for task in tasks:
        df = analyzer.load_results(task, include_plm=False)
        if not df.empty:
            df['task'] = task
            all_results.append(df)
    
    if not all_results:
        print("❌ No results found!")
        return
    
    # Combine and separate results
    combined_df = pd.concat(all_results, ignore_index=True)
    benchmark_df, modification_df = separate_benchmark_and_modifications(combined_df)
    
    print(f"\n✓ Total entries loaded: {len(combined_df)}")
    print(f"  • Main benchmark results: {len(benchmark_df)}")
    print(f"  • Modification tests: {len(modification_df)}")
    
    # Generate report
    report_lines = []
    report_lines.append("# FLUKE LLM Benchmark Analysis Report\n")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("\n---\n")
    
    # Main Benchmark Results Section
    report_lines.append("## Main Benchmark Results\n")
    report_lines.append("Performance on full benchmark datasets (not modification subsets):\n\n")
    
    # Create pivot table for main benchmarks
    benchmark_pivot = benchmark_df.pivot_table(
        values='accuracy',
        index='model',
        columns='modification',
        aggfunc='mean'
    ).round(2)
    
    # Calculate average for each model
    benchmark_pivot['Average'] = benchmark_pivot.mean(axis=1).round(2)
    benchmark_pivot = benchmark_pivot.sort_values('Average', ascending=False)
    
    # Model ranking table
    report_lines.append("### Model Performance Ranking\n")
    report_lines.append("| Rank | Model | Average | Tasks Evaluated |\n")
    report_lines.append("|------|-------|---------|----------------|\n")
    
    for rank, (model, row) in enumerate(benchmark_pivot.iterrows(), 1):
        tasks_count = row.dropna().count() - 1  # Exclude Average column
        avg = row['Average']
        report_lines.append(f"| {rank} | {model} | {avg:.2f}% | {tasks_count} |\n")
    
    # Detailed benchmark results
    report_lines.append("\n### Detailed Benchmark Results\n")
    report_lines.append("Accuracy (%) on main benchmark tasks:\n\n")
    
    # Format the pivot table as markdown
    cols = [col for col in benchmark_pivot.columns if col != 'Average']
    report_lines.append("| Model | " + " | ".join(cols) + " | Average |\n")
    report_lines.append("|-------|" + "---------|" * (len(cols) + 1) + "\n")
    
    for model, row in benchmark_pivot.iterrows():
        values = []
        for col in cols:
            val = row[col]
            values.append(f"{val:.2f}" if not pd.isna(val) else "-")
        values.append(f"{row['Average']:.2f}")
        report_lines.append(f"| {model} | " + " | ".join(values) + " |\n")
    
    # Task-specific analysis
    report_lines.append("\n## Task-Specific Analysis\n")
    
    task_mapping = {
        'coref': 'Coreference Resolution',
        'dialogue': 'Dialogue Understanding',
        'ner': 'Named Entity Recognition', 
        'sa': 'Sentiment Analysis'
    }
    
    for task, task_name in task_mapping.items():
        task_df = benchmark_df[benchmark_df['task'] == task]
        if not task_df.empty:
            report_lines.append(f"\n### {task_name}\n")
            
            # Get performance for main benchmark (not modifications)
            main_results = task_df[task_df['modification'].isin([task, 'sst2', f'cot-{task}', 'cot-sst2'])]
            
            if not main_results.empty:
                top_models = main_results.groupby('model')['accuracy'].max().sort_values(ascending=False)
                
                report_lines.append("Top Performers:\n")
                for i, (model, acc) in enumerate(top_models.head(5).items(), 1):
                    report_lines.append(f"{i}. **{model}**: {acc:.2f}%\n")
                
                # Task statistics
                report_lines.append(f"\nStatistics:\n")
                report_lines.append(f"- Average accuracy: {main_results['accuracy'].mean():.2f}%\n")
                report_lines.append(f"- Best accuracy: {main_results['accuracy'].max():.2f}%\n")
                report_lines.append(f"- Models evaluated: {main_results['model'].nunique()}\n")
    
    # Frontier Models Focus
    report_lines.append("\n## Frontier Models Analysis\n")
    
    # GPT-5 Analysis
    gpt5_benchmark = benchmark_df[benchmark_df['model'].str.contains('gpt-5|gpt5', case=False, na=False)]
    if not gpt5_benchmark.empty:
        report_lines.append("\n### GPT-5 Performance\n")
        gpt5_results = gpt5_benchmark.groupby('modification')['accuracy'].mean().sort_values(ascending=False)
        
        for mod, acc in gpt5_results.items():
            report_lines.append(f"- **{mod}**: {acc:.2f}%\n")
        
        report_lines.append(f"\n**GPT-5 Overall Average**: {gpt5_benchmark['accuracy'].mean():.2f}%\n")
    
    # DeepSeek R1 Analysis
    deepseek_benchmark = benchmark_df[benchmark_df['model'].str.contains('deepseek', case=False, na=False)]
    if not deepseek_benchmark.empty:
        report_lines.append("\n### DeepSeek R1 Performance\n")
        deepseek_results = deepseek_benchmark.groupby('modification')['accuracy'].mean().sort_values(ascending=False)
        
        for mod, acc in deepseek_results.items():
            report_lines.append(f"- **{mod}**: {acc:.2f}%\n")
        
        report_lines.append(f"\n**DeepSeek R1 Overall Average**: {deepseek_benchmark['accuracy'].mean():.2f}%\n")
    
    # GPT-5 vs GPT-4o comparison
    gpt4o_benchmark = benchmark_df[benchmark_df['model'] == 'gpt4o']
    if not gpt5_benchmark.empty and not gpt4o_benchmark.empty:
        report_lines.append("\n### GPT-5 vs GPT-4o Direct Comparison\n")
        
        # Find common benchmarks
        gpt5_mods = set(gpt5_benchmark['modification'].unique())
        gpt4o_mods = set(gpt4o_benchmark['modification'].unique())
        common_mods = gpt5_mods & gpt4o_mods
        
        if common_mods:
            report_lines.append("| Benchmark | GPT-5 | GPT-4o | Difference |\n")
            report_lines.append("|-----------|-------|--------|------------|\n")
            
            for mod in sorted(common_mods):
                gpt5_acc = gpt5_benchmark[gpt5_benchmark['modification'] == mod]['accuracy'].mean()
                gpt4o_acc = gpt4o_benchmark[gpt4o_benchmark['modification'] == mod]['accuracy'].mean()
                diff = gpt5_acc - gpt4o_acc
                sign = "+" if diff > 0 else ""
                report_lines.append(f"| {mod} | {gpt5_acc:.2f}% | {gpt4o_acc:.2f}% | {sign}{diff:.2f}% |\n")
    
    # Modification Robustness Analysis
    if not modification_df.empty:
        report_lines.append("\n## Modification Robustness Analysis\n")
        report_lines.append("Performance drop on linguistic modifications (100-sample tests):\n\n")
        
        # Calculate average drop per model
        mod_avg = modification_df.groupby('model')['accuracy'].mean().sort_values(ascending=False)
        
        report_lines.append("### Model Robustness Ranking\n")
        report_lines.append("Average accuracy across all modifications:\n\n")
        
        for i, (model, acc) in enumerate(mod_avg.head(10).items(), 1):
            report_lines.append(f"{i}. **{model}**: {acc:.2f}%\n")
    
    # Key Insights
    report_lines.append("\n## Key Insights\n")
    
    if not benchmark_pivot.empty:
        best_model = benchmark_pivot['Average'].idxmax()
        best_score = benchmark_pivot['Average'].max()
        report_lines.append(f"- **Best Overall Model**: {best_model} ({best_score:.2f}% average)\n")
        
        # Most consistent model
        model_std = benchmark_df.groupby('model')['accuracy'].std().sort_values()
        if not model_std.empty:
            most_consistent = model_std.index[0]
            report_lines.append(f"- **Most Consistent Model**: {most_consistent} (σ={model_std.iloc[0]:.2f})\n")
    
    # Task difficulty
    task_avg = benchmark_df.groupby('task')['accuracy'].mean().sort_values()
    hardest_task = task_mapping.get(task_avg.index[0], task_avg.index[0])
    easiest_task = task_mapping.get(task_avg.index[-1], task_avg.index[-1])
    
    report_lines.append(f"- **Most Challenging Task**: {hardest_task} ({task_avg.iloc[0]:.2f}% avg)\n")
    report_lines.append(f"- **Easiest Task**: {easiest_task} ({task_avg.iloc[-1]:.2f}% avg)\n")
    
    # Chain-of-thought impact
    cot_results = benchmark_df[benchmark_df['modification'].str.contains('cot', na=False)]
    if not cot_results.empty:
        non_cot = benchmark_df[~benchmark_df['modification'].str.contains('cot', na=False)]
        cot_avg = cot_results['accuracy'].mean()
        non_cot_avg = non_cot['accuracy'].mean()
        cot_improvement = cot_avg - non_cot_avg
        
        report_lines.append(f"- **Chain-of-Thought Impact**: {'+' if cot_improvement > 0 else ''}{cot_improvement:.2f}% average improvement\n")
    
    # Save outputs
    report_content = ''.join(report_lines)
    report_path = organizer.create_report(report_content, f"llm_benchmark_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    
    # Save data files
    benchmark_df.to_csv(organizer.get_output_path("benchmark_results.csv", "data"), index=False)
    modification_df.to_csv(organizer.get_output_path("modification_results.csv", "data"), index=False)
    benchmark_pivot.to_csv(organizer.get_output_path("benchmark_pivot.csv", "data"))
    
    # Print summary
    print("\n" + "="*80)
    print("📊 Analysis Complete!")
    print("="*80)
    
    print(f"\n✨ Key Findings:")
    if not benchmark_pivot.empty:
        print(f"  • Best model: {benchmark_pivot['Average'].idxmax()} ({benchmark_pivot['Average'].max():.2f}%)")
    
    if not gpt5_benchmark.empty:
        print(f"  • GPT-5 average: {gpt5_benchmark['accuracy'].mean():.2f}%")
    
    if not deepseek_benchmark.empty:
        print(f"  • DeepSeek R1 average: {deepseek_benchmark['accuracy'].mean():.2f}%")
    
    print(f"\n📁 Results saved to: {organizer.base_path}/")
    print(f"  • Report: {report_path}")
    print(f"  • Data: {organizer.base_path / 'data'}/")

if __name__ == "__main__":
    main()