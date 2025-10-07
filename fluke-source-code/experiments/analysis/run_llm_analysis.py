#!/usr/bin/env python
"""
LLM-Only Analysis Runner for FLUKE
Analyzes only Large Language Model results, excluding PLMs.
"""

import sys
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Import analysis modules
from consolidated_analysis import FLUKEAnalyzer
from visualization import FLUKEVisualizer
from utils import FileOrganizer, Config

def main():
    """Run LLM-only analysis."""
    print("\n" + "="*70)
    print("       FLUKE LLM Analysis - Frontier Models Focus")
    print("="*70)
    
    # Initialize components
    base_path = "../"
    analyzer = FLUKEAnalyzer(base_path)
    visualizer = FLUKEVisualizer()
    organizer = FileOrganizer("llm_analysis_output")
    
    # Tasks to analyze
    tasks = ['coref', 'dialogue', 'ner', 'sa']
    
    # Collect all results
    all_results = []
    task_results = {}
    
    print("\n📊 Loading LLM Results...")
    print("-" * 50)
    
    for task in tasks:
        print(f"\n{task.upper()} Task:")
        
        # Load only LLM results (include_plm=False)
        df = analyzer.load_results(task, include_plm=False)
        
        if not df.empty:
            task_results[task] = df
            df['task'] = task
            all_results.append(df)
            
            # Show summary
            unique_models = df['model'].unique()
            print(f"  ✓ Loaded {len(df)} entries from {len(unique_models)} models")
            
            # Check for frontier models
            has_gpt5 = any('gpt-5' in str(m) or 'gpt5' in str(m) for m in unique_models)
            has_deepseek = any('deepseek' in str(m) for m in unique_models)
            
            if has_gpt5:
                gpt5_count = len(df[df['model'].str.contains('gpt-5|gpt5', case=False, na=False)])
                print(f"  ✓ GPT-5: {gpt5_count} entries")
            
            if has_deepseek:
                deepseek_count = len(df[df['model'].str.contains('deepseek', case=False, na=False)])
                print(f"  ✓ DeepSeek R1: {deepseek_count} entries")
            
            # Models present
            models_list = sorted(set(str(m) for m in unique_models))
            print(f"  Models: {', '.join(models_list)}")
        else:
            print(f"  ⚠ No LLM results found")
    
    if not all_results:
        print("\n❌ No LLM results found to analyze!")
        return
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Generate comprehensive report
    print("\n" + "="*70)
    print("📝 Generating LLM Analysis Report")
    print("="*70)
    
    report_lines = []
    report_lines.append("# FLUKE LLM-Only Analysis Report\n")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("\n---\n")
    
    # Executive Summary
    report_lines.append("## Executive Summary\n")
    report_lines.append(f"- **Total LLM Experiments**: {len(combined_df)}\n")
    report_lines.append(f"- **LLM Models Evaluated**: {combined_df['model'].nunique()}\n")
    report_lines.append(f"- **Tasks Analyzed**: {', '.join([t.upper() for t in tasks])}\n")
    report_lines.append(f"- **Modifications Tested**: {combined_df['modification'].nunique()}\n")
    report_lines.append(f"- **Overall LLM Average Accuracy**: {combined_df['accuracy'].mean():.2f}%\n")
    
    # Best performing model overall
    best_model = combined_df.groupby('model')['accuracy'].mean().idxmax()
    best_acc = combined_df.groupby('model')['accuracy'].mean().max()
    report_lines.append(f"- **Best Performing LLM**: {best_model} ({best_acc:.2f}%)\n")
    
    # Frontier Models Summary
    gpt5_df = combined_df[combined_df['model'].str.contains('gpt-5|gpt5', case=False, na=False)]
    deepseek_df = combined_df[combined_df['model'].str.contains('deepseek', case=False, na=False)]
    
    if not gpt5_df.empty:
        report_lines.append(f"- **GPT-5 Average**: {gpt5_df['accuracy'].mean():.2f}%\n")
    
    if not deepseek_df.empty:
        report_lines.append(f"- **DeepSeek R1 Average**: {deepseek_df['accuracy'].mean():.2f}%\n")
    
    # Model Ranking
    report_lines.append("\n## LLM Model Ranking\n")
    model_ranking = combined_df.groupby('model')['accuracy'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
    
    report_lines.append("| Rank | Model | Avg Accuracy | Std Dev | Samples |\n")
    report_lines.append("|------|-------|-------------|---------|----------|\n")
    
    for rank, (model, row) in enumerate(model_ranking.iterrows(), 1):
        report_lines.append(f"| {rank} | {model} | {row['mean']:.2f}% | ±{row['std']:.2f} | {int(row['count'])} |\n")
    
    # Task-wise Analysis
    report_lines.append("\n## Task-wise Performance\n")
    
    for task in tasks:
        if task in task_results:
            task_df = task_results[task]
            report_lines.append(f"\n### {task.upper()} Task\n")
            
            task_stats = task_df.groupby('model')['accuracy'].mean().sort_values(ascending=False)
            
            report_lines.append(f"- **Task Average**: {task_df['accuracy'].mean():.2f}%\n")
            report_lines.append(f"- **Best Model**: {task_stats.index[0]} ({task_stats.iloc[0]:.2f}%)\n")
            report_lines.append(f"- **Models Tested**: {len(task_stats)}\n")
            
            # Top 3 models for this task
            report_lines.append("\nTop 3 Models:\n")
            for i, (model, acc) in enumerate(task_stats.head(3).items(), 1):
                report_lines.append(f"{i}. {model}: {acc:.2f}%\n")
    
    # Modification Impact Analysis
    report_lines.append("\n## Modification Impact Analysis\n")
    
    # Calculate average accuracy per modification
    mod_impact = combined_df.groupby('modification')['accuracy'].mean().sort_values()
    
    report_lines.append("\n### Most Challenging Modifications (Lowest Accuracy)\n")
    for mod, acc in mod_impact.head(5).items():
        report_lines.append(f"- **{mod}**: {acc:.2f}%\n")
    
    report_lines.append("\n### Easiest Modifications (Highest Accuracy)\n")
    for mod, acc in mod_impact.tail(5).items():
        report_lines.append(f"- **{mod}**: {acc:.2f}%\n")
    
    # Frontier Models Deep Dive
    if not gpt5_df.empty or not deepseek_df.empty:
        report_lines.append("\n## Frontier Models Analysis\n")
        
        if not gpt5_df.empty:
            report_lines.append("\n### GPT-5 Performance\n")
            gpt5_by_task = gpt5_df.groupby('task')['accuracy'].mean().sort_values(ascending=False)
            
            for task, acc in gpt5_by_task.items():
                report_lines.append(f"- {task.upper()}: {acc:.2f}%\n")
            
            # GPT-5 vs GPT-4o comparison
            gpt4o_df = combined_df[combined_df['model'] == 'gpt4o']
            if not gpt4o_df.empty:
                gpt4o_avg = gpt4o_df['accuracy'].mean()
                gpt5_avg = gpt5_df['accuracy'].mean()
                diff = gpt5_avg - gpt4o_avg
                
                report_lines.append(f"\n**GPT-5 vs GPT-4o**: ")
                if diff > 0:
                    report_lines.append(f"GPT-5 outperforms by {diff:.2f}%\n")
                else:
                    report_lines.append(f"GPT-4o outperforms by {abs(diff):.2f}%\n")
        
        if not deepseek_df.empty:
            report_lines.append("\n### DeepSeek R1 Performance\n")
            deepseek_by_task = deepseek_df.groupby('task')['accuracy'].mean().sort_values(ascending=False)
            
            for task, acc in deepseek_by_task.items():
                report_lines.append(f"- {task.upper()}: {acc:.2f}%\n")
    
    # Model Comparison Matrix
    report_lines.append("\n## Model Performance Matrix\n")
    report_lines.append("Average accuracy by model and task:\n\n")
    
    # Create pivot table
    pivot_df = combined_df.pivot_table(
        values='accuracy',
        index='model',
        columns='task',
        aggfunc='mean'
    ).round(2)
    
    # Convert to markdown table
    report_lines.append("| Model | " + " | ".join([t.upper() for t in pivot_df.columns]) + " | Average |\n")
    report_lines.append("|-------|" + "---------|" * (len(pivot_df.columns) + 1) + "\n")
    
    for model, row in pivot_df.iterrows():
        values = [f"{v:.1f}" if not pd.isna(v) else "-" for v in row.values]
        avg = row.mean()
        report_lines.append(f"| {model} | " + " | ".join(values) + f" | {avg:.1f} |\n")
    
    # Key Insights
    report_lines.append("\n## Key Insights\n")
    
    # Find most consistent model (lowest std)
    model_consistency = combined_df.groupby('model')['accuracy'].std().sort_values()
    most_consistent = model_consistency.index[0]
    
    report_lines.append(f"- **Most Consistent Model**: {most_consistent} (σ={model_consistency.iloc[0]:.2f})\n")
    
    # Find model with highest improvement potential
    model_range = combined_df.groupby('model')['accuracy'].agg(['min', 'max'])
    model_range['range'] = model_range['max'] - model_range['min']
    highest_range = model_range['range'].idxmax()
    
    report_lines.append(f"- **Highest Variation**: {highest_range} (range: {model_range.loc[highest_range, 'range']:.1f}%)\n")
    
    # Task difficulty ranking
    task_difficulty = combined_df.groupby('task')['accuracy'].mean().sort_values()
    report_lines.append(f"- **Most Challenging Task**: {task_difficulty.index[0].upper()} ({task_difficulty.iloc[0]:.2f}%)\n")
    report_lines.append(f"- **Easiest Task**: {task_difficulty.index[-1].upper()} ({task_difficulty.iloc[-1]:.2f}%)\n")
    
    # Save report
    report_content = ''.join(report_lines)
    report_path = organizer.create_report(report_content, f"llm_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    
    # Save data files
    print("\n💾 Saving Results...")
    combined_df.to_csv(organizer.get_output_path("llm_combined_results.csv", "data"), index=False)
    pivot_df.to_csv(organizer.get_output_path("llm_performance_matrix.csv", "data"))
    model_ranking.to_csv(organizer.get_output_path("llm_model_ranking.csv", "data"))
    
    # Summary statistics
    print("\n" + "="*70)
    print("📊 Analysis Summary")
    print("="*70)
    print(f"\n✓ Total LLM experiments analyzed: {len(combined_df)}")
    print(f"✓ Models evaluated: {combined_df['model'].nunique()}")
    print(f"✓ Best performing model: {best_model} ({best_acc:.2f}%)")
    
    if not gpt5_df.empty:
        print(f"✓ GPT-5 average accuracy: {gpt5_df['accuracy'].mean():.2f}%")
    
    if not deepseek_df.empty:
        print(f"✓ DeepSeek R1 average accuracy: {deepseek_df['accuracy'].mean():.2f}%")
    
    print(f"\n📁 Results saved to: {organizer.base_path}/")
    print(f"   • Report: {report_path}")
    print(f"   • Data files: {organizer.base_path / 'data'}/")
    
    print("\n✨ LLM Analysis Complete!")


if __name__ == "__main__":
    main()