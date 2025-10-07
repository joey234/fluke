#!/usr/bin/env python
"""
FLUKE Analysis Runner
Main script to run comprehensive FLUKE analysis with all components.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import analysis modules
from consolidated_analysis import FLUKEAnalyzer, NERAnalyzer, DialogueAnalyzer, SentimentAnalyzer
from visualization import FLUKEVisualizer, generate_all_visualizations
from utils import Config, FileOrganizer, DataLoader, MetricsCalculator, ResultsProcessor


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run FLUKE Analysis')
    
    parser.add_argument(
        '--task',
        type=str,
        choices=['all', 'coref', 'dialogue', 'ner', 'sa'],
        default='all',
        help='Task to analyze (default: all)'
    )
    
    parser.add_argument(
        '--base-path',
        type=str,
        default='../',
        help='Base path to experiments directory'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_output',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate detailed report'
    )
    
    parser.add_argument(
        '--statistical-tests',
        action='store_true',
        help='Run statistical significance tests'
    )
    
    parser.add_argument(
        '--export-format',
        type=str,
        choices=['csv', 'excel', 'json'],
        default='csv',
        help='Export format for results'
    )
    
    return parser.parse_args()


def analyze_task(task: str, analyzer: FLUKEAnalyzer, organizer: FileOrganizer, args):
    """Analyze a specific task."""
    print(f"\n{'='*60}")
    print(f"Analyzing {task.upper()} Task")
    print(f"{'='*60}")
    
    # Load results
    print(f"Loading {task} results...")
    df = analyzer.load_results(task)
    
    if df.empty:
        print(f"  ⚠️  No results found for {task}")
        return None
    
    print(f"  ✓ Loaded {len(df)} result entries")
    
    # Calculate performance drops
    print(f"Analyzing performance drops...")
    drops_df = analyzer.analyze_performance_drop(task)
    
    if not drops_df.empty:
        print(f"  ✓ Calculated drops for {len(drops_df)} configurations")
        
        # Find most challenging modifications
        top_drops = drops_df.nlargest(5, 'relative_drop_%')[['model', 'modification', 'relative_drop_%']]
        print("\n  Top 5 Performance Drops:")
        for _, row in top_drops.iterrows():
            print(f"    • {row['model']} on {row['modification']}: {row['relative_drop_%']:.1f}% drop")
    
    # Save results
    if args.export_format == 'csv':
        results_path = organizer.save_results(df, f"{task}_results.csv")
        if not drops_df.empty:
            drops_path = organizer.save_results(drops_df, f"{task}_performance_drops.csv")
    elif args.export_format == 'excel':
        results_path = organizer.save_results(df, f"{task}_results.xlsx")
        if not drops_df.empty:
            drops_path = organizer.save_results(drops_df, f"{task}_performance_drops.xlsx")
    else:  # json
        results_path = organizer.save_results(df, f"{task}_results.json")
        if not drops_df.empty:
            drops_path = organizer.save_results(drops_df, f"{task}_performance_drops.json")
    
    print(f"\n  📁 Results saved to {results_path}")
    
    # Statistical tests
    if args.statistical_tests and not df.empty:
        print(f"\nRunning statistical tests...")
        run_statistical_tests(df, task, organizer)
    
    # Task-specific analysis
    if task == 'ner':
        ner_analyzer = NERAnalyzer(args.base_path)
        # Additional NER-specific analysis can be added here
    elif task == 'dialogue':
        dialogue_analyzer = DialogueAnalyzer(args.base_path)
        # Additional dialogue-specific analysis can be added here
    elif task == 'sa':
        sentiment_analyzer = SentimentAnalyzer(args.base_path)
        # Additional sentiment-specific analysis can be added here
    
    return df, drops_df


def run_statistical_tests(df, task, organizer):
    """Run statistical significance tests."""
    from utils import StatisticalTester
    
    # Compare PLM vs LLM
    plm_acc = df[df['model_type'] == 'plm']['accuracy'].values
    llm_acc = df[df['model_type'] == 'llm']['accuracy'].values
    
    if len(plm_acc) > 0 and len(llm_acc) > 0:
        test_result = StatisticalTester.mann_whitney_test(plm_acc, llm_acc)
        
        print(f"  PLM vs LLM comparison:")
        print(f"    • Test: {test_result['test']}")
        print(f"    • p-value: {test_result['p_value']:.4f}")
        print(f"    • Significant: {'Yes' if test_result['significant'] else 'No'}")
        print(f"    • Effect size: {test_result['effect_size']:.3f}")
        
        # Save test results
        import json
        test_path = organizer.get_output_path(f"{task}_statistical_tests.json", "data")
        with open(test_path, 'w') as f:
            json.dump(test_result, f, indent=2)


def generate_report(analyzer, tasks, organizer, args):
    """Generate comprehensive analysis report."""
    print(f"\n{'='*60}")
    print("Generating Analysis Report")
    print(f"{'='*60}")
    
    report_lines = []
    report_lines.append("# FLUKE Comprehensive Analysis Report\n")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"Base Path: {args.base_path}\n")
    report_lines.append("\n---\n")
    
    # Executive Summary
    report_lines.append("## Executive Summary\n")
    
    all_results = []
    for task in tasks:
        df = analyzer.load_results(task)
        if not df.empty:
            all_results.append(df)
            
    if all_results:
        import pandas as pd
        combined = pd.concat(all_results, ignore_index=True)
        
        report_lines.append(f"- **Total Experiments**: {len(combined)}\n")
        report_lines.append(f"- **Models Evaluated**: {combined['model'].nunique()}\n")
        report_lines.append(f"- **Modifications Tested**: {combined['modification'].nunique()}\n")
        report_lines.append(f"- **Average Accuracy**: {combined['accuracy'].mean():.2f}%\n")
        report_lines.append(f"- **Best Performing Model**: {combined.groupby('model')['accuracy'].mean().idxmax()}\n")
        
        # Highlight new models if present
        if 'gpt-5' in combined['model'].values or 'gpt5' in combined['model'].values:
            gpt5_acc = combined[combined['model'].str.contains('gpt-5|gpt5', case=False, na=False)]['accuracy'].mean()
            report_lines.append(f"- **GPT-5 Average Accuracy**: {gpt5_acc:.2f}%\n")
        
        if 'deepseek' in combined['model'].str.lower().values:
            deepseek_acc = combined[combined['model'].str.contains('deepseek', case=False, na=False)]['accuracy'].mean()
            report_lines.append(f"- **DeepSeek R1 Average Accuracy**: {deepseek_acc:.2f}%\n")
        
        # Model comparison
        report_lines.append("\n## Model Performance Comparison\n")
        
        model_stats = combined.groupby(['model_type', 'model'])['accuracy'].agg(['mean', 'std', 'count'])
        model_stats = model_stats.sort_values('mean', ascending=False)
        
        report_lines.append("### Top Performing Models\n")
        report_lines.append("| Model | Type | Avg Accuracy | Std Dev | Samples |\n")
        report_lines.append("|-------|------|-------------|---------|----------|\n")
        
        for (model_type, model), row in model_stats.head(10).iterrows():
            report_lines.append(f"| {model} | {model_type.upper()} | {row['mean']:.2f}% | ±{row['std']:.2f} | {int(row['count'])} |\n")
        
        # Task-wise analysis
        report_lines.append("\n## Task-wise Analysis\n")
        
        for task in tasks:
            df = analyzer.load_results(task)
            if not df.empty:
                report_lines.append(f"\n### {task.upper()} Task\n")
                
                task_summary = df.groupby('model_type')['accuracy'].agg(['mean', 'std', 'min', 'max'])
                report_lines.append(f"- PLM Average: {task_summary.loc['plm', 'mean']:.2f}% (±{task_summary.loc['plm', 'std']:.2f})\n" 
                                  if 'plm' in task_summary.index else "")
                report_lines.append(f"- LLM Average: {task_summary.loc['llm', 'mean']:.2f}% (±{task_summary.loc['llm', 'std']:.2f})\n"
                                  if 'llm' in task_summary.index else "")
                
                # Most challenging modifications
                drops_df = analyzer.analyze_performance_drop(task)
                if not drops_df.empty:
                    top_challenges = drops_df.groupby('modification')['relative_drop_%'].mean().nlargest(3)
                    
                    report_lines.append("\n#### Most Challenging Modifications:\n")
                    for mod, drop in top_challenges.items():
                        report_lines.append(f"1. **{mod}**: {drop:.1f}% average drop\n")
        
        # Frontier Model Comparison (GPT-5 and DeepSeek R1)
        report_lines.append("\n## Frontier Model Analysis\n")
        
        gpt5_data = combined[combined['model'].str.contains('gpt-5|gpt5', case=False, na=False)]
        deepseek_data = combined[combined['model'].str.contains('deepseek', case=False, na=False)]
        
        if not gpt5_data.empty or not deepseek_data.empty:
            report_lines.append("### New Frontier Models Performance\n")
            
            if not gpt5_data.empty:
                gpt5_by_task = gpt5_data.groupby('task')['accuracy'].mean()
                report_lines.append("\n#### GPT-5 Performance by Task:\n")
                for task, acc in gpt5_by_task.items():
                    report_lines.append(f"- {task.upper()}: {acc:.2f}%\n")
                    
            if not deepseek_data.empty:
                deepseek_by_task = deepseek_data.groupby('task')['accuracy'].mean()
                report_lines.append("\n#### DeepSeek R1 Performance by Task:\n")
                for task, acc in deepseek_by_task.items():
                    report_lines.append(f"- {task.upper()}: {acc:.2f}%\n")
            
            # Compare with GPT-4o if available
            gpt4o_data = combined[combined['model'] == 'gpt4o']
            if not gpt4o_data.empty and not gpt5_data.empty:
                gpt4o_avg = gpt4o_data['accuracy'].mean()
                gpt5_avg = gpt5_data['accuracy'].mean()
                improvement = gpt5_avg - gpt4o_avg
                report_lines.append(f"\n#### GPT-5 vs GPT-4o:\n")
                report_lines.append(f"- GPT-5 shows {'an improvement' if improvement > 0 else 'a decrease'} of {abs(improvement):.2f}% compared to GPT-4o\n")
        
        # Insights and Recommendations
        report_lines.append("\n## Key Insights\n")
        
        # Calculate insights
        plm_avg = combined[combined['model_type'] == 'plm']['accuracy'].mean()
        llm_avg = combined[combined['model_type'] == 'llm']['accuracy'].mean()
        
        if plm_avg > 0 and llm_avg > 0:
            if llm_avg > plm_avg:
                report_lines.append(f"- LLMs outperform PLMs by {llm_avg - plm_avg:.1f}% on average\n")
            else:
                report_lines.append(f"- PLMs outperform LLMs by {plm_avg - llm_avg:.1f}% on average\n")
        
        # Find most robust model
        model_robustness = combined.groupby('model')['accuracy'].std()
        most_robust = model_robustness.idxmin()
        report_lines.append(f"- Most robust model (lowest variance): {most_robust} (σ={model_robustness[most_robust]:.2f})\n")
        
        # Modification impact
        all_drops = []
        for task in tasks:
            drops_df = analyzer.analyze_performance_drop(task)
            if not drops_df.empty:
                all_drops.append(drops_df)
        
        if all_drops:
            combined_drops = pd.concat(all_drops, ignore_index=True)
            worst_mod = combined_drops.groupby('modification')['relative_drop_%'].mean().idxmax()
            worst_drop = combined_drops.groupby('modification')['relative_drop_%'].mean().max()
            report_lines.append(f"- Most impactful modification: {worst_mod} (avg. {worst_drop:.1f}% drop)\n")
    
    # Save report
    report_content = ''.join(report_lines)
    report_path = organizer.create_report(report_content, f"fluke_analysis_report_{datetime.now().strftime('%Y%m%d')}.md")
    
    print(f"  ✓ Report saved to {report_path}")
    
    return report_content


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("\n" + "="*60)
    print("       FLUKE Analysis Framework v1.0")
    print("="*60)
    
    # Initialize components
    analyzer = FLUKEAnalyzer(args.base_path)
    organizer = FileOrganizer(args.output_dir)
    
    # Determine tasks to analyze
    if args.task == 'all':
        tasks = ['coref', 'dialogue', 'ner', 'sa']
    else:
        tasks = [args.task]
    
    # Analyze each task
    results = {}
    for task in tasks:
        task_results = analyze_task(task, analyzer, organizer, args)
        if task_results:
            results[task] = task_results
    
    # Generate visualizations
    if args.visualize:
        print(f"\n{'='*60}")
        print("Generating Visualizations")
        print(f"{'='*60}")
        
        visualizer = FLUKEVisualizer()
        
        for task, (df, drops_df) in results.items():
            if df is not None:
                print(f"\nCreating visualizations for {task.upper()}...")
                
                # Model comparison
                viz_path = organizer.get_output_path(f"{task}_model_comparison.png", "visualizations")
                visualizer.plot_model_comparison(df, task, save_path=viz_path)
                
                # Modification impact
                if drops_df is not None and not drops_df.empty:
                    viz_path = organizer.get_output_path(f"{task}_modification_impact.png", "visualizations")
                    visualizer.plot_modification_impact(drops_df, task, save_path=viz_path)
                
                print(f"  ✓ Visualizations saved")
    
    # Generate report
    if args.report:
        report = generate_report(analyzer, tasks, organizer, args)
    
    # Summary
    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"{'='*60}")
    print(f"\n📊 Results saved to: {organizer.base_path}")
    print(f"   • Data files: {organizer.base_path / 'data'}")
    if args.visualize:
        print(f"   • Visualizations: {organizer.base_path / 'visualizations'}")
    if args.report:
        print(f"   • Reports: {organizer.base_path / 'reports'}")
    
    print("\n✨ Analysis completed successfully!")


if __name__ == "__main__":
    main()