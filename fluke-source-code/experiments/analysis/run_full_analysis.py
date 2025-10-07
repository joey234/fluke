#!/usr/bin/env python
"""
FLUKE Full Analysis Runner
Unified script to run comprehensive FLUKE analysis with statistical tests and LaTeX table generation.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import analysis modules
from consolidated_analysis import FLUKEAnalyzer
from statistical_analysis import StatisticalAnalyzer
from generate_latex_tables import LaTeXTableGenerator
from visualization import FLUKEVisualizer
from utils import Config, FileOrganizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Complete FLUKE Analysis')
    
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
        default='full_analysis_output',
        help='Output directory for all results'
    )
    
    parser.add_argument(
        '--latex-only',
        action='store_true',
        help='Only generate LaTeX tables (skip other analysis)'
    )
    
    parser.add_argument(
        '--no-statistical-tests',
        action='store_true',
        help='Skip statistical significance tests'
    )
    
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--weighted-delta',
        action='store_true',
        default=True,
        help='Use weighted delta metric (default: True)'
    )
    
    return parser.parse_args()


def run_statistical_analysis(analyzer, tasks, organizer, args):
    """Run comprehensive statistical analysis for all tasks."""
    print(f"\n{'='*60}")
    print("Running Statistical Analysis")
    print(f"{'='*60}")
    
    stat_analyzer = StatisticalAnalyzer()
    all_stat_results = {}
    
    for task in tasks:
        print(f"\nRunning statistical tests for {task.upper()}...")
        
        # Get performance data for statistical testing
        df = analyzer.load_results(task)
        
        if df.empty:
            print(f"  ⚠️  No data found for {task}")
            continue
            
        # Run significance tests for each model in this task
        stat_results = {}
        unique_models = df['model'].unique()
        
        for model in unique_models:
            try:
                model_results = stat_analyzer.analyze_task_modifications(task, model)
                if not model_results.empty:
                    stat_results[f"{task}_{model}"] = model_results.to_dict('records')
            except Exception as e:
                print(f"    ⚠️  Error analyzing {model}: {e}")
                continue
        
        if stat_results:
            all_stat_results[task] = stat_results
            
            # Report significant findings - check nested structure
            significant_count = 0
            for model_key, model_results in stat_results.items():
                if isinstance(model_results, list):
                    significant_count += sum(1 for result in model_results 
                                           if isinstance(result, dict) and result.get('is_significant', False))
                elif isinstance(model_results, dict) and model_results.get('is_significant', False):
                    significant_count += 1
            
            print(f"  ✓ Found {significant_count} significant comparisons")
            
            # Save statistical results
            import json
            stat_path = organizer.get_output_path(f"{task}_statistical_analysis.json", "statistics")
            organizer.base_path.joinpath("statistics").mkdir(exist_ok=True)
            
            with open(stat_path, 'w') as f:
                json.dump(stat_results, f, indent=2, default=str)
                
            print(f"  📁 Statistical results saved to {stat_path}")
    
    return all_stat_results


def generate_latex_tables(analyzer, tasks, organizer, args):
    """Generate LaTeX tables with weighted delta and significance markers."""
    print(f"\n{'='*60}")
    print("Generating LaTeX Tables")
    print(f"{'='*60}")
    
    # Create latex_tables directory
    latex_dir = organizer.base_path / "latex_tables"
    latex_dir.mkdir(exist_ok=True)
    
    # Initialize LaTeX generator
    latex_generator = LaTeXTableGenerator(base_path=args.base_path)
    
    # Generate tables for each task
    generated_tables = []
    
    for task in tasks:
        print(f"\nGenerating LaTeX tables for {task.upper()}...")
        
        try:
            # Generate task-specific tables
            main_table = latex_generator.generate_main_results_table(task)
            mod_table = latex_generator.generate_modification_table(task)
            delta_table = latex_generator.generate_weighted_delta_table(task)
            
            # Save tables to files
            main_file = latex_dir / f"{task}_main_results.tex"
            mod_file = latex_dir / f"{task}_modifications.tex"
            delta_file = latex_dir / f"{task}_weighted_delta.tex"
            
            with open(main_file, 'w') as f:
                f.write(main_table)
            with open(mod_file, 'w') as f:
                f.write(mod_table)
            with open(delta_file, 'w') as f:
                f.write(delta_table)
                
            table_files = [str(main_file), str(mod_file), str(delta_file)]
            
            if table_files:
                generated_tables.extend(table_files)
                print(f"  ✓ Generated {len(table_files)} table(s)")
                for table_file in table_files:
                    print(f"    • {table_file}")
            else:
                print(f"  ⚠️  No tables generated for {task}")
                
        except Exception as e:
            print(f"  ❌ Error generating tables for {task}: {e}")
    
    # Generate cross-task comparison tables
    print(f"\nGenerating cross-task comparison tables...")
    try:
        comparison_table = latex_generator.generate_comparison_table(tasks)
        
        # Save comparison table
        comparison_file = latex_dir / "cross_task_comparison.tex"
        with open(comparison_file, 'w') as f:
            f.write(comparison_table)
            
        comparison_tables = [str(comparison_file)]
        
        if comparison_tables:
            generated_tables.extend(comparison_tables)
            print(f"  ✓ Generated {len(comparison_tables)} comparison table(s)")
            for table_file in comparison_tables:
                print(f"    • {table_file}")
                
    except Exception as e:
        print(f"  ❌ Error generating comparison tables: {e}")
    
    # Generate frontier models table (GPT-5 vs DeepSeek R1)
    print(f"\nGenerating frontier models comparison...")
    try:
        frontier_table = latex_generator.generate_frontier_models_table()
        
        # Save frontier models table
        frontier_file = latex_dir / "frontier_models.tex"
        with open(frontier_file, 'w') as f:
            f.write(frontier_table)
            
        frontier_tables = [str(frontier_file)]
        
        if frontier_tables:
            generated_tables.extend(frontier_tables)
            print(f"  ✓ Generated frontier models table")
            for table_file in frontier_tables:
                print(f"    • {table_file}")
                
    except Exception as e:
        print(f"  ❌ Error generating frontier models table: {e}")
    
    print(f"\n📁 All LaTeX tables saved to: {latex_dir}")
    return generated_tables


def generate_visualizations(analyzer, tasks, organizer, args):
    """Generate visualizations for analysis results."""
    print(f"\n{'='*60}")
    print("Generating Visualizations")
    print(f"{'='*60}")
    
    visualizer = FLUKEVisualizer()
    viz_dir = organizer.base_path / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    generated_plots = []
    
    for task in tasks:
        print(f"\nGenerating visualizations for {task.upper()}...")
        
        # Load data
        df = analyzer.load_results(task)
        
        if df.empty:
            print(f"  ⚠️  No data found for {task}")
            continue
            
        # Generate performance drops analysis
        drops_df = analyzer.analyze_performance_drop(task, use_weighted_delta=args.weighted_delta)
        
        try:
            # Model comparison plot
            model_plot = viz_dir / f"{task}_model_comparison.png"
            visualizer.plot_model_comparison(df, task, save_path=str(model_plot))
            generated_plots.append(str(model_plot))
            
            # Performance drop plot
            if not drops_df.empty:
                drop_plot = viz_dir / f"{task}_performance_drops.png"
                visualizer.plot_modification_impact(drops_df, task, save_path=str(drop_plot))
                generated_plots.append(str(drop_plot))
            
            # Weighted delta heatmap
            if not drops_df.empty and args.weighted_delta:
                heatmap_plot = viz_dir / f"{task}_weighted_delta_heatmap.png"
                visualizer.plot_weighted_delta_heatmap(drops_df, task, save_path=str(heatmap_plot))
                generated_plots.append(str(heatmap_plot))
            
            print(f"  ✓ Generated visualizations for {task}")
            
        except Exception as e:
            print(f"  ❌ Error generating visualizations for {task}: {e}")
    
    print(f"\n📁 All visualizations saved to: {viz_dir}")
    return generated_plots


def generate_comprehensive_report(analyzer, tasks, organizer, stat_results, latex_files, viz_files, args):
    """Generate comprehensive analysis report."""
    print(f"\n{'='*60}")
    print("Generating Comprehensive Report")
    print(f"{'='*60}")
    
    report_lines = []
    report_lines.append("# FLUKE Comprehensive Analysis Report\n\n")
    report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**Base Path**: {args.base_path}\n")
    report_lines.append(f"**Tasks Analyzed**: {', '.join(task.upper() for task in tasks)}\n")
    report_lines.append(f"**Weighted Delta**: {'Yes' if args.weighted_delta else 'No'}\n")
    report_lines.append(f"**Statistical Tests**: {'Yes' if not args.no_statistical_tests else 'No'}\n\n")
    report_lines.append("---\n\n")
    
    # Executive Summary
    report_lines.append("## Executive Summary\n\n")
    
    all_results = []
    for task in tasks:
        df = analyzer.load_results(task)
        if not df.empty:
            all_results.append(df)
            
    if all_results:
        import pandas as pd
        combined = pd.concat(all_results, ignore_index=True)
        
        report_lines.append(f"- **Total Experiments**: {len(combined):,}\n")
        report_lines.append(f"- **Models Evaluated**: {combined['model'].nunique()}\n")
        report_lines.append(f"- **Modifications Tested**: {combined['modification'].nunique()}\n")
        report_lines.append(f"- **Average Accuracy**: {combined['accuracy'].mean():.2f}%\n")
        report_lines.append(f"- **Best Performing Model**: {combined.groupby('model')['accuracy'].mean().idxmax()}\n")
        
        # Frontier models analysis
        gpt5_data = combined[combined['model'].str.contains('gpt-5|gpt5', case=False, na=False)]
        deepseek_data = combined[combined['model'].str.contains('deepseek', case=False, na=False)]
        
        if not gpt5_data.empty:
            gpt5_acc = gpt5_data['accuracy'].mean()
            report_lines.append(f"- **GPT-5 Average Accuracy**: {gpt5_acc:.2f}%\n")
        
        if not deepseek_data.empty:
            deepseek_acc = deepseek_data['accuracy'].mean()
            report_lines.append(f"- **DeepSeek R1 Average Accuracy**: {deepseek_acc:.2f}%\n")
    
    # Statistical Analysis Summary
    if stat_results and not args.no_statistical_tests:
        report_lines.append("\n## Statistical Analysis Summary\n\n")
        
        total_tests = 0
        significant_tests = 0
        for task_results in stat_results.values():
            for model_key, model_results in task_results.items():
                if isinstance(model_results, list):
                    total_tests += len(model_results)
                    significant_tests += sum(1 for result in model_results 
                                           if isinstance(result, dict) and result.get('is_significant', False))
                elif isinstance(model_results, dict):
                    total_tests += 1
                    if model_results.get('is_significant', False):
                        significant_tests += 1
        
        report_lines.append(f"- **Total Statistical Tests**: {total_tests}\n")
        report_lines.append(f"- **Significant Results**: {significant_tests} ({significant_tests/total_tests*100:.1f}%)\n")
        
        # Top significant findings
        significant_findings = []
        for task, task_results in stat_results.items():
            for comparison, result in task_results.items():
                if isinstance(result, dict) and result.get('significant', False):
                    significant_findings.append((task, comparison, result.get('p_value', 1.0)))
        
        if significant_findings:
            significant_findings.sort(key=lambda x: x[2])  # Sort by p-value
            report_lines.append("\n### Most Significant Findings:\n")
            for task, comparison, p_value in significant_findings[:5]:
                report_lines.append(f"- **{task.upper()}** - {comparison}: p = {p_value:.4f}\n")
    
    # Task-wise Performance Analysis
    report_lines.append("\n## Task-wise Performance Analysis\n\n")
    
    for task in tasks:
        df = analyzer.load_results(task)
        if not df.empty:
            report_lines.append(f"### {task.upper()} Task\n\n")
            
            # Performance drops analysis
            drops_df = analyzer.analyze_performance_drop(task, use_weighted_delta=args.weighted_delta)
            
            if not drops_df.empty:
                avg_drop = drops_df['weighted_delta' if args.weighted_delta else 'relative_drop_%'].mean()
                worst_mod = drops_df.loc[drops_df['weighted_delta' if args.weighted_delta else 'relative_drop_%'].idxmax()]
                
                metric_name = "Weighted Delta" if args.weighted_delta else "Relative Drop"
                report_lines.append(f"- **Average {metric_name}**: {avg_drop:.3f}\n")
                report_lines.append(f"- **Most Challenging Modification**: {worst_mod['modification']} "
                                  f"({worst_mod['weighted_delta' if args.weighted_delta else 'relative_drop_%']:.3f})\n")
                
                # Top challenging modifications
                top_mods = drops_df.nlargest(3, 'weighted_delta' if args.weighted_delta else 'relative_drop_%')
                report_lines.append(f"\n#### Top 3 Challenging Modifications:\n")
                for i, (_, row) in enumerate(top_mods.iterrows(), 1):
                    report_lines.append(f"{i}. **{row['modification']}** - {row['model']}: "
                                      f"{row['weighted_delta' if args.weighted_delta else 'relative_drop_%']:.3f}\n")
    
    # Generated Files Summary
    report_lines.append("\n## Generated Files\n\n")
    
    if latex_files:
        report_lines.append("### LaTeX Tables\n")
        for latex_file in latex_files:
            report_lines.append(f"- `{Path(latex_file).name}`\n")
        report_lines.append("\n")
    
    if viz_files and not args.no_visualizations:
        report_lines.append("### Visualizations\n")
        for viz_file in viz_files:
            report_lines.append(f"- `{Path(viz_file).name}`\n")
        report_lines.append("\n")
    
    # Usage Instructions
    report_lines.append("## Usage Instructions\n\n")
    report_lines.append("### LaTeX Tables\n")
    report_lines.append("Include the generated LaTeX files in your document:\n\n")
    report_lines.append("```latex\n")
    report_lines.append("\\usepackage{booktabs}\n")
    report_lines.append("\\usepackage{adjustbox}\n\n")
    report_lines.append("% Include tables\n")
    if latex_files:
        for latex_file in latex_files[:3]:  # Show first 3 as examples
            report_lines.append(f"\\input{{{Path(latex_file).name}}}\n")
    report_lines.append("```\n\n")
    
    if args.weighted_delta:
        report_lines.append("### Weighted Delta Metric\n")
        report_lines.append("The weighted delta metric is calculated as:\n\n")
        report_lines.append("```\n")
        report_lines.append("weighted_delta = (B - A) × log₁₀(A) / log₁₀(100)\n")
        report_lines.append("```\n\n")
        report_lines.append("Where:\n")
        report_lines.append("- A = Original accuracy\n")
        report_lines.append("- B = Modified accuracy\n")
        report_lines.append("- This metric gives higher weight to drops from higher baseline accuracies\n\n")
    
    # Save report
    report_content = ''.join(report_lines)
    report_path = organizer.base_path / "reports" / f"fluke_comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"  ✓ Comprehensive report saved to {report_path}")
    return report_content


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("\n" + "="*70)
    print("       FLUKE COMPREHENSIVE ANALYSIS FRAMEWORK")
    print("          Statistical Tests + LaTeX Tables + Visualizations")
    print("="*70)
    
    # Initialize components
    analyzer = FLUKEAnalyzer(args.base_path)
    organizer = FileOrganizer(args.output_dir)
    
    # Determine tasks to analyze
    if args.task == 'all':
        tasks = ['coref', 'dialogue', 'ner', 'sa']
    else:
        tasks = [args.task]
    
    print(f"\n📋 Analysis Configuration:")
    print(f"   • Tasks: {', '.join(task.upper() for task in tasks)}")
    print(f"   • Weighted Delta: {'Yes' if args.weighted_delta else 'No'}")
    print(f"   • Statistical Tests: {'Yes' if not args.no_statistical_tests else 'No'}")
    print(f"   • Visualizations: {'Yes' if not args.no_visualizations else 'No'}")
    print(f"   • LaTeX Only Mode: {'Yes' if args.latex_only else 'No'}")
    
    # Results containers
    stat_results = {}
    latex_files = []
    viz_files = []
    
    if args.latex_only:
        # Only generate LaTeX tables
        latex_files = generate_latex_tables(analyzer, tasks, organizer, args)
    else:
        # Full analysis pipeline
        
        # 1. Statistical Analysis
        if not args.no_statistical_tests:
            stat_results = run_statistical_analysis(analyzer, tasks, organizer, args)
        
        # 2. LaTeX Tables
        latex_files = generate_latex_tables(analyzer, tasks, organizer, args)
        
        # 3. Visualizations
        if not args.no_visualizations:
            viz_files = generate_visualizations(analyzer, tasks, organizer, args)
        
        # 4. Comprehensive Report
        report_content = generate_comprehensive_report(
            analyzer, tasks, organizer, stat_results, latex_files, viz_files, args
        )
    
    # Final Summary
    print(f"\n{'='*70}")
    print("🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 All results saved to: {organizer.base_path}")
    
    if latex_files:
        print(f"\n📊 LaTeX Tables ({len(latex_files)} files):")
        print(f"   📂 {organizer.base_path / 'latex_tables'}")
        
    if stat_results:
        print(f"\n📈 Statistical Analysis:")
        total_tests = 0
        significant_tests = 0
        for task_results in stat_results.values():
            for model_key, model_results in task_results.items():
                if isinstance(model_results, list):
                    total_tests += len(model_results)
                    significant_tests += sum(1 for result in model_results 
                                           if isinstance(result, dict) and result.get('is_significant', False))
                elif isinstance(model_results, dict):
                    total_tests += 1
                    if model_results.get('is_significant', False):
                        significant_tests += 1
        print(f"   • {total_tests} statistical tests performed")
        print(f"   • {significant_tests} significant results found")
        print(f"   📂 {organizer.base_path / 'statistics'}")
    
    if viz_files:
        print(f"\n🎨 Visualizations ({len(viz_files)} files):")
        print(f"   📂 {organizer.base_path / 'visualizations'}")
    
    if not args.latex_only:
        print(f"\n📋 Comprehensive Report:")
        print(f"   📂 {organizer.base_path / 'reports'}")
    
    print(f"\n✨ Ready for publication! ✨")
    
    return {
        'statistical_results': stat_results,
        'latex_files': latex_files,
        'visualization_files': viz_files,
        'output_directory': str(organizer.base_path)
    }


if __name__ == "__main__":
    main()