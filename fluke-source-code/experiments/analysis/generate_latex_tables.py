#!/usr/bin/env python
"""
Generate LaTeX tables for FLUKE LLM analysis results.
Creates publication-ready tables for each task.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from consolidated_analysis import FLUKEAnalyzer
from utils import FileOrganizer


class LaTeXTableGenerator:
    """Generate LaTeX tables for FLUKE results."""
    
    def __init__(self, base_path: str = "../"):
        """Initialize the LaTeX generator."""
        self.analyzer = FLUKEAnalyzer(base_path)
        self.organizer = FileOrganizer("latex_tables_output")
        
        # Define modification groups for organized presentation
        self.modification_groups = {
            'linguistic': [
                'active_to_passive', 'capitalization', 'punctuation', 
                'typo_bias', 'compound_word', 'grammatical_role'
            ],
            'semantic': [
                'negation', 'sentiment', 'concept_replacement',
                'derivation', 'coordinating_conjunction'
            ],
            'contextual': [
                'discourse', 'temporal_bias', 'geographical_bias',
                'length_bias', 'casual'
            ],
            'dialectal': ['dialectal']
        }
        
    def generate_main_results_table(self, task: str) -> str:
        """Generate LaTeX table for main benchmark results."""
        # Load results
        df = self.analyzer.load_results(task, include_plm=False)
        
        if df.empty:
            return f"% No results found for {task}\n"
        
        # For notebook-style data, use original_accuracy from any modification record
        if 'original_accuracy' in df.columns and not df['original_accuracy'].isna().all():
            # Get unique original accuracies by model
            main_df = df.groupby('model')['original_accuracy'].first().reset_index()
            main_df.rename(columns={'original_accuracy': 'accuracy'}, inplace=True)
            main_df['task'] = task
        else:
            # Fallback: Filter for main benchmark results (not modifications)
            main_patterns = {
                'coref': ['coref', 'cot-coref'],
                'dialogue': ['dialogue', 'cot-dialogue'],
                'ner': ['ner'],
                'sa': ['sst2', 'cot-sst2']
            }
            
            patterns = main_patterns.get(task, [task])
            main_df = df[df['modification'].isin(patterns)]
        
        if main_df.empty:
            return f"% No main benchmark results found for {task}\n"
        
        # Create pivot table - for main results we don't need modification columns
        if len(main_df['task'].unique()) == 1:
            # Single task - just reshape by model
            pivot = main_df.pivot_table(
                values='accuracy',
                index='model',
                columns='task',
                aggfunc='mean'
            )
        else:
            # Multiple tasks - pivot by task
            pivot = main_df.pivot_table(
                values='accuracy',
                index='model',
                columns='task',
                aggfunc='mean'
            )
        
        # Sort by average performance
        pivot['Average'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('Average', ascending=False)
        
        # Generate LaTeX
        latex = []
        latex.append(f"% Main Results Table for {task.upper()}\n")
        latex.append("\\begin{table}[htbp]\n")
        latex.append("\\centering\n")
        latex.append(f"\\caption{{Main Benchmark Results - {self._format_task_name(task)}}}\n")
        latex.append(f"\\label{{tab:{task}_main_results}}\n")
        
        # Adjust column format based on number of columns
        num_cols = len(pivot.columns)
        col_format = "l" + "r" * num_cols
        latex.append(f"\\begin{{tabular}}{{{col_format}}}\n")
        latex.append("\\toprule\n")
        
        # Header
        headers = ["Model"] + [self._format_column_name(col) for col in pivot.columns]
        latex.append(" & ".join(headers) + " \\\\\n")
        latex.append("\\midrule\n")
        
        # Data rows
        for model, row in pivot.iterrows():
            model_name = self._format_model_name(model)
            values = []
            for val in row.values:
                if pd.isna(val):
                    values.append("-")
                else:
                    values.append(f"{val:.1f}")
            
            # Highlight best value in each column
            row_str = model_name
            for i, val in enumerate(values):
                if val != "-" and i < len(values) - 1:  # Don't bold the average column
                    col_vals = pivot.iloc[:, i].dropna()
                    if len(col_vals) > 0 and float(val) == col_vals.max():
                        val = f"\\textbf{{{val}}}"
                row_str += f" & {val}"
            
            latex.append(row_str + " \\\\\n")
        
        latex.append("\\bottomrule\n")
        latex.append("\\end{tabular}\n")
        latex.append("\\end{table}\n")
        
        return ''.join(latex)
    
    def generate_modification_table(self, task: str) -> str:
        """Generate LaTeX table for modification results."""
        # Load results
        df = self.analyzer.load_results(task, include_plm=False)
        
        if df.empty:
            return f"% No results found for {task}\n"
        
        # Filter for linguistic modification results (exclude baseline and special variants)
        exclude_patterns = [
            f'^{task}$',  # baseline task results
            f'^cot-{task}$',  # chain-of-thought baseline
            f'^{task}_',  # task-specific variants like coref_capitalization
            f'^cot-{task}_',  # CoT variants
            f'^cot-{task}-',  # CoT dash variants
            'DP$',  # Direct Prompt variants
        ]
        
        mod_df = df.copy()
        for pattern in exclude_patterns:
            mod_df = mod_df[~mod_df['modification'].str.match(pattern, na=False)]
        
        if mod_df.empty:
            return f"% No modification results found for {task}\n"
        
        # Use modification names as-is
        mod_df['modification_clean'] = mod_df['modification']
        
        # Create pivot table
        pivot = mod_df.pivot_table(
            values='accuracy',
            index='model',
            columns='modification_clean',
            aggfunc='mean'
        )
        
        # Sort models by average performance
        pivot['Average'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('Average', ascending=False)
        
        # Generate LaTeX
        latex = []
        latex.append(f"% Modification Results Table for {task.upper()}\n")
        latex.append("\\begin{table}[htbp]\n")
        latex.append("\\centering\n")
        latex.append("\\small\n")  # Use smaller font for large tables
        latex.append(f"\\caption{{Linguistic Modification Results - {self._format_task_name(task)}}}\n")
        latex.append(f"\\label{{tab:{task}_modifications}}\n")
        
        # For wide tables, use adjustbox
        if len(pivot.columns) > 8:
            latex.append("\\adjustbox{width=\\textwidth}{\n")
        
        # Column format
        num_cols = len(pivot.columns)
        col_format = "l" + "r" * num_cols
        latex.append(f"\\begin{{tabular}}{{{col_format}}}\n")
        latex.append("\\toprule\n")
        
        # Header (abbreviated for space)
        headers = ["Model"] + [self._abbreviate_modification(col) for col in pivot.columns]
        latex.append(" & ".join(headers) + " \\\\\n")
        latex.append("\\midrule\n")
        
        # Data rows
        for model, row in pivot.iterrows():
            model_name = self._format_model_name(model)
            values = []
            for val in row.values:
                if pd.isna(val):
                    values.append("-")
                else:
                    values.append(f"{val:.1f}")
            
            row_str = model_name + " & " + " & ".join(values)
            latex.append(row_str + " \\\\\n")
        
        latex.append("\\bottomrule\n")
        latex.append("\\end{tabular}\n")
        
        if len(pivot.columns) > 8:
            latex.append("}\n")  # Close adjustbox
        
        latex.append("\\end{table}\n")
        
        return ''.join(latex)
    
    def generate_weighted_delta_table(self, task: str) -> str:
        """Generate LaTeX table for weighted delta values by modification."""
        # Load results from pre-processed CSV
        results_file = Path(f"../PLM/{self._get_task_folder(task)}/tmp/{self._get_task_results_file(task)}")
        
        if not results_file.exists():
            return f"% No weighted delta results found for {task}\n"
        
        df = pd.read_csv(results_file)
        
        if df.empty or 'weighted_delta' not in df.columns:
            return f"% No weighted delta data found for {task}\n"
        
        # Filter for linguistic modification results (exclude baseline)
        exclude_patterns = [
            f'^{task}$',  # baseline task results
            f'^cot-{task}$',  # chain-of-thought baseline
            f'^{task}_',  # task-specific variants like coref_capitalization
            f'^cot-{task}_',  # CoT variants
            f'^cot-{task}-',  # CoT dash variants
            'DP$',  # Direct Prompt variants
        ]
        
        mod_df = df.copy()
        for pattern in exclude_patterns:
            mod_df = mod_df[~mod_df['modification'].str.match(pattern, na=False)]
        
        if mod_df.empty:
            return f"% No modification results found for {task}\n"
        
        # Create pivot table for weighted delta values
        pivot = mod_df.pivot_table(
            values='weighted_delta',
            index='model',
            columns='modification',
            aggfunc='mean'
        )
        
        # Sort models by average weighted delta (most negative = most degradation)
        pivot['Average'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('Average', ascending=True)  # Most negative first
        
        # Generate LaTeX
        latex = []
        latex.append(f"% Weighted Delta Table for {task.upper()}\n")
        latex.append("\\begin{table}[htbp]\n")
        latex.append("\\centering\n")
        latex.append("\\small\n")
        latex.append(f"\\caption{{Weighted Delta Values - {self._format_task_name(task)}}}\n")
        latex.append(f"\\label{{tab:{task}_weighted_delta}}\n")
        
        # For wide tables, use adjustbox
        if len(pivot.columns) > 8:
            latex.append("\\adjustbox{width=\\textwidth}{\n")
        
        # Column format
        num_cols = len(pivot.columns)
        col_format = "l" + "r" * num_cols
        latex.append(f"\\begin{{tabular}}{{{col_format}}}\n")
        latex.append("\\toprule\n")
        
        # Header (abbreviated for space)
        headers = ["Model"] + [self._abbreviate_modification(col) for col in pivot.columns]
        latex.append(" & ".join(headers) + " \\\\\n")
        latex.append("\\midrule\n")
        
        # Data rows
        for model, row in pivot.iterrows():
            model_name = self._format_model_name(model)
            values = []
            for val in row.values:
                if pd.isna(val):
                    values.append("-")
                else:
                    # Format weighted delta with more precision and include sign
                    if val >= 0:
                        values.append(f"+{val:.1f}")
                    else:
                        values.append(f"{val:.1f}")
            
            row_str = model_name + " & " + " & ".join(values)
            latex.append(row_str + " \\\\\n")
        
        latex.append("\\bottomrule\n")
        latex.append("\\end{tabular}\n")
        
        if len(pivot.columns) > 8:
            latex.append("}\n")  # Close adjustbox
        
        latex.append("\\end{table}\n")
        
        return ''.join(latex)
    
    def _get_task_folder(self, task: str) -> str:
        """Get the folder name for a task."""
        task_folders = {
            'coref': 'coreference_resolution',
            'dialogue': 'dialogue',
            'ner': 'ner',
            'sa': 'sentiment_analysis'
        }
        return task_folders.get(task, task)
    
    def _get_task_results_file(self, task: str) -> str:
        """Get the results file name for a task."""
        task_files = {
            'coref': 'coreference_llm_results.csv',
            'dialogue': 'dialogue_llm_results.csv',
            'ner': 'ner_llm_results.csv',
            'sa': 'sentiment_llm_results.csv'
        }
        return task_files.get(task, f"{task}_llm_results.csv")
    
    def generate_comparison_table(self, tasks: list = None) -> str:
        """Generate cross-task comparison table."""
        if tasks is None:
            tasks = ['coref', 'dialogue', 'ner', 'sa']
        
        all_results = []
        for task in tasks:
            df = self.analyzer.load_results(task, include_plm=False)
            if not df.empty:
                # For notebook-style data, use original_accuracy
                if 'original_accuracy' in df.columns and not df['original_accuracy'].isna().all():
                    main_df = df.groupby('model')['original_accuracy'].first().reset_index()
                    main_df.rename(columns={'original_accuracy': 'accuracy'}, inplace=True)
                    main_df['task'] = task
                    all_results.append(main_df)
                else:
                    # Fallback: Get main benchmark results only
                    if task == 'sa':
                        main_df = df[df['modification'] == 'sst2']
                    else:
                        main_df = df[df['modification'] == task]
                    
                    if not main_df.empty:
                        main_df['task'] = task
                        all_results.append(main_df)
        
        if not all_results:
            return "% No results found for comparison\n"
        
        combined = pd.concat(all_results, ignore_index=True)
        
        # Create pivot table
        pivot = combined.pivot_table(
            values='accuracy',
            index='model',
            columns='task',
            aggfunc='mean'
        )
        
        # Add average and sort
        pivot['Average'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('Average', ascending=False)
        
        # Generate LaTeX
        latex = []
        latex.append("% Cross-Task Comparison Table\n")
        latex.append("\\begin{table}[htbp]\n")
        latex.append("\\centering\n")
        latex.append("\\caption{LLM Performance Across FLUKE Tasks}\n")
        latex.append("\\label{tab:cross_task_comparison}\n")
        latex.append("\\begin{tabular}{l" + "r" * (len(tasks) + 1) + "}\n")
        latex.append("\\toprule\n")
        
        # Header
        headers = ["Model"] + [self._format_task_name(t) for t in tasks] + ["Average"]
        latex.append(" & ".join(headers) + " \\\\\n")
        latex.append("\\midrule\n")
        
        # Data rows
        for model, row in pivot.iterrows():
            model_name = self._format_model_name(model)
            values = []
            
            # Task values
            for task in tasks:
                if task in row and not pd.isna(row[task]):
                    val = f"{row[task]:.1f}"
                    # Bold if best in column
                    col_vals = pivot[task].dropna()
                    if len(col_vals) > 0 and row[task] == col_vals.max():
                        val = f"\\textbf{{{val}}}"
                    values.append(val)
                else:
                    values.append("-")
            
            # Average
            if not pd.isna(row['Average']):
                avg_val = f"{row['Average']:.1f}"
                # Bold if best average
                if row['Average'] == pivot['Average'].max():
                    avg_val = f"\\textbf{{{avg_val}}}"
                values.append(avg_val)
            else:
                values.append("-")
            
            latex.append(f"{model_name} & " + " & ".join(values) + " \\\\\n")
        
        latex.append("\\bottomrule\n")
        latex.append("\\end{tabular}\n")
        latex.append("\\end{table}\n")
        
        return ''.join(latex)
    
    def generate_frontier_models_table(self) -> str:
        """Generate comparison table for frontier models (GPT-5 and DeepSeek R1)."""
        tasks = ['coref', 'dialogue', 'ner', 'sa']
        
        # Collect data for frontier models
        frontier_data = []
        
        for task in tasks:
            df = self.analyzer.load_results(task, include_plm=False)
            if not df.empty:
                # Filter for GPT-5 and DeepSeek
                frontier_df = df[
                    (df['model'].str.contains('gpt-5|gpt5', case=False, na=False)) |
                    (df['model'].str.contains('deepseek', case=False, na=False)) |
                    (df['model'] == 'gpt4o')  # Include GPT-4o for comparison
                ]
                
                if not frontier_df.empty:
                    # Get main benchmark results
                    if task == 'sa':
                        main_df = frontier_df[frontier_df['modification'] == 'sst2']
                    else:
                        main_df = frontier_df[frontier_df['modification'] == task]
                    
                    if not main_df.empty:
                        main_df['task'] = task
                        frontier_data.append(main_df)
        
        if not frontier_data:
            return "% No frontier model results found\n"
        
        combined = pd.concat(frontier_data, ignore_index=True)
        
        # Create pivot table
        pivot = combined.pivot_table(
            values='accuracy',
            index='model',
            columns='task',
            aggfunc='mean'
        )
        
        # Generate LaTeX
        latex = []
        latex.append("% Frontier Models Comparison Table\n")
        latex.append("\\begin{table}[htbp]\n")
        latex.append("\\centering\n")
        latex.append("\\caption{Frontier Models Performance Comparison}\n")
        latex.append("\\label{tab:frontier_models}\n")
        latex.append("\\begin{tabular}{l" + "r" * len(pivot.columns) + "r}\n")
        latex.append("\\toprule\n")
        
        # Header
        headers = ["Model"] + [self._format_task_name(t) for t in pivot.columns] + ["Average"]
        latex.append(" & ".join(headers) + " \\\\\n")
        latex.append("\\midrule\n")
        
        # Calculate averages
        pivot['Average'] = pivot.mean(axis=1)
        
        # Sort by model name for clear comparison
        model_order = ['gpt4o', 'gpt-5', 'deepseek-r1']
        pivot = pivot.reindex([m for m in model_order if m in pivot.index])
        
        # Data rows
        for model, row in pivot.iterrows():
            model_name = self._format_model_name(model)
            values = []
            
            for col in pivot.columns:
                if not pd.isna(row[col]):
                    val = f"{row[col]:.1f}"
                    # Bold if best in column
                    col_vals = pivot[col].dropna()
                    if len(col_vals) > 0 and row[col] == col_vals.max():
                        val = f"\\textbf{{{val}}}"
                    values.append(val)
                else:
                    values.append("-")
            
            latex.append(f"{model_name} & " + " & ".join(values) + " \\\\\n")
        
        # Add improvement row (GPT-5 vs GPT-4o)
        if 'gpt-5' in pivot.index and 'gpt4o' in pivot.index:
            latex.append("\\midrule\n")
            latex.append("\\textit{GPT-5 vs GPT-4o}")
            
            for col in pivot.columns[:-1]:  # Exclude Average column
                if col in pivot.columns:
                    gpt5_val = pivot.loc['gpt-5', col] if not pd.isna(pivot.loc['gpt-5', col]) else None
                    gpt4o_val = pivot.loc['gpt4o', col] if not pd.isna(pivot.loc['gpt4o', col]) else None
                    
                    if gpt5_val is not None and gpt4o_val is not None:
                        diff = gpt5_val - gpt4o_val
                        sign = "+" if diff > 0 else ""
                        latex.append(f" & {sign}{diff:.1f}")
                    else:
                        latex.append(" & -")
            
            # Average improvement
            avg_gpt5 = pivot.loc['gpt-5', 'Average']
            avg_gpt4o = pivot.loc['gpt4o', 'Average']
            avg_diff = avg_gpt5 - avg_gpt4o
            sign = "+" if avg_diff > 0 else ""
            latex.append(f" & {sign}{avg_diff:.1f}")
            latex.append(" \\\\\n")
        
        latex.append("\\bottomrule\n")
        latex.append("\\end{tabular}\n")
        latex.append("\\end{table}\n")
        
        return ''.join(latex)
    
    def _format_task_name(self, task: str) -> str:
        """Format task name for LaTeX."""
        mapping = {
            'coref': 'Coreference Resolution',
            'dialogue': 'Dialogue Understanding',
            'ner': 'Named Entity Recognition',
            'sa': 'Sentiment Analysis'
        }
        return mapping.get(task, task.upper())
    
    def _format_model_name(self, model: str) -> str:
        """Format model name for LaTeX."""
        # Escape underscores and special characters
        model = str(model).replace('_', '\\_')
        
        # Special formatting for certain models
        if 'gpt-5' in model.lower():
            return 'GPT-5'
        elif 'gpt4o' in model.lower():
            return 'GPT-4o'
        elif 'deepseek-r1' in model.lower():
            return 'DeepSeek-R1'
        elif 'claude' in model.lower():
            return 'Claude-3.5'
        elif 'llama3_405B' in model:
            return 'Llama-3-405B'
        elif 'llama3.1_70b' in model.lower():
            return 'Llama-3.1-70B'
        elif 'llama3.1_8b' in model.lower():
            return 'Llama-3.1-8B'
        elif 'llama' in model.lower():
            return 'Llama'
        elif 'mixtral' in model.lower():
            if '8x22b' in model.lower():
                return 'Mixtral-8x22B'
            return 'Mixtral'
        
        return model
    
    def _format_column_name(self, col: str) -> str:
        """Format column name for LaTeX header."""
        if col == 'Average':
            return '\\textbf{Avg}'
        
        # Clean up column names
        col = str(col).replace('_', ' ').replace('-', ' ')
        
        if 'cot' in col.lower():
            return 'CoT'
        elif col == 'coref':
            return 'Coref'
        elif col == 'dialogue':
            return 'Dialog'
        elif col == 'ner':
            return 'NER'
        elif col == 'sst2':
            return 'SST-2'
        
        return col.title()
    
    def _abbreviate_modification(self, mod: str) -> str:
        """Abbreviate modification names for space."""
        abbreviations = {
            'active_to_passive': 'A→P',
            'capitalization': 'Cap',
            'punctuation': 'Punc',
            'typo_bias': 'Typo',
            'compound_word': 'Comp',
            'grammatical_role': 'Gram',
            'negation': 'Neg',
            'sentiment': 'Sent',
            'concept_replacement': 'Conc',
            'derivation': 'Deriv',
            'coordinating_conjunction': 'Coord',
            'discourse': 'Disc',
            'temporal_bias': 'Temp',
            'geographical_bias': 'Geo',
            'length_bias': 'Len',
            'casual': 'Cas',
            'dialectal': 'Dial',
            'Average': 'Avg'
        }
        
        return abbreviations.get(mod, mod[:4])
    
    def generate_all_tables(self):
        """Generate all LaTeX tables and save to files."""
        tasks = ['coref', 'dialogue', 'ner', 'sa']
        
        # Ensure output directory exists
        tables_dir = self.organizer.base_path / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        
        all_latex = []
        
        # Add preamble
        preamble = """% LaTeX Tables for FLUKE LLM Analysis
% Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
% Required packages: booktabs, adjustbox

"""
        all_latex.append(preamble)
        
        # Generate cross-task comparison
        print("Generating cross-task comparison table...")
        all_latex.append(self.generate_comparison_table())
        all_latex.append("\n")
        
        # Generate frontier models table
        print("Generating frontier models comparison table...")
        all_latex.append(self.generate_frontier_models_table())
        all_latex.append("\n")
        
        # Generate task-specific tables
        for task in tasks:
            print(f"Generating tables for {task.upper()}...")
            
            # Main results
            all_latex.append(f"\n% ============ {task.upper()} TASK ============\n")
            all_latex.append(self.generate_main_results_table(task))
            all_latex.append("\n")
            
            # Modification results
            all_latex.append(self.generate_modification_table(task))
            all_latex.append("\n")
            
            # Weighted delta results
            all_latex.append(self.generate_weighted_delta_table(task))
            all_latex.append("\n")
        
        # Save to file
        output_path = self.organizer.get_output_path("fluke_llm_tables.tex", "tables")
        with open(output_path, 'w') as f:
            f.writelines(all_latex)
        
        print(f"\n✓ LaTeX tables saved to: {output_path}")
        
        # Also save individual tables
        for task in tasks:
            # Main results
            task_latex = self.generate_main_results_table(task)
            task_path = self.organizer.get_output_path(f"{task}_main_results.tex", "tables")
            with open(task_path, 'w') as f:
                f.write(task_latex)
            
            # Modification results
            mod_latex = self.generate_modification_table(task)
            mod_path = self.organizer.get_output_path(f"{task}_modifications.tex", "tables")
            with open(mod_path, 'w') as f:
                f.write(mod_latex)
            
            # Weighted delta results
            delta_latex = self.generate_weighted_delta_table(task)
            delta_path = self.organizer.get_output_path(f"{task}_weighted_delta.tex", "tables")
            with open(delta_path, 'w') as f:
                f.write(delta_latex)
        
        # Save comparison tables separately
        comparison_path = self.organizer.get_output_path("cross_task_comparison.tex", "tables")
        with open(comparison_path, 'w') as f:
            f.write(self.generate_comparison_table())
        
        frontier_path = self.organizer.get_output_path("frontier_models.tex", "tables")
        with open(frontier_path, 'w') as f:
            f.write(self.generate_frontier_models_table())
        
        return output_path


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("       FLUKE LaTeX Table Generator")
    print("="*60)
    
    generator = LaTeXTableGenerator()
    output_path = generator.generate_all_tables()
    
    print("\n" + "="*60)
    print("✨ LaTeX Generation Complete!")
    print("="*60)
    print(f"\nGenerated files in: {generator.organizer.base_path}/tables/")
    print("\nFiles created:")
    print("  • fluke_llm_tables.tex (all tables)")
    print("  • cross_task_comparison.tex")
    print("  • frontier_models.tex")
    print("  • [task]_main_results.tex (for each task)")
    print("  • [task]_modifications.tex (for each task)")
    print("\nTo use in LaTeX document:")
    print("  \\usepackage{booktabs}")
    print("  \\usepackage{adjustbox}")
    print("  \\input{fluke_llm_tables.tex}")


if __name__ == "__main__":
    main()
