#!/usr/bin/env python
"""
FLUKE Visualization Module
Creates comprehensive visualizations for FLUKE experiment results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class FLUKEVisualizer:
    """Main visualization class for FLUKE results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualizer with default figure size."""
        self.figsize = figsize
        self.colors = {
            'plm': '#3498db',  # Blue for PLMs
            'llm': '#e74c3c',  # Red for LLMs
            'bert': '#2ecc71',
            'gpt2': '#f39c12',
            't5': '#9b59b6',
            'gpt4o': '#e74c3c',
            'gpt-5': '#ff6b6b',  # Bright red for GPT-5
            'gpt5': '#ff6b6b',
            'claude': '#3498db',
            'claude-3-5-sonnet': '#3498db',
            'llama': '#1abc9c',
            'mixtral': '#34495e',
            'deepseek-r1': '#ff9f43',  # Orange for DeepSeek R1
            'deepseek': '#ff9f43',
            'o1': '#a29bfe',  # Light purple for O1
            'o3': '#6c5ce7'   # Dark purple for O3
        }
        
    def plot_model_comparison(self, df: pd.DataFrame, task: str, save_path: Optional[str] = None):
        """Create bar plot comparing model performances."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # PLM comparison
        plm_df = df[df['model_type'] == 'plm']
        if not plm_df.empty:
            plm_avg = plm_df.groupby('model')['accuracy'].mean().sort_values(ascending=False)
            ax1.bar(range(len(plm_avg)), plm_avg.values, 
                   color=[self.colors.get(m, '#95a5a6') for m in plm_avg.index])
            ax1.set_xticks(range(len(plm_avg)))
            ax1.set_xticklabels(plm_avg.index, rotation=45, ha='right')
            ax1.set_ylabel('Average Accuracy (%)')
            ax1.set_title(f'PLM Performance - {task.upper()}')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(plm_avg.values):
                ax1.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
        
        # LLM comparison
        llm_df = df[df['model_type'] == 'llm']
        if not llm_df.empty:
            llm_avg = llm_df.groupby('model')['accuracy'].mean().sort_values(ascending=False)
            ax2.bar(range(len(llm_avg)), llm_avg.values,
                   color=[self.colors.get(m, '#95a5a6') for m in llm_avg.index])
            ax2.set_xticks(range(len(llm_avg)))
            ax2.set_xticklabels(llm_avg.index, rotation=45, ha='right')
            ax2.set_ylabel('Average Accuracy (%)')
            ax2.set_title(f'LLM Performance - {task.upper()}')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(llm_avg.values):
                ax2.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
        
        plt.suptitle(f'Model Performance Comparison - {task.upper()} Task', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_modification_impact(self, drops_df: pd.DataFrame, task: str, save_path: Optional[str] = None):
        """Create heatmap showing impact of modifications on different models."""
        if drops_df.empty:
            print(f"No data available for {task}")
            return
            
        # Pivot data for heatmap
        pivot_df = drops_df.pivot_table(
            values='relative_drop_%',
            index='modification',
            columns='model',
            aggfunc='mean'
        )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdYlBu_r',
                   center=0, vmin=-50, vmax=50, cbar_kws={'label': 'Performance Drop (%)'},
                   ax=ax)
        ax.set_title(f'Modification Impact Heatmap - {task.upper()} Task', fontsize=14, pad=20)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Modification Type', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_performance_distribution(self, df: pd.DataFrame, task: str, save_path: Optional[str] = None):
        """Create violin plot showing performance distribution across modifications."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Prepare data
        plot_df = df[df['modification'] != 'original'].copy()
        
        if not plot_df.empty:
            # Create violin plot
            parts = ax.violinplot(
                [plot_df[plot_df['model'] == m]['accuracy'].values 
                 for m in plot_df['model'].unique()],
                positions=range(len(plot_df['model'].unique())),
                showmeans=True,
                showmedians=True
            )
            
            # Customize colors
            for pc in parts['bodies']:
                pc.set_facecolor('#3498db')
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(len(plot_df['model'].unique())))
            ax.set_xticklabels(plot_df['model'].unique(), rotation=45, ha='right')
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_title(f'Performance Distribution Across Modifications - {task.upper()}', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_plm_vs_llm(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Create comparison plot between PLMs and LLMs across all tasks."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        tasks = ['coref', 'dialogue', 'ner', 'sa']
        
        for i, task in enumerate(tasks):
            ax = axes[i]
            task_df = df[df['task'] == task] if 'task' in df.columns else pd.DataFrame()
            
            if not task_df.empty:
                # Calculate average performance by model type
                avg_perf = task_df.groupby('model_type')['accuracy'].agg(['mean', 'std'])
                
                # Create bar plot with error bars
                x = range(len(avg_perf))
                bars = ax.bar(x, avg_perf['mean'], yerr=avg_perf['std'],
                             color=['#3498db', '#e74c3c'], alpha=0.7, capsize=5)
                
                ax.set_xticks(x)
                ax.set_xticklabels(avg_perf.index, fontsize=11)
                ax.set_ylabel('Accuracy (%)', fontsize=11)
                ax.set_title(f'{task.upper()} Task', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar, (idx, row) in zip(bars, avg_perf.iterrows()):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + row['std'] + 1,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=10)
            else:
                ax.text(0.5, 0.5, f'No data for {task.upper()}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{task.upper()} Task', fontsize=12, fontweight='bold')
        
        plt.suptitle('PLM vs LLM Performance Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_modification_severity(self, drops_df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot modifications ranked by severity of performance drop."""
        if drops_df.empty:
            print("No performance drop data available")
            return
            
        # Calculate average drop per modification
        avg_drops = drops_df.groupby('modification')['relative_drop_%'].mean().sort_values(ascending=False)
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(avg_drops) * 0.3)))
        
        colors = ['#e74c3c' if d > 20 else '#f39c12' if d > 10 else '#3498db' 
                 for d in avg_drops.values]
        
        bars = ax.barh(range(len(avg_drops)), avg_drops.values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(avg_drops)))
        ax.set_yticklabels(avg_drops.index, fontsize=10)
        ax.set_xlabel('Average Performance Drop (%)', fontsize=12)
        ax.set_title('Modification Severity Ranking', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, avg_drops.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%', va='center', fontsize=9)
        
        # Add severity zones
        ax.axvline(x=10, color='orange', linestyle='--', alpha=0.5, label='Moderate impact')
        ax.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='Severe impact')
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_summary_dashboard(self, analyzer_results: Dict, save_path: Optional[str] = None):
        """Create a comprehensive dashboard with multiple visualizations."""
        fig = plt.figure(figsize=(16, 12))
        
        # Define grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Overall accuracy comparison
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_overall_accuracy(analyzer_results, ax1)
        
        # Task-wise performance
        ax2 = fig.add_subplot(gs[1, :2])
        self._plot_task_performance(analyzer_results, ax2)
        
        # Model type comparison
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_model_type_pie(analyzer_results, ax3)
        
        # Top challenging modifications
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_top_challenges(analyzer_results, ax4)
        
        # Performance matrix
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_performance_matrix(analyzer_results, ax5)
        
        plt.suptitle('FLUKE Analysis Dashboard', fontsize=18, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_overall_accuracy(self, results: Dict, ax):
        """Plot overall accuracy across all experiments."""
        # Implementation for overall accuracy plot
        ax.set_title('Overall Model Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        ax.grid(True, alpha=0.3)
        
    def _plot_task_performance(self, results: Dict, ax):
        """Plot task-wise performance comparison."""
        # Implementation for task performance plot
        ax.set_title('Task-wise Performance', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        ax.grid(True, alpha=0.3)
        
    def _plot_model_type_pie(self, results: Dict, ax):
        """Plot pie chart of model type distribution."""
        # Implementation for model type pie chart
        ax.set_title('Model Type Distribution', fontsize=12, fontweight='bold')
        
    def _plot_top_challenges(self, results: Dict, ax):
        """Plot top challenging modifications."""
        # Implementation for top challenges plot
        ax.set_title('Top 5 Challenging Modifications', fontsize=12, fontweight='bold')
        
    def _plot_performance_matrix(self, results: Dict, ax):
        """Plot performance matrix heatmap."""
        # Implementation for performance matrix
        ax.set_title('Performance Matrix', fontsize=12, fontweight='bold')


def generate_all_visualizations(base_path: str = "../"):
    """Generate all visualizations for FLUKE analysis."""
    from consolidated_analysis import FLUKEAnalyzer
    
    print("Generating FLUKE visualizations...")
    
    # Initialize components
    analyzer = FLUKEAnalyzer(base_path)
    visualizer = FLUKEVisualizer()
    
    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    tasks = ['coref', 'dialogue', 'ner', 'sa']
    
    for task in tasks:
        print(f"\nGenerating visualizations for {task.upper()}...")
        
        # Load data
        df = analyzer.load_results(task)
        if not df.empty:
            # Model comparison
            visualizer.plot_model_comparison(
                df, task, 
                save_path=output_dir / f"{task}_model_comparison.png"
            )
            
            # Performance drops
            drops_df = analyzer.analyze_performance_drop(task)
            if not drops_df.empty:
                # Modification impact heatmap
                visualizer.plot_modification_impact(
                    drops_df, task,
                    save_path=output_dir / f"{task}_modification_impact.png"
                )
                
                # Performance distribution
                visualizer.plot_performance_distribution(
                    df, task,
                    save_path=output_dir / f"{task}_performance_distribution.png"
                )
    
    # Generate cross-task visualizations
    print("\nGenerating cross-task visualizations...")
    
    # Combine all task results
    all_results = []
    for task in tasks:
        df = analyzer.load_results(task)
        if not df.empty:
            df['task'] = task
            all_results.append(df)
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # PLM vs LLM comparison
        visualizer.plot_plm_vs_llm(
            combined_df,
            save_path=output_dir / "plm_vs_llm_comparison.png"
        )
        
        # Overall modification severity
        all_drops = []
        for task in tasks:
            drops_df = analyzer.analyze_performance_drop(task)
            if not drops_df.empty:
                all_drops.append(drops_df)
        
        if all_drops:
            combined_drops = pd.concat(all_drops, ignore_index=True)
            visualizer.plot_modification_severity(
                combined_drops,
                save_path=output_dir / "modification_severity_ranking.png"
            )
    
    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    generate_all_visualizations()