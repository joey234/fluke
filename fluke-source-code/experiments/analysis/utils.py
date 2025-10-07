#!/usr/bin/env python
"""
FLUKE Analysis Utilities
Common utilities and helper functions for FLUKE analysis.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
import re
from datetime import datetime


class Config:
    """Configuration class for FLUKE analysis."""
    
    # Task mappings
    TASK_MAPPING = {
        'coref': 'coreference_resolution',
        'dialogue': 'dialogue_contradiction_detection', 
        'ner': 'named_entity_recognition',
        'sa': 'sentiment_analysis',
        'sentiment': 'sentiment_analysis'
    }
    
    # Modification types
    MODIFICATIONS = [
        'active_to_passive', 'capitalization', 'casual', 'compound_word',
        'concept_replacement', 'coordinating_conjunction', 'derivation',
        'dialectal', 'discourse', 'geographical_bias', 'grammatical_role',
        'length_bias', 'negation', 'punctuation', 'sentiment',
        'temporal_bias', 'typo_bias'
    ]
    
    # Model configurations
    PLM_MODELS = ['bert', 'bert-base-cased', 'gpt2', 't5', 't5-base']
    
    LLM_MODELS = [
        'gpt4o', 'gpt-4', 'gpt-5-standard', 'gpt-5',
        'claude-3-5-sonnet', 'claude',
        'llama', 'llama3', 'llama3.1',
        'mixtral', 'mixtral-8x22b',
        'deepseek-r1-deepseek', 'deepseek-r1', 'deepseek',
        'o1', 'o3'  # Adding other potential new models
    ]
    
    # File patterns
    RESULT_PATTERNS = {
        'plm': r'(.+)_predictions\.csv',
        'llm': r'(.+)-0shot-(.+)\.csv',
        'accuracy': r'(.+)_accuracy_comparison\.csv'
    }
    
    # Statistical thresholds
    SIGNIFICANCE_LEVEL = 0.05
    MIN_SAMPLES = 10
    
    # Visualization settings
    FIGURE_DPI = 300
    DEFAULT_FIGSIZE = (12, 8)
    COLOR_PALETTE = 'husl'


class DataLoader:
    """Utility class for loading various data formats."""
    
    @staticmethod
    def load_json(filepath: Union[str, Path]) -> Dict:
        """Load JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def load_csv(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load CSV file with error handling."""
        try:
            return pd.read_csv(filepath, **kwargs)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def load_predictions(filepath: Union[str, Path]) -> pd.DataFrame:
        """Load prediction results from various formats."""
        filepath = Path(filepath)
        
        if filepath.suffix == '.csv':
            return DataLoader.load_csv(filepath)
        elif filepath.suffix == '.json':
            data = DataLoader.load_json(filepath)
            return pd.DataFrame(data)
        elif filepath.suffix == '.txt':
            # Handle text format predictions
            return DataLoader._parse_text_predictions(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    @staticmethod
    def _parse_text_predictions(filepath: Path) -> pd.DataFrame:
        """Parse predictions from text file."""
        predictions = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Parse based on expected format
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        predictions.append({
                            'prediction': parts[0],
                            'ground_truth': parts[1] if len(parts) > 1 else None
                        })
        return pd.DataFrame(predictions)


class MetricsCalculator:
    """Calculate various metrics for model evaluation."""
    
    @staticmethod
    def accuracy(y_true: List, y_pred: List) -> float:
        """Calculate accuracy."""
        if len(y_true) != len(y_pred):
            raise ValueError("Prediction and ground truth lengths don't match")
        
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return (correct / len(y_true)) * 100 if y_true else 0
    
    @staticmethod
    def precision_recall_f1(y_true: List, y_pred: List, labels: Optional[List] = None) -> Dict:
        """Calculate precision, recall, and F1 scores."""
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average='weighted', zero_division=0
        )
        
        return {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'support': support
        }
    
    @staticmethod
    def confusion_matrix(y_true: List, y_pred: List, labels: Optional[List] = None) -> np.ndarray:
        """Calculate confusion matrix."""
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(y_true, y_pred, labels=labels)
    
    @staticmethod
    def performance_drop(baseline: float, modified: float, method: str = 'relative') -> float:
        """Calculate performance drop using various methods.
        
        Args:
            baseline: Original/baseline accuracy (0-100 scale)
            modified: Modified accuracy (0-100 scale)
            method: One of 'absolute', 'relative', 'log', or 'weighted_delta'
            
        Returns:
            Performance drop/delta value
        """
        if method == 'absolute':
            return baseline - modified
        elif method == 'relative':
            return ((baseline - modified) / baseline * 100) if baseline > 0 else 0
        elif method == 'log':
            import math
            if baseline > 0 and modified > 0:
                return math.log(baseline / modified)
            return 0
        elif method == 'weighted_delta':
            # FLUKE weighted delta metric: (B - A) * log10(A) / log10(100)
            import math
            if baseline > 0:
                return (modified - baseline) * math.log10(baseline) / math.log10(100)
            return 0
        else:
            raise ValueError(f"Unknown method: {method}")


class ResultsProcessor:
    """Process and format analysis results."""
    
    @staticmethod
    def normalize_model_name(model_name: str) -> str:
        """Normalize model names for consistency."""
        # Remove version numbers and special characters
        normalized = model_name.lower()
        normalized = re.sub(r'[-_]', '', normalized)
        
        # Map to standard names
        mappings = {
            'bertbasecased': 'bert',
            'bert-base-cased': 'bert',
            't5base': 't5',
            't5-base': 't5',
            'gpt4o': 'gpt4o',
            'gpt4': 'gpt4o',
            'gpt5standard': 'gpt5',
            'gpt5': 'gpt5',
            'claude35sonnet': 'claude',
            'llama3': 'llama',
            'llama31': 'llama',
            'mixtral8x22b': 'mixtral',
            'deepseekr1deepseek': 'deepseek-r1',
            'deepseekr1': 'deepseek-r1',
            'deepseek': 'deepseek-r1',
            'o1': 'o1',
            'o3': 'o3'
        }
        
        return mappings.get(normalized, normalized)
    
    @staticmethod
    def normalize_modification_name(mod_name: str) -> str:
        """Normalize modification names."""
        # Remove suffixes like _100
        normalized = re.sub(r'_\d+$', '', mod_name)
        # Convert to lowercase and replace underscores
        normalized = normalized.lower().replace('_', ' ')
        return normalized
    
    @staticmethod
    def aggregate_results(results_list: List[pd.DataFrame], 
                         group_by: List[str] = ['model', 'modification']) -> pd.DataFrame:
        """Aggregate multiple result dataframes."""
        if not results_list:
            return pd.DataFrame()
        
        combined = pd.concat(results_list, ignore_index=True)
        
        # Group and aggregate
        aggregated = combined.groupby(group_by).agg({
            'accuracy': ['mean', 'std', 'count'],
            'f1': 'mean',
            'precision': 'mean',
            'recall': 'mean'
        }).round(2)
        
        # Flatten column names
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
        aggregated.reset_index(inplace=True)
        
        return aggregated
    
    @staticmethod
    def create_summary_table(df: pd.DataFrame, metric: str = 'accuracy') -> pd.DataFrame:
        """Create a summary table with models as rows and modifications as columns."""
        if df.empty:
            return pd.DataFrame()
        
        pivot_table = df.pivot_table(
            values=metric,
            index='model',
            columns='modification',
            aggfunc='mean'
        ).round(2)
        
        # Add average column
        pivot_table['Average'] = pivot_table.mean(axis=1).round(2)
        
        # Sort by average performance
        pivot_table = pivot_table.sort_values('Average', ascending=False)
        
        return pivot_table

# Global Unrobustness helpers
def get_global_unrobustness_range() -> tuple:
    """Compute a global (min, max) range for unrobustness (U) across all tasks.
    Falls back to (0.0, 100.0) if no data found. Reads task-level LLM results CSVs
    emitted by the analysis scripts in this directory.
    """
    base = Path(__file__).parent
    csvs = [
        base / 'gsm_modification_results_llm.csv',
        base / 'sa_modification_results_llm.csv',
        base / 'ner_modification_results_llm.csv',
        base / 'dialogue_modification_results_llm.csv',
        base / 'coref_modification_results_llm.csv',
        base / 'ifeval_results_heatmap.png'  # sentinel; ignored if not CSV
    ]
    vals = []
    for p in csvs:
        try:
            if p.suffix.lower() == '.csv' and p.exists():
                df = pd.read_csv(p)
                if 'unrobustness' in df.columns:
                    vals.extend(pd.to_numeric(df['unrobustness'], errors='coerce').dropna().tolist())
        except Exception:
            continue
    if not vals:
        return (0.0, 100.0)
    vmin = float(np.nanmin(vals)) if len(vals) else 0.0
    vmax = float(np.nanmax(vals)) if len(vals) else 100.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return (0.0, 100.0)
    return (vmin, vmax)

def unrob_intensity(u: float, umin: float, umax: float) -> int:
    """Map a U value in [umin, umax] to a blue intensity [0,80] for LaTeX cellcolor."""
    try:
        u = float(u)
    except Exception:
        return 0
    if not np.isfinite(u) or not np.isfinite(umin) or not np.isfinite(umax) or umin >= umax:
        return 0
    t = (u - umin) / (umax - umin)
    if t < 0: t = 0.0
    elif t > 1: t = 1.0
    return int(round(80 * t))


class FileOrganizer:
    """Organize and manage analysis output files."""
    
    def __init__(self, base_path: Union[str, Path] = "analysis_output"):
        """Initialize file organizer."""
        self.base_path = Path(base_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure
        self._create_directories()
    
    def _create_directories(self):
        """Create organized directory structure."""
        dirs = [
            self.base_path,
            self.base_path / "reports",
            self.base_path / "visualizations",
            self.base_path / "data",
            self.base_path / "logs"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_output_path(self, filename: str, subdir: str = "data") -> Path:
        """Get output path for a file."""
        return self.base_path / subdir / filename
    
    def save_results(self, data: Union[pd.DataFrame, Dict], 
                    filename: str, subdir: str = "data") -> Path:
        """Save results to file."""
        output_path = self.get_output_path(filename, subdir)
        
        if isinstance(data, pd.DataFrame):
            if filename.endswith('.csv'):
                data.to_csv(output_path, index=False)
            elif filename.endswith('.xlsx'):
                data.to_excel(output_path, index=False)
            elif filename.endswith('.json'):
                data.to_json(output_path, orient='records', indent=2)
        elif isinstance(data, dict):
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        return output_path
    
    def create_report(self, content: str, filename: str = None) -> Path:
        """Create a markdown report."""
        if filename is None:
            filename = f"report_{self.timestamp}.md"
        
        output_path = self.get_output_path(filename, "reports")
        with open(output_path, 'w') as f:
            f.write(content)
        
        return output_path


class StatisticalTester:
    """Perform statistical tests on results."""
    
    @staticmethod
    def wilcoxon_test(x: List[float], y: List[float]) -> Dict:
        """Perform Wilcoxon signed-rank test."""
        from scipy import stats
        
        if len(x) != len(y):
            raise ValueError("Sample sizes must be equal for paired test")
        
        statistic, p_value = stats.wilcoxon(x, y)
        
        return {
            'test': 'Wilcoxon signed-rank',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < Config.SIGNIFICANCE_LEVEL,
            'effect_size': StatisticalTester._calculate_effect_size(x, y)
        }
    
    @staticmethod
    def mann_whitney_test(x: List[float], y: List[float]) -> Dict:
        """Perform Mann-Whitney U test."""
        from scipy import stats
        
        statistic, p_value = stats.mannwhitneyu(x, y, alternative='two-sided')
        
        return {
            'test': 'Mann-Whitney U',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < Config.SIGNIFICANCE_LEVEL,
            'effect_size': StatisticalTester._calculate_effect_size(x, y)
        }
    
    @staticmethod
    def _calculate_effect_size(x: List[float], y: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        x_mean, y_mean = np.mean(x), np.mean(y)
        x_std, y_std = np.std(x, ddof=1), np.std(y, ddof=1)
        
        pooled_std = np.sqrt((x_std**2 + y_std**2) / 2)
        
        if pooled_std == 0:
            return 0
        
        return (x_mean - y_mean) / pooled_std


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage string."""
    return f"{value:.{decimals}f}%"


def format_delta(value: float, decimals: int = 1) -> str:
    """Format delta value with +/- sign."""
    sign = '+' if value >= 0 else ''
    return f"{sign}{value:.{decimals}f}"


def create_latex_table(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """Convert DataFrame to LaTeX table."""
    latex = df.to_latex(index=True, escape=False, column_format='l' + 'c' * len(df.columns))
    
    if caption:
        latex = latex.replace('\\begin{tabular}', 
                            f'\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}')
    
    return latex


if __name__ == "__main__":
    # Test utilities
    print("FLUKE Analysis Utilities loaded successfully")
    print(f"Configured tasks: {list(Config.TASK_MAPPING.keys())}")
    print(f"Number of modifications: {len(Config.MODIFICATIONS)}")
    print(f"PLM models: {Config.PLM_MODELS}")
    print(f"LLM models: {Config.LLM_MODELS[:5]}...")  # Show first 5
