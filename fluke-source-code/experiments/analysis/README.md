# FLUKE Analysis Framework

A comprehensive, consolidated analysis framework for FLUKE (Few-shot Language Understanding and Knowledge Evaluation) experimental results.

## Overview

This framework consolidates and cleans up the analysis from the original Jupyter notebooks (`parse_coref_dialog.ipynb`, `parse_ner.ipynb`, `parse_sa.ipynb`) into a modular, maintainable Python package.

## Structure

```
analysis/
├── consolidated_analysis.py  # Main analysis engine
├── visualization.py          # Visualization components
├── utils.py                  # Utilities and helpers
├── run_analysis.py          # Main execution script
└── README.md                # This file
```

## Components

### 1. **consolidated_analysis.py**
Main analysis module containing:
- `FLUKEAnalyzer`: Core analyzer for all tasks
- `NERAnalyzer`: Specialized NER analysis
- `DialogueAnalyzer`: Dialogue task analysis
- `SentimentAnalyzer`: Sentiment analysis components

### 2. **visualization.py**
Comprehensive visualization module:
- Model performance comparisons
- Modification impact heatmaps
- Performance distribution plots
- PLM vs LLM comparisons
- Modification severity rankings

### 3. **utils.py**
Common utilities:
- `Config`: Configuration constants
- `DataLoader`: Multi-format data loading
- `MetricsCalculator`: Accuracy, F1, precision/recall
- `ResultsProcessor`: Result normalization and aggregation
- `FileOrganizer`: Output file management
- `StatisticalTester`: Statistical significance tests

### 4. **run_analysis.py**
Main execution script with CLI interface.

## Installation

```bash
# Install required dependencies
pip install pandas numpy scipy matplotlib seaborn scikit-learn tqdm
```

## Usage

### Basic Usage

```bash
# Analyze all tasks
python run_analysis.py

# Analyze specific task
python run_analysis.py --task ner

# Generate visualizations
python run_analysis.py --visualize

# Generate comprehensive report
python run_analysis.py --report

# Run statistical tests
python run_analysis.py --statistical-tests
```

### Advanced Options

```bash
# Full analysis with all features
python run_analysis.py \
    --task all \
    --base-path ../ \
    --output-dir analysis_results \
    --visualize \
    --report \
    --statistical-tests \
    --export-format excel
```

### Python API

```python
from consolidated_analysis import FLUKEAnalyzer
from visualization import FLUKEVisualizer

# Initialize analyzer
analyzer = FLUKEAnalyzer(base_path="../")

# Load results for a task
df = analyzer.load_results("ner")

# Analyze performance drops
drops_df = analyzer.analyze_performance_drop("ner")

# Statistical significance testing
result = analyzer.statistical_significance_test("ner", "bert", "gpt4o")

# Generate visualizations
visualizer = FLUKEVisualizer()
visualizer.plot_model_comparison(df, "ner", save_path="ner_comparison.png")

# Generate summary report
report = analyzer.generate_summary_report("analysis_report.md")
```

## Tasks Supported

1. **Coreference Resolution (coref)**
   - Pronoun resolution accuracy
   - Cross-sentence reference tracking

2. **Dialogue Contradiction Detection (dialogue)**
   - Contradiction identification
   - Context consistency analysis

3. **Named Entity Recognition (ner)**
   - Entity extraction accuracy
   - Entity type performance

4. **Sentiment Analysis (sa)**
   - Sentiment classification
   - Polarity shift analysis

## Modification Types

The framework analyzes 17 linguistic modifications:
- active_to_passive
- capitalization
- casual
- compound_word
- concept_replacement
- coordinating_conjunction
- derivation
- dialectal
- discourse
- geographical_bias
- grammatical_role
- length_bias
- negation
- punctuation
- sentiment
- singlish
- temporal_bias
- typo_bias

## Output Structure

```
analysis_output/
├── data/                    # Processed results
│   ├── *_results.csv
│   └── *_performance_drops.csv
├── visualizations/          # Generated plots
│   ├── *_model_comparison.png
│   └── *_modification_impact.png
├── reports/                 # Analysis reports
│   └── fluke_analysis_report_*.md
└── logs/                    # Execution logs
```

## Key Features

### 1. Consolidated Analysis
- Unified processing for PLM and LLM results
- Automatic model and modification detection
- Performance drop calculations
- Cross-task comparisons

### 2. Statistical Testing
- Wilcoxon signed-rank test (paired)
- Mann-Whitney U test (independent)
- Effect size calculations
- Significance level: p < 0.05

### 3. Visualization Suite
- Model performance bar charts
- Modification impact heatmaps
- Performance distribution violin plots
- PLM vs LLM comparisons
- Severity ranking plots

### 4. Comprehensive Reporting
- Executive summaries
- Task-wise analysis
- Model rankings
- Key insights
- Markdown and LaTeX output

## Supported Models

### Pre-trained Language Models (PLMs)
- BERT (bert-base-cased)
- GPT-2
- T5 (t5-base)

### Large Language Models (LLMs)
- **OpenAI Models**: GPT-4o, GPT-5
- **Anthropic Models**: Claude 3.5 Sonnet
- **Meta Models**: Llama 3, Llama 3.1
- **Mistral Models**: Mixtral, Mixtral-8x22b
- **DeepSeek Models**: DeepSeek R1
- **Other Models**: O1, O3

## Improvements Over Original Notebooks

1. **Modularity**: Separated concerns into distinct modules
2. **Reusability**: Functions can be imported and reused
3. **Maintainability**: Clear structure and documentation
4. **Extensibility**: Easy to add new tasks or models
5. **Automation**: CLI interface for batch processing
6. **Error Handling**: Robust error handling throughout
7. **Configuration**: Centralized configuration management
8. **Output Organization**: Structured output directory
9. **New Model Support**: Added GPT-5 and DeepSeek R1 integration

## Contributing

To add new analysis capabilities:

1. Extend task-specific analyzers in `consolidated_analysis.py`
2. Add new visualization types in `visualization.py`
3. Update configuration in `utils.py`
4. Document changes in this README

## License

This analysis framework is part of the FLUKE project.

## Contact

For questions or issues, please refer to the main FLUKE repository.