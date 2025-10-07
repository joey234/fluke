# FLUKE Comprehensive Analysis Report

**Generated**: 2025-08-19 10:13:41
**Base Path**: ../
**Tasks Analyzed**: COREF, DIALOGUE, NER, SA
**Weighted Delta**: Yes
**Statistical Tests**: Yes

---

## Executive Summary

- **Total Experiments**: 371
- **Models Evaluated**: 9
- **Modifications Tested**: 18
- **Average Accuracy**: 64.11%
- **Best Performing Model**: claude-3-5-sonnet
- **GPT-5 Average Accuracy**: 51.10%
- **DeepSeek R1 Average Accuracy**: 5.69%

## Statistical Analysis Summary

- **Total Statistical Tests**: 284
- **Significant Results**: 110 (38.7%)

## Task-wise Performance Analysis

### COREF Task

### DIALOGUE Task

### NER Task

### SA Task


## Generated Files

### LaTeX Tables
- `coref_main_results.tex`
- `coref_modifications.tex`
- `coref_weighted_delta.tex`
- `dialogue_main_results.tex`
- `dialogue_modifications.tex`
- `dialogue_weighted_delta.tex`
- `ner_main_results.tex`
- `ner_modifications.tex`
- `ner_weighted_delta.tex`
- `sa_main_results.tex`
- `sa_modifications.tex`
- `sa_weighted_delta.tex`
- `cross_task_comparison.tex`
- `frontier_models.tex`

### Visualizations
- `coref_model_comparison.png`
- `dialogue_model_comparison.png`
- `ner_model_comparison.png`
- `sa_model_comparison.png`

## Usage Instructions

### LaTeX Tables
Include the generated LaTeX files in your document:

```latex
\usepackage{booktabs}
\usepackage{adjustbox}

% Include tables
\input{coref_main_results.tex}
\input{coref_modifications.tex}
\input{coref_weighted_delta.tex}
```

### Weighted Delta Metric
The weighted delta metric is calculated as:

```
weighted_delta = (B - A) × log₁₀(A) / log₁₀(100)
```

Where:
- A = Original accuracy
- B = Modified accuracy
- This metric gives higher weight to drops from higher baseline accuracies

