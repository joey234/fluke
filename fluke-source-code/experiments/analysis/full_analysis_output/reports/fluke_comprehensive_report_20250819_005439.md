# FLUKE Comprehensive Analysis Report

**Generated**: 2025-08-19 00:54:39
**Base Path**: ../
**Tasks Analyzed**: COREF
**Weighted Delta**: Yes
**Statistical Tests**: Yes

---

## Executive Summary

- **Total Experiments**: 108
- **Models Evaluated**: 6
- **Modifications Tested**: 18
- **Average Accuracy**: 66.61%
- **Best Performing Model**: claude-3-5-sonnet

## Statistical Analysis Summary

- **Total Statistical Tests**: 57
- **Significant Results**: 39 (68.4%)

## Task-wise Performance Analysis

### COREF Task


## Generated Files

### LaTeX Tables
- `coref_main_results.tex`
- `coref_modifications.tex`
- `cross_task_comparison.tex`
- `frontier_models.tex`

## Usage Instructions

### LaTeX Tables
Include the generated LaTeX files in your document:

```latex
\usepackage{booktabs}
\usepackage{adjustbox}

% Include tables
\input{coref_main_results.tex}
\input{coref_modifications.tex}
\input{cross_task_comparison.tex}
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

