# FLUKE Comprehensive Analysis Report

**Generated**: 2025-08-19 00:47:10
**Base Path**: ../
**Tasks Analyzed**: COREF
**Weighted Delta**: Yes
**Statistical Tests**: Yes

---

## Executive Summary

- **Total Experiments**: 124
- **Models Evaluated**: 6
- **Modifications Tested**: 29
- **Average Accuracy**: 75.74%
- **Best Performing Model**: gpt-5
- **GPT-5 Average Accuracy**: 85.51%
- **DeepSeek R1 Average Accuracy**: 84.17%

## Statistical Analysis Summary

- **Total Statistical Tests**: 111
- **Significant Results**: 52 (46.8%)

## Task-wise Performance Analysis

### COREF Task

- **Average Weighted Delta**: 0.602
- **Most Challenging Modification**: cot-coref_capitalization (13.561)

#### Top 3 Challenging Modifications:
1. **cot-coref_capitalization** - gpt4o: 13.561
2. **cot-coref_punctuation** - gpt4o: 12.628
3. **coref_punctuation** - gpt4o: 11.695

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

