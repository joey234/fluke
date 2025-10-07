# FLUKE Comprehensive Analysis Report

**Generated**: 2025-08-19 00:42:47
**Base Path**: ../
**Tasks Analyzed**: COREF
**Weighted Delta**: Yes
**Statistical Tests**: Yes

---

## Executive Summary

- **Total Experiments**: 124
- **Models Evaluated**: 6
- **Modifications Tested**: 29
- **Average Accuracy**: 5.25%
- **Best Performing Model**: gpt4o
- **GPT-5 Average Accuracy**: 4.61%
- **DeepSeek R1 Average Accuracy**: 4.53%

## Statistical Analysis Summary

- **Total Statistical Tests**: 111
- **Significant Results**: 52 (46.8%)

## Task-wise Performance Analysis

### COREF Task

- **Average Weighted Delta**: -58.167
- **Most Challenging Modification**: cot-coref (4.405)

#### Top 3 Challenging Modifications:
1. **cot-coref** - gpt4o: 4.405
2. **concept_replacement** - llama: 0.000
3. **negation** - llama: 0.000

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

