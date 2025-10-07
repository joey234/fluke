# FLUKE Comprehensive Analysis Report

**Generated**: 2025-08-19 00:37:48
**Base Path**: ../
**Tasks Analyzed**: NER
**Weighted Delta**: Yes
**Statistical Tests**: Yes

---

## Executive Summary

- **Total Experiments**: 194
- **Models Evaluated**: 5
- **Modifications Tested**: 54
- **Average Accuracy**: 0.19%
- **Best Performing Model**: gpt-5
- **GPT-5 Average Accuracy**: 0.45%
- **DeepSeek R1 Average Accuracy**: 0.27%

## Statistical Analysis Summary

- **Total Statistical Tests**: 5
- **Significant Results**: 0 (0.0%)

## Task-wise Performance Analysis

### NER Task

- **Average Weighted Delta**: -3.280
- **Most Challenging Modification**: temporal_bias (-1.646)

#### Top 3 Challenging Modifications:
1. **temporal_bias** - deepseek-r1: -1.646
2. **derivation** - deepseek-r1: -1.646
3. **sentiment** - deepseek-r1: -1.646

## Generated Files

### LaTeX Tables
- `ner_main_results.tex`
- `ner_modifications.tex`
- `cross_task_comparison.tex`
- `frontier_models.tex`

## Usage Instructions

### LaTeX Tables
Include the generated LaTeX files in your document:

```latex
\usepackage{booktabs}
\usepackage{adjustbox}

% Include tables
\input{ner_main_results.tex}
\input{ner_modifications.tex}
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

