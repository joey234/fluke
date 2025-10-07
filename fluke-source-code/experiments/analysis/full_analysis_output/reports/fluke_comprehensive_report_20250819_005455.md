# FLUKE Comprehensive Analysis Report

**Generated**: 2025-08-19 00:54:54
**Base Path**: ../
**Tasks Analyzed**: COREF, DIALOGUE, NER, SA
**Weighted Delta**: Yes
**Statistical Tests**: Yes

---

## Executive Summary

- **Total Experiments**: 518
- **Models Evaluated**: 13
- **Modifications Tested**: 67
- **Average Accuracy**: 56.37%
- **Best Performing Model**: mixtral
- **GPT-5 Average Accuracy**: 49.74%
- **DeepSeek R1 Average Accuracy**: 5.64%

## Statistical Analysis Summary

- **Total Statistical Tests**: 302
- **Significant Results**: 114 (37.7%)

## Task-wise Performance Analysis

### COREF Task

### DIALOGUE Task

### NER Task

- **Average Weighted Delta**: 4.252
- **Most Challenging Modification**: dialectal_compare (20.369)

#### Top 3 Challenging Modifications:
1. **dialectal_compare** - claude: 20.369
2. **coordinating_conjunction_compare** - claude: 20.339
3. **active_to_passive_compare** - claude: 18.350
### SA Task


## Generated Files

### LaTeX Tables
- `coref_main_results.tex`
- `coref_modifications.tex`
- `dialogue_main_results.tex`
- `dialogue_modifications.tex`
- `ner_main_results.tex`
- `ner_modifications.tex`
- `sa_main_results.tex`
- `sa_modifications.tex`
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
\input{dialogue_main_results.tex}
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

