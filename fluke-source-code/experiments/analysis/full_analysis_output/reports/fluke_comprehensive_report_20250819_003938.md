# FLUKE Comprehensive Analysis Report

**Generated**: 2025-08-19 00:39:37
**Base Path**: ../
**Tasks Analyzed**: COREF, DIALOGUE, NER, SA
**Weighted Delta**: Yes
**Statistical Tests**: Yes

---

## Executive Summary

- **Total Experiments**: 548
- **Models Evaluated**: 9
- **Modifications Tested**: 85
- **Average Accuracy**: 4.24%
- **Best Performing Model**: llama3_405B
- **GPT-5 Average Accuracy**: 3.83%
- **DeepSeek R1 Average Accuracy**: 3.35%

## Statistical Analysis Summary

- **Total Statistical Tests**: 22
- **Significant Results**: 0 (0.0%)

## Task-wise Performance Analysis

### COREF Task

- **Average Weighted Delta**: -58.167
- **Most Challenging Modification**: cot-coref (4.405)

#### Top 3 Challenging Modifications:
1. **cot-coref** - gpt4o: 4.405
2. **concept_replacement** - llama: 0.000
3. **negation** - llama: 0.000
### DIALOGUE Task

- **Average Weighted Delta**: -89.113
- **Most Challenging Modification**: cot-dialogue (2.670)

#### Top 3 Challenging Modifications:
1. **cot-dialogue** - mixtral: 2.670
2. **cot-dialogue** - gpt4o: -2.359
3. **length_bias** - mixtral: -82.769
### NER Task

- **Average Weighted Delta**: -3.280
- **Most Challenging Modification**: temporal_bias (-1.646)

#### Top 3 Challenging Modifications:
1. **temporal_bias** - deepseek-r1: -1.646
2. **derivation** - deepseek-r1: -1.646
3. **sentiment** - deepseek-r1: -1.646
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

### Visualizations
- `coref_model_comparison.png`
- `coref_performance_drops.png`
- `dialogue_model_comparison.png`
- `dialogue_performance_drops.png`
- `ner_model_comparison.png`
- `ner_performance_drops.png`
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

