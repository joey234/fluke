# FLUKE LaTeX Tables Documentation

## Generated Tables

This directory contains LaTeX tables for the FLUKE LLM benchmark analysis, ready for inclusion in academic papers.

### Files Generated

1. **Main Compilation File**
   - `fluke_llm_tables.tex` - All tables in one file

2. **Comparison Tables**
   - `cross_task_comparison.tex` - Performance across all tasks
   - `frontier_models.tex` - GPT-5 vs DeepSeek R1 vs GPT-4o comparison

3. **Task-Specific Tables** (for each task: coref, dialogue, ner, sa)
   - `[task]_main_results.tex` - Main benchmark results
   - `[task]_modifications.tex` - Linguistic modification results

## Usage in LaTeX Document

### Required Packages

Add these to your LaTeX preamble:

```latex
\usepackage{booktabs}  % For professional table formatting
\usepackage{adjustbox}  % For wide tables
```

### Including Tables

#### Option 1: Include All Tables
```latex
\input{latex_tables_output/tables/fluke_llm_tables.tex}
```

#### Option 2: Include Specific Tables
```latex
% Main comparison
\input{latex_tables_output/tables/cross_task_comparison.tex}

% Frontier models
\input{latex_tables_output/tables/frontier_models.tex}

% Task-specific
\input{latex_tables_output/tables/coref_main_results.tex}
```

## Sample LaTeX Document

```latex
\documentclass{article}
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage{graphicx}

\begin{document}

\section{Experimental Results}

We evaluate the performance of frontier LLMs on the FLUKE benchmark suite, 
with particular focus on GPT-5 and DeepSeek R1.

\subsection{Overall Performance}

Table~\ref{tab:cross_task_comparison} presents the performance of all evaluated 
models across the four FLUKE tasks.

\input{latex_tables_output/tables/cross_task_comparison.tex}

\subsection{Frontier Models Analysis}

Table~\ref{tab:frontier_models} compares the performance of frontier models, 
showing GPT-5's consistent improvements over GPT-4o.

\input{latex_tables_output/tables/frontier_models.tex}

\subsection{Task-Specific Results}

\subsubsection{Coreference Resolution}

\input{latex_tables_output/tables/coref_main_results.tex}

The results show that GPT-5 (82.93\%) and DeepSeek R1 (81.48\%) achieve 
comparable performance on coreference resolution, significantly outperforming 
previous models.

\end{document}
```

## Key Results Summary

### Best Models by Task

| Task | Best Model | Score |
|------|------------|-------|
| Coreference | GPT-5 | 82.93% |
| Dialogue | Claude 3.5 | 94.83% |
| NER | Claude 3.5 | 8.37% |
| Sentiment | Llama | 95.53% |

### GPT-5 vs GPT-4o Improvements

| Task | Improvement |
|------|------------|
| Coreference | +9.46% |
| Dialogue | +0.19% |
| NER | +0.06% |
| Sentiment | +1.38% |
| **Average** | **+2.77%** |

### DeepSeek R1 Performance

- Coreference Resolution: **81.48%**
- Competitive with GPT-5 (82.93%)
- More evaluation needed on other tasks

## Table Customization

### Highlighting Best Results

Tables automatically bold the best result in each column using `\textbf{}`.

### Adjusting Table Width

For wide tables, the generator uses `\adjustbox{width=\textwidth}` automatically.

### Column Abbreviations

Modification tables use abbreviations to save space:
- A→P: Active to Passive
- Cap: Capitalization
- Punc: Punctuation
- Neg: Negation
- etc.

## Citation

When using these results, please cite:

```bibtex
@article{fluke2025,
  title={FLUKE: Few-shot Language Understanding and Knowledge Evaluation},
  author={...},
  year={2025}
}
```

## Notes

- All accuracy values are percentages
- Chain-of-Thought (CoT) results included where available
- Modification results based on 100-sample test sets
- Main benchmarks use full dataset evaluations