# FLUKE: A Task-Agnostic Framework for Linguistic Capability Testing

## Overview

This repository contains the source code for the FLUKE framework, a task‑agnostic robustness evaluation suite spanning multiple linguistic dimensions. FLUKE applies 17 types of linguistically motivated modifications to evaluate model behavior across the following tasks: **Coreference Resolution**, **Named Entity Recognition (NER)**, **Sentiment Analysis**, **Dialogue Understanding**, and **Grade School Math (GSM)**. In addition, we include an **Instruction Following (IFEval)** evaluation (LLM‑only) with constraint scoring.

## Repository Structure

```
├── data/                           # Core datasets and modifications
│   ├── modified_data/             # Generated linguistic modifications
│   │   ├── coref/                 # Coreference resolution modifications (17 types)
│   │   ├── dialogue/              # Dialogue understanding modifications (17 types)
│   │   ├── gsm/                   # GSM variations (incl. negation subtypes, styles)
│   │   ├── ifeval/                # IFEval prompts + constraint specs (JSONL)
│   │   ├── ner/                   # Named entity recognition modifications (17 types)
│   │   └── sa/                    # Sentiment analysis modifications (17 types)
│   └── train_dev_test_data/       # Original benchmark datasets
│       ├── coref/                 # OntoNotes 5.0 data
│       ├── dialog/                # PersonaChat data
│       ├── ner/                   # CoNLL-2003 data
│       ├── sentiment/             # Stanford Sentiment Treebank data
│       └── (task-specific)        # Additional sources as applicable (e.g., GSM)
│
├── data_generation/               # Scripts for generating linguistic modifications
│   ├── coref_prompt.ipynb         # Coreference modification generation
│   ├── dialogue_prompt.ipynb     # Dialogue modification generation
│   ├── ner_prompt.ipynb          # NER modification generation
│   └── sentiment_prompt.ipynb    # Sentiment modification generation
│
├── experiments/                   # Model evaluation code and results
│   ├── PLM/                      # Pre-trained Language Model experiments
│   │   ├── coreference_resolution/
│   │   ├── dialogue_contradiction_detection/
│   │   ├── ner/
│   │   └── sentiment_analysis/
│   ├── LLM/                      # Large Language Model experiments
│   │   ├── llm_coref_{model}.ipynb
│   │   ├── llm_dialogue_{model}.ipynb
│   │   ├── llm_ner_{model}.ipynb
│   │   ├── llm_sentiment_{model}.ipynb
│   │   ├── llm_gsm_{model}.ipynb
│   │   └── scripts/               # LLM scripts (incl. IFEval and analysis helpers)
│   └── analysis/                 # Results analysis and visualization
│       ├── parse_coref_dialog.ipynb
│       ├── parse_ner.ipynb
│       ├── parse_sa.ipynb
│       ├── gsm_analysis.py
│       └── ifeval_analysis.py
│
├── fluke_dataset/                # HuggingFace dataset preparation (legacy)
└── fluke_dataset_standard/       # Standardized dataset format
    ├── *.parquet                 # Final dataset files
    ├── hf_repo/                  # HuggingFace repository clone
    └── docs/                     # Dataset documentation
```

## Quick Start

### 1. Environment Setup
```bash
# Install required dependencies
pip install -r requirements.txt

# Set up environment variables (if using LLM experiments)
cp .env.example .env
# Edit .env with your API keys
```

### 2. Data Generation
To generate linguistic modifications using the FLUKE framework:

```bash
# Run data generation notebooks
jupyter notebook data_generation/
```

### 3. Model Evaluation

#### PLM Experiments
```bash
# Navigate to specific task directory
cd experiments/PLM/{task}/

# Run evaluation script
python eval_{model}.py
```

#### LLM Experiments (run scripts)
```bash
# Coreference
python fluke-source-code/experiments/LLM/scripts/run_coref_gpt5.py
# Dialogue
python fluke-source-code/experiments/LLM/scripts/run_dialogue_gpt5.py
# NER
python fluke-source-code/experiments/LLM/scripts/run_ner_gpt5.py
# Sentiment
python fluke-source-code/experiments/LLM/scripts/run_sentiment_gpt5.py

# Optional: with context-aware variants
python fluke-source-code/experiments/LLM/scripts/run_coref_gpt5_with_context.py
python fluke-source-code/experiments/LLM/scripts/run_dialogue_gpt5_with_context.py
python fluke-source-code/experiments/LLM/scripts/run_gsm_gpt5.py
python fluke-source-code/experiments/LLM/scripts/run_ifeval_gpt5.py

# Optional: OpenRouter variants
python fluke-source-code/experiments/LLM/scripts/run_coref_openrouter.py
python fluke-source-code/experiments/LLM/scripts/run_dialogue_openrouter.py
python fluke-source-code/experiments/LLM/scripts/run_ner_openrouter.py
python fluke-source-code/experiments/LLM/scripts/run_sentiment_openrouter.py
python fluke-source-code/experiments/LLM/scripts/run_gsm_openrouter.py
python fluke-source-code/experiments/LLM/scripts/run_ifeval_openrouter.py
```

GSM and IFEval
- GSM (Grade School Math) data and results are supported in LLM experiments; analysis lives in `experiments/analysis/gsm_analysis.py`.
- IFEval (instruction following) evaluation is LLM‑only; see the section below for end‑to‑end scoring and analysis.

### 4. Results Analysis (Python scripts)
Run the task-specific analysis scripts to aggregate metrics, generate tables and plots:
```bash
# From repo root
python fluke-source-code/experiments/analysis/coref_analysis.py
python fluke-source-code/experiments/analysis/ner_analysis.py
python fluke-source-code/experiments/analysis/sa_analysis.py
python fluke-source-code/experiments/analysis/dialogue_analysis.py
# Optional tasks
python fluke-source-code/experiments/analysis/gsm_analysis.py
python fluke-source-code/experiments/analysis/ifeval_analysis.py
```

### IFEval (Instruction Following) — LLM Only
This repo includes a lightweight evaluator for IFEval-style constraints to compare original vs modified prompts.

1) Prepare data and generations
- Dataset JSONL: `fluke-source-code/data/modified_data/ifeval/length_bias_100.jsonl` (contains both `text` and `modified` for each `key` plus constraint ids and kwargs)
- Model outputs CSV (per model):
  - `fluke-source-code/experiments/LLM/results/ifeval/<MODEL>-0shot-length_bias_100.csv`
  - Columns include: `key, original_text, text (modified), original_raw_output, original_reasoning, raw_output, reasoning`

2) Score constraints per side (run from `fluke-source-code/experiments/LLM/scripts`)
```bash
python ifeval_evaluate.py \
  --dataset ../../../data/modified_data/ifeval/length_bias_100.jsonl \
  --outputs ../results/ifeval/<MODEL>-0shot-length_bias_100.csv \
  --side original \
  --out_csv ../results/ifeval_scores/length_bias/<MODEL>_original.csv

python ifeval_evaluate.py \
  --dataset ../../../data/modified_data/ifeval/length_bias_100.jsonl \
  --outputs ../results/ifeval/<MODEL>-0shot-length_bias_100.csv \
  --side modified \
  --out_csv ../results/ifeval_scores/length_bias/<MODEL>_modified.csv
```

3) Analyze original vs modified (run from `fluke-source-code/experiments/LLM/scripts`)
```bash
python ifeval_analysis.py \
  --orig_csv ../results/ifeval_scores/length_bias/<MODEL>_original.csv \
  --mod_csv  ../results/ifeval_scores/length_bias/<MODEL>_modified.csv \
  --model <MODEL> \
  --mod length_bias \
  --out_csv ../results/ifeval_aggregates/length_bias/<MODEL>_comparison.csv
```

Outputs include: A/B compliance means, weighted delta (Δ), absolute_change (CSV only), unrobustness (U) on compliance and strict success, and paired significance (Wilcoxon + McNemar exact).

To generate model outputs via OpenRouter and analyze in one go (run from project root or anywhere):
```bash
python fluke-source-code/experiments/LLM/scripts/run_ifeval_openrouter.py \
  --mod length_bias \
  --model deepseek-r1 \
  --analyze
```

## Modification Types

FLUKE implements 17 types of linguistic modifications across different linguistic levels:

### Orthography
- **Capitalization**: Case sensitivity testing
- **Punctuation**: Punctuation mark variations  
- **Spelling (Typo)**: Character-level modifications

### Morphology
- **Derivation**: Morphologically related forms
- **Compound Words**: Compound vs. separate forms

### Syntax
- **Active to Passive**: Voice transformations
- **Grammatical Role**: Subject/object swapping
- **Coordinating Conjunction**: Adding conjunctions

### Semantics
- **Concept Replacement**: Synonym/hypernym substitutions
- **Negation**: Various negation types

### Discourse
- **Discourse Markers**: Discourse connective modifications
- **Sentiment**: Emotional tone changes

### Language Varieties
- **Dialectal**: Dialect variations (including Singlish)
- **Casual**: Formal to informal style changes

### Biases
- **Temporal Bias**: Old-fashioned vs. modern expressions
- **Geographical Bias**: Cultural variations
- **Length Bias**: Sentence length modifications

## Notes on Anonymity (for review)

This repository has been scrubbed to avoid author, affiliation, or contact details during anonymous review. Any identifying metadata, personal emails, or institutional references have been removed. A citation entry and contact information will be added after the review period.
