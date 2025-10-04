# FLUKE: A Task-Agnostic Framework for Linguistic Capability Testing

**Paper**: [FLUKE: A Linguistically-Driven and Task-Agnostic Framework for Robustness Evaluation](https://arxiv.org/abs/2504.17311)  
**Dataset**: [huggingface.co/datasets/joey234/fluke](https://huggingface.co/datasets/joey234/fluke)  
**Website**: [fluke-nlp.github.io](https://fluke-nlp.github.io)


## Overview

This repository contains the complete source code for the FLUKE framework, a comprehensive evaluation dataset designed to test the robustness of language models across multiple linguistic dimensions. FLUKE applies 17 different types of linguistic modifications to evaluate model performance across four core NLP tasks: **Coreference Resolution**, **Named Entity Recognition (NER)**, **Sentiment Analysis**, and **Dialogue Understanding**.

## Repository Structure

```
├── data/                           # Core datasets and modifications
│   ├── modified_data/             # Generated linguistic modifications
│   │   ├── coref/                 # Coreference resolution modifications (17 types)
│   │   ├── dialogue/              # Dialogue understanding modifications (17 types)
│   │   ├── ner/                   # Named entity recognition modifications (17 types)
│   │   └── sa/                    # Sentiment analysis modifications (17 types)
│   └── train_dev_test_data/       # Original benchmark datasets
│       ├── coref/                 # OntoNotes 5.0 data
│       ├── dialog/                # PersonaChat data
│       ├── ner/                   # CoNLL-2003 data
│       └── sentiment/             # Stanford Sentiment Treebank data
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
│   │   └── llm_sentiment_{model}.ipynb
│   └── analysis/                 # Results analysis and visualization
│       ├── parse_coref_dialog.ipynb
│       ├── parse_ner.ipynb
│       └── parse_sa.ipynb
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

#### LLM Experiments
```bash
# Run LLM evaluation notebooks
jupyter notebook experiments/LLM/
```

### 4. Results Analysis
```bash
# Generate analysis and visualizations
jupyter notebook experiments/analysis/
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

## Key Findings

- **Task-specific vulnerabilities**: Different tasks show different sensitivity patterns
- **Universal negation challenge**: All models struggle with negation across tasks
- **Surface-level dependencies**: Heavy reliance on orthographic cues (especially in NER)
- **LLM vs PLM trade-offs**: LLMs aren't always more robust than PLMs

## Citation

```bibtex
@article{otmakhova2025fluke,
    title={FLUKE: A Linguistically-Driven and Task-Agnostic Framework for Robustness Evaluation}, 
    author={Yulia Otmakhova and Hung Thinh Truong and Rahmad Mahendra and Zenan Zhai and Rongxin Zhu and Daniel Beck and Jey Han Lau},
    year={2025},
    eprint={2504.17311},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2504.17311}
}
```


## Contact

For questions or issues, please contact:
- Yulia Otmakhova: y.otmakhova@unimelb.edu.au
- Hung Thinh Truong: thinh.truong@unimelb.edu.au
