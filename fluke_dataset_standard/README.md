---
license: mit
task_categories:
- text-classification
- token-classification
- zero-shot-classification
language:
- en
tags:
- linguistics
- robustness
- evaluation
- nlp
- language-models
size_categories:
- 1K<n<10K
---

# FLUKE: A Task-Agnostic Framework for Linguistic Capability Testing

**Paper**: [FLUKE: A Linguistically-Driven and Task-Agnostic Framework for Robustness Evaluation](https://arxiv.org/abs/2504.17311)  
**Dataset**: [huggingface.co/datasets/joey234/fluke](https://huggingface.co/datasets/joey234/fluke)

**Authors**: Yulia Otmakhova¹*, Hung Thinh Truong¹*, Rahmad Mahendra², Zenan Zhai³, Rongxin Zhu¹'³, Daniel Beck², Jey Han Lau¹

¹The University of Melbourne, ²RMIT University, ³Oracle

*Equal contribution

## Dataset Description

FLUKE (Framework for Linguistic Capability Testing) is a comprehensive evaluation dataset designed to test the robustness of language models across multiple linguistic dimensions. This dataset contains systematically modified test data across four core NLP tasks: **Coreference Resolution**, **Named Entity Recognition (NER)**, **Sentiment Analysis**, and **Dialogue Understanding**.

### Dataset Summary

FLUKE applies 17 different types of linguistic modifications to evaluate model performance, revealing vulnerabilities that standard benchmarks might miss. Each modification type tests specific aspects of language understanding, from surface-level orthographic changes to deep semantic alterations.

### Supported Tasks

- **Coreference Resolution**: Testing pronoun and entity reference understanding
- **Named Entity Recognition (NER)**: Evaluating entity identification and classification
- **Sentiment Analysis**: Assessing sentiment classification robustness
- **Dialogue Understanding**: Testing conversational coherence and context understanding

### Modification Types

The dataset includes 17 types of linguistic modifications across different linguistic levels:

#### Orthography
- **Capitalization**: Testing case sensitivity (e.g., "Battlefield 3" → "battlefield 3")
- **Punctuation**: Adding/changing punctuation marks
- **Spelling (Typo)**: Character-level modifications (addition, omission, swapping)

#### Morphology
- **Derivation**: Using morphologically related forms (e.g., "killed" → "assassinated")
- **Compound Words**: Testing compound vs. separate word forms (e.g., "new" → "brand-new")

#### Syntax
- **Active to Passive**: Voice transformations (e.g., "Billy beat Tommy" → "Tommy was beaten by Billy")
- **Grammatical Role**: Swapping subject/object roles
- **Coordinating Conjunction**: Adding conjunctions for complexity

#### Semantics
- **Concept Replacement**: Synonym/hypernym substitutions
- **Negation**: Various negation types (verbal, lexical, double)

#### Discourse
- **Discourse Markers**: Adding/changing discourse connectives
- **Sentiment**: Emotional tone modifications

#### Language Varieties
- **Dialectal**: Dialect variations (including Singlish)
- **Casual**: Formal to informal style changes

#### Biases
- **Temporal Bias**: Old-fashioned vs. modern expressions
- **Geographical Bias**: Cultural name and entity variations
- **Length Bias**: Sentence length modifications

### Dataset Structure

```
├── train.parquet           # Combined dataset (all tasks)
├── coref.parquet          # Coreference Resolution
├── ner.parquet            # Named Entity Recognition  
├── sa.parquet             # Sentiment Analysis
└── dialogue.parquet       # Dialogue Understanding
```

### Data Fields

Each example contains the following fields:

- `original` (string): Original text before modification
- `modified` (string): Text after applying linguistic modification
- `label` (string): Ground truth label (task-specific)
- `modification_type` (string): Type of linguistic modification applied
- `task` (string): Task name (coref/ner/sa/dialogue)

### Data Splits

- **Train**: All examples are provided in the train split
- **Total examples**: 6,386 across all tasks and modifications
  - Coreference: 1,551 examples
  - NER: 1,549 examples  
  - Sentiment Analysis: 1,644 examples
  - Dialogue Understanding: 1,642 examples

### Dataset Creation

#### Source Data
The original data comes from standard NLP benchmarks:
- **Coreference**: OntoNotes 5.0
- **NER**: CoNLL-2003
- **Sentiment Analysis**: Stanford Sentiment Treebank
- **Dialogue**: PersonaChat

#### Annotation Process
Modifications were generated using a combination of:
1. **LLM-assisted generation**: GPT-4 for complex linguistic transformations
2. **Rule-based methods**: For systematic orthographic and morphological changes
3. **Human validation**: Quality control and verification

### Usage Example

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("joey234/fluke")

# Access specific task data
coref_data = load_dataset("joey234/fluke", data_files="coref.parquet")
ner_data = load_dataset("joey234/fluke", data_files="ner.parquet")
sa_data = load_dataset("joey234/fluke", data_files="sa.parquet")
dialogue_data = load_dataset("joey234/fluke", data_files="dialogue.parquet")

# Example: Filter by modification type
train_data = dataset["train"]
negation_examples = train_data.filter(lambda x: x["modification_type"] == "negation")

# Example usage
for example in negation_examples[:3]:
    print(f"Task: {example['task']}")
    print(f"Original: {example['original']}")
    print(f"Modified: {example['modified']}")
    print(f"Label: {example['label']}")
    print("---")
```

### Dataset Statistics

| Task | Examples | Modification Types | 
|------|----------|-------------------|
| Coreference | 1,551 | 17 |
| NER | 1,549 | 17 |  
| Sentiment Analysis | 1,644 | 17 |
| Dialogue Understanding | 1,642 | 17 |
| **Total** | **6,386** | **17** |

### Evaluation

FLUKE reveals several key findings:
- **Task-specific vulnerabilities**: Different tasks show different sensitivity patterns
- **Universal negation challenge**: All models struggle with negation across tasks
- **Surface-level dependencies**: Heavy reliance on orthographic cues (especially in NER)
- **LLM vs PLM trade-offs**: LLMs aren't always more robust than PLMs

### Citation

```bibtex
@misc{otmakhova2025fluke,
    title={FLUKE: A Linguistically-Driven and Task-Agnostic Framework for Robustness Evaluation}, 
    author={Yulia Otmakhova and Hung Thinh Truong and Rahmad Mahendra and Zenan Zhai and Rongxin Zhu and Daniel Beck and Jey Han Lau},
    year={2025},
    eprint={2504.17311},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2504.17311}
}
```

### License

This dataset is released under the MIT License.

### Dataset Card Authors

Yulia Otmakhova, Hung Thinh Truong, Rahmad Mahendra, Zenan Zhai, Rongxin Zhu, Daniel Beck, Jey Han Lau 