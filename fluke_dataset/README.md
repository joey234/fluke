---
license: mit
task_categories:
- text-classification
- token-classification
- question-answering
- conversational
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
data/
├── coref/                    # Coreference Resolution
├── ner/                      # Named Entity Recognition  
├── sa/                       # Sentiment Analysis
└── dialogue/                 # Dialogue Understanding
```

Each task directory contains JSON files named `{modification_type}_100.json`, where each file contains 100 modified examples for that specific modification type.

### Data Fields

Each JSON file contains a list of examples with the following structure:

```json
{
  "original": "Original text",
  "modified": "Modified text", 
  "label": "Ground truth label",
  "modification_type": "Type of modification applied",
  "task": "Task name (coref/ner/sa/dialogue)"
}
```

### Data Splits

- Each modification type contains 100 examples
- Total examples per task: ~1,700 (17 modification types × 100 examples)
- Total dataset size: ~6,800 examples across 4 tasks

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
import json
import os

# Load sentiment analysis data with negation modifications
with open("data/sa/negation_100.json", "r") as f:
    negation_data = json.load(f)

# Example usage
for example in negation_data[:5]:
    print(f"Original: {example['original']}")
    print(f"Modified: {example['modified']}")
    print(f"Label: {example['label']}")
    print("---")
```

### Evaluation

FLUKE reveals several key findings:
- **Task-specific vulnerabilities**: Different tasks show different sensitivity patterns
- **Universal negation challenge**: All models struggle with negation across tasks
- **Surface-level dependencies**: Heavy reliance on orthographic cues (especially in NER)
- **LLM vs PLM trade-offs**: LLMs aren't always more robust than PLMs

### Citation

```bibtex
@article{fluke2024,
  title={FLUKE: A Task-Agnostic Framework for Linguistic Capability Testing},
  author={Your Name and Co-author},
  journal={Conference/Journal Name},
  year={2024}
}
```

### License

This dataset is released under the MIT License.

### Contributions

We welcome contributions to expand FLUKE with additional tasks, modification types, or languages. Please see our [contribution guidelines](docs/CONTRIBUTING.md) for more information. 