# FLUKE Dataset Usage Guide

This guide provides detailed instructions on how to use the FLUKE dataset for evaluating language model robustness.

## Installation

```bash
pip install datasets transformers torch
```

## Loading the Dataset

### Load All Tasks

```python
from datasets import load_dataset

# Load all tasks combined
dataset = load_dataset("your-username/fluke", "all")

# Access individual tasks
coref_data = dataset["coref"]
ner_data = dataset["ner"] 
sa_data = dataset["sa"]
dialogue_data = dataset["dialogue"]
```

### Load Specific Task

```python
# Load only sentiment analysis
sa_dataset = load_dataset("your-username/fluke", "sa")

# Load only NER
ner_dataset = load_dataset("your-username/fluke", "ner")
```

## Data Structure

Each example contains:
- `original`: Original text before modification
- `modified`: Text after applying linguistic modification
- `label`: Ground truth label (task-specific)
- `modification_type`: Type of linguistic modification applied
- `task`: Task name (coref/ner/sa/dialogue)

## Example Usage

### Basic Evaluation

```python
import json
from datasets import load_dataset

# Load sentiment analysis data
sa_data = load_dataset("your-username/fluke", "sa")["train"]

# Filter by modification type
negation_examples = sa_data.filter(lambda x: x["modification_type"] == "negation")

print(f"Found {len(negation_examples)} negation examples")

# Example analysis
for example in negation_examples[:3]:
    print(f"Original: {example['original']}")
    print(f"Modified: {example['modified']}")
    print(f"Label: {example['label']}")
    print("---")
```

### Evaluation with Model

```python
from transformers import pipeline
from datasets import load_dataset

# Load a sentiment analysis model
classifier = pipeline("sentiment-analysis", 
                     model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Load FLUKE sentiment data
sa_data = load_dataset("your-username/fluke", "sa")["train"]

# Evaluate on original vs modified
results = []
for example in sa_data:
    original_pred = classifier(example["original"])[0]
    modified_pred = classifier(example["modified"])[0]
    
    results.append({
        "modification_type": example["modification_type"],
        "original_pred": original_pred["label"],
        "modified_pred": modified_pred["label"],
        "true_label": example["label"],
        "consistency": original_pred["label"] == modified_pred["label"]
    })

# Analysis
import pandas as pd
df = pd.DataFrame(results)
consistency_by_modification = df.groupby("modification_type")["consistency"].mean()
print("Consistency by modification type:")
print(consistency_by_modification.sort_values())
```

### Task-Specific Evaluation

#### Named Entity Recognition

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load NER model
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Load FLUKE NER data
ner_data = load_dataset("your-username/fluke", "ner")["train"]

# Evaluate entity consistency
for example in ner_data[:5]:
    original_entities = ner_pipeline(example["original"])
    modified_entities = ner_pipeline(example["modified"])
    
    print(f"Modification: {example['modification_type']}")
    print(f"Original: {example['original']}")
    print(f"Original entities: {original_entities}")
    print(f"Modified: {example['modified']}")
    print(f"Modified entities: {modified_entities}")
    print("---")
```

#### Coreference Resolution

```python
# For coreference, you might use spaCy or AllenNLP models
import spacy

nlp = spacy.load("en_core_web_sm")

coref_data = load_dataset("your-username/fluke", "coref")["train"]

for example in coref_data[:3]:
    original_doc = nlp(example["original"])
    modified_doc = nlp(example["modified"])
    
    print(f"Modification: {example['modification_type']}")
    print(f"Original: {example['original']}")
    print(f"Modified: {example['modified']}")
    # Add your coreference analysis here
    print("---")
```

## Modification Types Analysis

### Get Modification Statistics

```python
from collections import Counter
from datasets import load_dataset

# Load all data
dataset = load_dataset("your-username/fluke", "all")

# Count modifications per task
for task_name, task_data in dataset.items():
    modification_counts = Counter(task_data["modification_type"])
    print(f"\n{task_name.upper()} Task:")
    for mod_type, count in modification_counts.most_common():
        print(f"  {mod_type}: {count} examples")
```

### Filter by Specific Modifications

```python
# Focus on negation across all tasks
all_negation = []
for task_name, task_data in dataset.items():
    negation_examples = task_data.filter(lambda x: x["modification_type"] == "negation")
    for example in negation_examples:
        example["task"] = task_name
        all_negation.append(example)

print(f"Total negation examples across all tasks: {len(all_negation)}")
```

## Advanced Analysis

### Robustness Scoring

```python
def calculate_robustness_score(model_predictions, modification_type):
    """Calculate robustness score for a specific modification type."""
    consistent_predictions = sum(1 for pred in model_predictions if pred["consistent"])
    total_predictions = len(model_predictions)
    return consistent_predictions / total_predictions if total_predictions > 0 else 0

# Example usage with your model evaluation results
robustness_scores = {}
for mod_type in df["modification_type"].unique():
    mod_data = df[df["modification_type"] == mod_type]
    score = mod_data["consistency"].mean()
    robustness_scores[mod_type] = score

# Sort by robustness (lower score = more vulnerable)
sorted_robustness = sorted(robustness_scores.items(), key=lambda x: x[1])
print("Model vulnerabilities (least robust first):")
for mod_type, score in sorted_robustness:
    print(f"{mod_type}: {score:.3f}")
```

## Citation

If you use this dataset, please cite:

```bibtex
@article{fluke2024,
  title={FLUKE: A Task-Agnostic Framework for Linguistic Capability Testing},
  author={Your Name and Co-author},
  journal={Conference/Journal Name},
  year={2024}
}
``` 