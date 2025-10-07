# FLUKE O3 Notebook Template and Consistency Guide

## Unified Structure for All O3 Notebooks

This template ensures consistency across all FLUKE o3 reasoning model notebooks.

### 1. **Imports Section (Standardized)**
```python
# Standard imports
from datasets import load_dataset
import dspy
import openai
import os
import pandas as pd
import json
import glob
import time
from dotenv import load_dotenv
from dspy.evaluate import Evaluate

# Import unified FLUKE utilities
from fluke_o3_utils import (
    REASONING_MODELS, REASONING_CONFIGS,
    remove_space, extract_classification_prediction,
    aggregate_results, highlight_drops_and_significance,
    compare_models
)

# Task-specific imports (if needed)
# For NER: also import extract_ner_prediction, calculate_f1_ent, convert_string_to_entities
# For Dialogue: also import append_person
```

### 2. **Configuration Section (Unified)**
```python
# Load environment
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
# Set organization only if provided
_org = os.getenv('OPENAI_ORGANIZATION')
if _org:
    openai.organization = _org

# Model configuration
CONFIG_NAME = 'standard'  # Options: 'standard', 'detailed', 'efficient'
config = REASONING_CONFIGS[CONFIG_NAME]
MODEL_NAME = config['model']
MODEL_ID = REASONING_MODELS[MODEL_NAME]

# Configure DSPy
lm = dspy.LM(MODEL_ID)
dspy.configure(lm=lm)
```

### 3. **Data Loading (Task-Specific)**
```python
# Load dataset (varies by task)
# Sentiment: load_dataset('stanfordnlp/sst2')['validation']
# Dialogue: pd.read_json('../data/train_dev_test_data/dialog/test.json')
# NER: pd.read_json('../data/train_dev_test_data/ner/fewnerd_sample_test.json')
# Coref: pd.read_json('../data/train_dev_test_data/coref/test.json')

# Create examples using unified remove_space function
examples = [
    dspy.Example({...}).with_inputs(...)
    for r in ds
]
```

### 4. **Task Definition (Standardized Pattern)**
```python
class O3Task(dspy.Signature):
    """Task description with reasoning prompt."""
    # Input fields
    text = dspy.InputField()
    # Output field with clear prefix
    label = dspy.OutputField(prefix='Answer:')

class O3TaskModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(O3Task)
    
    def forward(self, **kwargs):
        return self.prog(**kwargs)

# Initialize module
o3_module = O3TaskModule()
```

### 5. **Evaluation Metric (Unified Pattern)**
```python
def eval_metric(true, prediction, trace=None):
    """Standardized evaluation metric."""
    pred = prediction.label
    
    # For classification tasks (sentiment, dialogue, coref)
    parsed_answer = extract_classification_prediction(pred)
    return parsed_answer == str(true.label)
    
    # For NER tasks
    # parsed_answer = extract_ner_prediction(pred)
    # gold_entities = ast.literal_eval(true.label)
    # _, _, f1_score = calculate_f1_ent(gold_entities, parsed_answer)
    # return f1_score
```

### 6. **Original Evaluation (Consistent Parameters)**
```python
# Fixed parameters for o3 evaluation
TEST_SIZE = 100  # Adjust based on budget
test_examples = examples[:TEST_SIZE]

evaluate = Evaluate(
    devset=test_examples,
    metric=eval_metric,
    num_threads=1,  # Always 1 for o3
    display_progress=True,
    display_table=10,
    return_outputs=True,
    return_all_scores=True
)

results = evaluate(o3_module)

# Standardized result saving
items = []
for sample in results[1]:
    items.append({
        'text': sample[0]['text'],
        'label': sample[0]['label'],
        'pred': extract_prediction_function(sample[1]['label']),
        'raw_output': sample[1]['label']
    })

df_result = pd.DataFrame(items)
output_file = f'results/{TASK_DIR}/{MODEL_NAME}-{CONFIG_NAME}-0shot-{TASK_NAME}.csv'
df_result.to_csv(output_file, index=False)
```

### 7. **Modification Evaluation (Unified Function)**
```python
def evaluate_modified_set(data, program, max_samples=30):
    """Standardized modification evaluation."""
    limited_data = data[:max_samples] if len(data) > max_samples else data
    
    # Create examples with both original and modified text
    mod_examples = [
        dspy.Example({...}).with_inputs(...)
        for r in limited_data
    ]
    
    evaluate = Evaluate(
        devset=mod_examples,
        metric=eval_metric,
        num_threads=1,
        display_progress=True,
        display_table=1,
        return_outputs=True,
        return_all_scores=True
    )
    
    return evaluate(program)
```

### 8. **Modification Testing Loop (Standardized)**
```python
# Standard modifications to test
test_modifications = ['typo_bias_100.json', 'capitalization_100.json', 'punctuation_100.json']
json_files = glob.glob(f'../data/modified_data/{TASK_DIR}/*_100.json')
json_files = [f for f in json_files if any(mod in f for mod in test_modifications)]

for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Task-specific preprocessing if needed
    # For dialogue: data = append_person(data)
    
    results_mod = evaluate_modified_set(data, o3_module, max_samples=25)
    
    # Process and save results
    # ... (standardized processing)
    
    time.sleep(5)  # Rate limiting
```

### 9. **Aggregation (Using Unified Function)**
```python
# Use unified aggregation function
result_files = glob.glob(f'results/{TASK_DIR}/{MODEL_NAME}-{CONFIG_NAME}-0shot-*_100.csv')

if result_files:
    results_df = aggregate_results(
        result_files,
        task_name=TASK_FULL_NAME,
        model_name=f'{MODEL_NAME}-{CONFIG_NAME}'
    )
    
    # Save and display
    output_file = f'results/{TASK_DIR}/{MODEL_NAME}-{CONFIG_NAME}-DP.csv'
    results_df.to_csv(output_file, index=False)
    
    styled_df = results_df.round(3).style.apply(highlight_drops_and_significance, axis=1)
    display(styled_df)
```

### 10. **Model Comparison (Using Unified Function)**
```python
# Standardized comparison
comparison_files = {
    'GPT-4o': f'results/{TASK_DIR}/gpt4o-0shot-{TASK_NAME}.csv',
    'Claude-3.5': f'results/{TASK_DIR}/claude-3-5-sonnet-0shot-{TASK_NAME}.csv',
    'Mixtral-8x22B': f'results/{TASK_DIR}/mixtral-8x22b-0shot-{TASK_NAME}.csv',
    f'{MODEL_NAME}-{CONFIG_NAME}': f'results/{TASK_DIR}/{MODEL_NAME}-{CONFIG_NAME}-0shot-{TASK_NAME}.csv'
}

comparison_df = compare_models(comparison_files, task_name=TASK_FULL_NAME)
```

## Key Consistency Rules

### 1. **Always Use Unified Utilities**
- Import from `fluke_o3_utils.py` instead of defining locally
- Use `remove_space()` for all text cleaning
- Use `extract_classification_prediction()` for binary tasks
- Use `extract_ner_prediction()` for NER tasks

### 2. **Consistent Naming Convention**
- Model classes: `O3{Task}` and `O3{Task}Module`
- Variables: `o3_{task}` for module instances
- Files: `{MODEL_NAME}-{CONFIG_NAME}-0shot-{modification}.csv`

### 3. **Fixed Parameters**
- `num_threads=1` for all o3 evaluations
- `max_samples=25-30` for modification testing
- `time.sleep(5)` between modifications
- `TEST_SIZE=100` for original evaluation (adjustable)

### 4. **Standard Directory Structure**
```
results/
├── sa/          # Sentiment analysis
├── coref/       # Coreference resolution  
├── dialogue/    # Dialogue contradiction
└── ner/         # Named entity recognition
```

### 5. **Required Output Files**
- Original: `{MODEL}-{CONFIG}-0shot-{task}.csv`
- Modifications: `{MODEL}-{CONFIG}-0shot-{modification}_100.csv`
- Aggregated: `{MODEL}-{CONFIG}-DP.csv`

### 6. **Error Handling Pattern**
```python
try:
    # Main processing
    results = evaluate(...)
except Exception as e:
    print(f"Error: {e}")
    continue  # Skip this modification
```

### 7. **Progress Reporting**
```python
print(f"Configuration: {CONFIG_NAME}")
print(f"Evaluating {len(test_examples)} examples...")
print(f"Accuracy: {results[0]:.3f}")
print(f"Saved to: {output_file}")
```

## Task-Specific Variations

### Sentiment Analysis
- Dataset: SST-2
- Metric: Accuracy
- Output: Binary (0/1)

### Dialogue Contradiction
- Dataset: Custom dialogue JSON
- Preprocessing: `append_person()`
- Metric: Accuracy
- Output: Binary (0/1)

### Named Entity Recognition
- Dataset: FewNERD
- Metric: F1 Score
- Functions: `calculate_f1_ent()`, `convert_string_to_entities()`
- Output: List of entities

### Coreference Resolution
- Dataset: Custom coref JSON
- Metric: Accuracy
- Output: Binary (0/1)
- Special: Pronoun and candidates handling

## Consolidation Checklist

- [ ] Import all functions from `fluke_o3_utils.py`
- [ ] Remove duplicate function definitions
- [ ] Standardize variable names
- [ ] Use consistent file paths
- [ ] Apply unified aggregation function
- [ ] Use common comparison function
- [ ] Ensure consistent error handling
- [ ] Apply standard progress reporting
- [ ] Verify output file naming
- [ ] Test with small sample size
