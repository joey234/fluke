# FLUKE Experiments with OpenRouter API

This directory contains standalone Python scripts for running all FLUKE experiments using OpenRouter API directly, without DSPy dependency. Includes sentiment analysis, coreference resolution, dialogue contradiction detection, and named entity recognition.

## Available Scripts

### Core Scripts
- **`run_sentiment_openrouter.py`** - Sentiment analysis (binary classification)
- **`run_coref_openrouter.py`** - Coreference resolution (pronoun resolution)
- **`run_dialogue_openrouter.py`** - Dialogue contradiction detection
- **`run_ner_openrouter.py`** - Named entity recognition

### Utility Scripts
- **`example_usage.py`** - Quick test examples for sentiment analysis
- **`fluke_reasoning_utils.py`** - Shared utility functions

## Features

- Direct OpenRouter API integration (no DSPy required)
- Support for DeepSeek R1 and other OpenRouter models
- **Reasoning tokens support** - captures model's step-by-step thinking process
- **Intelligent prediction caching** - automatically caches predictions to avoid duplicate API calls
- Automatic evaluation on FLUKE linguistic modifications
- Results aggregation and model comparison
- Rate limiting and error handling
- Consistent evaluation pipeline across all tasks

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements_openrouter.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file in this directory with your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

3. **Get an OpenRouter API key:**
   - Sign up at [OpenRouter](https://openrouter.ai/)
   - Get your API key from the dashboard
   - Add credits to your account for API usage

## Usage

### Quick Test (Sentiment Analysis)
```bash
python example_usage.py
```

### Full Evaluation for All Tasks
```bash
# Sentiment Analysis
python run_sentiment_openrouter.py

# Coreference Resolution
python run_coref_openrouter.py

# Dialogue Contradiction Detection
python run_dialogue_openrouter.py

# Named Entity Recognition
python run_ner_openrouter.py
```

### Script Configuration

All scripts can be configured by modifying these variables at the top of each file:

```python
CONFIG_NAME = 'deepseek'  # Options: 'deepseek', 'deepseek-lite'
# No TEST_SIZE needed - uses intelligent caching for efficient evaluation
```

**Note**: Unlike traditional approaches that require a separate base run, these scripts automatically manage predictions through intelligent caching. Simply run any script and it will efficiently handle both original and modified predictions without duplication.

Available models:
- `deepseek`: DeepSeek R1 (full model)
- `deepseek-lite`: DeepSeek R1 Lite (faster, cheaper)

### Task-Specific Settings

Each task evaluates different modification sets with intelligent caching:
- **Sentiment**: 6 modifications × 50 samples each (automatic caching)
- **Coreference**: 8 modifications × 50 samples each (automatic caching)  
- **Dialogue**: 8 modifications × 50 samples each (automatic caching)
- **NER**: 8 modifications × 50 samples each (automatic caching)

## Output

Each script generates several output files in the corresponding results directory:

### Output Files for Each Task
1. **Base results:** `{model}-{config}-0shot-{task}.csv`
2. **Modification results:** `{model}-{config}-0shot-{modification}_100.csv`
3. **Aggregated results:** `{model}-{config}-DP.csv`

**Note:** All output files include both the model's final answer and its reasoning process (when available from reasoning-capable models).

### Example Output Structure

```
../results/
├── sa/                                              # Sentiment Analysis
│   ├── deepseek-r1-deepseek-0shot-sst2.csv         # Base results + reasoning
│   ├── deepseek-r1-deepseek-0shot-typo_bias_100.csv
│   ├── deepseek-r1-deepseek-0shot-punctuation_100.csv
│   └── deepseek-r1-deepseek-DP.csv                  # Aggregated results
├── coref/                                           # Coreference Resolution
│   ├── deepseek-r1-deepseek-0shot-coref.csv
│   ├── deepseek-r1-deepseek-0shot-negation_100.csv
│   └── deepseek-r1-deepseek-DP.csv
├── dialogue/                                        # Dialogue Contradiction
│   ├── deepseek-r1-deepseek-0shot-dialogue.csv
│   ├── deepseek-r1-deepseek-0shot-casual_100.csv
│   └── deepseek-r1-deepseek-DP.csv
└── ner/                                            # Named Entity Recognition
    ├── deepseek-r1-deepseek-0shot-ner.csv
    ├── deepseek-r1-deepseek-0shot-dialectal_100.csv
    └── deepseek-r1-deepseek-DP.csv
```

## Supported Modifications

All scripts test robustness against these linguistic modifications:

### Common Modifications (All Tasks)
- `typo_bias_100`: Typos and spelling errors
- `capitalization_100`: Capitalization changes
- `punctuation_100`: Punctuation modifications
- `negation_100`: Negation changes
- `sentiment_100`: Sentiment word replacements
- `active_to_passive_100`: Voice changes

### Task-Specific Modifications
- **Coreference & Dialogue**: `grammatical_role_100`, `coordinating_conjunction_100`
- **Dialogue & NER**: `casual_100`, `dialectal_100`
- **All tasks**: Additional modifications like `compound_word_100`, `derivation_100`, etc.

## Reasoning Tokens

The scripts use OpenRouter's reasoning tokens feature to capture the model's step-by-step thinking process:

- **Automatic detection**: Reasoning tokens are enabled for compatible models (DeepSeek R1, o1, etc.)
- **High effort mode**: Uses 80% of available tokens for reasoning (2000 max reasoning tokens)
- **Saved reasoning**: All reasoning traces are saved in the `reasoning` column of output CSV files
- **Transparent process**: See exactly how the model arrives at its decisions

This provides valuable insights into model behavior and failure modes during robustness testing.

## Intelligent Caching Pipeline

The scripts use an efficient caching approach:

- **Dynamic prediction cache**: Automatically stores predictions in memory during evaluation
- **Automatic reuse**: If the same text appears in multiple modifications, prediction is retrieved from cache
- **Zero duplicate calls**: Never makes the same API call twice during a session
- **Complete data capture**: Both original and modified predictions with reasoning traces stored
- **Cross-modification efficiency**: Cache persists across all modifications in a single run

This approach significantly reduces API costs while ensuring perfect consistency between original and modified comparisons.

## API Costs

DeepSeek R1 via OpenRouter costs approximately:
- Input: ~$0.32 per million tokens
- Output: ~$1.28 per million tokens

### Estimated Costs per Task
- **Sentiment**: ~$1-3 (6×50 unique predictions with intelligent caching)
- **Coreference**: ~$2-4 (8×50 unique predictions with intelligent caching)  
- **Dialogue**: ~$3-6 (8×50 unique predictions with intelligent caching, longer texts)
- **NER**: ~$2-4 (8×50 unique predictions with intelligent caching)

**Cost Optimization**: Intelligent caching eliminates duplicate API calls entirely. Actual costs depend on text overlap between original and modified versions.

### Runtime Estimates (with rate limiting)
- **Per task**: 15-30 minutes
- **All tasks**: 1-2 hours total
- Adjust `TEST_SIZE` and modification samples to control cost/time

## Comparison with DSPy Version

### Advantages of Direct API:
- No DSPy dependency
- Simpler code structure
- Direct control over API calls
- Easier to modify and extend
- Better error handling

### DSPy Version Advantages:
- Built-in prompt optimization
- Automatic signature validation
- Integration with DSPy ecosystem
- Advanced evaluation metrics

## Troubleshooting

### Common Issues:

1. **API Key Error:**
   ```
   Error: OPENROUTER_API_KEY not found
   ```
   Solution: Check your `.env` file and API key

2. **Rate Limiting:**
   ```
   API Error: 429 Too Many Requests
   ```
   Solution: Increase sleep times in the script

3. **Insufficient Credits:**
   ```
   API Error: 402 Payment Required
   ```
   Solution: Add credits to your OpenRouter account

4. **Model Not Available:**
   ```
   API Error: 404 Model not found
   ```
   Solution: Check model availability on OpenRouter

## Extending the Script

### Adding New Models:
1. Add model to `REASONING_MODELS` in `fluke_reasoning_utils.py`
2. Add configuration to `REASONING_CONFIGS`  
3. Update `CONFIG_NAME` in any script

### Adding New Modifications:
1. Add JSON file to `test_modifications` list in the relevant script
2. Ensure the modification file exists in `../../../data/modified_data/{task}/`

### Custom Prompts:
Modify the `prompt_template` in each task's class:
- `SentimentAnalyzer` for sentiment analysis
- `CoreferenceResolver` for coreference resolution
- `DialogueContradictionDetector` for dialogue
- `NamedEntityRecognizer` for NER

### Running Specific Modifications:
Edit the `test_modifications` list in each script to run only specific modifications.

## Related Files

### Original Notebooks (DSPy-based)
- `llm_sentiment_deepseek.ipynb`: Sentiment analysis with DSPy
- `llm_coref_deepseek.ipynb`: Coreference resolution with DSPy
- `llm_dialogue_deepseek.ipynb`: Dialogue contradiction with DSPy
- `llm_ner_deepseek.ipynb`: Named entity recognition with DSPy

### OpenRouter Scripts (This Implementation)
- `run_sentiment_openrouter.py`: Sentiment analysis script
- `run_coref_openrouter.py`: Coreference resolution script  
- `run_dialogue_openrouter.py`: Dialogue contradiction script
- `run_ner_openrouter.py`: Named entity recognition script

### Utilities
- `fluke_reasoning_utils.py`: Shared utility functions
- `example_usage.py`: Simple usage examples
- `requirements_openrouter.txt`: Python dependencies

## Evaluation Metrics

- **Sentiment & Dialogue & Coreference**: Accuracy (percentage correct)
- **Named Entity Recognition**: F1 Score (precision/recall balance)
- **All tasks**: Robustness measured as performance drop under modifications