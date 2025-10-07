# FLUKE Experiments with OpenAI GPT-5

This directory contains standalone Python scripts for running all FLUKE experiments using OpenAI GPT-5 API directly, without DSPy dependency. Includes sentiment analysis, coreference resolution, dialogue contradiction detection, and named entity recognition.

## Available Scripts

### Core Scripts
- **`run_sentiment_gpt5.py`** - Sentiment analysis (binary classification)
- **`run_coref_gpt5.py`** - Coreference resolution (pronoun resolution)
- **`run_dialogue_gpt5.py`** - Dialogue contradiction detection
- **`run_ner_gpt5.py`** - Named entity recognition

### Utility Scripts
- **`fluke_gpt5_utils.py`** - Shared utility functions and GPT-5 client
- **`requirements_gpt5.txt`** - Python dependencies

## Features

- Direct OpenAI GPT-5 API integration (no DSPy required)
- Support for all GPT-5 model variants (gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-chat)
- **Direct prompting** - clean, straightforward prompts without chain-of-thought
- **GPT-5 reasoning support** - captures model's internal reasoning when available  
- **Intelligent prediction caching** - automatically caches predictions to avoid duplicate API calls
- Automatic evaluation on FLUKE linguistic modifications
- Rate limiting and error handling
- Consistent evaluation pipeline across all tasks

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements_gpt5.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file in this directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Get an OpenAI API key:**
   - Sign up at [OpenAI Platform](https://platform.openai.com/)
   - Get your API key from the API keys section
   - Add credits to your account for API usage

## Usage

### Full Evaluation for All Tasks
```bash
# Sentiment Analysis
python run_sentiment_gpt5.py

# Coreference Resolution
python run_coref_gpt5.py

# Dialogue Contradiction Detection
python run_dialogue_gpt5.py

# Named Entity Recognition
python run_ner_gpt5.py
```

### Script Configuration

All scripts can be configured by modifying the `CONFIG_NAME` variable at the top of each file:

```python
CONFIG_NAME = 'standard'  # Options: 'standard', 'fast', 'fastest', 'chat'
```

**Note**: These scripts use intelligent caching to automatically manage predictions without duplication. Simply run any script and it will efficiently handle both original and modified predictions.

Available GPT-5 models:
- `standard`: GPT-5 full reasoning model (best performance)
- `fast`: GPT-5 Mini (faster, cheaper)
- `fastest`: GPT-5 Nano (fastest, cheapest)
- `chat`: GPT-5 Chat (non-reasoning model)

### Task-Specific Settings

Each task evaluates different modification sets with intelligent caching:
- **Sentiment**: 6 modifications × all samples (~150-200 each) (automatic caching)
- **Coreference**: 8 modifications × all samples (~100 each) (automatic caching)
- **Dialogue**: 8 modifications × all samples (~100 each) (automatic caching)
- **NER**: 8 modifications × all samples (~100 each) (automatic caching)

## Output

Each script generates CSV files in the corresponding results directory:

### Output Files for Each Task
- **Modification results:** `{model}-{config}-0shot-{modification}_100.csv`

**Note:** All output files include both the model's final answer and its reasoning process (when available from GPT-5 reasoning models).

### Example Output Structure

```
../results/
├── sa/                                              # Sentiment Analysis
│   ├── gpt-5-standard-0shot-typo_bias_100.csv
│   ├── gpt-5-standard-0shot-punctuation_100.csv
│   └── gpt-5-standard-0shot-sentiment_100.csv
├── coref/                                           # Coreference Resolution
│   ├── gpt-5-standard-0shot-negation_100.csv
│   └── gpt-5-standard-0shot-grammatical_role_100.csv
├── dialogue/                                        # Dialogue Contradiction
│   ├── gpt-5-standard-0shot-casual_100.csv
│   └── gpt-5-standard-0shot-dialectal_100.csv
└── ner/                                            # Named Entity Recognition
    ├── gpt-5-standard-0shot-capitalization_100.csv
    └── gpt-5-standard-0shot-compound_word_100.csv
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

## GPT-5 Direct Prompting

The scripts use clean, direct prompts for maximum clarity and efficiency:

- **Simple prompts**: No complex chain-of-thought instructions
- **Clear instructions**: Direct task descriptions with minimal complexity
- **Automatic reasoning**: GPT-5's internal reasoning is captured when available
- **Consistent format**: All tasks use similar prompt structures for reliability

This approach provides clear results while allowing GPT-5 to leverage its built-in reasoning capabilities naturally.

## Intelligent Caching Pipeline

The scripts use an efficient caching approach:

- **Dynamic prediction cache**: Automatically stores predictions in memory during evaluation
- **Automatic reuse**: If the same text appears in multiple modifications, prediction is retrieved from cache
- **Zero duplicate calls**: Never makes the same API call twice during a session
- **Complete data capture**: Both original and modified predictions with reasoning traces stored
- **Cross-modification efficiency**: Cache persists across all modifications in a single run

This approach significantly reduces API costs while ensuring perfect consistency between original and modified comparisons.

## API Costs

OpenAI GPT-5 via OpenAI API costs approximately:
- GPT-5 Standard: ~$15 per million input tokens, ~$60 per million output tokens
- GPT-5 Mini: ~$3 per million input tokens, ~$12 per million output tokens
- GPT-5 Nano: ~$0.5 per million input tokens, ~$2 per million output tokens
- GPT-5 Chat: ~$5 per million input tokens, ~$15 per million output tokens

### Estimated Costs per Task
- **Sentiment**: ~$10-30 (6×all samples unique predictions with intelligent caching)
- **Coreference**: ~$8-20 (8×all samples unique predictions with intelligent caching)
- **Dialogue**: ~$10-25 (8×all samples unique predictions with intelligent caching, longer texts)
- **NER**: ~$8-20 (8×all samples unique predictions with intelligent caching)

**Cost Optimization**: Intelligent caching eliminates duplicate API calls entirely. Actual costs depend on text overlap between original and modified versions and chosen model variant.

### Runtime Estimates (with rate limiting)
- **Per task**: 20-45 minutes (processing all samples)
- **All tasks**: 80-180 minutes total
- Runtime scales with the total number of samples in each modification set

## Comparison with OpenRouter Version

### Advantages of OpenAI GPT-5:
- Access to latest GPT-5 models and features
- Official OpenAI API with best reliability
- GPT-5 reasoning capabilities when available
- Potential better performance on complex tasks

### OpenRouter Advantages:
- Access to multiple model providers
- Potentially lower costs for some models
- Model comparison capabilities
- DeepSeek R1 reasoning features

## Troubleshooting

### Common Issues:

1. **API Key Error:**
   ```
   Error: OPENAI_API_KEY not found
   ```
   Solution: Check your `.env` file and API key

2. **Rate Limiting:**
   ```
   API Error: 429 Too Many Requests
   ```
   Solution: Increase sleep times in the script or upgrade your OpenAI plan

3. **Insufficient Credits:**
   ```
   API Error: 402 Payment Required
   ```
   Solution: Add credits to your OpenAI account

4. **Model Not Available:**
   ```
   API Error: 404 Model not found
   ```
   Solution: Check model availability on OpenAI platform

## Extending the Scripts

### Adding New Models:
1. Add model to `GPT5_MODELS` in `fluke_gpt5_utils.py`
2. Add configuration to `GPT5_CONFIGS`
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

### OpenRouter Scripts (Alternative Implementation)
- `run_sentiment_openrouter.py`: Sentiment analysis with OpenRouter
- `run_coref_openrouter.py`: Coreference resolution with OpenRouter
- `run_dialogue_openrouter.py`: Dialogue contradiction with OpenRouter
- `run_ner_openrouter.py`: Named entity recognition with OpenRouter

### Original Notebooks (DSPy-based)
- `llm_sentiment_deepseek.ipynb`: Sentiment analysis with DSPy
- `llm_coref_deepseek.ipynb`: Coreference resolution with DSPy
- `llm_dialogue_deepseek.ipynb`: Dialogue contradiction with DSPy
- `llm_ner_deepseek.ipynb`: Named entity recognition with DSPy

## Evaluation Metrics

- **Sentiment & Dialogue & Coreference**: Accuracy (percentage correct)
- **Named Entity Recognition**: F1 Score (precision/recall balance)
- **All tasks**: Robustness measured as performance drop under modifications

## Getting Started

1. Set up your OpenAI API key in `.env`
2. Install dependencies: `pip install -r requirements_gpt5.txt`
3. Run a test: `python run_sentiment_gpt5.py`
4. Monitor the cache efficiency and adjust sample sizes as needed
5. Use different model configurations (`standard`, `fast`, `fastest`, `chat`) based on your speed/cost requirements

The intelligent caching system will automatically optimize API usage while providing comprehensive robustness testing results.