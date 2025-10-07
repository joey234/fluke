# FLUKE Experiments with OpenAI o3 and o1 Reasoning Models

This directory contains modified FLUKE notebooks and scripts for evaluating OpenAI's reasoning models (o3-2025-04-16, o1-preview, o1-mini, o1) on linguistic robustness tasks.

## 🧠 What's New with o3 and o1 Reasoning Models

OpenAI's o3 and o1 models are **advanced reasoning models** that:
- Automatically use sophisticated chain-of-thought reasoning
- Don't support temperature or system messages
- Have different API parameters than GPT-4
- Provide detailed reasoning traces
- Are optimized for complex reasoning and problem-solving tasks
- **o3-2025-04-16** represents the latest advancement in reasoning capabilities

## 📁 Files Added

### Notebooks
- `llm_sentiment_o1.ipynb` - Sentiment analysis with reasoning models (updated for o3)
- `llm_coref_o3.ipynb` - Coreference resolution with o3 model
- `run_o1_experiments.py` - Automated experiment runner (supports o3 and o1 models)

### Key Modifications from Original FLUKE

1. **Model Configuration**
   ```python
   # Original FLUKE
   lm = dspy.LM('openai/gpt-4o', temperature=0, max_tokens=250)
   
   # o3/o1 Version
   lm = dspy.LM('openai/o3-2025-04-16')  # No temperature/max_tokens
   # or
   lm = dspy.LM('openai/o1-preview')
   ```

2. **Reasoning-Aware Signatures**
   ```python
   class O3Sentiment(dspy.Signature):
       """Think step by step and analyze the emotional tone..."""
       # Enhanced reasoning prompts for o3 model
   ```

3. **Rate Limiting & Cost Management**
   - Reduced sample sizes (25-100 vs 500+)
   - Added delays between requests (especially for o3)
   - Enhanced cost estimation functionality

## 🚀 Quick Start

### Prerequisites
```bash
pip install dspy openai pandas datasets tqdm scipy
export OPENAI_API_KEY="your-api-key-here"
```

### Run Single Task
```bash
# Test sentiment analysis with latest o3 model
python run_o1_experiments.py --task sentiment --model o3-2025-04-16 --samples 25

# Use o1-preview for comparison
python run_o1_experiments.py --task sentiment --model o1-preview --samples 25

# Use efficient o1-mini for larger tests  
python run_o1_experiments.py --task sentiment --model o1-mini --samples 50

# Dry run to see plan and costs
python run_o1_experiments.py --task all --model o3-2025-04-16 --dry-run
```

### Run Individual Notebooks
```bash
# Start Jupyter and run manually for development/testing
jupyter notebook llm_sentiment_o1.ipynb
```

## 🔧 Configuration Options

### Available Models
- `o3-2025-04-16` - Latest o3 reasoning model (highest capability)
- `o1-preview` - Full o1 reasoning model (high cost, excellent performance)
- `o1-mini` - Efficient o1 reasoning model (lower cost, good performance)  
- `o1` - Latest o1 model (when available)

### Reasoning Modes
- `standard` - Standard reasoning approach
- `detailed` - Detailed step-by-step reasoning
- `efficient` - Concise reasoning (faster/cheaper)

### Example Configurations
```bash
# Latest o3 model with detailed reasoning
python run_o1_experiments.py --task coref --model o3-2025-04-16 --config detailed

# High-quality o1 reasoning
python run_o1_experiments.py --task coref --model o1-preview --config detailed

# Cost-efficient testing
python run_o1_experiments.py --task sentiment --model o1-mini --config efficient --samples 25

# Test specific modifications with o3
python run_o1_experiments.py --task sentiment --model o3-2025-04-16 --modifications negation_100 capitalization_100
```

## 💰 Cost Considerations

o3 and o1 models are more expensive than GPT-4. The runner provides cost estimates:

```
Estimated costs per task (50 samples):
- o3-2025-04-16: ~$4.00 (highest capability)
- o1-preview: ~$2.50
- o1-mini: ~$0.50
- Full evaluation (all tasks): ~$15-30 with o3
```

**Cost-saving tips:**
- Start with `o1-mini` for initial testing
- Use `o3-2025-04-16` for final high-quality evaluations
- Use `--samples 25` for exploration  
- Test specific modifications with `--modifications`
- Use `--dry-run` to see estimates first

## 📊 Output and Results

Results are saved with model-specific prefixes:
```
results/sa/o3-2025-04-16-standard-0shot-sst2.csv
results/sa/o1-preview-standard-0shot-sst2.csv
results/sa/o1-mini-efficient-0shot-negation_100.csv
results/coref/o3-2025-04-16-detailed-aggregated.csv
```

### Key Metrics
- **Base Accuracy** - Performance on original datasets
- **Robustness** - Performance drop on modified text
- **Reasoning Quality** - Analysis of reasoning traces
- **Cost Efficiency** - Accuracy per dollar spent

## 🔍 Research Questions

These experiments help answer:

1. **Do reasoning models show better robustness?**
   - Compare o3/o1 vs GPT-4 on FLUKE modifications
   
2. **What's the reasoning vs accuracy tradeoff?**
   - o3-2025-04-16 vs o1-preview vs o1-mini
   
3. **Which linguistic modifications challenge reasoning models?**
   - Negation, syntax changes, semantic shifts
   
4. **Is the cost worth it for robustness testing?**
   - Cost/benefit analysis for different use cases

## 🛠 Development

### Adding New Tasks

1. Copy an existing notebook (e.g., `llm_sentiment_o1.ipynb`)
2. Modify for your task's data format and evaluation
3. Add task configuration to `run_o1_experiments.py`
4. Test with small sample size first

### Customizing Reasoning Prompts

Edit the signature descriptions for task-specific reasoning:

```python
class O1CustomTask(dspy.Signature):
    """Analyze this text step by step. Consider X, Y, Z factors..."""
    text = dspy.InputField()
    result = dspy.OutputField(prefix='Analysis:')
```

## 📈 Expected Results

Based on initial testing:

- **o3-2025-04-16**: Highest baseline accuracy, most sophisticated reasoning traces
- **o1-preview**: Higher baseline accuracy, excellent reasoning traces
- **o1-mini**: Good efficiency/performance balance  
- **Robustness**: Mixed results - advanced reasoning helps with some modifications, not others
- **Cost**: 5-15x more expensive than GPT-4, but potentially more reliable

## 🤝 Contributing

To contribute improvements:

1. Test your changes with small samples first
2. Include cost estimates for new features
3. Document reasoning prompt design choices
4. Compare results with existing FLUKE baselines

## 📚 References

- [FLUKE Paper](https://arxiv.org/abs/2504.17311)
- [OpenAI o1 Documentation](https://platform.openai.com/docs/guides/reasoning)
- [DSPy Framework](https://github.com/stanfordnlp/dspy)

---

**Note**: o3 and o1 models are evolving rapidly. Update model IDs and configurations as new versions are released. The o3-2025-04-16 model represents the latest advancement in reasoning capabilities.