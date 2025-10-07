# DeepSeek R1 None Output Fix

## Problem Description

The DeepSeek R1 model is outputting `None` values in sentiment analysis tasks, even after instructing it to generate only 0 or 1. This is a common issue with DSPy when the model doesn't follow the expected output format consistently.

## Root Causes

1. **Vague Instructions**: The original signature doesn't clearly specify the exact output format
2. **Prefix Confusion**: Using `prefix='Answer:'` can confuse the model about where to put the actual answer
3. **Lack of Error Handling**: No fallback when the model returns invalid outputs
4. **Temperature Setting**: High temperature (1.0) can lead to inconsistent outputs

## Solutions Provided

### 1. Improved Signature (`create_improved_deepseek_signature`)

```python
class DeepSeekSentiment(dspy.Signature):
    """Classify sentiment of the given text. Analyze the emotional tone, word choice, and overall sentiment. 
    You must respond with ONLY a single digit: 1 for positive sentiment, 0 for negative sentiment. 
    Do not include any other text or explanation."""
    text = dspy.InputField()
    label = dspy.OutputField(desc="The sentiment label: 1 for positive, 0 for negative")
```

**Key improvements:**
- Removed confusing `prefix='Answer:'`
- Added explicit instruction: "ONLY a single digit"
- Used `desc` instead of `prefix` for clearer output specification
- Added "Do not include any other text or explanation"

### 2. Robust Evaluation Metric (`create_robust_eval_metric`)

```python
def eval_metric(true, prediction, trace=None):
    pred = prediction.label
    print(f"Raw prediction: {pred}")
    
    # Handle None or invalid predictions
    if pred is None:
        print("WARNING: Model returned None prediction")
        return False
    
    # Convert to string and clean
    pred_str = str(pred).strip()
    if not pred_str:
        print("WARNING: Model returned empty prediction")
        return False
    
    # Try to extract the classification
    parsed_answer = extract_classification_prediction(pred_str)
    if parsed_answer is None:
        print(f"WARNING: Could not parse prediction '{pred_str}'")
        return False
    
    result = parsed_answer == str(true.label)
    print(f"Parsed: {parsed_answer}, True: {true.label}, Correct: {result}")
    return result
```

**Key improvements:**
- Explicit None handling
- Empty string detection
- Better error messages
- Detailed logging for debugging

### 3. Chain-of-Thought Version (`create_chain_of_thought_version`)

```python
class CoTDeepSeekSentiment(dspy.Signature):
    """Classify sentiment of the given text. First think through the emotional tone, word choice, and overall sentiment.
    Then provide your final answer as ONLY a single digit: 1 for positive sentiment, 0 for negative sentiment."""
    text = dspy.InputField()
    reasoning = dspy.OutputField(desc="Your step-by-step reasoning about the sentiment")
    label = dspy.OutputField(desc="The final sentiment label: 1 for positive, 0 for negative")
```

**Key improvements:**
- Forces the model to think before answering
- Separates reasoning from final answer
- More structured output format

### 4. Fallback Signature (`create_fallback_signature`)

```python
class FallbackDeepSeekSentiment(dspy.Signature):
    """Classify sentiment of the given text. 
    
    Instructions:
    1. Analyze the emotional tone, word choice, and overall sentiment
    2. Respond with ONLY a single digit: 1 for positive, 0 for negative
    3. Do not include any other text, punctuation, or explanation
    
    Example output: 1
    """
    text = dspy.InputField()
    label = dspy.OutputField(desc="Single digit: 1=positive, 0=negative")
```

**Key improvements:**
- Numbered instructions for clarity
- Explicit example output
- Minimal description to avoid confusion

## Usage Instructions

### Step 1: Import the Fix Functions

```python
from fix_deepseek_signature import (
    create_improved_deepseek_signature,
    create_robust_eval_metric,
    create_chain_of_thought_version,
    create_fallback_signature
)
```

### Step 2: Replace Your Existing Signature

```python
# Instead of defining the signature inline, use:
deepseek_sentiment = create_improved_deepseek_signature()
```

### Step 3: Use the Robust Evaluation Metric

```python
# Replace your existing eval_metric with:
eval_metric = create_robust_eval_metric()
```

### Step 4: Adjust Model Configuration

```python
lm = dspy.LM(
    model="openrouter/deepseek/deepseek-r1",
    api_key=openrouter_api_key,
    api_base="https://openrouter.ai/api/v1",
    max_tokens=20_000,
    temperature=0  # Set to 0 for more consistent outputs
)
```

## Additional Recommendations

### 1. Set Temperature to 0
```python
temperature=0  # For consistent, deterministic outputs
```

### 2. Use System Messages
```python
# Add a system message to reinforce instructions
system_message = "You are a sentiment analysis expert. Always respond with only a single digit: 1 for positive, 0 for negative."
```

### 3. Implement Retry Logic
```python
def predict_with_retry(module, text, max_retries=3):
    for attempt in range(max_retries):
        try:
            pred = module(text=text)
            if pred.label is not None and str(pred.label).strip():
                return pred
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
    return None
```

### 4. Validate Outputs
```python
def validate_prediction(pred):
    if pred is None or pred.label is None:
        return False
    
    pred_str = str(pred.label).strip()
    return pred_str in ['0', '1']
```

## Testing

Run the test suite to verify the fix works:

```bash
cd fluke-source-code/experiments/LLM/scripts/
python test_deepseek_fix.py
```

## Expected Results

After implementing these fixes, you should see:
- No more `None` outputs
- Consistent 0/1 responses
- Better error handling and logging
- More reliable sentiment analysis results

## Troubleshooting

If you still get `None` outputs:

1. **Check API Key**: Ensure your OpenRouter API key is valid
2. **Reduce Temperature**: Set temperature to 0 for consistency
3. **Use Chain-of-Thought**: Try the CoT version for better reasoning
4. **Check Model Availability**: Verify DeepSeek R1 is available on OpenRouter
5. **Implement Retry Logic**: Add retry mechanism for failed requests

## Files Created

- `fix_deepseek_signature.py` - Main fix functions
- `test_deepseek_fix.py` - Test suite
- `README_deepseek_fix.md` - This documentation

## Integration with Existing Code

To integrate with your existing notebook:

1. Copy the fix functions to your notebook
2. Replace the signature definition
3. Update the evaluation metric
4. Test with a few examples before running the full evaluation
5. Monitor the outputs for any remaining issues 