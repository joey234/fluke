#!/usr/bin/env python3
"""
Test script for the fixed DeepSeek R1 signature.
This demonstrates how to use the improved signature to prevent None outputs.
"""

import os
import dspy
from dotenv import load_dotenv
from fix_deepseek_signature import (
    create_improved_deepseek_signature,
    create_robust_eval_metric,
    create_chain_of_thought_version,
    create_fallback_signature
)

# Load environment variables
load_dotenv()

def test_improved_signature():
    """Test the improved signature."""
    print("=== Testing Improved Signature ===")
    
    # Configure DSPy with DeepSeek R1 via OpenRouter
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    if not openrouter_api_key:
        print("Warning: OPENROUTER_API_KEY not found")
        return
    
    lm = dspy.LM(
        model="openrouter/deepseek/deepseek-r1",
        api_key=openrouter_api_key,
        api_base="https://openrouter.ai/api/v1",
        max_tokens=20_000,
        temperature=0  # Set to 0 for more consistent outputs
    )
    dspy.configure(lm=lm)
    
    # Create the improved module
    deepseek_sentiment = create_improved_deepseek_signature()
    
    # Test examples
    test_texts = [
        "it's a charming and often affecting journey.",
        "the movie was terrible and boring.",
        "this is absolutely fantastic!"
    ]
    
    for text in test_texts:
        try:
            pred = deepseek_sentiment(text=text)
            print(f"Text: {text}")
            print(f"Prediction: {pred.label}")
            print(f"Type: {type(pred.label)}")
            print("---")
        except Exception as e:
            print(f"Error with text '{text}': {e}")
            print("---")

def test_robust_eval_metric():
    """Test the robust evaluation metric."""
    print("\n=== Testing Robust Evaluation Metric ===")
    
    # Create a mock prediction object
    class MockPrediction:
        def __init__(self, label):
            self.label = label
    
    # Test various prediction types
    test_cases = [
        ("1", "1", True),
        ("0", "0", True),
        ("1", "0", False),
        (None, "1", False),  # None case
        ("", "1", False),     # Empty string case
        ("positive", "1", False),  # Invalid case
    ]
    
    eval_metric = create_robust_eval_metric()
    
    for pred_label, true_label, expected in test_cases:
        mock_pred = MockPrediction(pred_label)
        mock_true = MockPrediction(true_label)
        
        result = eval_metric(mock_true, mock_pred)
        print(f"Pred: {pred_label}, True: {true_label}, Expected: {expected}, Got: {result}")
        print("---")

def test_chain_of_thought():
    """Test the Chain-of-Thought version."""
    print("\n=== Testing Chain-of-Thought Version ===")
    
    # Create the CoT module
    cot_module = create_chain_of_thought_version()
    
    # Test with a simple example
    test_text = "this movie was absolutely wonderful!"
    
    try:
        # Note: This would need the actual LM configured to work
        print(f"CoT module created successfully for text: {test_text}")
        print("To test with actual LM, configure DSPy first")
    except Exception as e:
        print(f"Error creating CoT module: {e}")

def test_fallback_signature():
    """Test the fallback signature."""
    print("\n=== Testing Fallback Signature ===")
    
    # Create the fallback module
    fallback_module = create_fallback_signature()
    
    # Test with a simple example
    test_text = "this was the worst experience ever"
    
    try:
        # Note: This would need the actual LM configured to work
        print(f"Fallback module created successfully for text: {test_text}")
        print("To test with actual LM, configure DSPy first")
    except Exception as e:
        print(f"Error creating fallback module: {e}")

if __name__ == "__main__":
    print("DeepSeek R1 Signature Fix Test Suite")
    print("=" * 50)
    
    # Test the robust evaluation metric (doesn't need LM)
    test_robust_eval_metric()
    
    # Test module creation (doesn't need LM)
    test_chain_of_thought()
    test_fallback_signature()
    
    # Test with actual LM (requires API key)
    print("\n=== Testing with Actual LM ===")
    if os.getenv('OPENROUTER_API_KEY'):
        test_improved_signature()
    else:
        print("Skipping LM test - no API key found")
        print("Set OPENROUTER_API_KEY environment variable to test with actual LM")
    
    print("\n=== Test Complete ===")
    print("To use in your notebook:")
    print("1. Import the fix functions")
    print("2. Replace your existing signature with the improved one")
    print("3. Use the robust evaluation metric")
    print("4. Consider using Chain-of-Thought for better reasoning") 