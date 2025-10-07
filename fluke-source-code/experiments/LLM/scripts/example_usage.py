#!/usr/bin/env python3
"""
Example usage of the OpenRouter sentiment analysis script
"""

import os
from run_sentiment_openrouter import OpenRouterClient, SentimentAnalyzer
from fluke_reasoning_utils import extract_classification_prediction, REASONING_MODELS

def test_single_prediction():
    """Test a single sentiment prediction"""
    
    # Get API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable")
        return
    
    # Initialize client with DeepSeek R1
    model_id = REASONING_MODELS['deepseek-r1']
    client = OpenRouterClient(api_key, model_id)
    analyzer = SentimentAnalyzer(client, use_cot=False)
    
    # Test examples
    test_texts = [
        "This movie is absolutely fantastic!",
        "I hate this boring film.",
        "The weather today is okay, I guess.",
        "What a wonderful day to be alive!"
    ]
    
    print("Testing sentiment analysis with DeepSeek R1:")
    print("=" * 50)
    
    for text in test_texts:
        prediction = analyzer.predict(text)
        pred_label = extract_classification_prediction(prediction)
        
        print(f"Text: {text}")
        print(f"Prediction: {pred_label} ({'Positive' if pred_label == '1' else 'Negative'})")
        print(f"Raw output: {prediction[:100]}...")
        print("-" * 30)

def test_cot_prediction():
    """Test chain-of-thought sentiment prediction"""
    
    # Get API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable")
        return
    
    # Initialize client with Chain-of-Thought
    model_id = REASONING_MODELS['deepseek-r1']
    client = OpenRouterClient(api_key, model_id)
    cot_analyzer = SentimentAnalyzer(client, use_cot=True)
    
    test_text = "The movie had some good moments, but overall it was disappointing."
    
    print("\nTesting Chain-of-Thought reasoning:")
    print("=" * 50)
    
    prediction = cot_analyzer.predict(test_text)
    pred_label = extract_classification_prediction(prediction)
    
    print(f"Text: {test_text}")
    print(f"Prediction: {pred_label} ({'Positive' if pred_label == '1' else 'Negative'})")
    print(f"\nFull reasoning:")
    print(prediction)

if __name__ == "__main__":
    print("FLUKE Sentiment Analysis - OpenRouter Example")
    print("=" * 60)
    
    # Test basic predictions
    test_single_prediction()
    
    # Test chain-of-thought
    test_cot_prediction()
    
    print("\nTo run the full evaluation, use:")
    print("python run_sentiment_openrouter.py")