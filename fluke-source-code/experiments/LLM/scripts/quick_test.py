#!/usr/bin/env python3
"""
Quick test to verify the sentiment script is working
"""

import os
import json
from dotenv import load_dotenv

# Import our modules
from run_sentiment_openrouter import OpenRouterClient, SentimentAnalyzer
from fluke_reasoning_utils import extract_classification_prediction, REASONING_MODELS, REASONING_CONFIGS

# Load environment variables
load_dotenv()

def main():
    # Configuration
    CONFIG_NAME = 'deepseek'
    config = REASONING_CONFIGS[CONFIG_NAME]
    MODEL_NAME = config['model']
    MODEL_ID = REASONING_MODELS[MODEL_NAME]
    
    print(f"Testing: {MODEL_NAME} ({MODEL_ID})")
    
    # Get API key
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    if not openrouter_api_key:
        print("Error: OPENROUTER_API_KEY not found")
        return
    
    # Initialize client and analyzer
    client = OpenRouterClient(openrouter_api_key, MODEL_ID)
    analyzer = SentimentAnalyzer(client)
    
    # Test single example
    test_text = "This movie is absolutely fantastic!"
    print(f"\nTest text: {test_text}")
    
    response = analyzer.predict(test_text)
    pred_label = extract_classification_prediction(response['content'])
    
    print(f"Raw response: {response['content']}")
    print(f"Extracted prediction: {pred_label}")
    print(f"Expected: 1 (positive)")
    
    if pred_label == '1':
        print("✅ Test passed!")
    else:
        print("❌ Test failed - prediction doesn't match expected")

if __name__ == "__main__":
    main()