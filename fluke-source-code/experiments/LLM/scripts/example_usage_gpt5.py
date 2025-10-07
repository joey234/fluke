#!/usr/bin/env python3
"""
Example usage of FLUKE GPT-5 scripts
Quick test examples for sentiment analysis and other tasks.
"""

import os
from dotenv import load_dotenv
from fluke_gpt5_utils import GPT5Client, GPT5_MODELS, GPT5_CONFIGS, extract_classification_prediction, extract_ner_prediction

# Load environment variables
load_dotenv()

def test_sentiment():
    """Test GPT-5 sentiment analysis"""
    print("=" * 50)
    print("Testing GPT-5 Sentiment Analysis")
    print("=" * 50)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not found. Please set it in .env file")
        return
    
    # Initialize client
    client = GPT5Client(api_key, GPT5_MODELS['gpt-5'])
    
    # Test examples
    examples = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "This movie was terrible and boring. Complete waste of time.",
        "The weather is okay today, nothing special."
    ]
    
    for i, text in enumerate(examples):
        print(f"\nExample {i+1}: {text}")
        
        prompt = f"""Analyze the sentiment of the following text. Answer with 0 for negative sentiment, 1 for positive sentiment.

Text: {text}

Answer:"""
        
        response = client.generate(prompt)
        prediction = extract_classification_prediction(response['content'])
        
        print(f"Prediction: {prediction}")
        print(f"Raw output: {response['content'][:100]}...")
        if response['reasoning']:
            print(f"Reasoning: {response['reasoning'][:100]}...")

def test_ner():
    """Test GPT-5 named entity recognition"""
    print("\n" + "=" * 50)
    print("Testing GPT-5 Named Entity Recognition")
    print("=" * 50)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not found. Please set it in .env file")
        return
    
    # Initialize client with faster model for quick test
    client = GPT5Client(api_key, GPT5_MODELS['gpt-5-mini'])
    
    # Test example
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    print(f"\nExample text: {text}")
    
    prompt = f"""Extract named entities from the text. Possible entity types: ART, BUILDING, EVENT, LOCATION, ORGANIZATION, OTHER, PERSON, PRODUCT.

Text: {text}

Entities: [{{"text": "entity text span", "value": "entity type"}},]"""
    
    response = client.generate(prompt)
    entities = extract_ner_prediction(response['content'])
    
    print(f"Entities found: {entities}")
    print(f"Raw output: {response['content'][:200]}...")

def test_model_comparison():
    """Test different GPT-5 model variants"""
    print("\n" + "=" * 50)
    print("Testing GPT-5 Model Variants")
    print("=" * 50)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not found. Please set it in .env file")
        return
    
    test_text = "I really enjoyed this book, it was amazing!"
    
    # Test different model configurations
    models_to_test = ['fast', 'fastest', 'chat']  # Skip 'standard' to save costs
    
    for config_name in models_to_test:
        config = GPT5_CONFIGS[config_name]
        model_id = GPT5_MODELS[config['model']]
        
        print(f"\nTesting {config['model']} ({config['description']}):")
        
        client = GPT5Client(api_key, model_id)
        
        prompt = f"""Analyze the sentiment of the following text. Answer with 0 for negative sentiment, 1 for positive sentiment.

Text: {test_text}

Answer:"""
        
        response = client.generate(prompt)
        prediction = extract_classification_prediction(response['content'])
        
        print(f"  Prediction: {prediction}")
        print(f"  Response length: {len(response['content'])} chars")
        if response['reasoning']:
            print(f"  Has reasoning: Yes ({len(response['reasoning'])} chars)")
        else:
            print(f"  Has reasoning: No")

def main():
    """Run all example tests"""
    print("FLUKE GPT-5 Example Usage")
    print("This script demonstrates the GPT-5 integration")
    print("\nNote: This will make API calls to OpenAI GPT-5")
    print("Make sure you have:")
    print("1. Set OPENAI_API_KEY in .env file")
    print("2. Added credits to your OpenAI account")
    print("3. Installed dependencies: pip install -r requirements_gpt5.txt")
    
    # Check if user wants to proceed
    try:
        proceed = input("\nProceed with API tests? (y/n): ").lower().strip()
        if proceed != 'y':
            print("Skipping API tests.")
            return
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return
    
    try:
        test_sentiment()
        test_ner()
        test_model_comparison()
        
        print("\n" + "=" * 50)
        print("All tests completed!")
        print("=" * 50)
        print("\nTo run full FLUKE experiments:")
        print("  python run_sentiment_gpt5.py")
        print("  python run_coref_gpt5.py")
        print("  python run_dialogue_gpt5.py") 
        print("  python run_ner_gpt5.py")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        print("Please check your API key and internet connection.")

if __name__ == "__main__":
    main()