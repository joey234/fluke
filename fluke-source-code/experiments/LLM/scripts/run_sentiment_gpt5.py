#!/usr/bin/env python3
"""
FLUKE Sentiment Analysis with OpenAI GPT-5 (without DSPy)
Direct OpenAI GPT-5 API implementation for sentiment analysis robustness testing.
"""

import os
import json
import pandas as pd
import glob
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import FLUKE utilities
from fluke_gpt5_utils import (
    remove_space, extract_classification_prediction,
    aggregate_results, compare_models,
    GPT5_MODELS, GPT5_CONFIGS, GPT5Client
)

# Load environment variables
load_dotenv()

class SentimentAnalyzer:
    def __init__(self, client: GPT5Client, reasoning_effort: str = "medium"):
        self.client = client
        self.reasoning_effort = reasoning_effort
        self.prompt_template = """Analyze the sentiment of the following text. Answer with 0 for negative sentiment, 1 for positive sentiment.

Text: {text}

Answer:"""
    
    def predict(self, text: str) -> Dict[str, str]:
        """Predict sentiment for given text"""
        prompt = self.prompt_template.format(text=text)
        response = self.client.generate(prompt, reasoning_effort=self.reasoning_effort)
        return response

def evaluate_modified_set(analyzer: SentimentAnalyzer, data: List[Dict], prediction_cache: Dict[str, Dict], max_samples: int = None) -> List[Dict]:
    """Evaluate on modified dataset with prediction caching"""
    limited_data = data[:max_samples] if max_samples and len(data) > max_samples else data
    
    results = []
    for i, item in enumerate(limited_data):
        print(f"Processing modification {i+1}/{len(limited_data)}", end='\r')
        
        # Prepare text keys for caching
        original_text_clean = remove_space(item['original_text'])
        modified_text_clean = remove_space(item['modified_text'])
        
        # Get original prediction (from cache or new prediction)
        if original_text_clean in prediction_cache:
            original_response = prediction_cache[original_text_clean]
            original_pred_label = extract_classification_prediction(original_response['content'])
            original_raw_output = original_response['content']
            original_reasoning = original_response['reasoning']
        else:
            original_response = analyzer.predict(original_text_clean)
            original_pred_label = extract_classification_prediction(original_response['content'])
            original_raw_output = original_response['content']
            original_reasoning = original_response['reasoning']
            # Cache for future use
            prediction_cache[original_text_clean] = original_response
        
        # Get modified prediction (from cache or new prediction)
        if modified_text_clean in prediction_cache:
            modified_response = prediction_cache[modified_text_clean]
            modified_pred_label = extract_classification_prediction(modified_response['content'])
        else:
            modified_response = analyzer.predict(modified_text_clean)
            modified_pred_label = extract_classification_prediction(modified_response['content'])
            # Cache for future use
            prediction_cache[modified_text_clean] = modified_response
        
        result = {
            'text': modified_text_clean,
            'original_text': original_text_clean,
            'modified_label': int(item.get('modified_label', item['label'])),
            'original_label': int(item['label']),
            'modified_pred': modified_pred_label,
            'original_pred': original_pred_label,
            'raw_output': modified_response['content'],
            'reasoning': modified_response['reasoning'],
            'original_raw_output': original_raw_output,
            'original_reasoning': original_reasoning
        }
        
        results.append(result)
        
        # Rate limiting
        time.sleep(0.1)
    
    print()  # New line after progress
    return results

def main():
    # Configuration
    CONFIG_NAME = 'standard'  # Options: 'standard', 'fast', 'fastest', 'chat'
    config = GPT5_CONFIGS[CONFIG_NAME]
    MODEL_NAME = config['model']
    MODEL_ID = GPT5_MODELS[MODEL_NAME]
    
    print(f"Configuration: {CONFIG_NAME}")
    print(f"Model: {MODEL_NAME} ({MODEL_ID})")
    print(f"Description: {config['description']}")
    
    # Get API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        return
    
    # Initialize client and analyzer with reasoning effort from config
    client = GPT5Client(openai_api_key, MODEL_ID)
    analyzer = SentimentAnalyzer(client, reasoning_effort=config.get('reasoning_effort', 'medium'))
    
    # Load dataset
    data_path = '../../../data/train_dev_test_data/sent/test.json'
    with open(data_path, 'r') as f:
        ds = json.load(f)
    
    print(f"Dataset size: {len(ds)}")
    
    # Create examples
    examples = [
        {
            "text": remove_space(item["sentence"]),
            "label": item["label"]
        }
        for item in ds
    ]
    
    # Test single example
    example = examples[0]
    print(f"\nExample text: {example['text']}")
    print(f"Label: {example['label']}")
    
    test_response = analyzer.predict(example['text'])
    test_pred_label = extract_classification_prediction(test_response['content'])
    print(f"Prediction: {test_pred_label}")
    print(f"Raw output: {test_response['content'][:100]}...")
    if test_response['reasoning']:
        print(f"Reasoning: {test_response['reasoning'][:100]}...")
    
    # Initialize prediction cache for reusing predictions across modifications
    prediction_cache = {}
    os.makedirs('../results/sa', exist_ok=True)
    
    # Test modifications with caching approach
    json_files = glob.glob('../../../data/modified_data/sa/*_100.json')
    
    print(f"\nTesting {len(json_files)} modifications...")
    
    for json_file in json_files:
        print(f"\nProcessing: {json_file.split('/')[-1]}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Evaluate all samples using prediction cache
        results_mod = evaluate_modified_set(analyzer, data, prediction_cache)
        
        # Calculate accuracy
        mod_correct = sum(1 for r in results_mod if r['modified_pred'] == str(r['modified_label']))
        mod_accuracy = mod_correct / len(results_mod)
        
        # Save results
        df_mod = pd.DataFrame(results_mod)
        mod_name = json_file.split('/')[-1].replace('.json', '')
        output_file = f'../results/sa/{MODEL_NAME}-{CONFIG_NAME}-0shot-{mod_name}.csv'
        df_mod.to_csv(output_file, index=False)
        
        print(f"Accuracy: {mod_accuracy:.3f}")
        print(f"Saved to: {output_file}")
        
        time.sleep(2)  # Rate limiting
    
    print(f"\n{'='*60}")
    print(f"FLUKE Sentiment Analysis with {MODEL_NAME} Complete!")
    print(f"{'='*60}")
    print(f"Processed {len(json_files)} modifications with caching")
    print(f"Cache size: {len(prediction_cache)} unique predictions")
    print(f"Files saved in: ../results/sa/")
    print(f"OpenAI API endpoint: https://api.openai.com/v1")
    print(f"Model ID: {MODEL_ID}")
    print("Cached predictions avoid duplicate API calls for efficiency")

if __name__ == "__main__":
    main()