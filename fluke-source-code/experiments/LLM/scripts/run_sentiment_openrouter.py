#!/usr/bin/env python3
"""
FLUKE Sentiment Analysis with OpenRouter API (without DSPy)
Direct OpenRouter API implementation for sentiment analysis robustness testing.
"""

import os
import json
import pandas as pd
import glob
import time
import re
import ast
from typing import List, Dict, Any
from dotenv import load_dotenv
import requests

# Import FLUKE utilities
from fluke_reasoning_utils import (
    remove_space, extract_classification_prediction,
    aggregate_results, highlight_drops_and_significance,
    compare_models, REASONING_MODELS, REASONING_CONFIGS
)

# Load environment variables
load_dotenv()

class OpenRouterClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate(self, prompt: str, max_tokens: int = 20000, temperature: float = 0.2, use_reasoning: bool = True) -> Dict[str, str]:
        """Generate response from OpenRouter API with reasoning support"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Add reasoning support for compatible models
        if use_reasoning and ("deepseek" in self.model.lower()):
            # DeepSeek models on OpenRouter use include_reasoning parameter
            payload["include_reasoning"] = True
        elif use_reasoning and "o1" in self.model.lower():
            # o1 models use different reasoning format
            payload["reasoning"] = {
                "effort": "high",
                "max_tokens": 2000,
                "exclude": False
            }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            message = result["choices"][0]["message"]
            return {
                "content": message.get("content", ""),
                "reasoning": message.get("reasoning", "")
            }
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    print(f"Error details: {error_details}")
                except:
                    print(f"Response text: {e.response.text}")
            return {"content": "", "reasoning": ""}
        except Exception as e:
            print(f"API Error: {e}")
            return {"content": "", "reasoning": ""}

class SentimentAnalyzer:
    def __init__(self, client: OpenRouterClient):
        self.client = client
        # Simplified prompt for direct classification with reasoning
        # Use the same prompt template as GPT-5 for consistency
        self.prompt_template = """Analyze the sentiment of the following text. Respond with ONLY a single digit: 0 for negative sentiment, 1 for positive sentiment.

Text: {text}

Answer:"""
    
    def predict(self, text: str) -> Dict[str, str]:
        """Predict sentiment for given text with reasoning"""
        prompt = self.prompt_template.format(text=text)
        response = self.client.generate(prompt)
        return response

def evaluate_dataset(analyzer: SentimentAnalyzer, examples: List[Dict], max_samples: int = None) -> List[Dict]:
    """Evaluate sentiment analyzer on dataset"""
    if max_samples:
        examples = examples[:max_samples]
    
    results = []
    for i, example in enumerate(examples):
        print(f"Processing {i+1}/{len(examples)}", end='\r')
        
        response = analyzer.predict(example['text'])
        pred_label = extract_classification_prediction(response['content'])
        
        results.append({
            'text': example['text'],
            'label': example['label'],
            'pred': pred_label,
            'raw_output': response['content'],
            'reasoning': response['reasoning']
        })
        
        # Rate limiting
        time.sleep(0.1)
    
    print()  # New line after progress
    return results

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
            'modified_label': int(item.get('modified_label', item.get('label', 0))),
            'original_label': int(item.get('original_label', item.get('label', 0))),
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
    CONFIG_NAME = 'deepseek'  # Options: 'deepseek', 'deepseek-lite'
    if CONFIG_NAME == 'deepseek':
        config = {'model': 'deepseek', 'description': 'DeepSeek R1 - with reasoning tokens', 'use_reasoning': True}
        MODEL_NAME = 'deepseek'
        MODEL_ID = 'deepseek/deepseek-r1'
    elif CONFIG_NAME == 'deepseek-lite':
        config = {'model': 'deepseek-lite', 'description': 'DeepSeek R1 Lite', 'use_reasoning': True}
        MODEL_NAME = 'deepseek-lite'
        MODEL_ID = 'deepseek/deepseek-r1:free'
    else:
        config = REASONING_CONFIGS[CONFIG_NAME]
        MODEL_NAME = config['model']
        MODEL_ID = REASONING_MODELS[MODEL_NAME]
    
    print(f"Configuration: {CONFIG_NAME}")
    print(f"Model: {MODEL_NAME} ({MODEL_ID})")
    print(f"Description: {config['description']}")
    
    # Get API key
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    if not openrouter_api_key:
        print("Error: OPENROUTER_API_KEY not found in environment variables")
        print("Please set your OpenRouter API key in the .env file")
        return
    
    # Initialize client and analyzer
    client = OpenRouterClient(openrouter_api_key, MODEL_ID)
    analyzer = SentimentAnalyzer(client)
    
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
    print(f"Raw output: {test_response['content']}")
    print(f"Reasoning available: {bool(test_response.get('reasoning'))}")
    if test_response.get('reasoning'):
        print(f"Full Reasoning: {test_response['reasoning']}")
    else:
        print("No reasoning content found")
    
    # Initialize prediction cache for reusing predictions across modifications
    prediction_cache = {}
    os.makedirs('../results/sa', exist_ok=True)
    
    # Test modifications with caching approach
    json_files = glob.glob('../../../data/modified_data/sa/*_100.json')
    
    print(f"\nTesting {len(json_files)} modifications...")
    processed_mods = []
    
    for json_file in json_files:
        print(f"\nProcessing: {json_file.split('/')[-1]}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Evaluate all samples using prediction cache
        results_mod = evaluate_modified_set(analyzer, data, prediction_cache)
        
        # Handle any missing labels
        for item in results_mod:
            if item['modified_label'] is None:
                item['modified_label'] = item['original_label']
            if item['original_label'] is None:
                item['original_label'] = item['modified_label']
        
        # Calculate accuracy
        mod_correct = sum(1 for r in results_mod if r['modified_pred'] == str(r['modified_label']))
        mod_accuracy = mod_correct / len(results_mod)
        
        # Save results
        df_mod = pd.DataFrame(results_mod)
        print(f"DataFrame columns: {list(df_mod.columns)}")
        print(f"Sample reasoning content: {df_mod['reasoning'].iloc[0] if 'reasoning' in df_mod.columns else 'No reasoning column'}")
        mod_name = json_file.split('/')[-1].replace('.json', '')
        output_file = f'../results/sa/{MODEL_NAME}-{CONFIG_NAME}-0shot-{mod_name}.csv'
        df_mod.to_csv(output_file, index=False)
        
        print(f"Accuracy: {mod_accuracy:.3f}")
        print(f"Saved to: {output_file}")
        processed_mods.append(mod_name)
        
        time.sleep(2)  # Rate limiting
    
    print(f"\n{'='*60}")
    print(f"FLUKE Sentiment Analysis with {MODEL_NAME} Complete!")
    print(f"{'='*60}")
    print(f"Processed {len(json_files)} modifications with caching")
    print(f"Cache size: {len(prediction_cache)} unique predictions")
    print(f"Files saved in: ../results/sa/")
    print(f"OpenRouter API endpoint: https://openrouter.ai/api/v1")
    print(f"Model ID: {MODEL_ID}")
    print("Cached predictions avoid duplicate API calls for efficiency")
    try:
        uniq = sorted(set(processed_mods))
        print(f"Processed modifications ({len(uniq)}): {', '.join(uniq)}")
        print(f"Model: {MODEL_NAME}")
    except Exception:
        pass
    # Auto sanity report
    try:
        print("\nRunning sanity report...")
        os.system("python sanity_report.py")
    except Exception as e:
        print(f"Warning: sanity report failed: {e}")

if __name__ == "__main__":
    main()
