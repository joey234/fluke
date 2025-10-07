#!/usr/bin/env python3
"""
FLUKE Coreference Resolution with OpenRouter API (without DSPy)
Direct OpenRouter API implementation for coreference resolution robustness testing.
"""

import os
import json
import pandas as pd
import glob
import time
import random
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
        
        for attempt in range(3):
            try:
                response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=120)
                response.raise_for_status()
                try:
                    result = response.json()
                except json.JSONDecodeError as je:
                    if attempt < 2:
                        time.sleep(1 + attempt)
                        continue
                    print(f"API Error: JSON decode error: {je} | body preview: {response.text[:200]}")
                    return {"content": "", "reasoning": ""}
                message = result["choices"][0]["message"]
                return {
                    "content": message.get("content", ""),
                    "reasoning": message.get("reasoning", "")
                }
            except Exception as e:
                msg = str(e)
                if attempt < 2 and ("premature" in msg.lower() or "chunked" in msg.lower() or "timeout" in msg.lower() or isinstance(e, (requests.exceptions.ChunkedEncodingError, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError))):
                    time.sleep(1 + attempt)
                    continue
                print(f"API Error: {e}")
                return {"content": "", "reasoning": ""}
        return {"content": "", "reasoning": ""}

class CoreferenceResolver:
    def __init__(self, client: OpenRouterClient, use_cot: bool = False):
        self.client = client
        self.use_cot = use_cot
        
        # Use the same prompt template as GPT-5 for consistency
        self.prompt_template = """Determine the candidate that the given pronoun refers to in the text. Respond with ONLY a single digit: 0 or 1.

Text: {text}
Pronoun: {pronoun}
Candidates: {candidates}

Answer:"""
    
    def predict(self, text: str, pronoun: str, candidates: str) -> Dict[str, str]:
        """Predict coreference for given text and pronoun"""
        prompt = self.prompt_template.format(text=text, pronoun=pronoun, candidates=candidates)
        
        max_retries = 3
        for attempt in range(max_retries):
            response = self.client.generate(prompt)
            parsed_answer = extract_classification_prediction(response['content'])
            if parsed_answer and parsed_answer in ['0', '1']:
                return response
            time.sleep(0.5)  # Brief pause before retry
        
        return response  # Return last attempt even if parsing failed

def load_coref_data(data_path: str) -> List[Dict]:
    """Load and prepare coreference resolution dataset"""
    ds = pd.read_json(data_path)
    ds = ds.to_dict('records')
    
    print(f"Loaded {len(ds)} coreference samples")
    
    # Split by label for analysis
    positive_samples = []
    negative_samples = []
    for i, x in enumerate(ds):
        if x["label"] == 1:
            positive_samples.append((i, x))
        else:
            negative_samples.append((i, x))
    
    print(f"Positive samples (coreferent): {len(positive_samples)}")
    print(f"Negative samples (not coreferent): {len(negative_samples)}")
    
    # Combine and shuffle
    samples = positive_samples + negative_samples
    random.shuffle(samples)
    
    # Create examples
    examples = []
    for i, r in samples:
        example = {
            "text": remove_space(r["text"]),
            "pronoun": r["pronoun"],
            "candidates": '0: ' + r["candidates"][0] + ', 1: ' + r["candidates"][1],
            "label": r['label'],
            "index": i
        }
        examples.append(example)
    
    return examples

def evaluate_dataset(resolver: CoreferenceResolver, examples: List[Dict], max_samples: int = None) -> List[Dict]:
    """Evaluate coreference resolver on dataset"""
    if max_samples:
        examples = examples[:max_samples]
    
    results = []
    for i, example in enumerate(examples):
        print(f"Processing {i+1}/{len(examples)}", end='\r')
        
        prediction = resolver.predict(example['text'], example['pronoun'], example['candidates'])
        pred_label = extract_classification_prediction(prediction['content'])
        
        results.append({
            'text': example['text'],
            'pronoun': example['pronoun'],
            'candidates': example['candidates'],
            'label': example['label'],
            'pred': pred_label,
            'raw_output': prediction['content'],
            'reasoning': prediction['reasoning'],
            'index': example['index']
        })
        
        # Rate limiting
        time.sleep(0.1)
    
    print()  # New line after progress
    return results

def evaluate_modified_set(resolver: CoreferenceResolver, data: List[Dict], prediction_cache: Dict[str, Dict], max_samples: int = None) -> List[Dict]:
    """Evaluate on modified dataset with prediction caching"""
    limited_data = data[:max_samples] if max_samples and len(data) > max_samples else data
    
    results = []
    for i, item in enumerate(limited_data):
        print(f"Processing modification {i+1}/{len(limited_data)}", end='\r')
        
        # Extract original and modified data
        original_text = remove_space(item['original_text'])
        original_pronoun = item['original_pronoun']
        original_candidates = '0: ' + item["original_candidates"][0] + ', 1: ' + item["original_candidates"][1]
        
        modified_text = remove_space(item['modified_text'])
        modified_pronoun = item['modified_pronoun']
        modified_candidates = '0: ' + item["modified_candidates"][0] + ', 1: ' + item["modified_candidates"][1]
        
        # Create cache keys (combine text, pronoun, and candidates)
        original_cache_key = f"{original_text}|{original_pronoun}|{original_candidates}"
        modified_cache_key = f"{modified_text}|{modified_pronoun}|{modified_candidates}"
        
        # Get original prediction (from cache or new prediction)
        if original_cache_key in prediction_cache:
            original_response = prediction_cache[original_cache_key]
            original_pred_label = extract_classification_prediction(original_response['content'])
            original_raw_output = original_response['content']
            original_reasoning = original_response['reasoning']
        else:
            original_response = resolver.predict(original_text, original_pronoun, original_candidates)
            original_pred_label = extract_classification_prediction(original_response['content'])
            original_raw_output = original_response['content']
            original_reasoning = original_response['reasoning']
            # Cache for future use
            prediction_cache[original_cache_key] = original_response
        
        # Get modified prediction (from cache or new prediction)
        if modified_cache_key in prediction_cache:
            modified_response = prediction_cache[modified_cache_key]
            modified_pred_label = extract_classification_prediction(modified_response['content'])
        else:
            modified_response = resolver.predict(modified_text, modified_pronoun, modified_candidates)
            modified_pred_label = extract_classification_prediction(modified_response['content'])
            # Cache for future use
            prediction_cache[modified_cache_key] = modified_response
        
        result = {
            'text': modified_text,
            'original_text': original_text,
            'pronoun': modified_pronoun,
            'candidates': modified_candidates,
            'type': item.get('type', None),
            'modified_label': int(item.get('modified_label', item.get('label', 0))),
            'original_label': int(item.get('original_label', item.get('label', 0))),
            'modified_pred': modified_pred_label,
            'original_pred': original_pred_label,
            'raw_output': modified_response['content'],
            'reasoning': modified_response['reasoning'],
            'original_raw_output': original_raw_output,
            'original_reasoning': original_reasoning,
            'id': item['index']
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
    
    # Initialize client and resolver
    client = OpenRouterClient(openrouter_api_key, MODEL_ID)
    resolver = CoreferenceResolver(client, use_cot=False)
    
    # Load dataset
    data_path = '../../../data/train_dev_test_data/coref/test.json'
    examples = load_coref_data(data_path)
    
    # Test single example
    example = examples[0]
    print(f"\nExample text: {example['text']}")
    print(f"Pronoun: {example['pronoun']}")
    print(f"Candidates: {example['candidates']}")
    print(f"Label: {example['label']}")
    
    test_pred = resolver.predict(example['text'], example['pronoun'], example['candidates'])
    test_pred_label = extract_classification_prediction(test_pred['content'])
    print(f"Prediction: {test_pred_label}")
    print(f"Raw output: {test_pred['content']}")
    print(f"Reasoning available: {bool(test_pred.get('reasoning'))}")
    if test_pred.get('reasoning'):
        print(f"Full Reasoning: {test_pred['reasoning']}")
    else:
        print("No reasoning content found")
    
    # Initialize prediction cache for reusing predictions across modifications
    prediction_cache = {}
    os.makedirs('../results/coref', exist_ok=True)
    
    # Test modifications with caching approach
    json_files = glob.glob('../../../data/modified_data/coref/*_100.json')
    print(f"\nTesting {len(json_files)} modifications...")
    processed_mods = []
    
    for json_file in json_files:
        print(f"\nProcessing: {json_file.split('/')[-1]}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Evaluate all samples using prediction cache
        results_mod = evaluate_modified_set(resolver, data, prediction_cache)
        
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
        output_file = f'../results/coref/{MODEL_NAME}-{CONFIG_NAME}-0shot-{mod_name}.csv'
        df_mod.to_csv(output_file, index=False)
        
        print(f"Accuracy: {mod_accuracy:.3f}")
        print(f"Saved to: {output_file}")
        processed_mods.append(mod_name)
        
        time.sleep(2)  # Rate limiting
    
    print(f"\n{'='*60}")
    print(f"FLUKE Coreference Resolution with {MODEL_NAME} Complete!")
    print(f"{'='*60}")
    print(f"Processed {len(json_files)} modifications with caching")
    print(f"Cache size: {len(prediction_cache)} unique predictions")
    print(f"Files saved in: ../results/coref/")
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
