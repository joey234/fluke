#!/usr/bin/env python3
"""
FLUKE Named Entity Recognition with OpenRouter API (without DSPy)
Direct OpenRouter API implementation for NER robustness testing.
"""

import os
import json
import pandas as pd
import glob
import time
import ast
from typing import List, Dict, Any
from dotenv import load_dotenv
import requests

# Import FLUKE utilities
from fluke_reasoning_utils import (
    remove_space, extract_ner_prediction,
    aggregate_results, highlight_drops_and_significance,
    compare_models, calculate_f1_ent, convert_string_to_entities,
    REASONING_MODELS, REASONING_CONFIGS
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

class NamedEntityRecognizer:
    def __init__(self, client: OpenRouterClient, use_cot: bool = False):
        self.client = client
        self.use_cot = use_cot
        
        # Use the same prompt template as GPT-5 for consistency
        self.prompt_template = """Extract named entities from the text. Possible entity types: ART, BUILDING, EVENT, LOCATION, ORGANIZATION, OTHER, PERSON, PRODUCT.

Text: {text}

Return ONLY a JSON array in a fenced code block with objects of the form {{"text": "...", "value": "..."}}. No explanations.

```json
[]
```"""
    
    def predict(self, text: str) -> Dict[str, str]:
        """Predict named entities for given text"""
        prompt = self.prompt_template.format(text=text)
        response = self.client.generate(prompt)
        return response

def load_ner_data(data_path: str) -> List[Dict]:
    """Load and prepare NER dataset"""
    ds = pd.read_json(data_path, encoding_errors='replace')
    ds = ds.to_dict('records')
    
    print(f"Loaded {len(ds)} NER samples")
    print(f"Sample structure: {list(ds[0].keys())}")
    
    # Create examples
    examples = []
    for r in ds:
        example = {
            "text": r["text"],
            "label": str(r['label']),
            "index": r.get('id', 0)
        }
        examples.append(example)
    
    return examples

def evaluate_dataset(recognizer: NamedEntityRecognizer, examples: List[Dict], max_samples: int = None) -> List[Dict]:
    """Evaluate NER on dataset"""
    if max_samples:
        examples = examples[:max_samples]
    
    results = []
    for i, example in enumerate(examples):
        print(f"Processing {i+1}/{len(examples)}", end='\r')
        
        prediction = recognizer.predict(example['text'])
        
        results.append({
            'text': example['text'],
            'label': example['label'],
            'pred': prediction['content'],
            'raw_output': prediction['content'],
            'reasoning': prediction['reasoning'],
            'index': example['index']
        })
        
        # Rate limiting
        time.sleep(0.1)
    
    print()  # New line after progress
    return results

def evaluate_modified_set(recognizer: NamedEntityRecognizer, data: List[Dict], prediction_cache: Dict[str, Dict], max_samples: int = None) -> List[Dict]:
    """Evaluate on modified dataset with prediction caching"""
    limited_data = data[:max_samples] if max_samples and len(data) > max_samples else data
    
    results = []
    for i, item in enumerate(limited_data):
        print(f"Processing modification {i+1}/{len(limited_data)}", end='\r')
        
        # Prepare text keys for caching
        original_text_clean = remove_space(item['original_text'])
        modified_text_clean = remove_space(item["modified_text"])
        
        # Get original prediction (from cache or new prediction)
        if original_text_clean in prediction_cache:
            original_response = prediction_cache[original_text_clean]
            original_pred_content = original_response['content']
            original_raw_output = original_response['content']
            original_reasoning = original_response['reasoning']
        else:
            original_response = recognizer.predict(original_text_clean)
            original_pred_content = original_response['content']
            original_raw_output = original_response['content']
            original_reasoning = original_response['reasoning']
            # Cache for future use
            prediction_cache[original_text_clean] = original_response
        
        # Get modified prediction (from cache or new prediction)
        if modified_text_clean in prediction_cache:
            modified_response = prediction_cache[modified_text_clean]
        else:
            modified_response = recognizer.predict(modified_text_clean)
            # Cache for future use
            prediction_cache[modified_text_clean] = modified_response
        
        result = {
            'text': modified_text_clean,
            'original_text': original_text_clean,
            'modified_label': str(item.get('modified_label', item.get('label', []))),
            'original_label': str(item.get('original_label', item.get('label', []))),
            'modified_pred': modified_response['content'],
            'original_pred': original_pred_content,
            'index': item.get('index', i),
            'type': item.get('subtype', None),
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

def calculate_f1_score(true_label: str, pred_label: str) -> float:
    """Calculate F1 score for NER prediction"""
    try:
        # Parse true labels
        if isinstance(true_label, str):
            gold_entities = ast.literal_eval(true_label)
        else:
            gold_entities = true_label
        
        # Parse predicted entities
        parsed_answer = extract_ner_prediction(pred_label)
        if not parsed_answer:
            return 0.0
        
        # Calculate F1
        precision, recall, f1_score = calculate_f1_ent(gold_entities, parsed_answer)
        return f1_score
    except:
        return 0.0

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
    
    # Initialize client and recognizer
    client = OpenRouterClient(openrouter_api_key, MODEL_ID)
    recognizer = NamedEntityRecognizer(client, use_cot=False)
    
    # Load dataset
    data_path = '../../../data/train_dev_test_data/ner/fewnerd_sample_test.json'
    examples = load_ner_data(data_path)
    
    # Test single example
    example = examples[0]
    print(f"\nExample text: {example['text']}")
    print(f"Label: {example['label']}")
    
    test_pred = recognizer.predict(example['text'])
    test_f1 = calculate_f1_score(example['label'], test_pred['content'])
    print(f"Prediction: {test_pred['content']}")
    print(f"F1 Score: {test_f1:.3f}")
    print(f"Reasoning available: {bool(test_pred.get('reasoning'))}")
    if test_pred.get('reasoning'):
        print(f"Full Reasoning: {test_pred['reasoning']}")
    else:
        print("No reasoning content found")
    
    # Initialize prediction cache for reusing predictions across modifications
    prediction_cache = {}
    os.makedirs('../results/ner', exist_ok=True)
    
    # Test modifications with caching approach
    json_files = glob.glob('../../../data/modified_data/ner/*_100.json')
    
    print(f"\nTesting {len(json_files)} modifications...")
    processed_mods = []
    
    for json_file in json_files:
        print(f"\nProcessing: {json_file.split('/')[-1]}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Evaluate all samples using prediction cache
        results_mod = evaluate_modified_set(recognizer, data, prediction_cache)
        
        # Handle NaN and missing labels
        for item in results_mod:
            # Handle original_label
            if pd.isna(item['original_label']) or item['original_label'] == 'nan' or item['original_label'] == '[]':
                item['original_label'] = item['modified_label']
            
            # Handle modified_label  
            if pd.isna(item['modified_label']) or item['modified_label'] == 'nan' or item['modified_label'] == '[]':
                item['modified_label'] = item['original_label']
        
        # Calculate average F1 scores
        mod_f1_scores = []
        for r in results_mod:
            f1 = calculate_f1_score(r['modified_label'], r['modified_pred'])
            mod_f1_scores.append(f1)
        
        mod_avg_f1 = sum(mod_f1_scores) / len(mod_f1_scores) if mod_f1_scores else 0.0
        
        # Save results
        df_mod = pd.DataFrame(results_mod)
        print(f"DataFrame columns: {list(df_mod.columns)}")
        print(f"Sample reasoning content: {df_mod['reasoning'].iloc[0] if 'reasoning' in df_mod.columns else 'No reasoning column'}")
        mod_name = json_file.split('/')[-1].replace('.json', '')
        output_file = f'../results/ner/{MODEL_NAME}-{CONFIG_NAME}-0shot-{mod_name}.csv'
        df_mod.to_csv(output_file, index=False)
        
        print(f"Average F1: {mod_avg_f1:.3f}")
        print(f"Saved to: {output_file}")
        processed_mods.append(mod_name)
        
        time.sleep(2)  # Rate limiting
    
    print(f"\n{'='*60}")
    print(f"FLUKE NER with {MODEL_NAME} Complete!")
    print(f"{'='*60}")
    print(f"Processed {len(json_files)} modifications with caching")
    print(f"Cache size: {len(prediction_cache)} unique predictions")
    print(f"Files saved in: ../results/ner/")
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
