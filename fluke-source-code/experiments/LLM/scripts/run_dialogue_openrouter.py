#!/usr/bin/env python3
"""
FLUKE Dialogue Contradiction Detection with OpenRouter API (without DSPy)
Direct OpenRouter API implementation for dialogue contradiction detection robustness testing.
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
    compare_models, append_person, REASONING_MODELS, REASONING_CONFIGS
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

class DialogueContradictionDetector:
    def __init__(self, client: OpenRouterClient, use_cot: bool = False):
        self.client = client
        self.use_cot = use_cot
        
        # Use the same prompt template as GPT-5 for consistency
        self.prompt_template = """Given a dialogue with agent labels (agent 0 and agent 1 alternating), determine if the last utterance contradicts the dialogue context. Respond with ONLY a single digit: 1 if it contradicts, 0 if it does not.

Dialogue:
{dialogue}

Answer:"""
    
    def predict(self, dialogue: str) -> Dict[str, str]:
        """Predict dialogue contradiction for given dialogue"""
        prompt = self.prompt_template.format(dialogue=dialogue)
        response = self.client.generate(prompt)
        return response

def add_agent_labels(dialogue_list: List[str]) -> str:
    """Add agent 0/1 labels to each turn in the dialogue."""
    labeled_dialogue = []
    for i, turn in enumerate(dialogue_list):
        agent_label = f"agent {i % 2}: {turn}"
        labeled_dialogue.append(agent_label)
    return '\n'.join(labeled_dialogue)

def load_dialogue_data(data_path: str) -> List[Dict]:
    """Load and prepare dialogue contradiction dataset"""
    ds = pd.read_json(data_path)
    ds = ds.to_dict('records')
    
    print(f"Loaded {len(ds)} dialogue samples")
    
    # Split by contradiction type
    samples_with_contradiction = []
    samples_no_contradiction = []
    for i, x in enumerate(ds):
        if x["is_contradiction"]:
            samples_with_contradiction.append((i, x))
        else:
            samples_no_contradiction.append((i, x))
    
    print(f"Contradictions: {len(samples_with_contradiction)}, No contradictions: {len(samples_no_contradiction)}")
    
    # Combine and shuffle
    samples = samples_with_contradiction + samples_no_contradiction
    random.shuffle(samples)
    
    label_map = {'is_contradiction': 1, 'no_contradiction': 0}
    
    # Create examples with agent labels
    examples = []
    for i, r in samples:
        example = {
            "dialogue": remove_space(add_agent_labels(r["dialogue"])),
            "label": label_map[r['label']],
            "index": i
        }
        examples.append(example)
    
    return examples

def evaluate_dataset(detector: DialogueContradictionDetector, examples: List[Dict], max_samples: int = None) -> List[Dict]:
    """Evaluate dialogue contradiction detector on dataset"""
    if max_samples:
        examples = examples[:max_samples]
    
    results = []
    for i, example in enumerate(examples):
        print(f"Processing {i+1}/{len(examples)}", end='\r')
        
        prediction = detector.predict(example['dialogue'])
        pred_label = extract_classification_prediction(prediction['content'])
        
        results.append({
            'dialog': example['dialogue'],
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

def evaluate_modified_set(detector: DialogueContradictionDetector, data: List[Dict], prediction_cache: Dict[str, Dict], max_samples: int = None) -> List[Dict]:
    """Evaluate on modified dataset with prediction caching"""
    limited_data = data[:max_samples] if max_samples and len(data) > max_samples else data
    
    results = []
    for i, item in enumerate(limited_data):
        print(f"Processing modification {i+1}/{len(limited_data)}", end='\r')
        
        # Ensure item is a valid dictionary with required keys
        if not isinstance(item, dict) or 'dialog_context' not in item:
            continue
        
        # Create modified and original dialogues
        original_dialogue = add_agent_labels(item['dialog_context'] + [item['original_text']])
        modified_dialogue = add_agent_labels(item['dialog_context'] + [item['modified_text']])
        
        # Create cache keys
        original_dialogue_clean = remove_space(original_dialogue)
        modified_dialogue_clean = remove_space(modified_dialogue)
        
        # Get original prediction (from cache or new prediction)
        if original_dialogue_clean in prediction_cache:
            original_response = prediction_cache[original_dialogue_clean]
            original_pred_label = extract_classification_prediction(original_response['content'])
            original_raw_output = original_response['content']
            original_reasoning = original_response['reasoning']
        else:
            original_response = detector.predict(original_dialogue)
            original_pred_label = extract_classification_prediction(original_response['content'])
            original_raw_output = original_response['content']
            original_reasoning = original_response['reasoning']
            # Cache for future use
            prediction_cache[original_dialogue_clean] = original_response
        
        # Get modified prediction (from cache or new prediction)
        if modified_dialogue_clean in prediction_cache:
            modified_response = prediction_cache[modified_dialogue_clean]
            modified_pred_label = extract_classification_prediction(modified_response['content'])
        else:
            modified_response = detector.predict(modified_dialogue)
            modified_pred_label = extract_classification_prediction(modified_response['content'])
            # Cache for future use
            prediction_cache[modified_dialogue_clean] = modified_response
        
        result = {
            'original_dialog': original_dialogue_clean,
            'modified_dialog': modified_dialogue_clean,
            'modified_label': int(item.get('modified_label', item.get('label', 0))),
            'original_label': int(item.get('original_label', item.get('label', 0))),
            'modified_pred': modified_pred_label,
            'original_pred': original_pred_label,
            'raw_output': modified_response['content'],
            'reasoning': modified_response['reasoning'],
            'original_raw_output': original_raw_output,
            'original_reasoning': original_reasoning,
            'type': item.get('type', None),
            'id': i
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
    
    # Initialize client and detector
    client = OpenRouterClient(openrouter_api_key, MODEL_ID)
    detector = DialogueContradictionDetector(client, use_cot=False)
    
    # Load dataset
    data_path = '../../../data/train_dev_test_data/dialog/test.json'
    examples = load_dialogue_data(data_path)
    
    # Test single example
    example = examples[59]  # Use same example as notebook
    print(f"\nExample dialogue with agent labels:")
    print(f"{example['dialogue'][:500]}...")
    print(f"\nLabel: {example['label']} ({'Contradiction' if example['label'] == 1 else 'No contradiction'})")
    
    test_pred = detector.predict(example['dialogue'])
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
    os.makedirs('../results/dialogue', exist_ok=True)
    
    # Test modifications with caching approach
    json_files = glob.glob('../../../data/modified_data/dialogue/*_100.json')
    
    print(f"\nTesting {len(json_files)} modifications...")
    processed_mods = []
    
    for json_file in json_files:
        print(f"\nProcessing: {json_file.split('/')[-1]}")
        
        with open(json_file, 'r') as f:
            raw_data = json.load(f)
        
        # Extract the actual data from the [index, data_dict] structure
        data = []
        for item in raw_data:
            if isinstance(item, list) and len(item) >= 2:
                data.append(item[1])  # The data dict is the second element
            elif isinstance(item, dict):
                data.append(item)  # Already a dict
        
        print(f"Extracted {len(data)} samples from {len(raw_data)} raw items")
        if data:
            print(f"Sample data keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
        
        # Evaluate all samples using prediction cache
        results_mod = evaluate_modified_set(detector, data, prediction_cache)
        
        print(f"Results: {len(results_mod)} items processed")
        
        # Handle any missing labels
        for item in results_mod:
            if item['modified_label'] is None:
                item['modified_label'] = item['original_label']
            if item['original_label'] is None:
                item['original_label'] = item['modified_label']
        
        # Calculate accuracy
        if len(results_mod) == 0:
            print("Warning: No results processed!")
            mod_accuracy = 0.0
            mod_correct = 0
        else:
            mod_correct = sum(1 for r in results_mod if r['modified_pred'] == str(r['modified_label']))
            mod_accuracy = mod_correct / len(results_mod)
        
        # Save results
        df_mod = pd.DataFrame(results_mod)
        print(f"DataFrame columns: {list(df_mod.columns)}")
        print(f"Sample reasoning content: {df_mod['reasoning'].iloc[0] if 'reasoning' in df_mod.columns else 'No reasoning column'}")
        mod_name = json_file.split('/')[-1].replace('.json', '')
        output_file = f'../results/dialogue/{MODEL_NAME}-{CONFIG_NAME}-0shot-{mod_name}.csv'
        df_mod.to_csv(output_file, index=False)
        
        print(f"Accuracy: {mod_accuracy:.3f}")
        print(f"Saved to: {output_file}")
        processed_mods.append(mod_name)
        
        time.sleep(2)  # Rate limiting
    
    print(f"\n{'='*60}")
    print(f"FLUKE Dialogue Contradiction Detection with {MODEL_NAME} Complete!")
    print(f"{'='*60}")
    print(f"Processed {len(json_files)} modifications with caching")
    print(f"Cache size: {len(prediction_cache)} unique predictions")
    print(f"Files saved in: ../results/dialogue/")
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
