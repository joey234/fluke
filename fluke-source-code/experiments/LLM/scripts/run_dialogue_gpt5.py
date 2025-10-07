#!/usr/bin/env python3
"""
FLUKE Dialogue Contradiction Detection with OpenAI GPT-5 (without DSPy)
Direct OpenAI GPT-5 API implementation for dialogue contradiction detection robustness testing.
"""

import os
import json
import pandas as pd
import glob
import time
import random
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

class DialogueContradictionDetector:
    def __init__(self, client: GPT5Client, reasoning_effort: str = "medium"):
        self.client = client
        self.reasoning_effort = reasoning_effort
        self.prompt_template = """Given a dialogue with agent labels (agent 0 and agent 1 alternating), determine if the last utterance contradicts the dialogue context. Answer with 1 if it contradicts, 0 if it does not contradict.

Dialogue:
{dialogue}

Answer:"""
    
    def predict(self, dialogue: str) -> Dict[str, str]:
        """Predict dialogue contradiction for given dialogue"""
        prompt = self.prompt_template.format(dialogue=dialogue)
        response = self.client.generate(prompt, reasoning_effort=self.reasoning_effort)
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

def evaluate_modified_set(detector: DialogueContradictionDetector, data: List[Dict], prediction_cache: Dict[str, Dict], max_samples: int = None) -> List[Dict]:
    """Evaluate on modified dataset with prediction caching"""
    limited_data = data[:max_samples] if max_samples and len(data) > max_samples else data
    
    results = []
    for i, item_data in enumerate(limited_data):
        print(f"Processing modification {i+1}/{len(limited_data)}", end='\r')
        
        # Handle the data structure - it's [index, data_dict]
        if isinstance(item_data, list) and len(item_data) == 2:
            item_index, item = item_data
        else:
            item = item_data
        
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
            'modified_label': int(item.get('modified_label', item['label'])),
            'original_label': int(item['label']),
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
    CONFIG_NAME = 'standard'  # Options: 'standard', 'fast', 'fastest', 'chat'
    config = GPT5_CONFIGS[CONFIG_NAME]
    MODEL_NAME = config['model']
    MODEL_ID = GPT5_MODELS[MODEL_NAME]
    
    print(f"Configuration: {CONFIG_NAME}")
    print(f"Model: {MODEL_NAME} ({MODEL_ID})")
    print(f"Description: {config['description']}")
    print(f"Reasoning Effort: {config.get('reasoning_effort', 'medium')}")
    
    # Get API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        return
    
    # Initialize client and detector with reasoning effort from config
    client = GPT5Client(openai_api_key, MODEL_ID)
    detector = DialogueContradictionDetector(client, reasoning_effort=config.get('reasoning_effort', 'medium'))
    
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
    print(f"Debug info: {test_pred.get('debug_reasoning', {})}")
    
    # Initialize prediction cache for reusing predictions across modifications
    prediction_cache = {}
    os.makedirs('../results/dialogue', exist_ok=True)
    
    # Test modifications with caching approach
    json_files = glob.glob('../../../data/modified_data/dialogue/*_100.json')
    
    print(f"\nTesting {len(json_files)} modifications...")
    
    for json_file in json_files:
        print(f"\nProcessing: {json_file.split('/')[-1]}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Evaluate all samples using prediction cache (agent labels added in evaluation)
        results_mod = evaluate_modified_set(detector, data, prediction_cache)
        
        # Calculate accuracy
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
        
        time.sleep(2)  # Rate limiting
    
    print(f"\n{'='*60}")
    print(f"FLUKE Dialogue Contradiction Detection with {MODEL_NAME} Complete!")
    print(f"{'='*60}")
    print(f"Processed {len(json_files)} modifications with caching")
    print(f"Cache size: {len(prediction_cache)} unique predictions")
    print(f"Files saved in: ../results/dialogue/")
    print(f"OpenAI API endpoint: https://api.openai.com/v1")
    print(f"Model ID: {MODEL_ID}")
    print("Cached predictions avoid duplicate API calls for efficiency")

if __name__ == "__main__":
    main()