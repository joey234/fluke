#!/usr/bin/env python3
"""
FLUKE Named Entity Recognition with OpenAI GPT-5 (without DSPy)
Direct OpenAI GPT-5 API implementation for NER robustness testing.
"""

import os
import json
import pandas as pd
import glob
import time
import ast
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import FLUKE utilities
from fluke_gpt5_utils import (
    remove_space, extract_ner_prediction,
    aggregate_results, compare_models,
    calculate_f1_ent, convert_string_to_entities,
    GPT5_MODELS, GPT5_CONFIGS, GPT5Client
)

# Load environment variables
load_dotenv()

class NamedEntityRecognizer:
    def __init__(self, client: GPT5Client, reasoning_effort: str = "medium"):
        self.client = client
        self.reasoning_effort = reasoning_effort
        self.prompt_template = """Extract named entities from the text. Possible entity types: ART, BUILDING, EVENT, LOCATION, ORGANIZATION, OTHER, PERSON, PRODUCT.

Text: {text}

Entities: [{{"text": "entity text span", "value": "entity type"}},]"""
    
    def predict(self, text: str) -> Dict[str, str]:
        """Predict named entities for given text"""
        prompt = self.prompt_template.format(text=text)
        response = self.client.generate(prompt, reasoning_effort=self.reasoning_effort)
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
        # print(item['modified_label'])
        print(item)
        result = {
            'text': modified_text_clean,
            'original_text': original_text_clean,
            'modified_label': str(item['modified_label'] if item.get('modified_label') != None else item['label']),
            'original_label': str(item['original_label'] if item.get('original_label') != None else item['label']),
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
    
    # Initialize client and recognizer with reasoning effort from config
    client = GPT5Client(openai_api_key, MODEL_ID)
    recognizer = NamedEntityRecognizer(client, reasoning_effort=config.get('reasoning_effort', 'medium'))
    
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
    if test_pred['reasoning']:
        print(f"Reasoning: {test_pred['reasoning'][:100]}...")
    
    # Initialize prediction cache for reusing predictions across modifications
    prediction_cache = {}
    os.makedirs('../results/ner', exist_ok=True)
    
    # Test modifications with caching approach
    json_files = glob.glob('../../../data/modified_data/ner/*_100.json')
    test_modifications = [
        # 'typo_bias_100.json', 'capitalization_100.json', 'punctuation_100.json',
        # 'negation_100.json', 'sentiment_100.json', 'active_to_passive_100.json',
        # 'casual_100.json', 'dialectal_100.json', 
        'dialectal_100.json'
    ]
    json_files = [f for f in json_files if not any(mod in f for mod in test_modifications)]
    
    print(f"\nTesting {len(json_files)} modifications...")
    
    for json_file in json_files:
        print(f"\nProcessing: {json_file.split('/')[-1]}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Evaluate all samples using prediction cache
        results_mod = evaluate_modified_set(recognizer, data, prediction_cache)
        
        # Handle NaN in original_label
        for item in results_mod:
            if pd.isna(item['original_label']) or item['original_label'] == 'nan':
                item['original_label'] = item['modified_label']
        
        # Calculate average F1 scores
        mod_f1_scores = []
        for r in results_mod:
            f1 = calculate_f1_score(r['modified_label'], r['modified_pred'])
            mod_f1_scores.append(f1)
        
        mod_avg_f1 = sum(mod_f1_scores) / len(mod_f1_scores) if mod_f1_scores else 0.0
        
        # Save results
        df_mod = pd.DataFrame(results_mod)
        mod_name = json_file.split('/')[-1].replace('.json', '')
        output_file = f'../results/ner/{MODEL_NAME}-{CONFIG_NAME}-0shot-{mod_name}.csv'
        df_mod.to_csv(output_file, index=False)
        
        print(f"Average F1: {mod_avg_f1:.3f}")
        print(f"Saved to: {output_file}")
        
        time.sleep(2)  # Rate limiting
    
    print(f"\n{'='*60}")
    print(f"FLUKE NER with {MODEL_NAME} Complete!")
    print(f"{'='*60}")
    print(f"Processed {len(json_files)} modifications with caching")
    print(f"Cache size: {len(prediction_cache)} unique predictions")
    print(f"Files saved in: ../results/ner/")
    print(f"OpenAI API endpoint: https://api.openai.com/v1")
    print(f"Model ID: {MODEL_ID}")
    print("Cached predictions avoid duplicate API calls for efficiency")

if __name__ == "__main__":
    main()