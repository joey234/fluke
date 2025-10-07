#!/usr/bin/env python3
"""
FLUKE Named Entity Recognition with GPT-5 (with modification context)
Enhanced version that provides information about text modifications to the model.
"""

import os
import json
import pandas as pd
import glob
import time
import random
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

# Modification type descriptions (same as dialogue and coref)
MODIFICATION_DESCRIPTIONS = {
    'active_to_passive': 'transformed from active to passive voice',
    'capitalization': 'modified to have different capitalization patterns',
    'casual': 'modified to use more casual/informal language',
    'compound_word': 'modified to use compound words',
    'concept_replacement': 'modified with concept replacements',
    'coordinating_conjunction': 'modified to include coordinating conjunctions',
    'derivation': 'modified with word derivations',
    'dialectal': 'modified to use Singaporean English dialectal variations',
    'discourse': 'modified with discourse-level changes',
    'geographical_bias': 'modified to change some names or concepts to equivalent words from different geographical regions',
    'grammatical_role': 'modified with grammatical role changes',
    'length_bias': 'modified to have different text length',
    'negation': 'modified to add or remove negation',
    'punctuation': 'modified to have different punctuation',
    'sentiment': 'modified to change sentiment',
    'singlish': 'modified to use Singlish language patterns',
    'temporal_bias': 'modified to use words that were commonly used in the past',
    'typo_bias': 'modified to contain typos'
}

class NamedEntityRecognizerWithContext:
    def __init__(self, client: GPT5Client, reasoning_effort: str = "medium"):
        self.client = client
        self.reasoning_effort = reasoning_effort
        
        # Enhanced prompt template that includes modification context
        self.prompt_template = """Extract named entities from the text. Possible entity types: ART, BUILDING, EVENT, LOCATION, ORGANIZATION, OTHER, PERSON, PRODUCT.

Note: This text has been {modification_description}.

Text: {text}

Entities: [{{"text": "entity text span", "value": "entity type"}},]"""
        
        # Fallback template for when no modification context is available
        self.fallback_template = """Extract named entities from the text. Possible entity types: ART, BUILDING, EVENT, LOCATION, ORGANIZATION, OTHER, PERSON, PRODUCT.

Text: {text}

Entities: [{{"text": "entity text span", "value": "entity type"}},]"""
    
    def predict(self, text: str, modification_type: str = None) -> Dict[str, str]:
        """Predict named entities for given text with optional modification context"""
        if modification_type and modification_type in MODIFICATION_DESCRIPTIONS:
            modification_desc = MODIFICATION_DESCRIPTIONS[modification_type]
            prompt = self.prompt_template.format(text=text, modification_description=modification_desc)
        else:
            prompt = self.fallback_template.format(text=text)
            
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

def evaluate_modified_set(recognizer: NamedEntityRecognizerWithContext, data: List[Dict], prediction_cache: Dict[str, Dict], modification_type: str) -> List[Dict]:
    """Evaluate on modified dataset with prediction caching and modification context"""
    
    results = []
    for i, item in enumerate(data):
        print(f"Processing modification {i+1}/{len(data)}", end='\r')
        
        # Ensure item is a valid dictionary
        if not isinstance(item, dict):
            continue
        
        # Extract original and modified data
        original_text_clean = remove_space(item['original_text'])
        modified_text_clean = remove_space(item['modified_text'])
        
        # Get original prediction (from cache or new prediction) - no modification context
        if original_text_clean in prediction_cache:
            print(f"\n✓ CACHE HIT: Using cached original prediction")
            original_response = prediction_cache[original_text_clean]
            original_pred_content = original_response['content']
            original_raw_output = original_response['content']
            original_reasoning = original_response['reasoning']
        else:
            print(f"\n→ CACHE MISS: Generating new original prediction (no context)")
            original_response = recognizer.predict(original_text_clean)  # No modification context
            original_pred_content = original_response['content']
            original_raw_output = original_response['content']
            original_reasoning = original_response['reasoning']
            # Cache for future use
            prediction_cache[original_text_clean] = original_response
        
        # Get modified prediction (from cache or new prediction) - with modification context
        cache_key_with_context = f"{modified_text_clean}_{modification_type}"
        if cache_key_with_context in prediction_cache:
            print(f"✓ Using cached modified prediction ({modification_type})")
            modified_response = prediction_cache[cache_key_with_context]
        else:
            print(f"→ Generating new modified prediction with context ({modification_type})")
            modified_response = recognizer.predict(modified_text_clean, modification_type)  # Include modification context
            # Cache for future use
            prediction_cache[cache_key_with_context] = modified_response
        
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
            'original_reasoning': original_reasoning,
            'modification_type': modification_type,
            'modification_description': MODIFICATION_DESCRIPTIONS.get(modification_type, 'unknown')
        }
        
        results.append(result)
        
        # Rate limiting
        time.sleep(0.1)
    
    print()  # New line after progress
    return results

def main():
    # Configuration
    CONFIG_NAME = 'standard'  # Options: 'standard', 'turbo', 'efficient'
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
    
    # Initialize client and recognizer
    client = GPT5Client(openai_api_key, MODEL_ID)
    recognizer = NamedEntityRecognizerWithContext(client, reasoning_effort=config.get('reasoning_effort', 'medium'))
    
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
    # Seed cache from existing context-aware outputs if present
    seed_pattern = f"../results/ner/{MODEL_NAME}-{CONFIG_NAME}-context-aware-0shot-*.csv"
    seeded = 0
    for out_csv in glob.glob(seed_pattern):
        try:
            df_prev = pd.read_csv(out_csv)
        except Exception:
            continue
        # Original predictions
        if {'original_text','original_raw_output','original_reasoning'}.issubset(df_prev.columns):
            for _, r in df_prev.iterrows():
                key = remove_space(str(r.get('original_text','')))
                if key and key not in prediction_cache:
                    prediction_cache[key] = {'content': str(r.get('original_raw_output','')), 'reasoning': str(r.get('original_reasoning',''))}
                    seeded += 1
        # Modified predictions (context-aware key)
        if {'text','raw_output','reasoning','modification_type'}.issubset(df_prev.columns):
            for _, r in df_prev.iterrows():
                mtext = remove_space(str(r.get('text','')))
                mtype = str(r.get('modification_type',''))
                mkey = f"{mtext}_{mtype}"
                if mtext and mtype and mkey not in prediction_cache:
                    prediction_cache[mkey] = {'content': str(r.get('raw_output','')), 'reasoning': str(r.get('reasoning',''))}
                    seeded += 1
    if seeded:
        print(f"Seeded cache with {seeded} entries from previous outputs")
    os.makedirs('../results/ner', exist_ok=True)
    
    # Test modifications with context-aware prompts
    json_files = glob.glob('../../../data/modified_data/ner/*_100.json')
    
    print(f"\nTesting {len(json_files)} modifications with context-aware prompts...")
    
    for json_file in json_files:
        print(f"\nProcessing: {json_file.split('/')[-1]}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract modification type from filename
        modification_type = json_file.split('/')[-1].replace('_100.json', '')
        
        print(f"Processing {len(data)} samples with modification type: {modification_type}")
        print(f"Description: {MODIFICATION_DESCRIPTIONS.get(modification_type, 'unknown modification')}")
        
        # Evaluate all samples using prediction cache with modification context
        results_mod = evaluate_modified_set(recognizer, data, prediction_cache, modification_type)
        
        print(f"Results: {len(results_mod)} items processed")
        
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
        output_file = f'../results/ner/{MODEL_NAME}-{CONFIG_NAME}-context-aware-0shot-{mod_name}.csv'
        df_mod.to_csv(output_file, index=False)
        
        print(f"Average F1: {mod_avg_f1:.3f}")
        print(f"Saved to: {output_file}")
        
        time.sleep(2)  # Rate limiting
    
    print(f"\n{'='*60}")
    print(f"FLUKE Named Entity Recognition with Context-Aware Prompts Complete!")
    print(f"{'='*60}")
    print(f"Processed {len(json_files)} modifications with context information")
    print(f"Cache size: {len(prediction_cache)} unique predictions")
    print(f"Files saved in: ../results/ner/")
    print(f"OpenAI GPT-5 Responses API endpoint")
    print(f"Model ID: {MODEL_ID}")
    print("Context-aware prompts inform the model about text modifications")

if __name__ == "__main__":
    main()
