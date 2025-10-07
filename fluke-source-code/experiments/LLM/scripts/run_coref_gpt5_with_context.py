#!/usr/bin/env python3
"""
FLUKE Coreference Resolution with GPT-5 (with modification context)
Enhanced version that provides information about text modifications to the model.
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
from fluke_gpt5_utils import GPT5Client, GPT5_MODELS, GPT5_CONFIGS
from fluke_reasoning_utils import (
    remove_space, extract_classification_prediction,
    aggregate_results, highlight_drops_and_significance,
    compare_models
)

# Load environment variables
load_dotenv()

# Modification type descriptions (same as dialogue)
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

class CoreferenceResolverWithContext:
    def __init__(self, client: GPT5Client, reasoning_effort: str = "medium"):
        self.client = client
        self.reasoning_effort = reasoning_effort
        
        # Enhanced prompt template that includes modification context
        self.prompt_template = """Determine the candidate that the given pronoun refers to in the text. 

Note: This text has been {modification_description}.

Answer with the index of the candidate (0 or 1).

Text: {text}
Pronoun: {pronoun}
Candidates: {candidates}

Answer:"""
        
        # Fallback template for when no modification context is available
        self.fallback_template = """Determine the candidate that the given pronoun refers to in the text. Answer with the index of the candidate (0 or 1).

Text: {text}
Pronoun: {pronoun}
Candidates: {candidates}

Answer:"""
    
    def predict(self, text: str, pronoun: str, candidates: str, modification_type: str = None) -> Dict[str, str]:
        """Predict coreference for given text and pronoun with optional modification context"""
        if modification_type and modification_type in MODIFICATION_DESCRIPTIONS:
            modification_desc = MODIFICATION_DESCRIPTIONS[modification_type]
            prompt = self.prompt_template.format(text=text, pronoun=pronoun, candidates=candidates, modification_description=modification_desc)
        else:
            prompt = self.fallback_template.format(text=text, pronoun=pronoun, candidates=candidates)
        
        max_retries = 3
        for attempt in range(max_retries):
            response = self.client.generate(prompt, reasoning_effort=self.reasoning_effort)
            parsed_answer = extract_classification_prediction(response['content'])
            if parsed_answer and parsed_answer in ['0', '1']:
                return response
            time.sleep(0.5)  # Brief pause before retry
        
        return response  # Return last attempt even if parsing failed

def load_coref_data(data_path: str) -> List[Dict]:
    """Load and prepare coreference resolution dataset"""
    ds = pd.read_json(data_path)
    ds = ds.to_dict('records')
    
    print(f"Loaded {len(ds)} coref samples")
    
    # Create examples
    examples = []
    for i, r in enumerate(ds):
        candidates_str = '0: ' + r["candidates"][0] + ', 1: ' + r["candidates"][1]
        example = {
            "text": remove_space(r["text"]),
            "pronoun": r["pronoun"],
            "candidates": candidates_str,
            "label": int(r["label"]),
            "index": i
        }
        examples.append(example)
    
    return examples

def evaluate_modified_set(resolver: CoreferenceResolverWithContext, data: List[Dict], prediction_cache: Dict[str, Dict], modification_type: str) -> List[Dict]:
    """Evaluate on modified dataset with prediction caching and modification context"""
    
    results = []
    for i, item in enumerate(data):
        print(f"Processing modification {i+1}/{len(data)}", end='\r')
        
        # Ensure item is a valid dictionary with required keys
        if not isinstance(item, dict):
            continue
        
        # Extract original and modified data
        original_text = remove_space(item['original_text'])
        original_pronoun = item['original_pronoun']
        original_candidates = '0: ' + item["original_candidates"][0] + ', 1: ' + item["original_candidates"][1]
        
        modified_text = remove_space(item['modified_text'])
        modified_pronoun = item['modified_pronoun']
        modified_candidates = '0: ' + item["modified_candidates"][0] + ', 1: ' + item["modified_candidates"][1]
        
        # Create cache keys (combine text, pronoun, and candidates)
        original_cache_key = f"{original_text}|{original_pronoun}|{original_candidates}"
        modified_cache_key = f"{modified_text}|{modified_pronoun}|{modified_candidates}_{modification_type}"
        
        # Get original prediction (from cache or new prediction) - no modification context
        if original_cache_key in prediction_cache:
            print(f"\n✓ CACHE HIT: Using cached original prediction")
            original_response = prediction_cache[original_cache_key]
            original_pred_label = extract_classification_prediction(original_response['content'])
            original_raw_output = original_response['content']
            original_reasoning = original_response['reasoning']
        else:
            print(f"\n→ CACHE MISS: Generating new original prediction (no context)")
            original_response = resolver.predict(original_text, original_pronoun, original_candidates)  # No modification context
            original_pred_label = extract_classification_prediction(original_response['content'])
            original_raw_output = original_response['content']
            original_reasoning = original_response['reasoning']
            # Cache for future use
            prediction_cache[original_cache_key] = original_response
        
        # Get modified prediction (from cache or new prediction) - with modification context
        if modified_cache_key in prediction_cache:
            print(f"\n✓ CACHE HIT: Using cached modified prediction ({modification_type})")
            modified_response = prediction_cache[modified_cache_key]
            modified_pred_label = extract_classification_prediction(modified_response['content'])
        else:
            print(f"\n→ CACHE MISS: Generating new modified prediction with context ({modification_type})")
            modified_response = resolver.predict(modified_text, modified_pronoun, modified_candidates, modification_type)  # Include modification context
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
            'modification_type': modification_type,
            'modification_description': MODIFICATION_DESCRIPTIONS.get(modification_type, 'unknown'),
            'id': item.get('index', i)
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
    
    # Initialize client and resolver
    client = GPT5Client(openai_api_key, MODEL_ID)
    resolver = CoreferenceResolverWithContext(client, reasoning_effort=config.get('reasoning_effort', 'medium'))
    
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
    # Seed cache from existing context-aware outputs if present (modified side only; original key requires pronoun+candidates)
    seed_pattern = f"../results/coref/{MODEL_NAME}-{CONFIG_NAME}-context-aware-0shot-*.csv"
    seeded = 0
    for out_csv in glob.glob(seed_pattern):
        try:
            df_prev = pd.read_csv(out_csv)
        except Exception:
            continue
        # Modified predictions (context-aware key): needs text, pronoun, candidates, mod type
        if {'text','pronoun','candidates','raw_output','reasoning','modification_type'}.issubset(df_prev.columns):
            for _, r in df_prev.iterrows():
                mtext = remove_space(str(r.get('text','')))
                mpron = str(r.get('pronoun',''))
                mcand = str(r.get('candidates',''))
                mtype = str(r.get('modification_type',''))
                mkey = f"{mtext}|{mpron}|{mcand}_{mtype}"
                if mtext and mpron and mcand and mtype and mkey not in prediction_cache:
                    prediction_cache[mkey] = {'content': str(r.get('raw_output','')), 'reasoning': str(r.get('reasoning',''))}
                    seeded += 1
    if seeded:
        print(f"Seeded cache with {seeded} entries from previous outputs")
    os.makedirs('../results/coref', exist_ok=True)
    
    # Test modifications with context-aware prompts
    json_files = glob.glob('../../../data/modified_data/coref/*_100.json')
    
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
        results_mod = evaluate_modified_set(resolver, data, prediction_cache, modification_type)
        
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
        output_file = f'../results/coref/{MODEL_NAME}-{CONFIG_NAME}-context-aware-0shot-{mod_name}.csv'
        df_mod.to_csv(output_file, index=False)
        
        print(f"Accuracy: {mod_accuracy:.3f}")
        print(f"Saved to: {output_file}")
        
        time.sleep(2)  # Rate limiting
    
    print(f"\n{'='*60}")
    print(f"FLUKE Coreference Resolution with Context-Aware Prompts Complete!")
    print(f"{'='*60}")
    print(f"Processed {len(json_files)} modifications with context information")
    print(f"Cache size: {len(prediction_cache)} unique predictions")
    print(f"Files saved in: ../results/coref/")
    print(f"OpenAI GPT-5 Responses API endpoint")
    print(f"Model ID: {MODEL_ID}")
    print("Context-aware prompts inform the model about text modifications")

if __name__ == "__main__":
    main()
