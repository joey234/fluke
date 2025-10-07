#!/usr/bin/env python3
"""
FLUKE Dialogue Contradiction Detection with GPT-5 (with modification context)
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
    compare_models, append_person
)

# Load environment variables
load_dotenv()

# Modification type descriptions
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

class DialogueContradictionDetectorWithContext:
    def __init__(self, client: GPT5Client, reasoning_effort: str = "medium"):
        self.client = client
        self.reasoning_effort = reasoning_effort
        
        # Enhanced prompt template that includes modification context
        self.prompt_template = """Given a dialogue with agent labels (agent 0 and agent 1 alternating), determine if the last utterance contradicts the dialogue context. 

Note: This text has been {modification_description}.

Answer with 1 if it contradicts, 0 if it does not contradict.

Dialogue:
{dialogue}

Answer:"""
        
        # Fallback template for when no modification context is available
        self.fallback_template = """Given a dialogue with agent labels (agent 0 and agent 1 alternating), determine if the last utterance contradicts the dialogue context. Answer with 1 if it contradicts, 0 if it does not contradict.

Dialogue:
{dialogue}

Answer:"""
    
    def predict(self, dialogue: str, modification_type: str = None) -> Dict[str, str]:
        """Predict dialogue contradiction for given dialogue with optional modification context"""
        if modification_type and modification_type in MODIFICATION_DESCRIPTIONS:
            modification_desc = MODIFICATION_DESCRIPTIONS[modification_type]
            prompt = self.prompt_template.format(dialogue=dialogue, modification_description=modification_desc)
        else:
            prompt = self.fallback_template.format(dialogue=dialogue)
            
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

def evaluate_modified_set(detector: DialogueContradictionDetectorWithContext, data: List[Dict], prediction_cache: Dict[str, Dict], modification_type: str) -> List[Dict]:
    """Evaluate on modified dataset with prediction caching and modification context"""
    
    results = []
    for i, item in enumerate(data):
        print(f"Processing modification {i+1}/{len(data)}", end='\r')
        
        # Ensure item is a valid dictionary with required keys
        if not isinstance(item, dict) or 'dialog_context' not in item:
            continue
        
        # Create modified and original dialogues
        original_dialogue = add_agent_labels(item['dialog_context'] + [item['original_text']])
        modified_dialogue = add_agent_labels(item['dialog_context'] + [item['modified_text']])
        
        # Create cache keys
        original_dialogue_clean = remove_space(original_dialogue)
        modified_dialogue_clean = remove_space(modified_dialogue)
        
        # Get original prediction (from cache or new prediction) - no modification context
        if original_dialogue_clean in prediction_cache:
            print(f"\n✓ CACHE HIT: Using cached original prediction")
            original_response = prediction_cache[original_dialogue_clean]
            original_pred_label = extract_classification_prediction(original_response['content'])
            original_raw_output = original_response['content']
            original_reasoning = original_response['reasoning']
        else:
            print(f"\n→ CACHE MISS: Generating new original prediction (no context)")
            original_response = detector.predict(original_dialogue)  # No modification context for original
            original_pred_label = extract_classification_prediction(original_response['content'])
            original_raw_output = original_response['content']
            original_reasoning = original_response['reasoning']
            # Cache for future use
            prediction_cache[original_dialogue_clean] = original_response
        
        # Get modified prediction (from cache or new prediction) - with modification context
        cache_key_with_context = f"{modified_dialogue_clean}_{modification_type}"
        if cache_key_with_context in prediction_cache:
            print(f"✓ Using cached modified prediction ({modification_type})")
            modified_response = prediction_cache[cache_key_with_context]
            modified_pred_label = extract_classification_prediction(modified_response['content'])
        else:
            print(f"→ Generating new modified prediction with context ({modification_type})")
            modified_response = detector.predict(modified_dialogue, modification_type)  # Include modification context
            modified_pred_label = extract_classification_prediction(modified_response['content'])
            # Cache for future use
            prediction_cache[cache_key_with_context] = modified_response
        
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
            'modification_type': modification_type,
            'modification_description': MODIFICATION_DESCRIPTIONS.get(modification_type, 'unknown'),
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
    
    # Initialize client and detector
    client = GPT5Client(openai_api_key, MODEL_ID)
    detector = DialogueContradictionDetectorWithContext(client, reasoning_effort=config.get('reasoning_effort', 'medium'))
    
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
    # Seed cache from existing context-aware outputs if present
    seed_pattern = f"../results/dialogue/{MODEL_NAME}-{CONFIG_NAME}-context-aware-0shot-*.csv"
    seeded = 0
    for out_csv in glob.glob(seed_pattern):
        try:
            df_prev = pd.read_csv(out_csv)
        except Exception:
            continue
        # Original predictions
        if {'original_dialog','original_raw_output','original_reasoning'}.issubset(df_prev.columns):
            for _, r in df_prev.iterrows():
                key = str(r.get('original_dialog',''))
                if key and key not in prediction_cache:
                    prediction_cache[key] = {'content': str(r.get('original_raw_output','')), 'reasoning': str(r.get('original_reasoning',''))}
                    seeded += 1
        # Modified predictions (context-aware key)
        if {'modified_dialog','raw_output','reasoning','modification_type'}.issubset(df_prev.columns):
            for _, r in df_prev.iterrows():
                mtext = str(r.get('modified_dialog',''))
                mtype = str(r.get('modification_type',''))
                mkey = f"{mtext}_{mtype}"
                if mtext and mtype and mkey not in prediction_cache:
                    prediction_cache[mkey] = {'content': str(r.get('raw_output','')), 'reasoning': str(r.get('reasoning',''))}
                    seeded += 1
    if seeded:
        print(f"Seeded cache with {seeded} entries from previous outputs")
    os.makedirs('../results/dialogue', exist_ok=True)
    
    # Test modifications with context-aware prompts
    json_files = glob.glob('../../../data/modified_data/dialogue/*_100.json')
    
    print(f"\nTesting {len(json_files)} modifications with context-aware prompts...")
    
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
        
        # Extract modification type from filename
        modification_type = json_file.split('/')[-1].replace('_100.json', '')
        
        print(f"Extracted {len(data)} samples with modification type: {modification_type}")
        print(f"Description: {MODIFICATION_DESCRIPTIONS.get(modification_type, 'unknown modification')}")
        
        # Evaluate all samples using prediction cache with modification context
        results_mod = evaluate_modified_set(detector, data, prediction_cache, modification_type)
        
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
        output_file = f'../results/dialogue/{MODEL_NAME}-{CONFIG_NAME}-context-aware-0shot-{mod_name}.csv'
        df_mod.to_csv(output_file, index=False)
        
        print(f"Accuracy: {mod_accuracy:.3f}")
        print(f"Saved to: {output_file}")
        
        time.sleep(2)  # Rate limiting
    
    print(f"\n{'='*60}")
    print(f"FLUKE Dialogue Contradiction Detection with Context-Aware Prompts Complete!")
    print(f"{'='*60}")
    print(f"Processed {len(json_files)} modifications with context information")
    print(f"Cache size: {len(prediction_cache)} unique predictions")
    print(f"Files saved in: ../results/dialogue/")
    print(f"OpenAI GPT-5 Responses API endpoint")
    print(f"Model ID: {MODEL_ID}")
    print("Context-aware prompts inform the model about text modifications")

if __name__ == "__main__":
    main()
