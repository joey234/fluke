#!/usr/bin/env python3
"""
FLUKE GSM (Grade School Math) with GPT-5 (with modification context)
Enhanced version that provides information about text modifications to the model.
"""

import os
import json
import pandas as pd
import glob
import time
import random
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import FLUKE utilities
from fluke_gpt5_utils import GPT5Client, GPT5_MODELS, GPT5_CONFIGS, extract_step_by_step_reasoning
from fluke_reasoning_utils import (
    remove_space, extract_answer_prediction,
    aggregate_results, highlight_drops_and_significance,
    compare_models
)

# Load environment variables
load_dotenv()

# Modification type descriptions for GSM task
MODIFICATION_DESCRIPTIONS = {
    'active_to_passive': 'transformed from active to passive voice',
    'capitalization': 'modified to have different capitalization patterns',
    'casual': 'modified to use more casual/informal language',
    'concept_replacement': 'modified with concept replacements',
    'coordinating_conjunction': 'modified to include coordinating conjunctions',
    'dialectal': 'modified to use Singaporean English dialectal variations',
    'geographical_bias': 'modified to change some names or concepts to equivalent words from different geographical regions',
    'length_bias': 'modified to have different text length',
    'negation': 'modified to add or remove negation',
    'punctuation': 'modified to have different punctuation',
    'sentiment': 'modified to change sentiment',
    'temporal_bias': 'modified to use words that were commonly used in the past',
    'typo_bias': 'modified to contain typos'
}

class GSMSolverWithContext:
    def __init__(self, client: GPT5Client, reasoning_effort: str = "medium"):
        self.client = client
        self.reasoning_effort = reasoning_effort
        
        # Enhanced prompt template that includes modification context
        self.prompt_template = """Solve the following math word problem step by step.

Note: This problem has been {modification_description}. However, the mathematical solution should remain the same.

Problem: {text}

Let me work through this step by step.

End your response with #### followed by the final numerical answer only.
"""
        
        # Fallback template for when no modification context is available
        self.fallback_template = """Solve the following math word problem step by step.

Problem: {text}

Let me work through this step by step.

End your response with #### followed by the final numerical answer only.
"""
    
    def predict(self, text: str, modification_type: str = None) -> Dict[str, str]:
        """Predict answer for given math problem with optional modification context"""
        if modification_type and modification_type in MODIFICATION_DESCRIPTIONS:
            modification_desc = MODIFICATION_DESCRIPTIONS[modification_type]
            prompt = self.prompt_template.format(text=text, modification_description=modification_desc)
        else:
            prompt = self.fallback_template.format(text=text)
            
        response = self.client.generate(prompt, reasoning_effort=self.reasoning_effort)
        return response

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def get_modification_type_from_filename(filename: str) -> str:
    """Extract modification type from filename"""
    # Remove .jsonl and _100 suffixes
    mod_type = filename.replace('.jsonl', '').replace('_100', '')
    return mod_type

def evaluate_modified_set_with_context(
    solver: GSMSolverWithContext,
    data: List[Dict],
    modification_type: str,
    prediction_cache: Dict[str, Dict],
    max_samples: int = None,
    force_regenerate: bool = False,
) -> List[Dict]:
    """Evaluate on modified dataset with context awareness"""
    limited_data = data[:max_samples] if max_samples and len(data) > max_samples else data
    
    results = []
    for i, item in enumerate(limited_data):
        print(f"Processing modification {i+1}/{len(limited_data)}", end='\r')
        
        # Prepare text keys for caching
        original_text_clean = remove_space(item['text'])
        modified_text_clean = remove_space(item['modified'])
        
        # Create cache keys that include context information
        original_cache_key = f"{original_text_clean}_no_context"
        modified_cache_key = f"{modified_text_clean}_context_{modification_type}"
        
        # Helper: strict #### extractor (prefer last #### <number>)
        import re as _re
        def _extract_last_hash_num(s: str) -> str | None:
            if not s:
                return None
            m = None
            for m in _re.finditer(r"####\s*[$€£¥₹₽]?\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?|[+-]?\d+\s*/\s*\d+)", str(s)):
                pass
            return m.group(1).replace(',', '') if m else None

        # Get original prediction (without context)
        if (not force_regenerate) and (original_cache_key in prediction_cache):
            original_response = prediction_cache[original_cache_key]
            # Prefer strict #### parse over generic extractor
            original_pred_answer = _extract_last_hash_num(original_response['content']) or extract_answer_prediction(original_response['content'])
            original_raw_output = original_response['content']
            original_reasoning = original_response['reasoning']
        else:
            original_response = solver.predict(original_text_clean, modification_type=None)  # No context for original
            original_pred_answer = _extract_last_hash_num(original_response['content']) or extract_answer_prediction(original_response['content'])
            original_raw_output = original_response['content']
            original_reasoning = original_response['reasoning']
            # Cache for future use
            if not force_regenerate:
                prediction_cache[original_cache_key] = original_response
        
        # Get modified prediction (with context)
        if (not force_regenerate) and (modified_cache_key in prediction_cache):
            modified_response = prediction_cache[modified_cache_key]
            modified_pred_answer = _extract_last_hash_num(modified_response['content']) or extract_answer_prediction(modified_response['content'])
        else:
            modified_response = solver.predict(modified_text_clean, modification_type=modification_type)  # With context
            modified_pred_answer = _extract_last_hash_num(modified_response['content']) or extract_answer_prediction(modified_response['content'])
            # Cache for future use
            if not force_regenerate:
                prediction_cache[modified_cache_key] = modified_response
        
        # Map gold answers depending on modification type (negation-aware)
        mod_type = str(modification_type or '')
        if mod_type.startswith('negation_change'):
            original_answer = str(item.get('original_answer', item.get('short_answer', '')))
            modified_answer = str(item.get('short_answer', original_answer))
        elif mod_type.startswith('negation'):
            original_answer = str(item.get('original_answer', item.get('short_answer', '')))
            modified_answer = original_answer
        else:
            original_answer = str(item.get('short_answer', ''))
            modified_answer = original_answer

        result = {
            'text': modified_text_clean,
            'original_text': original_text_clean,
            'modified_answer': modified_answer,
            'original_answer': original_answer,
            'modified_pred': modified_pred_answer,
            'original_pred': original_pred_answer,
            'raw_output': modified_response['content'],
            'reasoning': modified_response['reasoning'],  # API reasoning tokens
            'step_by_step_reasoning': extract_step_by_step_reasoning(modified_response['content']),  # CoT reasoning
            'original_raw_output': original_raw_output,
            'original_reasoning': original_reasoning,
            'original_step_by_step_reasoning': extract_step_by_step_reasoning(original_raw_output),
            'modification_type': modification_type,
            'context_provided': modification_type in MODIFICATION_DESCRIPTIONS,
            'index': item.get('index', i),
            'type': item.get('type', ''),
            'negation_subtype': item.get('negation_subtype', item.get('type', ''))
        }
        
        results.append(result)
        
        # Rate limiting
        time.sleep(0.1)
    
    print()  # New line after progress
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='FLUKE GSM with GPT-5 (Context-Aware)')
    parser.add_argument('--config', '-c', default='standard', 
                        choices=['standard', 'minimal', 'fast', 'fastest', 'chat'],
                        help='Configuration to use (default: standard; minimal = gpt-5 with minimal reasoning)')
    parser.add_argument('--mod', default='all', help="GSM modification name (e.g., negation_100). Use 'all' to run every *_100.jsonl in data/modified_data/gsm. You may also pass a path to a single .jsonl file.")
    parser.add_argument('--force_regenerate', action='store_true', help='Ignore any cached predictions and regenerate all outputs')
    args = parser.parse_args()
    
    # Configuration
    CONFIG_NAME = args.config
    config = GPT5_CONFIGS[CONFIG_NAME]
    MODEL_NAME = config['model']
    MODEL_ID = GPT5_MODELS[MODEL_NAME]
    
    print(f"Configuration: {CONFIG_NAME}")
    print(f"Model: {MODEL_NAME} ({MODEL_ID})")
    print(f"Description: {config['description']}")
    print("🔍 Context-Aware Mode: Providing modification context to the model")
    
    # Get API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        return
    
    # Initialize client and solver with reasoning effort from config
    client = GPT5Client(openai_api_key, MODEL_ID)
    solver = GSMSolverWithContext(client, reasoning_effort=config.get('reasoning_effort', 'medium'))
    
    # Test with sample data first
    print("\nTesting context-aware prompting...")
    test_problem = "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market. How many eggs does Janet sell per day?"
    
    # Test without context
    test_response_no_context = solver.predict(test_problem, modification_type=None)
    print(f"Without context: {extract_answer_prediction(test_response_no_context['content'])}")
    print(f"Reasoning available (no context): {bool(test_response_no_context.get('reasoning'))}")
    
    # Test with context
    test_response_with_context = solver.predict(test_problem, modification_type='active_to_passive')
    print(f"With context: {extract_answer_prediction(test_response_with_context['content'])}")
    print(f"Reasoning available (with context): {bool(test_response_with_context.get('reasoning'))}")
    if test_response_with_context.get('reasoning'):
        print(f"Context reasoning sample: {test_response_with_context['reasoning'][:100]}...")
    else:
        print("No reasoning content found")
    
    # Initialize prediction cache for reusing predictions across modifications
    prediction_cache = {}
    # Seed cache from existing context-aware outputs if present (unless force regenerating)
    if not args.force_regenerate:
        seed_pattern = f"../results/gsm/{MODEL_NAME}-{CONFIG_NAME}-context-aware-0shot-*.csv"
        seeded = 0
        for out_csv in glob.glob(seed_pattern):
            try:
                df_prev = pd.read_csv(out_csv)
            except Exception:
                continue
            # Original predictions
            if {'original_text','original_raw_output','original_reasoning'}.issubset(df_prev.columns):
                for _, r in df_prev.iterrows():
                    key = remove_space(str(r.get('original_text',''))) + "_no_context"
                    if key and key not in prediction_cache:
                        prediction_cache[key] = {'content': str(r.get('original_raw_output','')), 'reasoning': str(r.get('original_reasoning',''))}
                        seeded += 1
            # Modified predictions (context-aware key)
            if {'text','raw_output','reasoning','modification_type'}.issubset(df_prev.columns):
                for _, r in df_prev.iterrows():
                    mtext = remove_space(str(r.get('text','')))
                    mtype = str(r.get('modification_type',''))
                    mkey = f"{mtext}_context_{mtype}"
                    if mtext and mtype and mkey not in prediction_cache:
                        prediction_cache[mkey] = {'content': str(r.get('raw_output','')), 'reasoning': str(r.get('reasoning',''))}
                        seeded += 1
        if seeded:
            print(f"Seeded cache with {seeded} entries from previous outputs")
    os.makedirs('../results/gsm', exist_ok=True)
    
    # Test modifications with context-aware caching approach
    if args.mod and args.mod.lower() != 'all':
        # Accept direct .jsonl path or a mod name
        if args.mod.endswith('.jsonl') and os.path.exists(args.mod):
            jsonl_files = [args.mod]
        else:
            base = '../../../data/modified_data/gsm'
            candidate = os.path.join(base, args.mod if args.mod.endswith('.jsonl') else args.mod + '.jsonl')
            if not os.path.exists(candidate):
                # Try stem variant if user passed just 'negation' (append _100)
                alt = os.path.join(base, f"{args.mod}_100.jsonl")
                jsonl_files = [alt] if os.path.exists(alt) else []
            else:
                jsonl_files = [candidate]
        if not jsonl_files:
            print(f"Error: could not resolve modification '{args.mod}' to a dataset file.")
            return
    else:
        jsonl_files = glob.glob('../../../data/modified_data/gsm/*_100.jsonl')
    
    print(f"\nTesting {len(jsonl_files)} modification(s) with context awareness...")
    
    for jsonl_file in jsonl_files:
        filename = jsonl_file.split('/')[-1]
        modification_type = get_modification_type_from_filename(filename)
        
        print(f"\nProcessing: {filename}")
        print(f"Modification type: {modification_type}")
        
        if modification_type in MODIFICATION_DESCRIPTIONS:
            print(f"Context: '{MODIFICATION_DESCRIPTIONS[modification_type]}'")
        else:
            print("⚠️  No context description available for this modification type")
        
        data = load_jsonl(jsonl_file)
        
        # Evaluate all samples using context-aware prediction cache
        results_mod = evaluate_modified_set_with_context(
            solver, data, modification_type, prediction_cache,
            force_regenerate=args.force_regenerate
        )
        
        # Save results with context indicator
        df_mod = pd.DataFrame(results_mod)
        print(f"DataFrame columns: {list(df_mod.columns)}")
        print(f"Context provided: {results_mod[0]['context_provided'] if results_mod else 'N/A'}")
        print(f"Sample reasoning content: {df_mod['reasoning'].iloc[0] if 'reasoning' in df_mod.columns else 'No reasoning column'}")
        mod_name = filename.replace('.jsonl', '')
        output_file = f'../results/gsm/{MODEL_NAME}-{CONFIG_NAME}-context-aware-0shot-{mod_name}.csv'
        df_mod.to_csv(output_file, index=False)
        
        print(f"Processed {len(results_mod)} samples")
        print(f"Saved to: {output_file}")
        
        time.sleep(2)  # Rate limiting
    
    print(f"\n{'='*60}")
    print(f"FLUKE GSM with Context-Aware {MODEL_NAME} Complete!")
    print(f"{'='*60}")
    print(f"Processed {len(jsonl_files)} modifications with context awareness")
    print(f"Cache size: {len(prediction_cache)} unique predictions")
    print(f"Files saved in: ../results/gsm/")
    print(f"OpenAI API endpoint: https://api.openai.com/v1")
    print(f"Model ID: {MODEL_ID}")
    print("🔍 Context-aware mode provides modification information to improve robustness")

if __name__ == "__main__":
    main()
