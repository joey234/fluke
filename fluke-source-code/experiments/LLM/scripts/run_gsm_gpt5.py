#!/usr/bin/env python3
"""
FLUKE GSM (Grade School Math) with OpenAI GPT-5 (without DSPy)
Direct OpenAI GPT-5 API implementation for math word problem robustness testing.
"""

import os
import json
import pandas as pd
import glob
import time
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import FLUKE utilities
from fluke_gpt5_utils import (
    remove_space, extract_answer_prediction, extract_step_by_step_reasoning,
    aggregate_results, compare_models,
    GPT5_MODELS, GPT5_CONFIGS, GPT5Client
)

# Load environment variables
load_dotenv()

class GSMSolver:
    def __init__(self, client: GPT5Client, reasoning_effort: str = "medium"):
        self.client = client
        self.reasoning_effort = reasoning_effort
        self.prompt_template = """Solve the following math word problem step by step.

Problem: {text}

Let me work through this step by step:

End your response with #### followed by the final numerical answer only.

"""
    
    def predict(self, text: str) -> Dict[str, str]:
        """Predict answer for given math problem"""
        prompt = self.prompt_template.format(text=text)
        response = self.client.generate(prompt, reasoning_effort=self.reasoning_effort)
        return response

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def evaluate_modified_set(solver: GSMSolver, data: List[Dict], prediction_cache: Dict[str, Dict], max_samples: int = None) -> List[Dict]:
    """Evaluate on modified dataset with prediction caching"""
    limited_data = data[:max_samples] if max_samples and len(data) > max_samples else data
    
    results = []
    for i, item in enumerate(limited_data):
        print(f"Processing modification {i+1}/{len(limited_data)}", end='\r')
        
        # Prepare text keys for caching
        original_text_clean = remove_space(item['text'])
        modified_text_clean = remove_space(item['modified'])
        
        # Get original prediction (from cache or new prediction)
        if original_text_clean in prediction_cache:
            original_response = prediction_cache[original_text_clean]
            original_pred_answer = extract_answer_prediction(original_response['content'])
            original_raw_output = original_response['content']
            original_reasoning = original_response['reasoning']
        else:
            original_response = solver.predict(original_text_clean)
            original_pred_answer = extract_answer_prediction(original_response['content'])
            original_raw_output = original_response['content']
            original_reasoning = original_response['reasoning']
            # Cache for future use
            prediction_cache[original_text_clean] = original_response
        
        # Get modified prediction (from cache or new prediction)
        if modified_text_clean in prediction_cache:
            modified_response = prediction_cache[modified_text_clean]
            modified_pred_answer = extract_answer_prediction(modified_response['content'])
        else:
            modified_response = solver.predict(modified_text_clean)
            modified_pred_answer = extract_answer_prediction(modified_response['content'])
            # Cache for future use
            prediction_cache[modified_text_clean] = modified_response
        
        # Negation subtype handling (for evaluation/reporting)
        neg_subtype = ''
        expected_flip = None
        tval = item.get('type', '')
        if isinstance(tval, str) and tval.startswith('negation_'):
            neg_subtype = tval.split('_', 1)[1]
            expected_flip = (neg_subtype not in ['approximate', 'double'])

        negation_flip_correct = None
        if expected_flip is not None:
            original_label = str(item.get('original_answer', item.get('short_answer', ''))).strip()
            mp = str(modified_pred_answer).strip()
            norm = lambda s: s.replace(',', '').strip()
            negation_flip_correct = (norm(mp) != norm(original_label)) if expected_flip else (norm(mp) == norm(original_label))

        result = {
            'text': modified_text_clean,
            'original_text': original_text_clean,
            'modified_answer': item['short_answer'],
            'original_answer': item['short_answer'],  # GSM modifications don't change answers
            'modified_pred': modified_pred_answer,
            'original_pred': original_pred_answer,
            'raw_output': modified_response['content'],
            'reasoning': modified_response['reasoning'],  # API reasoning tokens
            'step_by_step_reasoning': extract_step_by_step_reasoning(modified_response['content']),  # CoT reasoning
            'original_raw_output': original_raw_output,
            'original_reasoning': original_reasoning,
            'original_step_by_step_reasoning': extract_step_by_step_reasoning(original_raw_output),
            'index': item.get('index', i),
            'type': item.get('type', ''),  # Include negation type for evaluation
            'negation_subtype': neg_subtype,
            'negation_expected_flip': expected_flip if expected_flip is not None else '',
            'negation_flip_correct': negation_flip_correct if negation_flip_correct is not None else ''
        }
        
        results.append(result)
        
        # Rate limiting
        time.sleep(0.1)
    
    print()  # New line after progress
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='FLUKE GSM with GPT-5')
    parser.add_argument('--config', '-c', default='standard', 
                        choices=['standard', 'fast', 'fastest', 'chat'],
                        help='Configuration to use (default: standard)')
    parser.add_argument('--mod', default='all', help="GSM modification name (e.g., negation_100). Use 'all' to run every *.jsonl in data/modified_data/gsm")
    parser.add_argument('--dataset', type=str, help='Path to a single GSM JSONL dataset file')
    parser.add_argument('--outputs_root', type=str, default='../results/gsm', help='Directory to write result CSVs')
    args = parser.parse_args()
    
    # Configuration
    CONFIG_NAME = args.config
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
    
    # Initialize client and solver with reasoning effort from config
    client = GPT5Client(openai_api_key, MODEL_ID)
    solver = GSMSolver(client, reasoning_effort=config.get('reasoning_effort', 'medium'))
    
    # Test with sample data first
    print("\nTesting with sample GSM problem...")
    test_problem = "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market. How many eggs does Janet sell per day?"
    test_response = solver.predict(test_problem)
    test_pred_answer = extract_answer_prediction(test_response['content'])
    print(f"Problem: {test_problem}")
    print(f"Prediction: {test_pred_answer}")
    print(f"Raw output: {test_response['content']}")
    print(f"Reasoning available: {bool(test_response.get('reasoning'))}")
    if test_response.get('reasoning'):
        print(f"Full Reasoning: {test_response['reasoning']}")
    else:
        print("No reasoning content found")
    
    # Initialize prediction cache for reusing predictions across modifications
    prediction_cache = {}
    os.makedirs(args.outputs_root, exist_ok=True)
    
    # Determine which datasets to run
    if args.dataset:
        jsonl_files = [args.dataset]
    elif args.mod and args.mod.lower() != 'all':
        candidate = f"../../../data/modified_data/gsm/{args.mod if args.mod.endswith('.jsonl') else args.mod + '.jsonl'}"
        if not os.path.exists(candidate):
            candidate2 = f"../../../data/modified_data/gsm/{args.mod}_100.jsonl"
            jsonl_files = [candidate2]
        else:
            jsonl_files = [candidate]
    else:
        jsonl_files = glob.glob('../../../data/modified_data/gsm/*_100.jsonl')
    
    print(f"\nTesting {len(jsonl_files)} modifications...")
    processed_mods = []
    
    for jsonl_file in jsonl_files:
        print(f"\nProcessing: {jsonl_file.split('/')[-1]}")
        
        data = load_jsonl(jsonl_file)
        
        # Evaluate all samples using prediction cache
        results_mod = evaluate_modified_set(solver, data, prediction_cache)
        
        # Save results
        df_mod = pd.DataFrame(results_mod)
        print(f"DataFrame columns: {list(df_mod.columns)}")
        print(f"Sample reasoning content: {df_mod['reasoning'].iloc[0] if 'reasoning' in df_mod.columns else 'No reasoning column'}")
        mod_name = jsonl_file.split('/')[-1].replace('.jsonl', '')
        output_file = f"{args.outputs_root.rstrip('/')}/{MODEL_NAME}-{CONFIG_NAME}-0shot-{mod_name}.csv"
        df_mod.to_csv(output_file, index=False)
        
        print(f"Processed {len(results_mod)} samples")
        print(f"Saved to: {output_file}")
        processed_mods.append(mod_name)
        
        time.sleep(2)  # Rate limiting
    
    print(f"\n{'='*60}")
    print(f"FLUKE GSM with {MODEL_NAME} Complete!")
    print(f"{'='*60}")
    print(f"Processed {len(jsonl_files)} modifications with caching")
    print(f"Cache size: {len(prediction_cache)} unique predictions")
    print(f"Files saved in: ../results/gsm/")
    print(f"OpenAI API endpoint: https://api.openai.com/v1")
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
