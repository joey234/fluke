#!/usr/bin/env python3
"""
FLUKE GSM (Grade School Math) with OpenRouter API (without DSPy)
Direct OpenRouter API implementation for math word problem robustness testing.
"""

import os
import json
import pandas as pd
import glob
import time
import re
import ast
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv
import requests

# Import FLUKE utilities
from fluke_reasoning_utils import (
    remove_space, extract_answer_prediction, extract_step_by_step_reasoning,
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
                    data = response.json()
                except json.JSONDecodeError as je:
                    if attempt < 2:
                        time.sleep(1 + attempt)
                        continue
                    print(f"HTTP Error: JSON decode error: {je} | body preview: {response.text[:200]}")
                    return {"content": "", "reasoning": "", "debug_fields": ["error"], "model": self.model}
            except Exception as e:
                msg = str(e)
                if attempt < 2 and ("premature" in msg.lower() or "chunked" in msg.lower() or "timeout" in msg.lower() or isinstance(e, (requests.exceptions.ChunkedEncodingError, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError))):
                    time.sleep(1 + attempt)
                    continue
                print(f"HTTP Error: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_details = e.response.json()
                        print(f"Error details: {error_details}")
                    except:
                        try:
                            print(f"Response text: {e.response.text}")
                        except Exception:
                            pass
                return {"content": "", "reasoning": "", "debug_fields": ["error"], "model": self.model}
            choice = data['choices'][0]
            message = choice.get('message', {})
            content = message.get('content', '')
            
            # Extract reasoning if available
            reasoning = ""
            if use_reasoning:
                # For DeepSeek models - check message first (like coref implementation)
                if 'reasoning' in message:
                    reasoning = message.get('reasoning', '')
                elif 'reasoning' in choice:
                    # For o1-style models
                    reasoning_data = choice.get('reasoning', {})
                    if isinstance(reasoning_data, dict):
                        reasoning = reasoning_data.get('content', '')
                    else:
                        reasoning = str(reasoning_data)
                elif 'thinking' in choice:
                    # For some models that use 'thinking'
                    reasoning = choice.get('thinking', '')
            
            return {
                "content": content,
                "reasoning": reasoning,
                "debug_fields": ["model", "usage"] if "usage" in data else ["model"],
                "model": self.model
            }
        # fallback empty
        return {"content": "", "reasoning": "", "debug_fields": ["error"], "model": self.model}

class GSMSolver:
    def __init__(self, client: OpenRouterClient, use_reasoning: bool = True):
        self.client = client
        self.use_reasoning = use_reasoning
        self.prompt_template = """Solve the following math word problem step by step.

Problem: {text}

End your response with #### followed by the final numerical answer only.

"""
    
    def predict(self, text: str) -> Dict[str, str]:
        """Predict answer for given math problem"""
        prompt = self.prompt_template.format(text=text)
        response = self.client.generate(prompt, use_reasoning=self.use_reasoning)
        return response

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_existing_results(results_file_path: str) -> Dict[str, Dict]:
    """Load existing results and create lookup by modified text"""
    if not os.path.exists(results_file_path):
        return {}
    
    try:
        df = pd.read_csv(results_file_path)
        results_lookup = {}
        def _valid(x):
            try:
                return not pd.isna(x) and str(x).strip().lower() not in ("", "nan", "none", "null")
            except Exception:
                return False
        
        for _, row in df.iterrows():
            modified_text_clean = remove_space(row['text'])
            raw_out = row.get('raw_output', '')
            orig_raw = row.get('original_raw_output', '')
            # Only cache valid rows; invalid ones will be re-run
            if _valid(raw_out) and _valid(orig_raw):
                results_lookup[modified_text_clean] = {
                    'modified_pred': row.get('modified_pred', ''),
                    'original_pred': row.get('original_pred', ''),
                    'raw_output': raw_out,
                    'reasoning': row.get('reasoning', ''),
                    'original_raw_output': orig_raw,
                    'original_reasoning': row.get('original_reasoning', ''),
                    'index': row.get('index', -1)
                }
        
        return results_lookup
    except Exception as e:
        print(f"Error loading existing results: {e}")
        return {}

def evaluate_modified_set(solver: GSMSolver, data: List[Dict], prediction_cache: Dict[str, Dict], existing_results: Dict[str, Dict], max_samples: int = None, file_name: str = "", force_regenerate: bool = False) -> List[Dict]:
    """Evaluate on modified dataset with prediction caching and result comparison"""
    limited_data = data[:max_samples] if max_samples and len(data) > max_samples else data
    
    # Identify negation variants based on filename
    fname_lower = (file_name or '').lower()
    is_negation_change = 'negation_change' in fname_lower
    is_negation = ('negation' in fname_lower) and not is_negation_change
    
    # Filter data to only include samples that need to be run
    filtered_data = []
    reused_results = []
    
    for i, item in enumerate(limited_data):
        modified_text_clean = remove_space(item['modified'])
        
        # Check if we have existing results for this modified text (unless force regenerating)
        if not force_regenerate and modified_text_clean in existing_results:
            # Compare the modified text in results with the modified text in data
            # If they match exactly, we can reuse the existing result
            existing_result = existing_results[modified_text_clean].copy()
            # Handle different answer fields based on file type
            if is_negation_change:
                # negation_change: original answer provided in dataset, modified answer is short_answer
                original_answer = str(item.get('original_answer', item.get('short_answer', '')))
                modified_answer = str(item.get('short_answer', original_answer))
            elif is_negation:
                # negation: both sides use the original (unmodified) gold answer
                original_answer = str(item.get('original_answer', item.get('short_answer', '')))
                modified_answer = original_answer
            else:
                # other GSM mods: original/modified share the same short_answer
                original_answer = str(item.get('short_answer', ''))
                modified_answer = original_answer
            
            # Coerce possibly non-string cached fields to strings before extracting reasoning
            _raw_output_cached = existing_result.get('raw_output', '')
            _orig_raw_output_cached = existing_result.get('original_raw_output', '')

            existing_result.update({
                'text': modified_text_clean,
                'original_text': remove_space(item['text']),
                'modified_answer': modified_answer,
                'original_answer': original_answer,
                'step_by_step_reasoning': extract_step_by_step_reasoning(str(_raw_output_cached)),
                'original_step_by_step_reasoning': extract_step_by_step_reasoning(str(_orig_raw_output_cached)),
                'index': item.get('index', i),
                'type': item.get('type', '')  # Include negation type for evaluation
            })
            reused_results.append(existing_result)
            print(f"Reusing existing result for sample {i+1} (index: {item.get('index', i)})")
        else:
            # This sample needs to be processed
            filtered_data.append(item)
    
    print(f"Found {len(reused_results)} existing results to reuse")
    print(f"Need to process {len(filtered_data)} new samples")
    
    # Process only the filtered data that needs new predictions
    new_results = []
    for i, item in enumerate(filtered_data):
        print(f"Processing new sample {i+1}/{len(filtered_data)}", end='\r')
        
        # Prepare text keys for caching
        original_text_clean = remove_space(item['text'])
        modified_text_clean = remove_space(item['modified'])
        
        # Get original prediction (from cache or new prediction, unless force regenerating)
        if not force_regenerate and original_text_clean in prediction_cache:
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
        
        # Get modified prediction (from cache or new prediction, unless force regenerating)
        if not force_regenerate and modified_text_clean in prediction_cache:
            modified_response = prediction_cache[modified_text_clean]
            modified_pred_answer = extract_answer_prediction(modified_response['content'])
        else:
            modified_response = solver.predict(modified_text_clean)
            modified_pred_answer = extract_answer_prediction(modified_response['content'])
            # Cache for future use
            prediction_cache[modified_text_clean] = modified_response
        
        # Handle different answer fields based on file type
        if is_negation_change:
            original_answer = str(item.get('original_answer', item.get('short_answer', '')))
            modified_answer = str(item.get('short_answer', original_answer))
        elif is_negation:
            original_answer = str(item.get('original_answer', item.get('short_answer', '')))
            modified_answer = original_answer
        else:
            original_answer = str(item.get('short_answer', ''))
            modified_answer = original_answer
        
        # Negation subtype handling (for evaluation/reporting):
        neg_subtype = ''
        expected_flip = None
        tval = item.get('type', '')
        if isinstance(tval, str) and tval.startswith('negation_'):
            neg_subtype = tval.split('_', 1)[1]
            expected_flip = (neg_subtype not in ['approximate', 'double'])

        # Compute negation flip correctness relative to original label (answer)
        negation_flip_correct = None
        if expected_flip is not None:
            # Original label from dataset
            original_label = str(item.get('original_answer', item.get('short_answer', ''))).strip()
            mp = str(modified_pred_answer).strip()
            # Normalize simple numeric forms by removing commas/spaces
            norm = lambda s: s.replace(',', '').strip()
            negation_flip_correct = (norm(mp) != norm(original_label)) if expected_flip else (norm(mp) == norm(original_label))

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
            'index': item.get('index', i),
            'type': item.get('type', ''),  # Include negation type for evaluation
            'negation_subtype': neg_subtype,
            'negation_expected_flip': expected_flip if expected_flip is not None else '',
            'negation_flip_correct': negation_flip_correct if negation_flip_correct is not None else ''
        }
        
        new_results.append(result)
        
        # Rate limiting
        time.sleep(0.1)
    
    if len(filtered_data) > 0:
        print()  # New line after progress
    
    # Combine reused and new results, maintaining original order by index
    all_results = reused_results + new_results
    all_results.sort(key=lambda x: x.get('index', 0))
    
    return all_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='FLUKE GSM with OpenRouter')
    parser.add_argument('--config', '-c', default='gpt4o', 
                        choices=['gpt4o', 'claude', 'llama', 'deepseek', 'gpt5'],
                        help='Model configuration to use (default: gpt4o)')
    parser.add_argument('--mod', default='all', help="GSM modification name (e.g., negation_100). Use 'all' to run every *.jsonl in data/modified_data/gsm")
    parser.add_argument('--dataset', type=str, help='Path to a single GSM JSONL dataset file')
    parser.add_argument('--outputs_root', type=str, default='../results/gsm', help='Directory to write result CSVs')
    parser.add_argument('--force-regenerate', '-f', action='store_true',
                        help='Force regeneration of all results, ignoring existing cache')
    args = parser.parse_args()
    
    # Configuration - Handle DeepSeek separately since it's not in the standard configs
    CONFIG_NAME = args.config
    if CONFIG_NAME == 'deepseek':
        # DeepSeek configuration (similar to coref implementation)
        config = {
            'model': 'deepseek',
            'description': 'DeepSeek R1 - with reasoning tokens',
            'use_reasoning': True
        }
        MODEL_NAME = 'deepseek'
        MODEL_ID = 'deepseek/deepseek-r1'
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
    
    # Initialize client and solver
    client = OpenRouterClient(openrouter_api_key, MODEL_ID)
    solver = GSMSolver(client, use_reasoning=config.get('use_reasoning', True))
    
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
        # Also allow passing a base mod like 'negation_100' without .jsonl
        if not os.path.exists(candidate):
            # Try stem variant if user passed just 'negation' (append _100)
            candidate2 = f"../../../data/modified_data/gsm/{args.mod}_100.jsonl"
            jsonl_files = [candidate2]
        else:
            jsonl_files = [candidate]
    else:
        jsonl_files = glob.glob('../../../data/modified_data/gsm/*_100.jsonl')
    
    print(f"\nTesting {len(jsonl_files)} modifications...")
    processed_mods = []
    for jsonl_file in jsonl_files:
        # if 'geographical' not in jsonl_file and 'concept' not in jsonl_file and 'casual' not in jsonl_file:
        # if 'length' in jsonl_file or 'dialectal' in jsonl_file or 'temporal' in jsonl_file:
            # continue
        print(f"\nProcessing: {jsonl_file.split('/')[-1]}")
        
        data = load_jsonl(jsonl_file)
        
        # Load existing results if they exist (unless force regenerating)
        mod_name = jsonl_file.split('/')[-1].replace('.jsonl', '')
        output_file = f"{args.outputs_root.rstrip('/')}/{MODEL_NAME}-{CONFIG_NAME}-0shot-{mod_name}.csv"
        existing_results = {} if args.force_regenerate else load_existing_results(output_file)
        
        if args.force_regenerate:
            print("Force regenerate mode: ignoring existing results")
        else:
            print(f"Found {len(existing_results)} existing results in {output_file}")
        
        # Evaluate samples, comparing with existing results
        results_mod = evaluate_modified_set(solver, data, prediction_cache, existing_results, file_name=jsonl_file, force_regenerate=args.force_regenerate)
        
        # Save results
        df_mod = pd.DataFrame(results_mod)
        print(f"DataFrame columns: {list(df_mod.columns)}")
        print(f"Sample reasoning content: {df_mod['reasoning'].iloc[0] if 'reasoning' in df_mod.columns else 'No reasoning column'}")
        df_mod.to_csv(output_file, index=False)
        
        print(f"Processed {len(results_mod)} total samples ({len([r for r in results_mod if r.get('index', 0) not in [e.get('index', -1) for e in existing_results.values()]])} new)")
        print(f"Saved to: {output_file}")
        processed_mods.append(mod_name)
        
        time.sleep(2)  # Rate limiting
    
    print(f"\n{'='*60}")
    print(f"FLUKE GSM with {MODEL_NAME} Complete!")
    print(f"{'='*60}")
    print(f"Processed {len(jsonl_files)} modifications with caching")
    print(f"Cache size: {len(prediction_cache)} unique predictions")
    print(f"Files saved in: ../results/gsm/")
    print(f"OpenRouter API endpoint: {client.base_url}")
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
