"""
FLUKE O3 Reasoning Model Utilities
Consolidated utility functions for all o3 notebooks to ensure consistency
"""

import re
import ast
import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Any, Tuple, Optional

# ============================================================================
# Model Configuration
# ============================================================================

REASONING_MODELS = {
    'gpt4o': 'openai/gpt-4o-2024-11-20',
    'claude': 'anthropic/claude-3.5-sonnet',
    'llama': 'meta-llama/llama-3.1-405b-instruct',
    'deepseek': 'deepseek/deepseek-r1',
    'deepseek-lite': 'deepseek/deepseek-r1:free',
    'gpt5': 'openai/gpt-5',
}

REASONING_CONFIGS = {
    'gpt4o': {
        'model': 'gpt4o',
        'instruction_style': 'standard',
        'description': 'GPT-4o (2024-11-20) - no reasoning tokens',
        'use_reasoning': False
    },
    'claude': {
        'model': 'claude',
        'instruction_style': 'standard',
        'description': 'Claude 3.5 Sonnet - no reasoning tokens',
        'use_reasoning': False
    },
    'llama': {
        'model': 'llama',
        'instruction_style': 'standard',
        'description': 'Llama 3.1 405B - no reasoning tokens',
        'use_reasoning': False
    },
    'deepseek': {
        'model': 'deepseek',
        'instruction_style': 'strict',
        'description': 'DeepSeek R1 - reasoning tokens enabled',
        'use_reasoning': True
    },
    'deepseek-lite': {
        'model': 'deepseek-lite',
        'instruction_style': 'strict',
        'description': 'DeepSeek R1 Lite - faster, reasoning tokens enabled when available',
        'use_reasoning': True
    },
    'gpt5': {
        'model': 'gpt5',
        'instruction_style': 'standard',
        'description': 'OpenAI GPT-5 via OpenRouter',
        'use_reasoning': False
    }
}

# ============================================================================
# Text Processing Functions
# ============================================================================

def remove_space(text: str) -> str:
    """
    Clean up spacing and formatting in text.
    Standardized across all FLUKE tasks.
    """
    if not text:
        return ""
        
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Remove multiple spaces
        cleaned = ' '.join(line.split())
        
        # Fix spacing around punctuation
        cleaned = re.sub(r'\s+([.,!?:;])', r'\1', cleaned)
        cleaned = re.sub(r'([.,!?:;])\s+', r'\1 ', cleaned)
        
        # Fix contractions
        cleaned = re.sub(r'\s*\'\s*s\b', "'s", cleaned)
        cleaned = re.sub(r'\s*n\s*\'\s*t\b', "n't", cleaned)
        cleaned = re.sub(r'\s*\'\s*ve\b', "'ve", cleaned)
        cleaned = re.sub(r'\s*\'\s*re\b', "'re", cleaned)
        cleaned = re.sub(r'\s*\'\s*ll\b', "'ll", cleaned)
        cleaned = re.sub(r'\s*\'\s*d\b', "'d", cleaned)
        cleaned = re.sub(r'\s*\'\s*m\b', "'m", cleaned)
        
        # Fix spaces around parentheses
        cleaned = re.sub(r'\(\s+', '(', cleaned)
        cleaned = re.sub(r'\s+\)', ')', cleaned)
        
        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()
        cleaned_lines.append(cleaned)
        
    return '\n'.join(cleaned_lines)

# ============================================================================
# Prediction Extraction Functions
# ============================================================================

def extract_classification_prediction(text: str) -> str:
    """Extract binary classification prediction (0 or 1) from model output.
    Robust to extra text by searching for digits and falling back to keywords."""
    if not text:
        return "0"
    patterns = [
        r'Answer:\s*([01])',
        r'Final answer:\s*([01])',
        r'Label:\s*([01])',
        r'Prediction:\s*([01])',
        r'Classification:\s*([01])',
        r'\b([01])\b',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1]
    text_lower = text.lower()
    if any(w in text_lower for w in ['yes', 'true', 'contradiction', 'positive']):
        return '1'
    if any(w in text_lower for w in ['no', 'false', 'no contradiction', 'negative']):
        return '0'
    return '0'

def extract_ner_prediction(pred: str) -> List[Dict[str, str]]:
    """Extract NER predictions from model output.
    Supports fenced code blocks and multiline JSON lists of {text,value}."""
    if not pred:
        return []
    # Prefer fenced code blocks first
    fence = re.findall(r"```(?:json)?\n([\s\S]*?)```", pred)
    candidates = []
    if fence:
        candidates.extend(fence)
    # Also scan for bracketed list across lines
    m = re.findall(r"\[\s*\{[\s\S]*?\}\s*\]", pred)
    candidates.extend(m)
    for blob in candidates:
        try:
            data = ast.literal_eval(blob)
            if isinstance(data, list):
                normalized = []
                for e in data:
                    if isinstance(e, dict) and 'text' in e and 'value' in e:
                        normalized.append({'text': str(e['text']), 'value': str(e['value'])})
                if normalized:
                    return normalized
        except Exception:
            continue
    return []

def extract_answer_prediction(text: str) -> str:
    """Extract numerical answer from model output for math problems (CoT-aware).

    Priority rules:
    1) #### <number> (supports optional currency, commas, fractions, sci-notation)
    2) final answer/answer: <number> (same numeric support)
    3) Emphasized numbers/currencies near the answer cue
    4) Numbers at line starts/ends
    5) Fallback to last number
    """
    if not text:
        return "0"

    # Strategy 0: Look for #### pattern first (highest priority for GSM problems)
    # Allow optional currency symbols and robust numeric forms
    hash_pattern = r"####\s*[$€£¥₹₽]?\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?|[+-]?\d+\s*/\s*\d+)"
    hash_matches = re.findall(hash_pattern, text)
    if hash_matches:
        return hash_matches[-1].replace(',', ' ' if False else '').replace(',', '')
    
    # Strategy 1: Look for explicit answer patterns and prioritize emphasized numbers in remaining text
    answer_patterns = [
        (r'[Ff]inal\s+[Aa]nswer[:\s]*', 'final_answer'),
        (r'[Aa]nswer:\s*', 'answer_colon'),
        (r'[Aa]nswer\s*=\s*', 'answer_equals'), 
        (r'[Aa]nswer\s+is\s+', 'answer_is'),
        (r'[Tt]herefore,?\s*', 'therefore'),
        (r'[Ss]o\s+the\s+answer\s+is\s+', 'so_answer_is')
    ]
    
    for pattern, pattern_name in answer_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            # Take the last occurrence of this pattern
            last_match = matches[-1]
            remaining_text = text[last_match.end():]
            
            # IMPROVED: In the remaining text, prioritize emphasized numbers first
            # Look for emphasized numbers in the remaining text (handles **$975**, **26 marbles** format)
            emphasis_patterns = [
                # With currency symbols: **$975**, **€150 euros**, etc.
                r'\*\*[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\s*\w*\*\*',  # **$975 dollars**
                r'\*\*[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\*\*',        # **$975**
                r'\*[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\s*\w*\*',      # *$975 dollars*
                r'\*[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\*',            # *$975*
                r'__[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\s*\w*__',      # __$975 dollars__
                r'__[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)__',            # __$975__
                r'_[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\s*\w*_',        # _$975 dollars_
                r'_[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)_',              # _$975_
                
                # Without currency symbols (original patterns)
                r'\*\*([+-]?\d+(?:\.\d+)?)\s*\w*\*\*',  # **26 marbles**
                r'\*\*([+-]?\d+(?:\.\d+)?)\*\*',        # **26**
                r'\*([+-]?\d+(?:\.\d+)?)\s*\w*\*',      # *26 marbles*
                r'\*([+-]?\d+(?:\.\d+)?)\*',            # *26*
                r'__([+-]?\d+(?:\.\d+)?)\s*\w*__',      # __26 marbles__
                r'__([+-]?\d+(?:\.\d+)?)__',            # __26__
                r'_([+-]?\d+(?:\.\d+)?)\s*\w*_',        # _26 marbles_
                r'_([+-]?\d+(?:\.\d+)?)_'               # _26_
            ]
            
            for emp_pattern in emphasis_patterns:
                emp_matches = re.findall(emp_pattern, remaining_text)
                if emp_matches:
                    return emp_matches[-1]  # Return the last emphasized number
            
            # NEW: If no emphasized numbers, look for plain currency amounts (like $975, €150, £45)
            currency_patterns = [
                r'[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)',  # $975, €150, £45, etc.
            ]
            
            for curr_pattern in currency_patterns:
                curr_matches = re.findall(curr_pattern, remaining_text)
                if curr_matches:
                    # Return the largest currency amount found
                    try:
                        currency_values = [(float(n), n) for n in curr_matches]
                        largest_currency = max(currency_values, key=lambda x: x[0])
                        return largest_currency[1]
                    except:
                        return curr_matches[-1]  # Fallback to last match
            
            # If no currency amounts, use smart number selection strategy
            # Accept numbers with commas, decimals, or simple fractions
            all_numbers = re.findall(r'([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?|[+-]?\d+\s*/\s*\d+)', remaining_text)
            if all_numbers:
                # Strategy A: If there's a percentage sign (%), prioritize the number before it
                percentage_match = re.search(r'(\d+(?:\.\d+)?)%', remaining_text)
                if percentage_match:
                    return percentage_match.group(1)
                
                # Strategy B: Look for numbers in the first sentence (more likely to be the answer)
                first_sentence = remaining_text.split('.')[0] if '.' in remaining_text else remaining_text.split('\n')[0]
                first_sentence_numbers = re.findall(r'([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?|[+-]?\d+\s*/\s*\d+)', first_sentence)
                if first_sentence_numbers:
                    # If there's only one number in the first sentence, it's likely the answer
                    if len(first_sentence_numbers) == 1:
                        return first_sentence_numbers[0].replace(',', '')
                    # If multiple numbers in first sentence, take the largest (handles cases like "24,500 fils")
                    try:
                        def to_float(x):
                            x2 = x.replace(',', '')
                            if '/' in x2:
                                num, den = x2.split('/')
                                return float(num)/float(den) if float(den) != 0 else float('nan')
                            return float(x2)
                        number_values = [(to_float(n), n.replace(',', '')) for n in first_sentence_numbers]
                        largest_in_sentence = max(number_values, key=lambda x: x[0])
                        return largest_in_sentence[1]
                    except:
                        return first_sentence_numbers[0].replace(',', '')
                
                # Strategy C: Fallback to largest number in remaining text (original logic)
                if len(all_numbers) > 1:
                    try:
                        def to_float(x):
                            x2 = x.replace(',', '')
                            if '/' in x2:
                                num, den = x2.split('/')
                                return float(num)/float(den) if float(den) != 0 else float('nan')
                            return float(x2)
                        number_values = [(to_float(n), n.replace(',', '')) for n in all_numbers]
                        largest_number = max(number_values, key=lambda x: x[0])
                        return largest_number[1]
                    except:
                        return all_numbers[-1].replace(',', '')
                else:
                    return all_numbers[0].replace(',', '')
    
    # Strategy 2: Look for numbers in bold/emphasis in the entire text (as fallback)
    emphasis_patterns = [
        # With currency symbols
        r'\*\*[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\s*\w*\*\*',  # **$975 dollars**
        r'\*\*[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\*\*',        # **$975**
        r'\*[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\s*\w*\*',      # *$975 dollars*
        r'\*[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\*',            # *$975*
        r'__[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\s*\w*__',      # __$975 dollars__
        r'__[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)__',            # __$975__
        r'_[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\s*\w*_',        # _$975 dollars_
        r'_[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)_',              # _$975_
        
        # Without currency symbols
        r'\*\*([+-]?\d+(?:\.\d+)?)\s*\w*\*\*',  # **26 marbles**
        r'\*\*([+-]?\d+(?:\.\d+)?)\*\*',        # **26**
        r'\*([+-]?\d+(?:\.\d+)?)\s*\w*\*',      # *26 marbles*
        r'\*([+-]?\d+(?:\.\d+)?)\*',            # *26*
        r'__([+-]?\d+(?:\.\d+)?)\s*\w*__',      # __26 marbles__
        r'__([+-]?\d+(?:\.\d+)?)__',            # __26__
        r'_([+-]?\d+(?:\.\d+)?)\s*\w*_',        # _26 marbles_
        r'_([+-]?\d+(?:\.\d+)?)_'               # _26_
    ]
    
    for pattern in emphasis_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]  # Return the last emphasized number
    
    # NEW Strategy 2.5: Look for plain currency amounts in the entire text  
    currency_patterns = [
        r'[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)',  # $975, €150, £45, etc.
    ]
    
    for curr_pattern in currency_patterns:
        curr_matches = re.findall(curr_pattern, text)
        if curr_matches:
            # Return the last currency amount found (most likely to be the final answer)
            return curr_matches[-1]
    
    # Strategy 3: Look for standalone numbers at the beginning of lines (likely answers)
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line:
            # Check if line starts with a number (possibly the answer)
            start_number_match = re.match(r'^([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)', line)
            if start_number_match:
                return start_number_match.group(1).replace(',', '')
    
    # Strategy 4: Look for numbers at the end of the text (last line) - but prioritize larger numbers
    for line in reversed(lines):
        line = line.strip()
        if line:
            # Find all numbers in this line
            numbers = re.findall(r'([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)', line)
            if numbers:
                # Convert to float for comparison, return the largest number
                try:
                    number_values = [(float(n.replace(',', '')), n.replace(',', '')) for n in numbers]
                    largest_number = max(number_values, key=lambda x: x[0])
                    return largest_number[1]
                except:
                    return numbers[-1].replace(',', '')  # Fallback to last number
    
    # Strategy 5: Fall back to any number in the text (last occurrence)
    numbers = re.findall(r'([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)', text)
    if numbers:
        return numbers[-1].replace(',', '')
    
    return "0"  # Default if no number found

def extract_step_by_step_reasoning(text: str) -> str:
    """Extract the step-by-step reasoning from CoT response, excluding the final answer"""
    if not text:
        return ""
    
    # Split by lines and process
    lines = text.strip().split('\n')
    reasoning_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip lines that look like they contain the final answer
        if any(keyword in line.lower() for keyword in ['answer:', 'final answer:', 'therefore', 'so the answer is']):
            # Check if this line has reasoning before the answer
            answer_patterns = [
                r'[Aa]nswer:\s*[+-]?\d+(?:\.\d+)?',
                r'[Ff]inal\s+answer:\s*[+-]?\d+(?:\.\d+)?',
                r'[Tt]herefore,?\s*[+-]?\d+(?:\.\d+)?',
                r'[Ss]o\s+the\s+answer\s+is\s+[+-]?\d+(?:\.\d+)?'
            ]
            
            # Extract the part before the answer if it contains reasoning
            for pattern in answer_patterns:
                if re.search(pattern, line):
                    # Split at the answer and keep the reasoning part
                    reasoning_part = re.split(pattern, line)[0].strip()
                    if reasoning_part and len(reasoning_part) > 5:  # Only add if substantial
                        reasoning_lines.append(reasoning_part)
                    break
            else:
                # If no answer pattern matched but contains answer keywords, skip
                continue
        else:
            # Regular reasoning line
            reasoning_lines.append(line)
    
    return '\n'.join(reasoning_lines).strip()

# ============================================================================
# NER-Specific Functions
# ============================================================================

def convert_string_to_entities(entity_str: Any) -> List[Dict[str, str]]:
    """
    Convert string representation of entities to proper format.
    Handles various input formats consistently.
    """
    if isinstance(entity_str, str):
        try:
            # Convert string to list of dicts
            entities = ast.literal_eval(entity_str)
            
            # Handle nested lists by flattening
            if isinstance(entities, list):
                # Handle double nested lists
                if len(entities) > 0 and isinstance(entities[0], list):
                    entities = entities[0]
                    
                # Handle list of dicts with text/value format
                if len(entities) > 0 and isinstance(entities[0], dict):
                    # Handle format with text/value keys
                    if 'text' in entities[0]:
                        return entities
                        
                    # Handle format with single key-value pair
                    if len(entities[0]) == 1:
                        converted = []
                        for e in entities:
                            for text, value in e.items():
                                converted.append({'text': text, 'value': value})
                        return converted
                        
                    # Handle format with multiple key-value pairs
                    converted = []
                    for e in entities:
                        for text, value in e.items():
                            if isinstance(value, str):
                                converted.append({'text': text, 'value': value})
                    return converted
                    
            return entities if isinstance(entities, list) else []
        except:
            return []
    
    if isinstance(entity_str, list):
        return entity_str
        
    return []

def calculate_f1_ent(gold_entities: Any, predicted_entities: Any) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score for NER.
    Handles various input formats and edge cases.
    """
    if predicted_entities is None:
        return 0.0, 0.0, 0.0

    true_entities = {}
    pred_entities = {}
    
    # Convert to empty list if NaN
    def handle_nan(entities):
        if isinstance(entities, list):
            return entities
        # Handle pandas/numpy types
        if isinstance(entities, (pd.Series, np.ndarray)):
            nan_check = pd.isna(entities)
            if isinstance(nan_check, (pd.Series, np.ndarray)):
                if nan_check.any():
                    return []
            elif nan_check:
                return []
        # Handle single values
        elif pd.isna(entities):
            return []
        return entities

    # Handle NaN cases
    gold_entities = handle_nan(gold_entities)
    predicted_entities = handle_nan(predicted_entities)
            
    # Parse strings if needed
    if isinstance(gold_entities, str):
        try:
            gold_entities = ast.literal_eval(gold_entities)
        except:
            gold_entities = []
    if isinstance(predicted_entities, str):
        try:
            predicted_entities = ast.literal_eval(predicted_entities)
        except:
            predicted_entities = []

    # Process gold entities
    for entity in gold_entities:
        if isinstance(entity, str):
            try:
                entity = ast.literal_eval(entity)
            except:
                continue
        if isinstance(entity, dict):
            if entity.get('text') is not None:
                true_entities[entity['text']] = entity.get('value', 'UNKNOWN')
            else:
                for key, value in entity.items():
                    true_entities[key] = value
    
    # Process predicted entities
    for entity in predicted_entities:
        if isinstance(entity, str):
            try:
                entity = ast.literal_eval(entity)
            except:
                continue
        if isinstance(entity, dict):
            if entity.get('text') is not None:
                pred_entities[entity['text']] = entity.get('value', 'UNKNOWN')
            else:
                for key, value in entity.items():
                    pred_entities[key] = value

    # Calculate metrics
    true_positives = sum(1 for text in true_entities 
                        if text in pred_entities and true_entities[text] == pred_entities[text])
    false_positives = sum(1 for text in pred_entities if text not in true_entities)
    false_negatives = sum(1 for text in true_entities if text not in pred_entities)

    if true_positives == 0:
        return 0.0, 0.0, 0.0

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score

# ============================================================================
# Dialogue-Specific Functions
# ============================================================================

def append_person(ds: List[Dict]) -> List[Dict]:
    """
    Add agent labels to dialogue turns.
    Used for dialogue contradiction detection.
    """
    for i, sample in enumerate(ds):
        # Handle modified and original dialogues
        if 'dialog_context' in sample:
            modified_dialog = sample['dialog_context'] + [sample.get('modified_text', '')]
            original_dialog = sample['dialog_context'] + [sample.get('original_text', '')]
            
            # Add agent labels to modified dialogue
            for j, turn in enumerate(modified_dialog):
                turn = 'agent ' + str(j % 2) + ': ' + turn
                modified_dialog[j] = turn
                
            # Add agent labels to original dialogue
            for j, turn in enumerate(original_dialog):
                turn = 'agent ' + str(j % 2) + ': ' + turn
                original_dialog[j] = turn
                
            ds[i]['original_dialog'] = '\n'.join(original_dialog)
            ds[i]['original_dialog'] = remove_space(ds[i]['original_dialog'])
            ds[i]['modified_dialog'] = '\n'.join(modified_dialog)
            ds[i]['modified_dialog'] = remove_space(ds[i]['modified_dialog'])
            
    return ds

def append_person_dialogue(dialogue: str) -> str:
    """
    Add agent labels to dialogue turns.
    Used for dialogue contradiction detection.
    """
    return dialogue.replace('agent 0:', '').replace('agent 1:', '')


# ============================================================================
# Aggregation and Statistics Functions
# ============================================================================

def aggregate_results(result_files: List[str], task_name: str, model_name: str) -> pd.DataFrame:
    """
    Aggregate results across all modifications for any task.
    Returns a DataFrame with aggregated statistics.
    """
    aggregated_results = []
    
    for file in result_files:
        # Extract modification type from filename
        mod_type = file.split('-')[-1].replace('.csv', '')
        
        try:
            # Read results file
            df = pd.read_csv(file)
            
            # Handle missing columns gracefully
            if 'original_pred' not in df.columns or df['original_pred'].isna().all():
                print(f"Warning: {file} missing original predictions")
                continue
            
            # Task-specific accuracy calculation
            if task_name == 'named_entity_recognition':
                # NER uses F1 score
                all_original_labels = []
                all_original_preds = []
                all_modified_labels = []
                all_modified_preds = []
                
                for idx, row in df.iterrows():
                    original_label = convert_string_to_entities(row['original_label'])
                    original_pred = convert_string_to_entities(row['original_pred'])
                    modified_label = convert_string_to_entities(row['modified_label'])
                    modified_pred = convert_string_to_entities(row['modified_pred'])
                    
                    all_original_labels.extend(original_label)
                    all_original_preds.extend(original_pred)
                    all_modified_labels.extend(modified_label)
                    all_modified_preds.extend(modified_pred)
                
                original_precision, original_recall, original_score = calculate_f1_ent(
                    all_original_labels, all_original_preds)
                modified_precision, modified_recall, modified_score = calculate_f1_ent(
                    all_modified_labels, all_modified_preds)
            else:
                # Other tasks use accuracy
                original_correct = (df['original_pred'] == df['original_label']).sum()
                modified_correct = (df['modified_pred'] == df['modified_label']).sum()
                total = len(df)
                
                if total == 0:
                    continue
                    
                original_score = original_correct / total
                modified_score = modified_correct / total
            
            # Calculate differences
            difference = -round(original_score - modified_score, 3)
            
            # Calculate percentage difference
            if original_score > 0:
                pct_difference = -round((original_score - modified_score) / original_score * 100, 2)
            else:
                pct_difference = 0
            
            # Perform t-test if we have enough data
            try:
                t_stat, p_value = stats.ttest_ind(
                    (df['original_pred'] == df['original_label']).astype(float),
                    (df['modified_pred'] == df['modified_label']).astype(float)
                )
            except:
                p_value = None
            
            result_dict = {
                'task': task_name,
                'model': model_name,
                'modification': mod_type,
                'original_res': round(original_score, 3),
                'modified_res': round(modified_score, 3),
                'difference': difference,
                'pct_difference': pct_difference,
                'p_value': p_value,
                'samples': len(df)
            }
            
            # Add NER-specific metrics
            if task_name == 'named_entity_recognition':
                result_dict.update({
                    'original_precision': round(original_precision, 3),
                    'original_recall': round(original_recall, 3),
                    'modified_precision': round(modified_precision, 3),
                    'modified_recall': round(modified_recall, 3)
                })
            
            aggregated_results.append(result_dict)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    if not aggregated_results:
        return pd.DataFrame()
    
    # Create results dataframe
    results_df = pd.DataFrame(aggregated_results)
    
    # Sort by modification type
    modification_order = [
        'temporal_bias_100', 'geographical_bias_100', 'length_bias_100',
        'typo_bias_100', 'capitalization_100', 'punctuation_100',
        'derivation_100', 'compound_word_100', 'active_to_passive_100',
        'grammatical_role_100', 'coordinating_conjunction_100',
        'concept_replacement_100', 'negation_100', 'discourse_100',
        'sentiment_100', 'casual_100', 'dialectal_100', 'singlish_100'
    ]
    
    existing_mods = results_df['modification'].unique()
    modification_order = [mod for mod in modification_order if mod in existing_mods]
    
    results_df['modification'] = pd.Categorical(
        results_df['modification'], categories=modification_order, ordered=True)
    results_df = results_df.sort_values(by='modification')
    
    # Calculate averages
    avg_original = results_df['original_res'].mean()
    avg_modified = results_df['modified_res'].mean()
    avg_difference = avg_original - avg_modified
    avg_pct_difference = results_df['pct_difference'].mean()
    
    avg_row = {
        'task': task_name,
        'model': model_name,
        'modification': 'average',
        'original_res': round(avg_original, 3),
        'modified_res': round(avg_modified, 3),
        'difference': -round(avg_difference, 3),
        'pct_difference': round(avg_pct_difference, 2),
        'p_value': None,
        'samples': results_df['samples'].sum()
    }
    
    # Add NER-specific averages
    if task_name == 'named_entity_recognition' and 'original_precision' in results_df.columns:
        avg_row.update({
            'original_precision': round(results_df['original_precision'].mean(), 3),
            'original_recall': round(results_df['original_recall'].mean(), 3),
            'modified_precision': round(results_df['modified_precision'].mean(), 3),
            'modified_recall': round(results_df['modified_recall'].mean(), 3)
        })
    
    # Add average row
    results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)
    
    return results_df

def highlight_drops_and_significance(row: pd.Series) -> List[str]:
    """
    Styling function to highlight performance drops and significant p-values.
    Returns list of style strings for DataFrame styling.
    """
    colors = [''] * len(row)
    if row['original_res'] > row['modified_res']:
        colors = ['background-color: red'] * len(row)
        # If p-value < 0.05, add bold text
        if 'p_value' in row and row['p_value'] is not None and row['p_value'] < 0.05:
            colors = ['background-color: red; font-weight: bold'] * len(row)
    return colors

# ============================================================================
# Model Comparison Functions
# ============================================================================

def compare_models(comparison_files: Dict[str, str], task_name: str) -> pd.DataFrame:
    """
    Compare performance across different models for a given task.
    Returns a DataFrame with model comparison statistics.
    """
    model_scores = {}
    
    for model_name, file_path in comparison_files.items():
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path)
            
            if task_name == 'named_entity_recognition':
                # Calculate F1 scores for NER
                all_labels = []
                all_preds = []
                
                for idx, row in df.iterrows():
                    labels = convert_string_to_entities(row.get('label', []))
                    preds = convert_string_to_entities(row.get('pred', []))
                    all_labels.extend(labels)
                    all_preds.extend(preds)
                
                precision, recall, f1_score = calculate_f1_ent(all_labels, all_preds)
                model_scores[model_name] = {
                    'score': f1_score,
                    'precision': precision,
                    'recall': recall,
                    'metric': 'F1 Score'
                }
            else:
                # Calculate accuracy for other tasks
                pred_col = 'pred' if 'pred' in df.columns else 'prediction'
                if pred_col in df.columns and 'label' in df.columns:
                    accuracy = (df[pred_col] == df['label']).mean()
                    model_scores[model_name] = {
                        'score': accuracy,
                        'metric': 'Accuracy'
                    }
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    if not model_scores:
        return pd.DataFrame()
    
    # Create comparison dataframe
    comparison_data = []
    for model, scores in model_scores.items():
        row = {
            'Model': model,
            scores['metric']: scores['score'],
            'Performance': f"{scores['score']:.1%}"
        }
        if 'precision' in scores:
            row['Precision'] = scores['precision']
            row['Recall'] = scores['recall']
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    metric_col = list(model_scores.values())[0]['metric']
    comparison_df = comparison_df.sort_values(metric_col, ascending=False)
    
    return comparison_df

# ============================================================================
# Export all functions for notebook use
# ============================================================================

__all__ = [
    # Configuration
    'REASONING_MODELS',
    'REASONING_CONFIGS',
    
    # Text processing
    'remove_space',
    
    # Prediction extraction
    'extract_classification_prediction',
    'extract_ner_prediction',
    
    # NER functions
    'convert_string_to_entities',
    'calculate_f1_ent',
    
    # Dialogue functions
    'append_person',
    
    # Aggregation and statistics
    'aggregate_results',
    'highlight_drops_and_significance',
    
    # Model comparison
    'compare_models'
]
