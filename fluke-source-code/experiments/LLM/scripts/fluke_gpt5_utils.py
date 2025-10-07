#!/usr/bin/env python3
"""
FLUKE GPT-5 Utilities
OpenAI GPT-5 API client and utility functions for FLUKE experiments.
"""

import ast
import json
import re
import pandas as pd
from typing import List, Dict, Any, Tuple
import openai
from openai import OpenAI

# GPT-5 Model configurations (Based on OpenAI's latest model guide)
GPT5_MODELS = {
    'gpt-5': 'gpt-5',  # GPT-5 full model
    'gpt-5-mini': 'gpt-5-mini',  # GPT-5 Mini (faster, cheaper)
    'gpt-5-nano': 'gpt-5-nano',  # GPT-5 Nano (fastest, cheapest)
    'gpt-5-chat': 'gpt-5-chat-latest'  # GPT-5 Chat (non-reasoning)
}

GPT5_CONFIGS = {
    'standard': {
        'model': 'gpt-5',
        'description': 'GPT-5 with medium reasoning - best for complex tasks',
        'reasoning_effort': 'medium',
        'temperature': 1.0,
        'max_tokens': 4096
    },
    'minimal': {
        'model': 'gpt-5',
        'description': 'GPT-5 with minimal reasoning (no chat) - fast, deterministic parsing',
        'reasoning_effort': 'minimal',
        'temperature': 1.0,
        'max_tokens': 4096
    },
    'fast': {
        'model': 'gpt-5-mini', 
        'description': 'GPT-5 Mini with low reasoning - faster and cheaper',
        'reasoning_effort': 'low',
        'temperature': 1.0,
        'max_tokens': 4096
    },
    'fastest': {
        'model': 'gpt-5-nano',
        'description': 'GPT-5 Nano with minimal reasoning - fastest and cheapest',
        'reasoning_effort': 'minimal',
        'temperature': 1.0,
        'max_tokens': 4096
    },
    'high_reasoning': {
        'model': 'gpt-5',
        'description': 'GPT-5 with high reasoning - for most complex tasks',
        'reasoning_effort': 'high',
        'temperature': 1.0,
        'max_tokens': 4096
    },
    'chat': {
        'model': 'gpt-5-chat-latest',
        'description': 'GPT-5 Chat - non-reasoning model for simple tasks',
        'reasoning_effort': 'minimal',
        'temperature': 1.0,
        'max_tokens': 4096
    }
}

class GPT5Client:
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate(self, prompt: str, max_tokens: int = 4096, temperature: float = 1.0, reasoning_effort: str = "medium") -> Dict[str, str]:
        """Generate response from OpenAI GPT-5 Responses API"""
        try:
            # GPT-5 uses the Responses API with correct format from documentation
            params = {
                "model": self.model,
                "input": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "reasoning": {
                    "effort": reasoning_effort,
                    "summary": "auto"  # Request reasoning summary
                },
                "text": {"verbosity": "medium"},
                # Ensure long enough completions
                "max_output_tokens": max_tokens
            }

            # Use the Responses API endpoint for GPT-5
            response = self.client.responses.create(**params)
            
            # Parse the new GPT-5 Responses API structure
            content = ""
            reasoning_content = ""
            
            # The response.output is an array of items
            if hasattr(response, 'output') and response.output:
                for item in response.output:
                    # Extract content from message items
                    if hasattr(item, 'type') and item.type == 'message':
                        if hasattr(item, 'content') and item.content:
                            # Content is an array of content items
                            for content_item in item.content:
                                if hasattr(content_item, 'text'):
                                    content += content_item.text
                    
                    # Extract reasoning from reasoning items
                    elif hasattr(item, 'type') and item.type == 'reasoning':
                        # Extract reasoning from summary field
                        if hasattr(item, 'summary') and item.summary:
                            reasoning_parts = []
                            for summary_item in item.summary:
                                if hasattr(summary_item, 'text'):
                                    reasoning_parts.append(summary_item.text)
                            if reasoning_parts:
                                reasoning_content = " ".join(reasoning_parts)
                        
                        # Fallback: check content field if no summary
                        elif hasattr(item, 'content') and item.content:
                            if isinstance(item.content, str):
                                reasoning_content = item.content
                            elif hasattr(item.content, '__iter__'):
                                content_parts = []
                                for content_item in item.content:
                                    if hasattr(content_item, 'text'):
                                        content_parts.append(content_item.text)
                                    elif isinstance(content_item, str):
                                        content_parts.append(content_item)
                                if content_parts:
                                    reasoning_content = " ".join(content_parts)
            
            # Fallback: check for legacy structure
            if not content and hasattr(response, 'output_text'):
                content = response.output_text or ""
            
            # If no reasoning found, check if it's embedded in content
            if not reasoning_content and content:
                if any(word in content.lower() for word in ["reasoning:", "because", "therefore", "let me think", "step by step", "analysis:"]):
                    reasoning_content = "Reasoning embedded in response content"
            
            # Debug info
            available_fields = []
            debug_reasoning_info = {}
            try:
                available_fields = [attr for attr in dir(response) if not attr.startswith('_') and not callable(getattr(response, attr))]
                
                # Debug the new output structure
                debug_reasoning_info['has_output'] = bool(hasattr(response, 'output') and response.output)
                debug_reasoning_info['output_items_count'] = len(response.output) if hasattr(response, 'output') and response.output else 0
                
                if hasattr(response, 'output') and response.output:
                    output_types = []
                    reasoning_items_found = 0
                    message_items_found = 0
                    
                    for item in response.output:
                        if hasattr(item, 'type'):
                            output_types.append(item.type)
                            if item.type == 'reasoning':
                                reasoning_items_found += 1
                                if hasattr(item, 'summary') and item.summary:
                                    debug_reasoning_info['reasoning_summary_items'] = len(item.summary)
                            elif item.type == 'message':
                                message_items_found += 1
                    
                    debug_reasoning_info['output_types'] = output_types
                    debug_reasoning_info['reasoning_items_found'] = reasoning_items_found
                    debug_reasoning_info['message_items_found'] = message_items_found
                
                debug_reasoning_info['reasoning_content_length'] = len(reasoning_content) if reasoning_content else 0
                debug_reasoning_info['content_length'] = len(content) if content else 0
                        
            except Exception as e:
                available_fields = ["debug_failed"]
                debug_reasoning_info = {"error": str(e)}
            
            return {
                "content": content,
                "reasoning": reasoning_content,
                "debug_fields": available_fields,
                "debug_reasoning": debug_reasoning_info,
                "reasoning_effort": reasoning_effort
            }
            
        except Exception as e:
            print(f"OpenAI GPT-5 API Error: {e}")
            # Fallback to Chat Completions API if Responses API fails
            try:
                print("Falling back to Chat Completions API...")
                params = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "reasoning_effort": reasoning_effort,
                    "verbosity": "medium"
                }
                
                # Try max_completion_tokens first
                try:
                    params["max_completion_tokens"] = max_tokens
                    response = self.client.chat.completions.create(**params)
                except:
                    params["max_tokens"] = max_tokens
                    del params["max_completion_tokens"]
                    response = self.client.chat.completions.create(**params)
                
                message = response.choices[0].message
                return {
                    "content": message.content or "",
                    "reasoning": "Using Chat Completions API fallback",
                    "debug_fields": ["fallback_mode"],
                    "reasoning_effort": reasoning_effort
                }
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                return {"content": "", "reasoning": "", "debug_fields": ["both_apis_failed"], "reasoning_effort": reasoning_effort}

# Utility functions from original FLUKE code
def remove_space(text: str) -> str:
    """Remove extra spaces from text"""
    return ' '.join(text.split())

def extract_classification_prediction(text: str) -> str:
    """Extract binary classification prediction (0 or 1) from model output"""
    if not text:
        return "0"
    
    # Look for explicit 0 or 1 in the text
    matches = re.findall(r'\b[01]\b', text)
    if matches:
        return matches[-1]  # Return the last match
    
    # Fallback: look for keywords
    text_lower = text.lower()
    if any(word in text_lower for word in ['yes', 'true', 'contradiction', 'positive']):
        return "1"
    elif any(word in text_lower for word in ['no', 'false', 'no contradiction', 'negative']):
        return "0"
    
    # Default fallback
    return "0"

def extract_ner_prediction(text: str) -> List[Dict[str, str]]:
    """Extract NER predictions from model output"""
    if not text:
        return []
    
    try:
        # Look for JSON-like structures
        json_pattern = r'\[.*?\]'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                entities = ast.literal_eval(match)
                if isinstance(entities, list):
                    valid_entities = []
                    for entity in entities:
                        if isinstance(entity, dict) and 'text' in entity and 'value' in entity:
                            valid_entities.append({
                                'text': str(entity['text']),
                                'value': str(entity['value'])
                            })
                    if valid_entities:
                        return valid_entities
            except:
                continue
        
        # Fallback: try to parse line by line
        lines = text.split('\n')
        entities = []
        for line in lines:
            if ':' in line and any(ent_type in line.upper() for ent_type in ['PERSON', 'LOCATION', 'ORGANIZATION', 'ART', 'BUILDING', 'EVENT', 'OTHER', 'PRODUCT']):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    entity_text = parts[0].strip()
                    entity_type = parts[1].strip().upper()
                    entities.append({'text': entity_text, 'value': entity_type})
        
        if entities:
            return entities
            
    except Exception as e:
        print(f"Error parsing NER prediction: {e}")
    
    return []

def extract_answer_prediction(text: str) -> str:
    """Extract numerical answer from model output for math problems (CoT-aware)"""
    if not text:
        return "0"
    
    # Strategy 0: Look for #### pattern first (highest priority for GSM problems)
    hash_pattern = r'####\s*([+-]?\d+(?:\.\d+)?)'
    hash_matches = re.findall(hash_pattern, text)
    if hash_matches:
        return hash_matches[-1]  # Return the last #### answer found
    
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
            all_numbers = re.findall(r'([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)', remaining_text)
            if all_numbers:
                # Strategy A: If there's a percentage sign (%), prioritize the number before it
                percentage_match = re.search(r'(\d+(?:\.\d+)?)%', remaining_text)
                if percentage_match:
                    return percentage_match.group(1)
                
                # Strategy B: Look for numbers in the first sentence (more likely to be the answer)
                first_sentence = remaining_text.split('.')[0] if '.' in remaining_text else remaining_text.split('\n')[0]
                first_sentence_numbers = re.findall(r'([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)', first_sentence)
                if first_sentence_numbers:
                    # If there's only one number in the first sentence, it's likely the answer
                    if len(first_sentence_numbers) == 1:
                        return first_sentence_numbers[0].replace(',', '')
                    # If multiple numbers in first sentence, take the largest (handles cases like "24,500 fils")
                    try:
                        number_values = [(float(n.replace(',', '')), n.replace(',', '')) for n in first_sentence_numbers]
                        largest_in_sentence = max(number_values, key=lambda x: x[0])
                        return largest_in_sentence[1]
                    except:
                        return first_sentence_numbers[0].replace(',', '')
                
                # Strategy C: Fallback to largest number in remaining text (original logic)
                if len(all_numbers) > 1:
                    try:
                        number_values = [(float(n.replace(',', '')), n.replace(',', '')) for n in all_numbers]
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

def convert_string_to_entities(entity_string: str) -> List[Dict[str, str]]:
    """Convert string representation to entities list"""
    try:
        if isinstance(entity_string, str):
            return ast.literal_eval(entity_string)
        return entity_string
    except:
        return []

def calculate_f1_ent(gold_entities: List[Dict], pred_entities: List[Dict]) -> Tuple[float, float, float]:
    """Calculate F1 score for NER entities"""
    if not gold_entities and not pred_entities:
        return 1.0, 1.0, 1.0
    
    if not pred_entities:
        return 0.0, 0.0, 0.0
    
    if not gold_entities:
        return 0.0, 1.0, 0.0
    
    # Convert to sets of (text, value) tuples for comparison
    gold_set = set((ent['text'], ent['value']) for ent in gold_entities if isinstance(ent, dict))
    pred_set = set((ent['text'], ent['value']) for ent in pred_entities if isinstance(ent, dict))
    
    true_positives = len(gold_set & pred_set)
    
    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(gold_set) if gold_set else 0.0
    
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1_score

def append_person(data: List[Dict]) -> List[Dict]:
    """Add person labels to dialogue data"""
    for item in data:
        if 'dialog_context' in item:
            # Add person labels for dialogue turns
            item['dialog_context'] = [f"Person {i%2}: {turn}" for i, turn in enumerate(item['dialog_context'])]
    return data

def aggregate_results(result_files: List[str], task_name: str, model_name: str) -> pd.DataFrame:
    """Aggregate results from multiple modification files"""
    results = []
    
    for file_path in result_files:
        try:
            df = pd.read_csv(file_path)
            modification = file_path.split('/')[-1].replace('.csv', '').split('-')[-1]
            
            if task_name == 'named_entity_recognition':
                # Calculate F1 scores for NER
                original_f1s = []
                modified_f1s = []
                
                for _, row in df.iterrows():
                    # Original F1
                    try:
                        gold_entities = convert_string_to_entities(row['original_label'])
                        pred_entities = extract_ner_prediction(row['original_pred'])
                        _, _, orig_f1 = calculate_f1_ent(gold_entities, pred_entities)
                        original_f1s.append(orig_f1)
                    except:
                        original_f1s.append(0.0)
                    
                    # Modified F1
                    try:
                        gold_entities = convert_string_to_entities(row['modified_label'])
                        pred_entities = extract_ner_prediction(row['modified_pred'])
                        _, _, mod_f1 = calculate_f1_ent(gold_entities, pred_entities)
                        modified_f1s.append(mod_f1)
                    except:
                        modified_f1s.append(0.0)
                
                original_res = sum(original_f1s) / len(original_f1s) if original_f1s else 0.0
                modified_res = sum(modified_f1s) / len(modified_f1s) if modified_f1s else 0.0
            else:
                # Binary classification tasks
                original_correct = sum(1 for _, row in df.iterrows() if str(row['original_pred']) == str(row['original_label']))
                modified_correct = sum(1 for _, row in df.iterrows() if str(row['modified_pred']) == str(row['modified_label']))
                
                original_res = original_correct / len(df)
                modified_res = modified_correct / len(df)
            
            results.append({
                'modification': modification,
                'original_res': original_res,
                'modified_res': modified_res,
                'difference': original_res - modified_res,
                'samples': len(df)
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return pd.DataFrame(results)

def compare_models(model_files: Dict[str, str], task_name: str) -> pd.DataFrame:
    """Compare model performance"""
    results = []
    
    for model_name, file_path in model_files.items():
        try:
            df = pd.read_csv(file_path)
            
            if task_name == 'named_entity_recognition':
                f1_scores = []
                for _, row in df.iterrows():
                    try:
                        gold_entities = convert_string_to_entities(row['label'])
                        pred_entities = extract_ner_prediction(row['pred'])
                        _, _, f1 = calculate_f1_ent(gold_entities, pred_entities)
                        f1_scores.append(f1)
                    except:
                        f1_scores.append(0.0)
                
                performance = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
            else:
                correct = sum(1 for _, row in df.iterrows() if str(row['pred']) == str(row['label']))
                performance = correct / len(df) if len(df) > 0 else 0.0
            
            results.append({
                'model': model_name,
                'performance': performance,
                'samples': len(df)
            })
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
    
    return pd.DataFrame(results)
