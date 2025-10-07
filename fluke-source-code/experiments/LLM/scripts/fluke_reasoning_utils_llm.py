#!/usr/bin/env python3
"""
Enhanced version of fluke_reasoning_utils with LLM-based answer extraction fallback
"""

import re
import os
from typing import List, Dict
from llm_answer_extractor import LLMAnswerExtractor, create_llm_extractor

# Global LLM extractor instance (initialized lazily)
_llm_extractor = None

def get_llm_extractor():
    """Get or create the global LLM extractor instance"""
    global _llm_extractor
    if _llm_extractor is None:
        # Try to get API key from environment variable
        api_key = os.environ.get('OPENROUTER_API_KEY')
        if api_key:
            try:
                _llm_extractor = LLMAnswerExtractor(api_key)
                print("✅ LLM extractor initialized for robust answer parsing")
            except Exception as e:
                print(f"⚠️ Failed to initialize LLM extractor: {e}")
                _llm_extractor = None
        else:
            print("⚠️ OPENROUTER_API_KEY not set, using regex fallback only")
            _llm_extractor = None
    return _llm_extractor

def extract_answer_prediction_hybrid(text: str, question_text: str = "") -> str:
    """
    Hybrid answer extraction: Try LLM first, fallback to smart regex
    
    Args:
        text: The raw output text containing the solution
        question_text: The original question (optional, helps LLM understand context)
    
    Returns:
        String containing the extracted numerical answer
    """
    if not text:
        return "0"
    
    # Try LLM extraction first if available
    llm_extractor = get_llm_extractor()
    if llm_extractor and question_text:
        try:
            llm_result = llm_extractor.extract_answer(question_text, text)
            if llm_result and llm_result != "0":
                return llm_result
        except Exception as e:
            print(f"LLM extraction failed, using regex fallback: {e}")
    
    # Fallback to smart regex extraction
    return extract_answer_prediction_smart_regex(text)

def extract_answer_prediction_smart_regex(text: str) -> str:
    """Smart regex-based answer extraction (our previous improved logic)"""
    if not text:
        return "0"
    
    # Strategy 1: Look for explicit answer patterns
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
            
            # Look for emphasized numbers first
            emphasis_patterns = [
                # With currency symbols
                r'\*\*[$€£¥₹₽]([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*\w*\*\*',
                r'\*\*[$€£¥₹₽]([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\*\*',
                r'\*[$€£¥₹₽]([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*\w*\*',
                r'\*[$€£¥₹₽]([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\*',
                r'__[$€£¥₹₽]([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*\w*__',
                r'__[$€£¥₹₽]([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)__',
                r'_[$€£¥₹₽]([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*\w*_',
                r'_[$€£¥₹₽]([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)_',
                
                # Without currency symbols
                r'\*\*([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*\w*\*\*',
                r'\*\*([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\*\*',
                r'\*([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*\w*\*',
                r'\*([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\*',
                r'__([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*\w*__',
                r'__([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)__',
                r'_([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*\w*_',
                r'_([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)_'
            ]
            
            for emp_pattern in emphasis_patterns:
                emp_matches = re.findall(emp_pattern, remaining_text)
                if emp_matches:
                    return emp_matches[-1].replace(',', '')
            
            # Look for plain currency amounts
            currency_patterns = [r'[$€£¥₹₽]([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)']
            
            for curr_pattern in currency_patterns:
                curr_matches = re.findall(curr_pattern, remaining_text)
                if curr_matches:
                    try:
                        currency_values = [(float(n.replace(',', '')), n.replace(',', '')) for n in curr_matches]
                        largest_currency = max(currency_values, key=lambda x: x[0])
                        return largest_currency[1]
                    except:
                        return curr_matches[-1].replace(',', '')
            
            # Smart number selection strategy
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
                    if len(first_sentence_numbers) == 1:
                        return first_sentence_numbers[0].replace(',', '')
                    try:
                        number_values = [(float(n.replace(',', '')), n.replace(',', '')) for n in first_sentence_numbers]
                        largest_in_sentence = max(number_values, key=lambda x: x[0])
                        return largest_in_sentence[1]
                    except:
                        return first_sentence_numbers[0].replace(',', '')
                
                # Strategy C: Fallback to largest number in remaining text
                if len(all_numbers) > 1:
                    try:
                        number_values = [(float(n.replace(',', '')), n.replace(',', '')) for n in all_numbers]
                        largest_number = max(number_values, key=lambda x: x[0])
                        return largest_number[1]
                    except:
                        return all_numbers[-1].replace(',', '')
                else:
                    return all_numbers[0].replace(',', '')
    
    # Additional fallback strategies...
    # Strategy 2: Look for emphasized numbers in entire text
    emphasis_patterns = [
        r'\*\*([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\*\*',
        r'\*([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\*'
    ]
    
    for pattern in emphasis_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].replace(',', '')
    
    # Strategy 3: Look for currency amounts in entire text
    currency_matches = re.findall(r'[$€£¥₹₽]([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)', text)
    if currency_matches:
        return currency_matches[-1].replace(',', '')
    
    # Strategy 4: Look for percentage in entire text
    percentage_match = re.search(r'(\d+(?:\.\d+)?)%', text)
    if percentage_match:
        return percentage_match.group(1)
    
    # Strategy 5: Fall back to any number in the text (last occurrence)
    numbers = re.findall(r'([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)', text)
    if numbers:
        return numbers[-1].replace(',', '')
    
    return "0"

# Keep the original function name for backward compatibility
def extract_answer_prediction(text: str, question_text: str = "") -> str:
    """
    Main answer extraction function with LLM enhancement
    This is the function that should be called by other scripts
    """
    return extract_answer_prediction_hybrid(text, question_text)

# Other functions from the original file remain unchanged...
def extract_classification_prediction(text: str) -> str:
    """Extract classification prediction (unchanged)"""
    # ... existing implementation
    pass

def extract_ner_prediction(pred: str) -> List[Dict[str, str]]:
    """Extract NER prediction (unchanged)"""  
    # ... existing implementation
    pass

# ... other existing functions