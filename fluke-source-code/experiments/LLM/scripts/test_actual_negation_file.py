#!/usr/bin/env python3
"""
Test processing the actual negation_change_100.jsonl file
"""

import json
import os
import sys

# Import the functions from run_gsm_openrouter
sys.path.append('.')

def load_jsonl(file_path: str):
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def test_negation_file_processing():
    """Test that the negation_change file loads correctly"""
    
    negation_file = '../../../data/modified_data/gsm/negation_change_100.jsonl'
    
    if not os.path.exists(negation_file):
        print(f"File not found: {negation_file}")
        return
    
    print(f"Loading {negation_file}...")
    data = load_jsonl(negation_file)
    
    print(f"Loaded {len(data)} samples")
    
    # Test the logic we added to the main script
    file_name = negation_file
    is_negation_change = 'negation_change' in file_name.lower()
    
    print(f"Is negation_change file: {is_negation_change}")
    
    # Process first few samples
    for i, item in enumerate(data[:3]):
        print(f"\nSample {i+1}:")
        print(f"  Index: {item['index']}")
        print(f"  Has 'text': {'text' in item}")
        print(f"  Has 'modified': {'modified' in item}")
        print(f"  Has 'short_answer': {'short_answer' in item}")
        print(f"  Has 'original_answer': {'original_answer' in item}")
        
        # Test the answer processing logic
        if is_negation_change:
            modified_answer = item['short_answer']
            original_answer = item.get('original_answer', item['short_answer'])
        else:
            modified_answer = item['short_answer']
            original_answer = item['short_answer']
        
        print(f"  Processed original_answer: {original_answer}")
        print(f"  Processed modified_answer: {modified_answer}")
        print(f"  Original text: {item['text'][:60]}...")
        print(f"  Modified text: {item['modified'][:60]}...")

if __name__ == "__main__":
    test_negation_file_processing()