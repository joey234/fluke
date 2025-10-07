#!/usr/bin/env python3
"""
Test the updated negation_change handling logic
"""

import json

# Mock the functions we need to test the logic
def remove_space(text):
    return text.strip()

def extract_answer_prediction(content):
    # Simple mock - extract last number
    words = content.split()
    for word in reversed(words):
        try:
            return int(word.replace(',', ''))
        except:
            continue
    return "N/A"

def extract_step_by_step_reasoning(content):
    return content[:100] + "..." if len(content) > 100 else content

# Test data mimicking the negation_change structure
test_data = [
    {
        "index": 109,
        "text": "Original question about gas station with cashback",
        "modified": "Modified question about gas station WITHOUT cashback", 
        "short_answer": "30",
        "original_answer": "28"
    },
    {
        "index": 448,
        "text": "Original question about salesman selling sneakers",
        "modified": "Modified question about salesman NOT selling sneakers",
        "short_answer": "199", 
        "original_answer": "539"
    }
]

# Test the logic for negation_change file
def test_negation_change_logic():
    print("Testing negation_change file handling...")
    
    for i, item in enumerate(test_data):
        # This is the logic from our updated script
        is_negation_change = True  # Simulating 'negation_change' in filename
        
        if is_negation_change:
            modified_answer = item['short_answer']
            original_answer = item.get('original_answer', item['short_answer'])
        else:
            modified_answer = item['short_answer']
            original_answer = item['short_answer']
        
        print(f"\nSample {i+1} (Index {item['index']}):")
        print(f"  Original text: {item['text']}")
        print(f"  Modified text: {item['modified']}")
        print(f"  Original answer: {original_answer}")
        print(f"  Modified answer: {modified_answer}")
        print(f"  Answers different: {original_answer != modified_answer}")

# Test the logic for regular GSM file
def test_regular_gsm_logic():
    print("\n" + "="*50)
    print("Testing regular GSM file handling...")
    
    regular_data = {
        "index": 100,
        "text": "Original question",
        "modified": "Modified question with same answer",
        "short_answer": "42"
        # No 'original_answer' field
    }
    
    is_negation_change = False  # Simulating regular GSM file
    
    if is_negation_change:
        modified_answer = regular_data['short_answer']
        original_answer = regular_data.get('original_answer', regular_data['short_answer'])
    else:
        modified_answer = regular_data['short_answer']
        original_answer = regular_data['short_answer']  # Same answer for regular GSM
    
    print(f"Original answer: {original_answer}")
    print(f"Modified answer: {modified_answer}")
    print(f"Answers different: {original_answer != modified_answer}")

if __name__ == "__main__":
    test_negation_change_logic()
    test_regular_gsm_logic()