#!/usr/bin/env python3
"""
Final analysis and mapping for negation_change_100.jsonl.

It appears this file contains 200 unique samples, potentially all modified with negation changes.
Let's create a proper index mapping to find original questions from the GSM dataset.
"""

import json
import pandas as pd
import glob
from typing import Dict, List

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file"""
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except FileNotFoundError:
        pass
    return data

def create_comprehensive_gsm_index():
    """Create a comprehensive index of all GSM questions from all modification files"""
    
    # Load all GSM modification files to build a comprehensive original question index
    gsm_files = glob.glob('../../../data/modified_data/gsm/*.jsonl')
    
    original_questions = {}
    
    for file_path in gsm_files:
        if 'negation_change' in file_path:
            continue  # Skip the negation_change file itself
            
        print(f"Processing {file_path.split('/')[-1]}...")
        data = load_jsonl(file_path)
        
        for item in data:
            if 'index' in item and 'text' in item:
                idx = item['index']
                if idx not in original_questions:
                    original_questions[idx] = {
                        'original_question': item['text'],
                        'short_answer': item['short_answer'],
                        'source_file': file_path.split('/')[-1]
                    }
    
    print(f"Built index of {len(original_questions)} original questions")
    return original_questions

def analyze_negation_change_file():
    """Analyze the negation_change file and map to original questions"""
    
    # Load the negation change file
    negation_change = load_jsonl('../../../data/modified_data/gsm/negation_change_100.jsonl')
    print(f"Loaded {len(negation_change)} samples from negation_change_100.jsonl")
    
    # Build comprehensive original question index
    original_questions = create_comprehensive_gsm_index()
    
    # Map each negation_change sample
    results = []
    found_originals = 0
    
    for pos, item in enumerate(negation_change):
        idx = item['index']
        
        result = {
            'position': pos,
            'index': idx,
            'negation_question': item['question'],
            'negation_answer': item['answer'],
            'short_answer': item['short_answer'],
            'original_question': None,
            'found_original': False,
            'source_file': None
        }
        
        # Try to find original question
        if idx in original_questions:
            orig = original_questions[idx]
            result.update({
                'original_question': orig['original_question'],
                'found_original': True,
                'source_file': orig['source_file']
            })
            found_originals += 1
        
        results.append(result)
    
    print(f"Found original questions for {found_originals}/{len(negation_change)} samples")
    
    # Save comprehensive results
    df = pd.DataFrame(results)
    output_file = '../results/gsm/negation_change_final_mapping.csv'
    df.to_csv(output_file, index=False)
    
    # Show some examples where we found originals
    print("\\nExamples with original questions found:")
    print("=" * 60)
    
    found_examples = df[df['found_original'] == True].head(3)
    for _, row in found_examples.iterrows():
        print(f"\\nIndex {row['index']} (Position {row['position']}):")
        print(f"Original:  {row['original_question'][:80]}...")
        print(f"Negation:  {row['negation_question'][:80]}...")
        print(f"Answer:    {row['short_answer']}")
        print(f"Same?:     {row['original_question'] == row['negation_question']}")
    
    # Check for actual differences
    same_questions = 0
    different_questions = 0
    
    for _, row in df[df['found_original'] == True].iterrows():
        if row['original_question'] == row['negation_question']:
            same_questions += 1
        else:
            different_questions += 1
    
    print(f"\\nQuestion Analysis:")
    print(f"Same as original: {same_questions}")
    print(f"Different from original: {different_questions}")
    
    if different_questions > 0:
        print("\\nExamples of actual differences:")
        different_examples = df[(df['found_original'] == True) & 
                               (df['original_question'] != df['negation_question'])].head(2)
        
        for _, row in different_examples.iterrows():
            print(f"\\nIndex {row['index']}:")
            print(f"Original:  {row['original_question']}")
            print(f"Modified:  {row['negation_question']}")
            print(f"Answer:    {row['short_answer']}")
    
    # Statistics
    print(f"\\nFinal Statistics:")
    print(f"Total samples in negation_change: {len(results)}")
    print(f"Found originals: {found_originals} ({found_originals/len(results)*100:.1f}%)")
    print(f"Missing originals: {len(results) - found_originals}")
    print(f"Questions identical to original: {same_questions}")
    print(f"Questions modified from original: {different_questions}")
    
    print(f"\\nResults saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    results = analyze_negation_change_file()