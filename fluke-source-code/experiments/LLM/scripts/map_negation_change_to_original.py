#!/usr/bin/env python3
"""
Map indices from negation_change_100.jsonl to find the ORIGINAL questions and answers.
The negation_change file contains modified questions, we need to find the originals.
"""

import json
import pandas as pd
from typing import Dict, List, Optional

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def find_original_gsm_questions(negation_change_file: str) -> Dict[int, Dict]:
    """
    Find original GSM questions by looking up indices in other GSM modification files.
    The negation_change file has modified questions, we need to find the originals.
    """
    # Load negation_change data (these are MODIFIED questions)
    negation_data = load_jsonl(negation_change_file)
    
    # Try to find original questions from other modification files that have both text and modified fields
    gsm_files = [
        '../../../data/modified_data/gsm/geographical_bias_100.jsonl',
        '../../../data/modified_data/gsm/casual_100.jsonl',
        '../../../data/modified_data/gsm/concept_replacement_100.jsonl',
        '../../../data/modified_data/gsm/active_to_passive_100.jsonl',
        '../../../data/modified_data/gsm/capitalization_100.jsonl'
    ]
    
    # Create a master lookup of index -> original question
    original_questions_lookup = {}
    
    for gsm_file in gsm_files:
        try:
            print(f"Loading {gsm_file.split('/')[-1]}...")
            data = load_jsonl(gsm_file)
            for item in data:
                if 'index' in item and 'text' in item:
                    index = item['index']
                    if index not in original_questions_lookup:
                        original_questions_lookup[index] = {
                            'original_question': item['text'],
                            'short_answer': item['short_answer'],
                            'source_file': gsm_file.split('/')[-1]
                        }
        except FileNotFoundError:
            print(f"File not found: {gsm_file}")
            continue
    
    print(f"Found {len(original_questions_lookup)} original questions from reference files")
    
    # Now map the negation_change data to original questions
    mapping = {}
    found_count = 0
    
    for pos, item in enumerate(negation_data):
        index = item['index']
        
        negation_entry = {
            'position_in_negation_file': pos,
            'modified_question': item['question'],  # This is the modified question
            'modified_answer': item['answer'],
            'short_answer': item['short_answer'],
            'original_question': None,
            'found_original': False
        }
        
        # Look up the original question using the index
        if index in original_questions_lookup:
            original = original_questions_lookup[index]
            negation_entry.update({
                'original_question': original['original_question'],
                'found_original': True,
                'source_file': original['source_file']
            })
            found_count += 1
        
        mapping[index] = negation_entry
    
    print(f"Successfully mapped {found_count}/{len(negation_data)} questions to originals")
    
    return mapping

def main():
    negation_change_file = '../../../data/modified_data/gsm/negation_change_100.jsonl'
    
    print("Mapping negation_change questions to original GSM questions...")
    print("=" * 70)
    
    mapping = find_original_gsm_questions(negation_change_file)
    
    # Show examples
    print("\nExamples of Original vs Modified Questions:")
    print("=" * 50)
    
    count = 0
    for index in sorted(mapping.keys()):
        if mapping[index]['found_original'] and count < 3:
            item = mapping[index]
            print(f"\nIndex {index}:")
            print(f"Original:  {item['original_question'][:100]}...")
            print(f"Modified:  {item['modified_question'][:100]}...")
            print(f"Answer:    {item['short_answer']}")
            count += 1
    
    # Save detailed mapping to CSV
    df_data = []
    for index in sorted(mapping.keys()):
        item = mapping[index]
        df_data.append({
            'index': index,
            'position_in_negation_file': item['position_in_negation_file'],
            'found_original': item['found_original'],
            'original_question': item.get('original_question', 'NOT FOUND'),
            'modified_question': item['modified_question'],
            'short_answer': item['short_answer'],
            'source_file': item.get('source_file', 'N/A')
        })
    
    df = pd.DataFrame(df_data)
    output_file = '../results/gsm/negation_change_original_mapping.csv'
    df.to_csv(output_file, index=False)
    
    # Statistics
    found_originals = df['found_original'].sum()
    total_samples = len(df)
    
    print(f"\nMapping Results:")
    print("=" * 30)
    print(f"Total negation_change samples: {total_samples}")
    print(f"Found original questions: {found_originals}")
    print(f"Missing original questions: {total_samples - found_originals}")
    print(f"Success rate: {found_originals/total_samples*100:.1f}%")
    
    if found_originals < total_samples:
        missing_indices = df[~df['found_original']]['index'].tolist()
        print(f"\nMissing indices: {missing_indices[:10]}{'...' if len(missing_indices) > 10 else ''}")
    
    print(f"\nDetailed mapping saved to: {output_file}")
    
    return mapping

if __name__ == "__main__":
    mapping = main()