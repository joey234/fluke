#!/usr/bin/env python3
"""
Map indices from negation_change_100.jsonl to find original questions and answers.
This script creates a mapping to understand which samples correspond to which original GSM problems.
"""

import json
import pandas as pd
from typing import Dict, List

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def create_index_mapping(negation_change_file: str) -> Dict[int, Dict]:
    """Create mapping from index to original question and answer"""
    data = load_jsonl(negation_change_file)
    
    index_mapping = {}
    for item in data:
        index = item['index']
        index_mapping[index] = {
            'original_question': item['question'],
            'original_answer': item['answer'],
            'short_answer': item['short_answer'],
            'position_in_file': len(index_mapping)  # 0-based position in the negation_change file
        }
    
    return index_mapping

def main():
    # Load the negation_change file
    negation_change_file = '../../../data/modified_data/gsm/negation_change_100.jsonl'
    
    print("Loading negation_change_100.jsonl file...")
    index_mapping = create_index_mapping(negation_change_file)
    
    print(f"Found {len(index_mapping)} samples in negation_change file")
    print("\nIndex Mapping Summary:")
    print("=" * 80)
    
    # Show first 10 mappings as examples
    sorted_indices = sorted(index_mapping.keys())
    for i, index in enumerate(sorted_indices[:10]):
        item = index_mapping[index]
        print(f"Index {index} (Position {item['position_in_file']}):")
        print(f"  Original Question: {item['original_question'][:100]}...")
        print(f"  Short Answer: {item['short_answer']}")
        print()
    
    if len(sorted_indices) > 10:
        print(f"... and {len(sorted_indices) - 10} more samples")
    
    print("\nComplete Index List:")
    print("=" * 40)
    print("Indices in order of appearance:", sorted_indices)
    
    # Save to CSV for easy reference
    df_data = []
    for index in sorted_indices:
        item = index_mapping[index]
        df_data.append({
            'index': index,
            'position_in_file': item['position_in_file'],
            'original_question': item['original_question'],
            'short_answer': item['short_answer'],
            'original_answer': item['original_answer']
        })
    
    df = pd.DataFrame(df_data)
    output_file = '../results/gsm/negation_change_index_mapping.csv'
    df.to_csv(output_file, index=False)
    print(f"\nIndex mapping saved to: {output_file}")
    
    # Show statistics
    print("\nStatistics:")
    print("=" * 30)
    print(f"Total samples: {len(index_mapping)}")
    print(f"Index range: {min(sorted_indices)} to {max(sorted_indices)}")
    print(f"Unique indices: {len(set(sorted_indices))}")
    
    return index_mapping

if __name__ == "__main__":
    mapping = main()