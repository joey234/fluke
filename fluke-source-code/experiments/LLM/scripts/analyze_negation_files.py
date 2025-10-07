#!/usr/bin/env python3
"""
Analyze the relationship between negation_change_100.jsonl and negation_100.jsonl files
to understand which samples are original vs modified.
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

def analyze_negation_files():
    """Analyze the relationship between the two negation files"""
    
    # Load both files
    negation_change = load_jsonl('../../../data/modified_data/gsm/negation_change_100.jsonl')
    negation_regular = load_jsonl('../../../data/modified_data/gsm/negation_100.jsonl')
    
    print("File Analysis:")
    print("=" * 50)
    print(f"negation_change_100.jsonl: {len(negation_change)} samples")
    print(f"negation_100.jsonl: {len(negation_regular)} samples")
    
    # Create lookups
    change_by_index = {item['index']: item for item in negation_change}
    regular_by_index = {item['index']: item for item in negation_regular}
    
    # Find overlapping indices
    change_indices = set(change_by_index.keys())
    regular_indices = set(regular_by_index.keys())
    overlapping = change_indices & regular_indices
    
    print(f"Overlapping indices: {len(overlapping)}")
    print(f"Only in negation_change: {len(change_indices - regular_indices)}")
    print(f"Only in negation_100: {len(regular_indices - change_indices)}")
    
    # Analyze the overlapping samples to understand the pattern
    print("\nAnalyzing overlapping samples:")
    print("=" * 40)
    
    analysis_results = []
    
    for idx in sorted(list(overlapping))[:5]:  # Check first 5 overlapping samples
        change_item = change_by_index[idx]
        regular_item = regular_by_index[idx]
        
        # Compare questions
        change_question = change_item['question']
        regular_original = regular_item['text']
        regular_modified = regular_item['modified']
        
        # Check which one matches
        matches_original = change_question == regular_original
        matches_modified = change_question == regular_modified
        
        print(f"\nIndex {idx}:")
        print(f"Change question matches regular original: {matches_original}")
        print(f"Change question matches regular modified: {matches_modified}")
        
        if matches_original:
            print("  -> negation_change contains ORIGINAL question")
        elif matches_modified:
            print("  -> negation_change contains MODIFIED question")
        else:
            print("  -> negation_change contains DIFFERENT question")
            print(f"  Change:   {change_question[:80]}...")
            print(f"  Original: {regular_original[:80]}...")
            print(f"  Modified: {regular_modified[:80]}...")
        
        analysis_results.append({
            'index': idx,
            'matches_original': matches_original,
            'matches_modified': matches_modified,
            'change_question': change_question,
            'regular_original': regular_original,
            'regular_modified': regular_modified
        })
    
    # Check the pattern for all overlapping samples
    original_matches = sum(1 for idx in overlapping if change_by_index[idx]['question'] == regular_by_index[idx]['text'])
    modified_matches = sum(1 for idx in overlapping if change_by_index[idx]['question'] == regular_by_index[idx]['modified'])
    
    print(f"\nOverall pattern for {len(overlapping)} overlapping samples:")
    print(f"Questions matching original: {original_matches}")
    print(f"Questions matching modified: {modified_matches}")
    print(f"Questions not matching either: {len(overlapping) - original_matches - modified_matches}")
    
    # Now analyze the non-overlapping samples in negation_change
    non_overlapping_indices = change_indices - regular_indices
    print(f"\nNon-overlapping samples in negation_change: {len(non_overlapping_indices)}")
    
    if len(non_overlapping_indices) > 0:
        print("These might be the modified versions of the overlapping samples")
        print("Sample non-overlapping indices:", sorted(list(non_overlapping_indices))[:10])
    
    # Create a comprehensive mapping
    mapping_data = []
    
    # Process overlapping samples
    for idx in overlapping:
        change_item = change_by_index[idx]
        regular_item = regular_by_index[idx]
        
        matches_original = change_item['question'] == regular_item['text']
        matches_modified = change_item['question'] == regular_item['modified']
        
        mapping_data.append({
            'index': idx,
            'in_regular_file': True,
            'question_type': 'original' if matches_original else ('modified' if matches_modified else 'different'),
            'question': change_item['question'],
            'answer': change_item['answer'],
            'short_answer': change_item['short_answer'],
            'regular_original': regular_item['text'] if matches_original else None,
            'regular_modified': regular_item['modified'] if matches_modified else None
        })
    
    # Process non-overlapping samples
    for idx in non_overlapping_indices:
        change_item = change_by_index[idx]
        mapping_data.append({
            'index': idx,
            'in_regular_file': False,
            'question_type': 'unknown',
            'question': change_item['question'],
            'answer': change_item['answer'],
            'short_answer': change_item['short_answer'],
            'regular_original': None,
            'regular_modified': None
        })
    
    # Save comprehensive mapping
    df = pd.DataFrame(mapping_data)
    df = df.sort_values('index')
    output_file = '../results/gsm/negation_change_comprehensive_analysis.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\nComprehensive analysis saved to: {output_file}")
    
    return mapping_data

if __name__ == "__main__":
    results = analyze_negation_files()