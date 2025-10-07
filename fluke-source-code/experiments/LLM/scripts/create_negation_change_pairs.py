#!/usr/bin/env python3
"""
Create proper original-modified question pairs for negation_change_100.jsonl.

Understanding:
- negation_change_100.jsonl has 200 samples
- 100 samples have indices that match negation_100.jsonl (these are ORIGINAL questions)  
- 100 samples have different indices (these are likely the MODIFIED versions)
- We need to pair them up correctly
"""

import json
import pandas as pd
from typing import Dict, List, Tuple

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def create_negation_change_pairs():
    """Create original-modified pairs from negation_change file"""
    
    # Load the files
    negation_change = load_jsonl('../../../data/modified_data/gsm/negation_change_100.jsonl')
    negation_regular = load_jsonl('../../../data/modified_data/gsm/negation_100.jsonl')
    
    # Create lookups
    change_by_index = {item['index']: item for item in negation_change}
    regular_by_index = {item['index']: item for item in negation_regular}
    
    # Find overlapping and non-overlapping indices
    change_indices = set(change_by_index.keys())
    regular_indices = set(regular_by_index.keys())
    overlapping = change_indices & regular_indices  # These should be original questions
    non_overlapping = change_indices - regular_indices  # These should be modified questions
    
    print(f"Found {len(overlapping)} original questions and {len(non_overlapping)} potentially modified questions")
    
    # Since we have equal numbers, let's try to pair them by position in the file
    # First, let's examine the order
    original_samples = [item for item in negation_change if item['index'] in overlapping]
    modified_samples = [item for item in negation_change if item['index'] in non_overlapping]
    
    print(f"Original samples by file position: {[item['index'] for item in negation_change if item['index'] in overlapping][:10]}")
    print(f"Modified samples by file position: {[item['index'] for item in negation_change if item['index'] in non_overlapping][:10]}")
    
    # Let's try pairing by alternating pattern (assuming they are interleaved)
    pairs = []
    used_modified = set()
    
    # Method 1: Try to pair by looking at the regular negation file
    # For each sample in regular file, find its original in change file and try to find corresponding modified
    for regular_item in negation_regular:
        original_idx = regular_item['index']
        original_question = change_by_index[original_idx]['question']
        
        # Now find the modified version by comparing with regular_item['modified']
        target_modified = regular_item['modified']
        
        # Look for a matching modified question in the non-overlapping samples
        modified_match = None
        modified_idx = None
        
        for mod_idx in non_overlapping:
            if mod_idx in used_modified:
                continue
            mod_question = change_by_index[mod_idx]['question']
            if mod_question == target_modified:
                modified_match = change_by_index[mod_idx]
                modified_idx = mod_idx
                used_modified.add(mod_idx)
                break
        
        if modified_match:
            pairs.append({
                'original_index': original_idx,
                'modified_index': modified_idx,
                'original_question': original_question,
                'modified_question': modified_match['question'],
                'original_answer': change_by_index[original_idx]['answer'],
                'modified_answer': modified_match['answer'],
                'short_answer': change_by_index[original_idx]['short_answer'],
                'pairing_method': 'exact_match'
            })
        else:
            # If no exact match found, we'll leave it unpaired for now
            pairs.append({
                'original_index': original_idx,
                'modified_index': None,
                'original_question': original_question,
                'modified_question': None,
                'original_answer': change_by_index[original_idx]['answer'],
                'modified_answer': None,
                'short_answer': change_by_index[original_idx]['short_answer'],
                'pairing_method': 'no_match'
            })
    
    # Check how many we successfully paired
    successful_pairs = [p for p in pairs if p['modified_index'] is not None]
    print(f"Successfully paired {len(successful_pairs)} out of {len(pairs)} samples")
    
    # Show some examples
    print("\nExample pairs:")
    print("=" * 50)
    for i, pair in enumerate(successful_pairs[:3]):
        print(f"Pair {i+1}:")
        print(f"Original (idx {pair['original_index']}): {pair['original_question'][:80]}...")
        print(f"Modified (idx {pair['modified_index']}): {pair['modified_question'][:80]}...")
        print(f"Answer: {pair['short_answer']}")
        print()
    
    # For unpaired samples, try positional pairing as fallback
    unpaired_original = [p for p in pairs if p['modified_index'] is None]
    unused_modified = non_overlapping - used_modified
    
    if len(unpaired_original) == len(unused_modified):
        print(f"Attempting positional pairing for {len(unpaired_original)} remaining samples...")
        
        # Sort both by their position in the original file
        original_positions = {}
        modified_positions = {}
        
        for pos, item in enumerate(negation_change):
            if item['index'] in [p['original_index'] for p in unpaired_original]:
                original_positions[item['index']] = pos
            elif item['index'] in unused_modified:
                modified_positions[item['index']] = pos
        
        # Pair by position
        unpaired_orig_sorted = sorted(unpaired_original, key=lambda x: original_positions[x['original_index']])
        unused_mod_sorted = sorted(unused_modified, key=lambda x: modified_positions[x])
        
        for i, orig_pair in enumerate(unpaired_orig_sorted):
            if i < len(unused_mod_sorted):
                mod_idx = unused_mod_sorted[i]
                orig_pair['modified_index'] = mod_idx
                orig_pair['modified_question'] = change_by_index[mod_idx]['question']
                orig_pair['modified_answer'] = change_by_index[mod_idx]['answer']
                orig_pair['pairing_method'] = 'positional'
    
    # Save results
    df = pd.DataFrame(pairs)
    output_file = '../results/gsm/negation_change_question_pairs.csv'
    df.to_csv(output_file, index=False)
    
    # Final statistics
    exact_matches = len([p for p in pairs if p['pairing_method'] == 'exact_match'])
    positional_matches = len([p for p in pairs if p['pairing_method'] == 'positional'])
    no_matches = len([p for p in pairs if p['pairing_method'] == 'no_match'])
    
    print(f"\nFinal Results:")
    print(f"Exact matches: {exact_matches}")
    print(f"Positional matches: {positional_matches}")
    print(f"No matches: {no_matches}")
    print(f"Total pairs: {len(pairs)}")
    print(f"\nResults saved to: {output_file}")
    
    return pairs

if __name__ == "__main__":
    pairs = create_negation_change_pairs()