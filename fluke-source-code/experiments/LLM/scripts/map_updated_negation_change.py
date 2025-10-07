#!/usr/bin/env python3
"""
Map the updated negation_change_100.jsonl file which now has the proper structure:
- text: original question
- modified: modified question with negation changes  
- short_answer: answer for modified question
- original_answer: answer for original question
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

def analyze_updated_negation_change():
    """Analyze the updated negation_change file structure"""
    
    negation_data = load_jsonl('../../../data/modified_data/gsm/negation_change_100.jsonl')
    
    print(f"Loaded {len(negation_data)} samples from updated negation_change_100.jsonl")
    print("New file structure:")
    print("- text: original question")
    print("- modified: modified question with negation changes")
    print("- short_answer: answer for modified question")
    print("- original_answer: answer for original question")
    print()
    
    # Create comprehensive mapping
    mapping_data = []
    
    for pos, item in enumerate(negation_data):
        mapping_data.append({
            'position': pos,
            'index': item['index'],
            'type': item.get('type', 'negation_verbal'),
            'original_question': item['text'],
            'modified_question': item['modified'],
            'original_answer': item['original_answer'],
            'modified_answer': item['short_answer'],
            'questions_identical': item['text'] == item['modified'],
            'answers_identical': item['original_answer'] == item['short_answer']
        })
    
    df = pd.DataFrame(mapping_data)
    
    # Analysis
    identical_questions = df['questions_identical'].sum()
    different_questions = len(df) - identical_questions
    identical_answers = df['answers_identical'].sum()
    different_answers = len(df) - identical_answers
    
    print("Analysis Results:")
    print("=" * 40)
    print(f"Total samples: {len(df)}")
    print(f"Questions identical to original: {identical_questions}")
    print(f"Questions different from original: {different_questions}")
    print(f"Answers identical: {identical_answers}")
    print(f"Answers different: {different_answers}")
    
    # Show examples of modifications
    print("\nExamples of negation modifications:")
    print("=" * 50)
    
    different_examples = df[~df['questions_identical']].head(3)
    for i, (_, row) in enumerate(different_examples.iterrows(), 1):
        print(f"\nExample {i} (Index {row['index']}):")
        print(f"Original:  {row['original_question']}")
        print(f"Modified:  {row['modified_question']}")
        print(f"Original Answer: {row['original_answer']}")
        print(f"Modified Answer: {row['modified_answer']}")
        
        # Highlight the changes
        orig_words = row['original_question'].split()
        mod_words = row['modified_question'].split()
        if len(orig_words) != len(mod_words):
            print("Note: Different word counts - structural changes made")
    
    # Show types of modifications
    if 'type' in df.columns:
        type_counts = df['type'].value_counts()
        print(f"\nModification types:")
        for mod_type, count in type_counts.items():
            print(f"- {mod_type}: {count} samples")
    
    # Save comprehensive mapping
    output_file = '../results/gsm/negation_change_updated_mapping.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\nMapping saved to: {output_file}")
    
    # Create index lookup for the script
    index_lookup = {}
    for _, row in df.iterrows():
        index_lookup[row['index']] = {
            'position': row['position'],
            'original_question': row['original_question'],
            'modified_question': row['modified_question'],
            'original_answer': row['original_answer'],
            'modified_answer': row['modified_answer'],
            'type': row['type']
        }
    
    # Save index lookup as JSON for easy use in other scripts
    lookup_file = '../results/gsm/negation_change_index_lookup.json'
    with open(lookup_file, 'w') as f:
        json.dump(index_lookup, f, indent=2)
    
    print(f"Index lookup saved to: {lookup_file}")
    
    return mapping_data, index_lookup

def show_index_examples(index_lookup: Dict):
    """Show how to use the index lookup"""
    
    print("\nHow to use the index mapping:")
    print("=" * 35)
    
    # Show first few indices
    sample_indices = list(index_lookup.keys())[:3]
    
    for idx in sample_indices:
        data = index_lookup[idx]
        print(f"\nIndex {idx}:")
        print(f"  Position in file: {data['position']}")
        print(f"  Original Q: {data['original_question'][:80]}...")
        print(f"  Modified Q: {data['modified_question'][:80]}...")
        print(f"  Original A: {data['original_answer']}")
        print(f"  Modified A: {data['modified_answer']}")
    
    print(f"\nTotal indices available: {len(index_lookup)}")
    print(f"Index range: {min(index_lookup.keys())} to {max(index_lookup.keys())}")

if __name__ == "__main__":
    mapping_data, index_lookup = analyze_updated_negation_change()
    show_index_examples(index_lookup)