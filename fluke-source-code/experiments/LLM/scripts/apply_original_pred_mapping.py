#!/usr/bin/env python3
"""
Apply original prediction mapping to remaining scripts
"""

import re

# For dialogue script
dialogue_script = 'run_dialogue_openrouter.py'

print(f"Updating {dialogue_script}...")

with open(dialogue_script, 'r') as f:
    content = f.read()

# Update evaluate_modified_set function for dialogue
old_dialogue_func = r'''def evaluate_modified_set\(detector: DialogueContradictionDetector, data: List\[Dict\], max_samples: int = 50\) -> List\[Dict\]:
    """Evaluate on modified dataset"""
    limited_data = data\[:max_samples\] if len\(data\) > max_samples else data
    
    results = \[\]
    for i, item in enumerate\(limited_data\):
        print\(f"Processing modification \{i\+1\}/\{len\(limited_data\)\}", end='\\r'\)
        
        # Create modified and original dialogues
        modified_dialogue = add_agent_labels\(item\['dialog_context'\] \+ \[item\['modified_text'\]\]\)
        original_dialogue = add_agent_labels\(item\['dialog_context'\] \+ \[item\['original_text'\]\]\)
        
        # Modified prediction
        modified_pred = detector\.predict\(modified_dialogue\)
        modified_pred_label = extract_classification_prediction\(modified_pred\)
        
        result = \{
            'original_dialog': remove_space\(original_dialogue\),
            'modified_dialog': remove_space\(modified_dialogue\),
            'modified_label': int\(item\.get\('modified_label', item\['label'\]\)\),
            'original_label': int\(item\['label'\]\),
            'modified_pred': modified_pred_label,
            'raw_output': modified_pred,
            'type': item\.get\('type', None\),
            'id': i
        \}
        
        results\.append\(result\)
        
        # Rate limiting
        time\.sleep\(0\.1\)
    
    print\(\)  # New line after progress
    return results'''

new_dialogue_func = '''def evaluate_modified_set(detector: DialogueContradictionDetector, data: List[Dict], max_samples: int = 50) -> List[Dict]:
    """Evaluate on modified dataset with original predictions"""
    limited_data = data[:max_samples] if len(data) > max_samples else data
    
    results = []
    for i, item in enumerate(limited_data):
        print(f"Processing modification {i+1}/{len(limited_data)}", end='\\r')
        
        # Create modified and original dialogues
        modified_dialogue = add_agent_labels(item['dialog_context'] + [item['modified_text']])
        original_dialogue = add_agent_labels(item['dialog_context'] + [item['original_text']])
        
        # Get both predictions
        original_response = detector.predict(original_dialogue)
        original_pred_label = extract_classification_prediction(original_response['content'])
        
        modified_response = detector.predict(modified_dialogue)
        modified_pred_label = extract_classification_prediction(modified_response['content'])
        
        result = {
            'original_dialog': remove_space(original_dialogue),
            'modified_dialog': remove_space(modified_dialogue),
            'modified_label': int(item.get('modified_label', item['label'])),
            'original_label': int(item['label']),
            'modified_pred': modified_pred_label,
            'original_pred': original_pred_label,
            'raw_output': modified_response['content'],
            'reasoning': modified_response['reasoning'],
            'original_raw_output': original_response['content'],
            'original_reasoning': original_response['reasoning'],
            'type': item.get('type', None),
            'id': i
        }
        
        results.append(result)
        
        # Rate limiting
        time.sleep(0.1)
    
    print()  # New line after progress
    return results'''

content = re.sub(old_dialogue_func, new_dialogue_func, content, flags=re.DOTALL)

# Remove original prediction mapping
content = content.replace(
    '''        # Evaluate with sample limit
        results_mod = evaluate_modified_set(detector, data, max_samples=50)
        
        # Add original predictions
        for item in results_mod:
            matches = original_pred_ds[original_pred_ds['dialog'] == item['original_dialog']]
            item['original_pred'] = matches.iloc[0]['pred'] if not matches.empty else None
        
        # Calculate accuracy''',
    '''        # Evaluate with sample limit (includes original predictions)
        results_mod = evaluate_modified_set(detector, data, max_samples=50)
        
        # Calculate accuracy'''
)

content = content.replace(
    '''    print(f"\\nAccuracy: {accuracy:.3f}")
    print(f"Results saved to: {output_file}")
    
    # Load original predictions for modification evaluation
    original_pred_ds = df_result.copy()
    original_pred_ds['dialog'] = original_pred_ds['dialog'].apply(remove_space)
    
    # Test modifications''',
    '''    print(f"\\nAccuracy: {accuracy:.3f}")
    print(f"Results saved to: {output_file}")
    
    # Test modifications (original predictions generated on-the-fly)'''
)

with open(dialogue_script, 'w') as f:
    f.write(content)

print(f"Updated {dialogue_script}")

print("Done! All scripts now generate original predictions on-the-fly.")