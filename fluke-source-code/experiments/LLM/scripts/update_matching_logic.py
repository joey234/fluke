#!/usr/bin/env python3
"""
Update dialogue and NER scripts to use matching instead of double prediction
"""

import re

def update_dialogue_script():
    script_path = 'run_dialogue_openrouter.py'
    print(f"Updating {script_path}...")
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Update function signature
    content = re.sub(
        r'def evaluate_modified_set\(detector: DialogueContradictionDetector, data: List\[Dict\], max_samples: int = 50\)',
        'def evaluate_modified_set(detector: DialogueContradictionDetector, data: List[Dict], original_pred_ds: pd.DataFrame, max_samples: int = 50)',
        content
    )
    
    # Update function docstring
    content = content.replace(
        '"""Evaluate on modified dataset with original predictions"""',
        '"""Evaluate on modified dataset using matched original predictions"""'
    )
    
    # Update the prediction logic - remove original prediction generation
    old_pred_logic = r'''        # Get both predictions
        original_response = detector\.predict\(original_dialogue\)
        original_pred_label = extract_classification_prediction\(original_response\['content'\]\)
        
        modified_response = detector\.predict\(modified_dialogue\)
        modified_pred_label = extract_classification_prediction\(modified_response\['content'\]\)'''
    
    new_pred_logic = '''        # Get modified prediction only
        modified_response = detector.predict(modified_dialogue)
        modified_pred_label = extract_classification_prediction(modified_response['content'])
        
        # Match original prediction from base results
        original_dialogue_clean = remove_space(original_dialogue)
        matches = original_pred_ds[original_pred_ds['dialog'] == original_dialogue_clean]
        
        if not matches.empty:
            original_pred_label = matches.iloc[0]['pred']
            original_raw_output = matches.iloc[0]['raw_output']
            original_reasoning = matches.iloc[0].get('reasoning', '')
        else:
            # Fallback: run prediction if no match found
            print(f"\\nWarning: No match found for original dialogue, running prediction...")
            original_response = detector.predict(original_dialogue)
            original_pred_label = extract_classification_prediction(original_response['content'])
            original_raw_output = original_response['content']
            original_reasoning = original_response['reasoning']'''
    
    content = re.sub(old_pred_logic, new_pred_logic, content, flags=re.DOTALL)
    
    # Update result dict to use matched data
    content = content.replace(
        "'original_raw_output': original_response['content'],",
        "'original_raw_output': original_raw_output,"
    )
    content = content.replace(
        "'original_reasoning': original_response['reasoning'],",
        "'original_reasoning': original_reasoning,"
    )
    
    # Update main function calls
    content = content.replace(
        '    # Test modifications (original predictions generated on-the-fly)',
        '''    # Load original predictions for modification evaluation
    original_pred_ds = df_result.copy()
    original_pred_ds['dialog'] = original_pred_ds['dialog'].apply(remove_space)
    
    # Test modifications using matched original predictions'''
    )
    
    content = content.replace(
        'results_mod = evaluate_modified_set(detector, data, max_samples=50)',
        'results_mod = evaluate_modified_set(detector, data, original_pred_ds, max_samples=50)'
    )
    
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"Updated {script_path}")

def update_ner_script():
    script_path = 'run_ner_openrouter.py'
    print(f"Updating {script_path}...")
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Update function signature
    content = re.sub(
        r'def evaluate_modified_set\(recognizer: NamedEntityRecognizer, data: List\[Dict\], max_samples: int = 50\)',
        'def evaluate_modified_set(recognizer: NamedEntityRecognizer, data: List[Dict], original_pred_ds: pd.DataFrame, max_samples: int = 50)',
        content
    )
    
    # Update function docstring
    content = content.replace(
        '"""Evaluate on modified dataset with original predictions"""',
        '"""Evaluate on modified dataset using matched original predictions"""'
    )
    
    # Update the prediction logic
    old_pred_logic = r'''        # Get both original and modified predictions
        original_response = recognizer\.predict\(remove_space\(item\['original_text'\]\)\)
        modified_response = recognizer\.predict\(remove_space\(item\["modified_text"\]\)\)'''
    
    new_pred_logic = '''        # Get modified prediction only
        modified_response = recognizer.predict(remove_space(item["modified_text"]))
        
        # Match original prediction from base results using index
        original_idx = item.get('index', i)
        matches = original_pred_ds[original_pred_ds['index'] == original_idx]
        
        if not matches.empty:
            original_pred_content = matches.iloc[0]['pred']
            original_raw_output = matches.iloc[0]['raw_output']
            original_reasoning = matches.iloc[0].get('reasoning', '')
        else:
            # Fallback: run prediction if no match found
            print(f"\\nWarning: No match found for index {original_idx}, running prediction...")
            original_response = recognizer.predict(remove_space(item['original_text']))
            original_pred_content = original_response['content']
            original_raw_output = original_response['content']
            original_reasoning = original_response['reasoning']'''
    
    content = re.sub(old_pred_logic, new_pred_logic, content, flags=re.DOTALL)
    
    # Update result dict
    content = content.replace(
        "'original_pred': original_response['content'],",
        "'original_pred': original_pred_content,"
    )
    content = content.replace(
        "'original_raw_output': original_response['content'],",
        "'original_raw_output': original_raw_output,"
    )
    content = content.replace(
        "'original_reasoning': original_response['reasoning']",
        "'original_reasoning': original_reasoning"
    )
    
    # Update main function calls
    content = content.replace(
        '    # Test modifications (original predictions generated on-the-fly)',
        '''    # Load original predictions for modification evaluation
    original_pred_ds = df_result.copy()
    
    # Test modifications using matched original predictions'''
    )
    
    content = content.replace(
        'results_mod = evaluate_modified_set(recognizer, data, max_samples=50)',
        'results_mod = evaluate_modified_set(recognizer, data, original_pred_ds, max_samples=50)'
    )
    
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"Updated {script_path}")

if __name__ == "__main__":
    update_dialogue_script()
    update_ner_script()
    print("All scripts updated to use matching logic instead of double predictions!")