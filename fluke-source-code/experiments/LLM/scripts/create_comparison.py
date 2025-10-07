#!/usr/bin/env python3
"""
Create per-model and combined comparison CSV files across all tasks.
Adds explicit `model` and `task` columns per row and normalizes names.
"""

import pandas as pd
import glob
import os
import re

def extract_model_and_modification_from_filename(filename):
    """Extract model name and modification type from filename"""
    # Model patterns with their corresponding model names
    patterns = [
        (r'gpt-5-standard-context-aware-0shot-(.+)\.csv', 'gpt-5-standard-context-aware'),
        (r'gpt-5-standard-0shot-(.+)\.csv', 'gpt-5-standard'), 
        (r'claude-3-5-sonnet-0shot-(.+)\.csv', 'claude-3-5-sonnet'),
        (r'deepseek-r1-deepseek-0shot-(.+)\.csv', 'deepseek-r1'),
        (r'deepseek-deepseek-0shot-(.+)\.csv', 'deepseek-r1'),
        (r'gpt4o-0shot-(.+)\.csv', 'gpt4o'),
        (r'llama-llama-0shot-(.+)\.csv', 'llama'),
        (r'llama-0shot-(.+)\.csv', 'llama'),
        (r'mixtral-0shot-(.+)\.csv', 'mixtral'),
        (r'gpt4o-gpt4o-0shot-(.+)\.csv', 'gpt4o')  # GSM files
    ]
    
    for pattern, model_name in patterns:
        match = re.search(pattern, filename)
        if match:
            mod = match.group(1)
            # Normalize common suffixes and aliases
            if mod.endswith('_100_new'):
                mod = mod[:-4]
            if mod.startswith('singlish'):
                mod = mod.replace('singlish', 'dialectal', 1)
            return model_name, mod
    return None, None

def extract_modification_from_filename(filename):
    """Extract modification type from filename (backward compatibility)"""
    _, modification = extract_model_and_modification_from_filename(filename)
    return modification

# Indices to ignore per modification across tasks (by dataset index)
IGNORED_INDICES = {
    'capitalization': {349},
    'concept_replacement': {3369},
    'punctuation': {2337},
    'sentiment': {2337, 1129},  # Appraisal
    'temporal_bias': {3633},
    'typo_bias': {2337, 1129, 349},  # Spelling
}

def _norm_mod_key(mod: str) -> str:
    s = str(mod or '').strip()
    if s.endswith('_100'):
        s = s[:-4]
    if s.startswith('negation'):
        return 'negation'
    return s

def process_task(task_name, results_dir):
    """Process a single task and create comparison CSV for all models"""
    print(f"\nProcessing {task_name} task...")
    
    # Find all CSV files for this task
    pattern = f"{results_dir}/{task_name}/*.csv"
    # Ignore backup and negation_change helper files
    all_files = [
        f for f in glob.glob(pattern)
        if ('_backup.csv' not in f and 'negation_change' not in f)
    ]
    
    # Filter to only include files that match our model patterns
    files = []
    for file in all_files:
        filename = os.path.basename(file)
        model_name, modification = extract_model_and_modification_from_filename(filename)
        if model_name and modification:
            files.append(file)
    
    if not files:
        print(f"No matching model files found for {task_name}")
        return
    
    # Group files by model for separate processing
    model_files = {}
    for file in files:
        filename = os.path.basename(file)
        model_name, _ = extract_model_and_modification_from_filename(filename)
        if model_name not in model_files:
            model_files[model_name] = []
        model_files[model_name].append(file)
    
    print(f"Found models: {list(model_files.keys())}")
    
    # Process each model separately
    global_comparisons = []
    for model_name, model_file_list in model_files.items():
        print(f"\n  Processing model: {model_name}")
        all_comparisons = []
        processed_mods = set()
        
        for file in model_file_list:
            filename = os.path.basename(file)
            modification = extract_modification_from_filename(filename)
            
            if not modification:
                continue
                
            print(f"    Processing {filename}...")
            
            try:
                df = pd.read_csv(file)
            except Exception as e:
                print(f"      Error reading {filename}: {e}")
                continue
            # Drop ignored dataset indices for this modification if present
            try:
                mod_key = _norm_mod_key(modification)
                if 'index' in df.columns and mod_key in IGNORED_INDICES:
                    df = df[~df['index'].astype(str).str.extract(r'(\d+)').astype(float).astype('Int64').isin(IGNORED_INDICES[mod_key])]
            except Exception:
                pass
                
            # For GSM negation, load gold answers by index from dataset JSONL (esp. for GPT-5 w. context)
            answer_by_index = {}
            neg_type_by_index = {}
            if task_name == 'gsm' and modification and str(modification).startswith('negation'):
                try:
                    from pathlib import Path as _Path
                    script_dir = _Path(__file__).resolve().parent
                    ds_path = script_dir / f"../../../data/modified_data/gsm/{modification}.jsonl"
                    if not ds_path.exists():
                        alt = script_dir / f"../../../data/modified_data/gsm/{str(modification).replace('_100','')}_100.jsonl"
                        ds_path = alt if alt.exists() else ds_path
                    if ds_path.exists():
                        import json as _json
                        with open(ds_path, 'r', encoding='utf-8') as fds:
                            for line in fds:
                                s = line.strip()
                                if not s:
                                    continue
                                try:
                                    obj = _json.loads(s)
                                except Exception:
                                    continue
                                idx = obj.get('index')
                                if idx is None:
                                    continue
                                oa = obj.get('original_answer', obj.get('short_answer', ''))
                                answer_by_index[int(idx)] = str(oa)
                                typ = obj.get('negation_subtype', obj.get('type', ''))
                                if typ:
                                    neg_type_by_index[int(idx)] = str(typ)
                except Exception as e:
                    print(f"      Warning: failed to load GSM dataset for {modification}: {e}")

            # Track modification processed
            processed_mods.add(modification)
            # Process each row
            for idx, row in df.iterrows():
                # Handle different column names for different tasks
                if task_name == 'dialogue':
                    original_text = row.get('original_dialog', '')
                    modified_text = row.get('modified_dialog', '')
                else:
                    original_text = row.get('original_text', '')
                    modified_text = row.get('text', '')

                # Skip rows with empty text data or missing predictions
                if (pd.isna(original_text) or pd.isna(modified_text) or
                    str(original_text).strip() == '' or str(modified_text).strip() == '' or
                    pd.isna(row.get('original_pred')) or pd.isna(row.get('modified_pred'))):
                    continue

                # Get predictions and labels based on task
            if task_name == 'gsm':
                # GSM uses numerical answers
                norm = lambda s: str(s).replace(',', '').strip()
                original_pred = norm(row.get('original_pred', ''))
                modified_pred = norm(row.get('modified_pred', ''))
                # Support both original_answer and legacy short_answer in CSVs
                # Prefer dataset-mapped original answer for negation (by index); otherwise CSV
                ds_idx = row.get('index')
                try:
                    ds_idx = int(ds_idx)
                except Exception:
                    ds_idx = None
                if str(modification).startswith('negation') and ds_idx is not None and ds_idx in answer_by_index:
                    original_answer = norm(answer_by_index[ds_idx])
                    modified_answer = original_answer
                else:
                    original_answer = norm(row.get('original_answer', row.get('short_answer', '')))
                    modified_answer = norm(row.get('modified_answer', row.get('short_answer', '')))
                # Prefer the LAST #### <number> across CoT/raw for all models
                try:
                    def _extract_last_hash(texts):
                        # Prefer plain digits first; only treat thousands when comma present
                        pat = re.compile(r"####\s*[$€£¥₹₽]?\s*([+-]?(?:\d+(?:\\.\d+)?(?:[eE][+-]?\d+)?|\d{1,3}(?:,\d{3})+(?:\\.\d+)?|\d+\s*/\s*\d+))")
                        last = None
                        for t in texts:
                            s = str(t) if t is not None else ''
                            for m in pat.finditer(s):
                                last = m.group(1)
                        if not last:
                            return None
                        val = last.replace(',', '')
                        # Remove non-digit markers (e.g., %)
                        val = re.sub(r'[^0-9+\-eE./]', '', val)
                        return val
                    op_hash = _extract_last_hash([
                        row.get('original_step_by_step_reasoning', ''),
                        row.get('original_raw_output', ''),
                        row.get('original_reasoning', ''),
                    ])
                    mp_hash = _extract_last_hash([
                        row.get('modified_step_by_step_reasoning', row.get('step_by_step_reasoning', '')),
                        row.get('raw_output', ''),
                        row.get('modified_reasoning', row.get('reasoning', '')),
                    ])
                    if op_hash:
                        original_pred = op_hash
                    if mp_hash:
                        modified_pred = mp_hash
                except Exception:
                    pass
                # Simplified GSM rule: take everything after the last '####' if present in CoT/raw
                try:
                    def _after_hash(text: str):
                        if not isinstance(text, str):
                            return None
                        i = text.rfind('####')
                        if i == -1:
                            return None
                        tail = text[i+4:]
                        tail = re.sub(r'^[\s:,-]+', '', tail)
                        tail = tail.splitlines()[0].strip()
                        tail = re.sub(r'^[\$€£¥₹₽]\s*', '', tail)
                        tail = tail.replace(',', '')
                        tail = re.sub(r'[^0-9+\-eE./]', '', tail)
                        return tail or None
                    op_final = _after_hash(str(row.get('original_step_by_step_reasoning', ''))) or \
                               _after_hash(str(row.get('original_raw_output', ''))) or \
                               _after_hash(str(row.get('original_reasoning', '')))
                    mp_final = _after_hash(str(row.get('modified_step_by_step_reasoning', row.get('step_by_step_reasoning', '')))) or \
                               _after_hash(str(row.get('raw_output', ''))) or \
                               _after_hash(str(row.get('modified_reasoning', row.get('reasoning', ''))))
                    if op_final is not None:
                        original_pred = op_final
                    if mp_final is not None:
                        modified_pred = mp_final
                except Exception:
                    pass
                # Prefer extracting from CoT/outputs when available (handles #### and numeric in text)
                try:
                    orig_was_overridden = original_pred != norm(row.get('original_pred', ''))
                    mod_was_overridden = modified_pred != norm(row.get('modified_pred', ''))
                    def _parse_hash(text: str):
                        if not isinstance(text, str):
                            return None
                        m = re.search(r"####\s*[$€£¥₹₽]?\s*([+-]?(?:\d+(?:\\.\d+)?(?:[eE][+-]?\d+)?|\d{1,3}(?:,\d{3})+(?:\\.\d+)?|\d+\s*/\s*\d+))", text)
                        if not m:
                            return None
                        val = m.group(1).replace(',', '')
                        return re.sub(r'[^0-9+\-eE./]', '', val)
                    # Original side
                    if not orig_was_overridden:
                        op_hash = _parse_hash(str(row.get('original_step_by_step_reasoning', ''))) or \
                                  _parse_hash(str(row.get('original_reasoning', ''))) or \
                                  _parse_hash(str(row.get('original_raw_output', '')))
                        if not op_hash:
                            # Fallback: any numeric token in CoT/outputs
                            m = re.findall(r'[+-]?\d+(?:\.\d+)?', (str(row.get('original_step_by_step_reasoning', '')) + ' ' + str(row.get('original_reasoning', '')) + ' ' + str(row.get('original_raw_output', ''))).replace(',', ''))
                            if m:
                                op_hash = m[-1]
                        if op_hash:
                            original_pred = op_hash
                    # Modified side
                    if not mod_was_overridden:
                        mp_hash = _parse_hash(str(row.get('modified_step_by_step_reasoning', ''))) or \
                                  _parse_hash(str(row.get('modified_reasoning', row.get('reasoning', '')))) or \
                                  _parse_hash(str(row.get('raw_output', '')))
                        if not mp_hash:
                            m2 = re.findall(r'[+-]?\d+(?:\.\d+)?', (str(row.get('modified_step_by_step_reasoning', '')) + ' ' + str(row.get('modified_reasoning', row.get('reasoning', ''))) + ' ' + str(row.get('raw_output', ''))).replace(',', ''))
                            if m2:
                                mp_hash = m2[-1]
                        if mp_hash:
                            modified_pred = mp_hash
                except Exception:
                    pass

                    # Check if this is negation data with type field
                    negation_type = row.get('type', '')

                    # Convert to float for comparison if possible
                    try:
                        original_correct = float(original_pred) == float(original_answer)

                        # Apply different correctness criteria based on negation type
                        if negation_type in ['negation_approximate', 'negation_double']:
                            # Approximate/double: prediction equals gold is correct
                            modified_correct = float(modified_pred) == float(modified_answer)
                        elif negation_type in ['negation_verbal', 'negation_absolute', 'negation_lexical']:
                            # Other negations: flip correctness
                            modified_correct = float(modified_pred) != float(modified_answer)
                        else:
                            # Default behavior for non-negation GSM data
                            modified_correct = float(modified_pred) == float(modified_answer)

                    except (ValueError, TypeError):
                        # Fall back to string comparison
                        original_correct = original_pred == original_answer

                        if negation_type in ['negation_approximate', 'negation_double']:
                            modified_correct = modified_pred == modified_answer
                        elif negation_type in ['negation_verbal', 'negation_absolute', 'negation_lexical']:
                            modified_correct = modified_pred != modified_answer
                        else:
                            modified_correct = modified_pred == modified_answer
                else:
                    # Other tasks use string labels
                    original_pred = str(row.get('original_pred', '')).strip().lower()
                    modified_pred = str(row.get('modified_pred', '')).strip().lower()

                    if task_name == 'dialogue':
                        original_label = str(row.get('original_dialog_label', '')).strip().lower()
                        modified_label = str(row.get('modified_dialog_label', '')).strip().lower()
                    elif task_name == 'ner':
                        original_label = str(row.get('original_entities', '')).strip().lower()
                        modified_label = str(row.get('modified_entities', '')).strip().lower()
                    elif task_name == 'sa':
                        original_label = str(row.get('original_label', '')).strip().lower()
                        modified_label = str(row.get('modified_label', '')).strip().lower()
                    elif task_name == 'coref':
                        original_label = str(row.get('original_clusters', '')).strip().lower()
                        modified_label = str(row.get('modified_clusters', '')).strip().lower()
                    else:
                        # Generic fallback
                        original_label = str(row.get('original_label', '')).strip().lower()
                        modified_label = str(row.get('modified_label', '')).strip().lower()

                    original_correct = original_pred == original_label
                    modified_correct = modified_pred == modified_label

                # Determine performance category
                if original_correct and modified_correct:
                    performance = 'both_correct'
                elif original_correct and not modified_correct:
                    performance = 'original_better'
                elif not original_correct and modified_correct:
                    performance = 'modified_better'
                else:
                    performance = 'both_wrong'

                comparison_row = {
                    'task': task_name,
                    'model': model_name,
                    'modification': modification,
                    'original_text': original_text,
                    'modified_text': modified_text,
                    # Store computed/normalized predictions
                    'original_pred': original_pred,
                    'modified_pred': modified_pred,
                    'original_correct': original_correct,
                    'modified_correct': modified_correct,
                    'performance': performance,
                    'original_reasoning': row.get('original_reasoning', ''),
                    'modified_reasoning': row.get('modified_reasoning', row.get('reasoning', '')),
                    'original_step_by_step_reasoning': row.get('original_step_by_step_reasoning', ''),
                    'modified_step_by_step_reasoning': row.get('modified_step_by_step_reasoning', row.get('step_by_step_reasoning', '')),
                    'file': filename,
                    'row_index': idx
                }

                # Add task-specific fields
                if task_name == 'gsm':
                    # Echo the answers actually used for scoring into the output
                    comparison_row['original_answer'] = original_answer
                    comparison_row['modified_answer'] = modified_answer
                    # Carry forward negation subtype if present
                    # Prefer dataset subtype if available for negation
                    if str(modification).startswith('negation') and ds_idx is not None and ds_idx in neg_type_by_index:
                        comparison_row['negation_subtype'] = neg_type_by_index.get(ds_idx, '')
                        comparison_row['type'] = neg_type_by_index.get(ds_idx, '')
                    else:
                        if 'negation_subtype' in row:
                            comparison_row['negation_subtype'] = row.get('negation_subtype', '')
                        if 'type' in row:
                            comparison_row['type'] = row.get('type', '')
                elif task_name == 'dialogue':
                    comparison_row['original_dialog_label'] = row.get('original_dialog_label', '')
                    comparison_row['modified_dialog_label'] = row.get('modified_dialog_label', '')
                elif task_name == 'ner':
                    comparison_row['original_entities'] = row.get('original_entities', '')
                    comparison_row['modified_entities'] = row.get('modified_entities', '')
                elif task_name == 'sa':
                    comparison_row['original_label'] = row.get('original_label', '')
                    comparison_row['modified_label'] = row.get('modified_label', '')
                elif task_name == 'coref':
                    comparison_row['original_clusters'] = row.get('original_clusters', '')
                    comparison_row['modified_clusters'] = row.get('modified_clusters', '')

                all_comparisons.append(comparison_row)
    
        if all_comparisons:
            # Create DataFrame and save
            df_comparison = pd.DataFrame(all_comparisons)
        
        # Sort by performance and modification
        df_comparison = df_comparison.sort_values(['performance', 'modification'])
        
        # Save per-model file
        per_model_out = f"{model_name}_comparison_{task_name}.csv"
        df_comparison.to_csv(per_model_out, index=False)

        print(f"  Saved {len(all_comparisons)} samples to {per_model_out}")
        
        # Print summary
        perf_counts = df_comparison['performance'].value_counts()
        print(f"  Performance breakdown: {dict(perf_counts)}")
        try:
            print(f"  Modifications processed ({len(processed_mods)}): {', '.join(sorted(processed_mods))}")
        except Exception:
            pass

        # Append to global list
        global_comparisons.extend(all_comparisons)

    # Also write a combined file across models to match viewer expectations
    if global_comparisons:
        df_global = pd.DataFrame(global_comparisons)
        df_global = df_global.sort_values(['model', 'performance', 'modification'])
        output_file = f"all_models_comparison_{task_name}.csv"
        df_global.to_csv(output_file, index=False)
        print(f"\nSaved combined comparison for {task_name}: {output_file} ({len(df_global)} rows)")
        
    else:
        print(f"  No valid comparisons found for {task_name}")

def main():
    results_dir = "../results"
    
    # List of tasks to process
    tasks = ['ner', 'dialogue', 'sa', 'coref', 'gsm']
    
    print("Creating comparison CSV files for gpt-5-standard-context-aware model")
    print("=" * 70)
    
    for task in tasks:
        process_task(task, results_dir)
    
    print("\n" + "=" * 70)
    print("Comparison files created successfully!")

if __name__ == "__main__":
    main()
