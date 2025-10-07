#!/usr/bin/env python3
"""
Create comparison CSV files for all available models across all tasks
"""

import pandas as pd
import glob
import os
import re
from pathlib import Path
from pathlib import Path
from typing import Optional

try:
    from .llm_utils import llm_extract_number_from_texts, load_dotenv_if_present  # type: ignore
except Exception:
    try:
        from llm_utils import llm_extract_number_from_texts, load_dotenv_if_present  # type: ignore
    except Exception:
        llm_extract_number_from_texts = None
        def load_dotenv_if_present():
            pass

def extract_model_and_modification_from_filename(filename):
    """Extract model name and modification type from filename"""
    # Model patterns with their corresponding model names
    patterns = [
        (r'gpt-5-standard-context-aware-0shot-(.+)\.csv', 'gpt-5-standard-context-aware'),
        (r'gpt-5-context-aware-0shot-(.+)\.csv', 'gpt-5-standard-context-aware'),
        (r'gpt-5-standard-0shot-(.+)\.csv', 'gpt-5-standard'), 
        (r'gpt-5-0shot-(.+)\.csv', 'gpt-5-standard'),
        (r'claude-3-5-sonnet-0shot-(.+)\.csv', 'claude-3-5-sonnet'),
        (r'claude-0shot-(.+)\.csv', 'claude-3-5-sonnet'),  # Claude NER files use different naming
        # DeepSeek R1 variants across tasks
        (r'deepseek-r1-deepseek-0shot-(.+)\.csv', 'deepseek-r1'),
        (r'deepseek-deepseek-0shot-(.+)\.csv', 'deepseek-r1'),
        (r'deepseek-r1-0shot-(.+)\.csv', 'deepseek-r1'),
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
                mod = mod[:-4]  # drop _new
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

def _extract_float(value):
    """Robust extraction of a float from a value/string.
    Handles:
    - comma-separated numbers (1,596 -> 1596)
    - #### answer markers, 'final answer:' patterns
    - simple fractions like '3/4'
    - scientific notation
    Returns None if no numeric can be parsed.
    """
    if value is None:
        return None
    s = str(value).strip()
    if s == '' or s.lower() == 'nan':
        return None
    # Normalize
    s_no_commas = s.replace(',', '')
    # 1) #### <number> pattern
    m = re.search(r'####\s*[$€£¥₹₽]?\s*([+-]?(?:\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?|[+-]?\d+\s*/\s*\d+)', s_no_commas)
    if m:
        token = m.group(1)
        # fraction
        mf = re.fullmatch(r'\s*([+-]?\d+)\s*/\s*(\d+)\s*', token)
        if mf:
            a = float(mf.group(1)); b = float(mf.group(2))
            return a / b if b != 0 else None
        try:
            return float(token)
        except Exception:
            pass
    # 2) 'final answer: <number>' style (case-insensitive)
    m = re.search(r'(?:final\s+answer|answer)\s*[:\-]?\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[+-]?\d+\s*/\s*\d+)', s_no_commas, flags=re.IGNORECASE)
    if m:
        token = m.group(1)
        mf = re.fullmatch(r'\s*([+-]?\d+)\s*/\s*(\d+)\s*', token)
        if mf:
            a = float(mf.group(1)); b = float(mf.group(2))
            return a / b if b != 0 else None
        try:
            return float(token)
        except Exception:
            pass
    # 3) fraction anywhere
    mf = re.findall(r'[+-]?\d+\s*/\s*\d+', s_no_commas)
    if mf:
        num, den = mf[-1].split('/')
        try:
            den_v = float(den)
            return float(num) / den_v if den_v != 0 else None
        except Exception:
            pass
    # 4) last numeric token (integer/decimal/scientific)
    m = re.findall(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', s_no_commas)
    if m:
        try:
            return float(m[-1])
        except Exception:
            return None
    return None

def process_task(task_name, results_dir):
    """Process a single task and create comparison CSV for all models"""
    # Ensure .env is loaded and defaults are set
    try:
        load_dotenv_if_present()
    except Exception:
        pass
    print(f"\nProcessing {task_name} task...")
    
    # Find all LLM CSV files for this task
    pattern = f"{results_dir}/{task_name}/*.csv"
    all_files = [
        f for f in glob.glob(pattern)
        if ('_backup.csv' not in f and 'negation_change' not in f)
    ]
    
    # Filter to only include files that match our model patterns
    files = []
    for file in all_files:
        filename = os.path.basename(file)
        if '_backup.csv' in filename or 'negation_change' in filename:
            continue
        model_name, modification = extract_model_and_modification_from_filename(filename)
        if model_name and modification:
            files.append(file)
    
    # Also gather PLM files (all tasks except GSM)
    plm_files = []
    if task_name != 'gsm':
        plm_files = collect_plm_files(task_name)
        files.extend([pf['path'] for pf in plm_files])
    
    if not files and not plm_files:
        print(f"No matching model files found for {task_name}")
        return
    
    # Group files by model for separate processing
    model_files = {}
    for file in files:
        filename = os.path.basename(file)
        # Check if PLM file entry exists
        plm_entry = next((pf for pf in plm_files if pf['path'] == file), None)
        if plm_entry:
            model_name = plm_entry['model']
        else:
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
        
        for file in model_file_list:
            filename = os.path.basename(file)
            # Resolve modification and loader source
            plm_entry = next((pf for pf in plm_files if pf['path'] == file), None)
            if plm_entry:
                modification = plm_entry['modification']
                source = 'plm'
                plm_task = plm_entry['task']
                plm_model = plm_entry['model']
            else:
                modification = extract_modification_from_filename(filename)
                source = 'llm'
            
            if not modification:
                continue
                
            print(f"    Processing {filename}...")
            
            try:
                df = pd.read_csv(file)
            except Exception as e:
                print(f"      Error reading {filename}: {e}")
                continue
                
            # Prepare GSM gold answers (and negation subtype) from dataset JSONL for negation only
            answer_by_index = {}
            neg_type_by_index = {}
            if task_name == 'gsm' and modification and str(modification).startswith('negation'):
                try:
                    script_dir = Path(__file__).resolve().parent
                    # modification already includes _100; dataset is under ../../../data/modified_data/gsm/<mod>.jsonl
                    ds_path = script_dir / f"../../../data/modified_data/gsm/{modification}.jsonl"
                    if not ds_path.exists():
                        # try without _100 suffix
                        alt = script_dir / f"../../../data/modified_data/gsm/{str(modification).replace('_100','')}_100.jsonl"
                        ds_path = alt if alt.exists() else ds_path
                    if ds_path.exists():
                        import json
                        with open(ds_path, 'r', encoding='utf-8') as fds:
                            for ln, line in enumerate(fds, 1):
                                s = line.strip()
                                if not s:
                                    continue
                                try:
                                    obj = json.loads(s)
                                except Exception:
                                    continue
                                idx = obj.get('index')
                                if idx is None:
                                    continue
                                # Prefer original_answer; fallback to short_answer for legacy sets
                                ans = obj.get('original_answer', obj.get('short_answer', ''))
                                answer_by_index[int(idx)] = str(ans)
                                typ = obj.get('negation_subtype', obj.get('type', ''))
                                if typ:
                                    neg_type_by_index[int(idx)] = str(typ)
                except Exception as e:
                    print(f"      Warning: failed to load GSM dataset for {modification}: {e}")

            # Process each row
            for idx, row in df.iterrows():
                # Skip specific problematic dataset indices by modification
                try:
                    mod_key = _norm_mod_key(modification)
                    idx_val = row.get('index', None)
                    if idx_val is not None:
                        idx_int = int(idx_val)
                        if mod_key in IGNORED_INDICES and idx_int in IGNORED_INDICES[mod_key]:
                            continue
                except Exception:
                    pass
                # Handle different column names for different tasks
                if task_name == 'dialogue':
                    if source == 'plm':
                        original_text = row.get('dialog_context', '')
                        modified_text = row.get('modified_text', '')
                    else:
                        original_text = row.get('original_dialog', '')
                        modified_text = row.get('modified_dialog', '')
                elif task_name == 'coref':
                    # Coref: LLM and PLM use original_text/modified_text
                    original_text = row.get('original_text', '')
                    modified_text = row.get('modified_text', row.get('text', ''))
                elif task_name == 'ner':
                    # NER: PLM parsed results and LLM both include fields
                    original_text = row.get('original_text', '')
                    modified_text = row.get('modified_text', row.get('text', ''))
                elif task_name == 'sa':
                    # SA PLM compare files include original_text/modified_text
                    original_text = row.get('original_text', '')
                    modified_text = row.get('modified_text', row.get('text', ''))
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
                    # Prefer precomputed parsed columns when present (for non-DeepSeek/LLaMA models)
                    if model_name not in ('deepseek-r1','llama') and 'parsed_original_pred' in df.columns:
                        original_pred = norm(row.get('parsed_original_pred', row.get('original_pred', '')))
                    else:
                        original_pred = norm(row.get('original_pred', ''))
                    if model_name not in ('deepseek-r1','llama') and 'parsed_modified_pred' in df.columns:
                        modified_pred = norm(row.get('parsed_modified_pred', row.get('modified_pred', '')))
                    else:
                        modified_pred = norm(row.get('modified_pred', ''))
                    # Prefer the LAST #### <number> in CoT/raw for all models
                    try:
                        def _extract_last_hash(texts):
                            # Prefer plain digits first; only treat thousands when comma is present
                            pat = re.compile(r"####\s*[$€£¥₹₽]?\s*([+-]?(?:\d+(?:\\.\d+)?(?:[eE][+-]?\d+)?|\d{1,3}(?:,\d{3})+(?:\\.\d+)?|\d+\s*/\s*\d+))")
                            last = None
                            for t in texts:
                                s = str(t) if t is not None else ''
                                for m in pat.finditer(s):
                                    last = m.group(1)
                            if not last:
                                return None
                            val = last.replace(',', '')
                            # Strip any non-numeric markers like % or text
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
                    # Simplified GSM rule: only apply to DeepSeek and LLaMA — take everything after last '####'
                    if model_name in ('deepseek-r1', 'llama'):
                        try:
                            def _after_hash(text: str):
                                if not isinstance(text, str):
                                    return None
                                i = text.rfind('####')
                                if i == -1:
                                    return None
                                tail = text[i+4:]
                                # Trim leading separators and keep only up to newline
                                tail = re.sub(r'^[\s:,-]+', '', tail)
                                tail = tail.splitlines()[0].strip()
                                # Drop currency and commas
                                tail = re.sub(r'^[\$€£¥₹₽]\s*', '', tail)
                                tail = tail.replace(',', '')
                                # Remove non-digit markers like % before comparing
                                tail = re.sub(r'[^0-9+\-eE./]', '', tail)
                                return tail or None
                            # Check likely columns for final answers
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
                    # Deterministic fallback: if still not overridden, parse strictly from CoT/raw with '####' or final markers
                    # Deterministic fallback only for DeepSeek/LLaMA as well
                    if model_name in ('deepseek-r1', 'llama'):
                        def _override_from_candidates(cands):
                            for txt in cands:
                                if not isinstance(txt, str) or not txt.strip():
                                    continue
                                m = re.search(r"####\s*[$€£¥₹₽]?\s*([+-]?(?:\d+(?:\\.\d+)?(?:[eE][+-]?\d+)?|\d{1,3}(?:,\\d{3})+(?:\\.\\d+)?|\\d+\\s*/\\s*\\d+))", txt)
                                if m:
                                    val = m.group(1).replace(',', '')
                                    return re.sub(r'[^0-9+\-eE./]', '', val)
                                m = re.search(r"(?i)(?:final\s+answer|answer)\s*[:\-]?\s*([+-]?(?:\d+(?:\\.\d+)?(?:[eE][+-]?\d+)?|\d{1,3}(?:,\d{3})+(?:\\.\d+)?|\d+\s*/\s*\d+))", txt)
                                if m:
                                    return m.group(1).replace(',', '')
                                lines = [ln.strip() for ln in str(txt).splitlines() if ln.strip()]
                                if lines:
                                    last = lines[-1].rstrip('.;:')
                                    if re.fullmatch(r"[+-]?(?:\d+(?:\\.\d+)?(?:[eE][+-]?\d+)?|\d{1,3}(?:,\d{3})+(?:\\.\d+)?)", last) or \
                                       re.fullmatch(r"[+-]?\d+\s*/\s*\d+", last):
                                        return last.replace(',', '')
                            return None
                        if not original_pred or original_pred == norm(row.get('original_pred', '')):
                            op2 = _override_from_candidates([
                                row.get('original_step_by_step_reasoning', ''),
                                row.get('original_raw_output', ''),
                            ])
                            if op2 is not None:
                                original_pred = op2
                        if not modified_pred or modified_pred == norm(row.get('modified_pred', '')):
                            mp2 = _override_from_candidates([
                                row.get('modified_step_by_step_reasoning', row.get('step_by_step_reasoning', '')),
                                row.get('raw_output', ''),
                            ])
                            if mp2 is not None:
                                modified_pred = mp2

                    # LLM-assisted parsing for other models (optional; requires USE_LLM_GSM_PARSE and OPENAI_API_KEY)
                    if model_name not in ('deepseek-r1', 'llama') and llm_extract_number_from_texts is not None:
                        if os.environ.get('USE_LLM_GSM_PARSE'):
                            # Only try if we don't already have a good-looking numeric
                            def _looks_ok(v: str) -> bool:
                                return bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\d+\s*/\s*\d+)", str(v)))
                            if not _looks_ok(original_pred):
                                op_llm = llm_extract_number_from_texts([
                                    str(row.get('original_step_by_step_reasoning', '')),
                                    str(row.get('original_raw_output', '')),
                                    str(row.get('original_reasoning', '')),
                                ])
                                if op_llm:
                                    original_pred = op_llm
                            if not _looks_ok(modified_pred):
                                mp_llm = llm_extract_number_from_texts([
                                    str(row.get('modified_step_by_step_reasoning', row.get('step_by_step_reasoning', ''))),
                                    str(row.get('raw_output', '')),
                                    str(row.get('modified_reasoning', row.get('reasoning', ''))),
                                ])
                                if mp_llm:
                                    modified_pred = mp_llm
                    # For negation: prefer dataset gold answer via index; otherwise use CSV
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
                        modified_answer = original_answer
                    
                    # Check if this is negation data with type field
                    # Negation subtype: prefer dataset map by index for negation; fallback to CSV column
                    if str(modification).startswith('negation') and ds_idx is not None and ds_idx in neg_type_by_index:
                        negation_type = neg_type_by_index.get(ds_idx, '')
                    else:
                        negation_type = row.get('type', '')
                    
                    # Convert to float for comparison if possible (with tolerance)
                    tol = 1e-6
                    op = _extract_float(original_pred)
                    oa = _extract_float(original_answer)
                    mp = _extract_float(modified_pred)
                    ma = _extract_float(modified_answer)

                    if op is not None and oa is not None:
                        original_correct = abs(op - oa) < tol
                    else:
                        original_correct = original_pred == original_answer

                    # For negation modification, evaluate modified prediction against SHORT ANSWER (original answer)
                    if modification and modification.startswith('negation'):
                        if mp is not None and oa is not None:
                            st = str(negation_type).lower()
                            if ('approximate' in st) or ('double' in st):
                                modified_correct = abs(mp - oa) < tol
                            else:
                                modified_correct = abs(mp - oa) >= tol
                        else:
                            st = str(negation_type).lower()
                            if ('approximate' in st) or ('double' in st):
                                modified_correct = modified_pred == original_answer
                            else:
                                modified_correct = modified_pred != original_answer
                    else:
                        # Non-negation: standard equality vs short answer
                        if mp is not None and oa is not None:
                            modified_correct = abs(mp - oa) < tol
                        else:
                            modified_correct = modified_pred == original_answer
                else:
                    # Other tasks use string labels, but NER needs special handling
                    if task_name == 'ner':
                        # NER data contains JSON structures, need to normalize for comparison
                        def normalize_ner_entities(entities_str):
                            if not entities_str or entities_str.strip() == '':
                                return {}
                            try:
                                import ast
                                entities = ast.literal_eval(str(entities_str).strip())
                                if not isinstance(entities, list):
                                    return {}
                                
                                normalized = {}
                                for entity in entities:
                                    if isinstance(entity, dict):
                                        if 'text' in entity and 'value' in entity:
                                            # Format: [{'text': 'Name', 'value': 'TYPE'}]
                                            normalized[entity['text']] = entity['value']
                                        else:
                                            # Format: [{'Name': 'TYPE'}] or {'Name': 'TYPE'}
                                            for key, value in entity.items():
                                                normalized[key] = value
                                return normalized
                            except:
                                return {}
                        
                        original_pred_norm = normalize_ner_entities(row.get('original_pred', ''))
                        modified_pred_norm = normalize_ner_entities(row.get('modified_pred', ''))
                        original_label_norm = normalize_ner_entities(row.get('original_label', ''))
                        modified_label_norm = normalize_ner_entities(row.get('modified_label', ''))
                        
                        original_correct = original_pred_norm == original_label_norm
                        modified_correct = modified_pred_norm == modified_label_norm
                        
                        # Store the original string values for display
                        original_pred = str(row.get('original_pred', '')).strip()
                        modified_pred = str(row.get('modified_pred', '')).strip()
                        original_label = str(row.get('original_label', '')).strip()
                        modified_label = str(row.get('modified_label', '')).strip()
                    else:
                        # Other tasks use simple string comparison
                        original_pred = str(row.get('original_pred', '')).strip().lower()
                        modified_pred = str(row.get('modified_pred', '')).strip().lower()
                        # Dialogue + DeepSeek: refine by parsing final decision from raw outputs, avoiding agent ids
                        if task_name == 'dialogue' and model_name == 'deepseek-r1':
                            try:
                                def _parse_final_binary(text: str):
                                    s = str(text) if text is not None else ''
                                    # Prefer explicit keywords
                                    kw = re.findall(r"(?i)(?:final\s*answer|answer|label|prediction|result)\s*[:\-]?\s*([01])", s)
                                    if kw:
                                        return int(kw[-1])
                                    # Last line that is exactly 0 or 1
                                    for line in [ln.strip() for ln in s.splitlines() if ln.strip()][::-1]:
                                        if re.fullmatch(r"[01]", line):
                                            return int(line)
                                    # Fallback: last digit not immediately preceded by 'agent'
                                    cand = None
                                    for m in re.finditer(r"([01])", s):
                                        i = m.start()
                                        prev = s[max(0, i-8):i].lower()
                                        if prev.endswith('agent ') or prev.endswith('agent\t') or prev.endswith('agent'):
                                            continue
                                        cand = int(m.group(1))
                                    return cand
                                op = _parse_final_binary(row.get('original_raw_output', ''))
                                mp = _parse_final_binary(row.get('raw_output', ''))
                                if op is not None:
                                    original_pred = op
                                if mp is not None:
                                    modified_pred = mp
                            except Exception:
                                pass
                        # Coref + DeepSeek: override preds by parsing the final digit from raw outputs
                        if task_name == 'coref' and model_name == 'deepseek-r1':
                            try:
                                def _last_digit(text: str):
                                    s = str(text) if text is not None else ''
                                    m = re.findall(r"(\d)", s)
                                    return m[-1] if m else None
                                op = _last_digit(row.get('original_raw_output', ''))
                                mp = _last_digit(row.get('raw_output', ''))
                                if op is not None:
                                    original_pred = op
                                if mp is not None:
                                    modified_pred = mp
                            except Exception:
                                pass
                        # SA PLM files may use modified_label/original_label names already
                        ol = row.get('original_label', row.get('label', ''))
                        ml = row.get('modified_label', row.get('label', ''))
                        original_label = str(ol).strip().lower()
                        modified_label = str(ml).strip().lower()
                        
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
                    # Store the computed/normalized predictions, not raw CSV cells
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
                    # Persist answers; dataset-derived for negation, CSV-derived otherwise
                    comparison_row['original_answer'] = original_answer
                    comparison_row['modified_answer'] = modified_answer
                    # Include negation subtype tag if present
                    if str(modification).startswith('negation') and negation_type:
                        comparison_row['negation_subtype'] = negation_type
                    # Also carry dataset index for debugging/traceability
                    if 'index' in row:
                        comparison_row['index'] = row.get('index')
                elif task_name == 'dialogue':
                    comparison_row['original_dialog_label'] = row.get('original_label', '')
                    comparison_row['modified_dialog_label'] = row.get('modified_label', '')
                elif task_name == 'ner':
                    comparison_row['original_entities'] = row.get('original_label', row.get('original_gold', ''))
                    comparison_row['modified_entities'] = row.get('modified_label', row.get('modified_gold', ''))
                elif task_name == 'sa':
                    comparison_row['original_label'] = row.get('original_label', '')
                    comparison_row['modified_label'] = row.get('modified_label', '')
                elif task_name == 'coref':
                    comparison_row['original_clusters'] = row.get('original_label', '')
                    comparison_row['modified_clusters'] = row.get('modified_label', '')
                    
                all_comparisons.append(comparison_row)
        
        if all_comparisons:
            # Create DataFrame and save
            df_comparison = pd.DataFrame(all_comparisons)
            
            # Sort by performance and modification
            df_comparison = df_comparison.sort_values(['performance', 'modification'])
            
            output_file = f"{model_name}_comparison_{task_name}.csv"
            df_comparison.to_csv(output_file, index=False)
            
            print(f"    Saved {len(all_comparisons)} samples to {output_file}")
            
            # Print summary
            perf_counts = df_comparison['performance'].value_counts()
            print(f"    Performance breakdown: {dict(perf_counts)}")
            
        else:
            print(f"    No valid comparisons found for {model_name} on {task_name}")

        # Accumulate for combined file
        global_comparisons.extend(all_comparisons)

    # Write a combined file across models for this task
    if global_comparisons:
        df_global = pd.DataFrame(global_comparisons)
        # Ensure normalized predictions are present (we already store computed ones)
        df_global = df_global.sort_values(['model', 'performance', 'modification'])
        out_combined = f"all_models_comparison_{task_name}.csv"
        df_global.to_csv(out_combined, index=False)
        print(f"\nSaved combined comparison for {task_name}: {out_combined} ({len(df_global)} rows)")

def main():
    script_dir = Path(__file__).resolve().parent
    results_dir = str((script_dir / "../results").resolve())
    
    # List of tasks to process
    tasks = ['ner', 'dialogue', 'sa', 'coref', 'gsm']
    
    print("Creating comparison CSV files for all available models")
    print("=" * 70)
    
    for task in tasks:
        process_task(task, results_dir)
    
    # Also process IFEval outputs if present
    try:
        from create_ifeval_comparison import main as ifeval_main
        print("\nProcessing IFEval task...")
        ifeval_main()
    except Exception as e:
        print(f"Skipping IFEval comparison generation due to error: {e}")
    
    print("\n" + "=" * 70)
    print("Comparison files created successfully!")

def collect_plm_files(task_name):
    """Collect PLM prediction files for a task and return list of dicts with keys: path, model, modification, task"""
    results = []
    # Resolve PLM base relative to this script
    script_dir = Path(__file__).resolve().parent
    base = script_dir.parents[1] / 'PLM'
    if task_name == 'coref':
        task_dir = base / 'coreference_resolution'
        models = {
            'bert': task_dir / 'coref_bert',
            'gpt2': task_dir / 'coref_gpt2',
            't5': task_dir / 'coref_t5'
        }
        for model, mdir in models.items():
            files = list(mdir.glob('*_predictions.csv'))
            mods = set(f.stem.replace('_predictions', '') for f in files)
            for fp in files:
                mod = fp.stem.replace('_predictions', '')
                if mod == 'dialectal' and 'singlish' in mods:
                    continue
                if mod == 'singlish':
                    mod = 'dialectal'
                results.append({'path': str(fp), 'model': model, 'modification': f'{mod}_100', 'task': task_name})
    elif task_name == 'dialogue':
        task_dir = base / 'dialogue_contradiction_detection'
        models = {
            'bert': task_dir / 'dialog_bert',
            'gpt2': task_dir / 'dialog_gpt2',
            't5': task_dir / 'dialog_t5'
        }
        for model, mdir in models.items():
            files = list(mdir.glob('*_predictions.csv'))
            mods = set(f.stem.replace('_predictions', '') for f in files)
            for fp in files:
                mod = fp.stem.replace('_predictions', '')
                if mod == 'dialectal' and 'singlish' in mods:
                    continue
                if mod == 'singlish':
                    mod = 'dialectal'
                results.append({'path': str(fp), 'model': model, 'modification': f'{mod}_100', 'task': task_name})
    elif task_name == 'sa':
        task_dir = base / 'sentiment_analysis' / 'sa_plm_compare'
        models = {
            'bert': task_dir / 'bert',
            'gpt2': task_dir / 'gpt2',
            't5': task_dir / 't5'
        }
        for model, mdir in models.items():
            # Prefer singlish from tmp if present
            tmp_dir = base / 'sentiment_analysis' / 'tmp' / model
            sing_fp = tmp_dir / 'singlish_comparison.csv'
            used_sing = False
            if sing_fp.exists():
                results.append({'path': str(sing_fp), 'model': model, 'modification': 'dialectal_100', 'task': task_name})
                used_sing = True
            for fp in mdir.glob('*_comparison.csv'):
                mod = fp.stem.replace('_comparison', '')
                if mod == 'dialectal' and used_sing:
                    continue
                results.append({'path': str(fp), 'model': model, 'modification': f'{mod}_100', 'task': task_name})
    elif task_name == 'ner':
        task_dir = base / 'ner'
        models = {
            'bert': task_dir / 'parsed_NER_BERT',
            'gpt2': task_dir / 'parsed_NER_GPT2',
            't5': task_dir / 'parsed_NER_T5'
        }
        for model, mdir in models.items():
            files = list(mdir.glob(f'{model}_ner_results_*.csv'))
            mods = [f.stem.replace(f'{model}_ner_results_', '') for f in files]
            for fp in files:
                mod = fp.stem.replace(f'{model}_ner_results_', '')
                if mod == 'dialectal' and 'singlish' in mods:
                    continue
                if mod == 'singlish':
                    mod = 'dialectal'
                results.append({'path': str(fp), 'model': model, 'modification': f'{mod}', 'task': task_name})
    return results

if __name__ == "__main__":
    main()
