#!/usr/bin/env python3
"""
Clean GSM (Grade School Math) Analysis Script
Processes LLM GSM results using accuracy metrics for mathematical reasoning
"""

import json
import argparse
import numpy as np
from scipy import stats
import pandas as pd
from utils import get_global_unrobustness_range, unrob_intensity
import os
import glob
from pathlib import Path
from tqdm import tqdm
try:
    # Load .env if present and default USE_LLM_GSM_PARSE
    from ..LLM.scripts.llm_utils import load_dotenv_if_present  # type: ignore
except Exception:
    try:
        from LLM.scripts.llm_utils import load_dotenv_if_present  # type: ignore
    except Exception:
        def load_dotenv_if_present():
            pass
load_dotenv_if_present()
import re
import os

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).parent
_U_MIN, _U_MAX = get_global_unrobustness_range()

def safe_wilcoxon(x, y, **kwargs):
    import numpy as _np
    from scipy.stats import wilcoxon as _wilcoxon
    xa = _np.asarray(x)
    ya = _np.asarray(y)
    mask = _np.isfinite(xa) & _np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    if xa.size == 0 or ya.size == 0 or xa.size != ya.size:
        return 0.0, 1.0
    diff = xa - ya
    if _np.all(diff == 0):
        return 0.0, 1.0
    try:
        return _wilcoxon(xa, ya, zero_method=kwargs.get('zero_method', 'wilcox'),
                         alternative=kwargs.get('alternative', 'two-sided'))
    except Exception:
        return 0.0, 1.0


def load_jsonl_file(file_path):
    """Load JSONL file (one JSON object per line)"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_numerical_answer(answer_text):
    """Extract the final numerical answer from a prediction string.

    Heuristics (in order):
    1) Prefer trailing '#### <number>' pattern.
    2) Prefer 'final answer: <number>' style.
    3) Prefer a number-only last line.
    4) Handle simple fractions like '3/4'.
    5) Fallback: last numeric token (supports commas and scientific notation).
    """
    if answer_text is None or (isinstance(answer_text, float) and pd.isna(answer_text)):
        return None

    s = str(answer_text).strip()
    if not s:
        return None

    import re

    def _parse_number(token: str):
        if token is None:
            return None
        tok = token.strip().replace(',', '')
        # fraction
        mfr = re.fullmatch(r'\s*([+-]?\d+)\s*/\s*(\d+)\s*', tok)
        if mfr:
            num = float(mfr.group(1))
            den = float(mfr.group(2))
            if den != 0:
                return num / den
        # plain/scientific
        try:
            return float(tok)
        except Exception:
            return None

    # 1) Simplified: take everything after the last '####' on that line
    if '####' in s:
        tail = s[s.rfind('####')+4:]
        tail = re.sub(r'^\s*[:,-]?\s*', '', tail)
        tail = re.sub(r'^[\$€£¥₹₽]\s*', '', tail)
        tail = tail.splitlines()[0].strip()
        # Strip non-digit markers like %
        tail = re.sub(r'[^0-9+\-eE./]', '', tail)
        v = _parse_number(tail)
        if v is not None:
            return v

    # 2) 'final answer: <number>' style
    m = re.search(r'(?:final\s+answer|answer)\s*[:\-]?\s*([+-]?(?:\d+|\d{1,3}(?:,\d{3})*)(?:\.\d+)?(?:[eE][+-]?\d+)?|[+-]?\d+\s*/\s*\d+)', s, flags=re.IGNORECASE)
    if m:
        v = _parse_number(m.group(1))
        if v is not None:
            return v

    # 3) last non-empty line number-only
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        last = lines[-1].strip().rstrip('.;:')
        # if last line contains only a number/fraction
        if re.fullmatch(r'[+-]?(?:\d+|\d{1,3}(?:,\d{3})*)(?:\.\d+)?(?:[eE][+-]?\d+)?', last) or re.fullmatch(r'[+-]?\d+\s*/\s*\d+', last):
            v = _parse_number(last)
            if v is not None:
                return v

    # 4) last fraction anywhere
    fracs = re.findall(r'[+-]?\d+\s*/\s*\d+', s)
    if fracs:
        v = _parse_number(fracs[-1])
        if v is not None:
            return v

    # 5) fallback to last numeric token (commas/scientific ok)
    nums = re.findall(r'[+-]?(?:\d+|\d{1,3}(?:,\d{3})*)(?:\.\d+)?(?:[eE][+-]?\d+)?', s)
    if nums:
        v = _parse_number(nums[-1])
        if v is not None:
            return v

    return None

def _final_num_from_row(row, side: str, model: str):
    """Extract numeric answer from CoT/raw.
    For non-GPT-5 models: prefer the LAST '#### <number>' across all relevant fields.
    Otherwise: fall back to first-available per-field extraction.
    Returns a float or None.
    """
    try:
        # Only exclude GPT-5 standard from forced #### preference; allow GPT-5 context-aware
        is_gpt5_std = model == 'gpt-5-standard'
        if side == 'original':
            fields = [
                row.get('original_step_by_step_reasoning', ''),
                row.get('original_raw_output', ''),
                row.get('original_reasoning', ''),
            ]
        else:
            fields = [
                row.get('modified_step_by_step_reasoning', row.get('step_by_step_reasoning', '')),
                row.get('raw_output', ''),
                row.get('modified_reasoning', row.get('reasoning', '')),
            ]
        if not is_gpt5_std:
            # Prefer last #### across concatenated fields
            import re as _re
            cat = ' '.join([str(x) for x in fields if isinstance(x, str) and x.strip()])
            last = None
            for m in _re.finditer(r"####\s*[$€£¥₹₽]?\s*([+-]?(?:\d+(?:\\.\d+)?(?:[eE][+-]?\d+)?|\d{1,3}(?:,\d{3})+(?:\\.\d+)?|\d+\s*/\s*\d+))", cat):
                last = m.group(1)
            if last:
                # Use exactly the numeric token after the last #### occurrence
                return extract_numerical_answer(last)
        # Fallback: first per-field parse
        for val in fields:
            v = extract_numerical_answer(val)
            if v is not None:
                return v
    except Exception:
        pass
    return None

def _llm_extract_number_from_texts(texts, model=None):
    """LLM-assisted numeric extraction (optional, requires USE_LLM_GSM_PARSE and OPENAI_API_KEY).
    Uses gpt-4.1 via OpenAI SDK if available. Returns float or None.
    """
    if not os.environ.get('USE_LLM_GSM_PARSE') or 'OPENAI_API_KEY' not in os.environ:
        return None
    model = model or os.environ.get('LLM_PREDICT_MODEL', 'gpt-4.1')
    payload = '\n\n'.join([str(t) for t in texts if isinstance(t, str) and str(t).strip()])
    if not payload:
        return None
    prompt = (
        "You will be given an assistant's reasoning and/or output for a GSM-style math question. "
        "Extract the assistant's final numeric answer only. Output NOTHING except the number. "
        "Allowed formats: integer, decimal, scientific notation, or a simple fraction like 3/4."
    )
    try:
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI()
            resp = client.responses.create(
                model=model,
                input=[{"role":"system","content":prompt},{"role":"user","content":payload}],
                temperature=0.0,
                max_output_tokens=10,
            )
            content = getattr(resp, 'output_text', None)
        except Exception:
            import openai  # type: ignore
            openai.api_key = os.environ['OPENAI_API_KEY']
            chat = openai.ChatCompletion.create(
                model=model,
                messages=[{"role":"system","content":prompt},{"role":"user","content":payload}],
                temperature=0.0,
                max_tokens=10,
            )
            content = chat.choices[0].message.get('content') if chat.choices else None
        if content:
            s = content.strip().splitlines()[0].strip()
            s = re.sub(r'^[$€£¥₹₽]\s*', '', s).replace(',', '')
            m = re.search(r'[+-]?(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\d+\s*/\s*\d+)', s)
            if m:
                return extract_numerical_answer(m.group(0))
    except Exception:
        return None
    return None


def _map_model_from_filename(filename: str) -> str:
    base = os.path.basename(filename)
    if base.startswith('gpt-5-standard-context-aware-'):
        return 'gpt-5-standard-context-aware'
    if base.startswith('gpt-5-standard-'):
        return 'gpt-5-standard'
    if base.startswith('gpt4o-') or base.startswith('gpt-4o-'):
        return 'gpt4o'
    if base.startswith('claude-'):
        return 'claude'
    if base.startswith('deepseek-'):
        # Normalize DeepSeek R1 naming for consistency across analyses
        return 'deepseek-r1'
    if base.startswith('llama-'):
        return 'llama'
    parts = base.split('-')
    model = parts[0]
    if model == 'gpt':
        return 'gpt-5-standard-context-aware' if 'context' in base else 'gpt-5-standard'
    return model


def _format_num(n: float) -> str:
    try:
        if n is None:
            return ''
        if abs(n - round(n)) < 1e-9:
            return str(int(round(n)))
        return str(n)
    except Exception:
        return str(n)


def _rewrite_predictions_in_df(df: pd.DataFrame, model: str) -> tuple[pd.DataFrame, int]:
    """Return a copy of df with original_pred/modified_pred overwritten using strict extraction.
    Uses last #### from CoT/raw for all except GPT-5 standard; falls back to existing logic otherwise.
    Returns (new_df, changed_count).
    """
    if df is None or df.empty:
        return df, 0
    dfc = df.copy()
    changed = 0
    n = len(dfc)
    for i in range(n):
        row = dfc.iloc[i]
        try:
            op_num = _final_num_from_row(row, 'original', model)
            mp_num = _final_num_from_row(row, 'modified', model)
            # Fallbacks if strict parse failed
            if op_num is None:
                op_num = extract_numerical_answer(row.get('original_pred', ''))
            if mp_num is None:
                mp_num = extract_numerical_answer(row.get('modified_pred', ''))
            # Apply if available and different
            if op_num is not None:
                new_op = _format_num(op_num)
                if str(row.get('original_pred', '')) != new_op:
                    dfc.at[i, 'original_pred'] = new_op
                    changed += 1
            if mp_num is not None:
                new_mp = _format_num(mp_num)
                if str(row.get('modified_pred', '')) != new_mp:
                    dfc.at[i, 'modified_pred'] = new_mp
                    changed += 1
        except Exception:
            continue
    return dfc, changed


def process_llm_results(results_dir, rewrite_predictions: bool = False):
    """Process LLM results from CSV files"""
    results_files = [
        f for f in glob.glob(os.path.join(results_dir, '*.csv'))
        if ('_backup.csv' not in f and 'negation_change' not in f)
    ]
    
    results_data = []
    
    print("Processing LLM results...")
    
    for results_file in tqdm(results_files, desc="Processing files"):
        filename = os.path.basename(results_file)
        
        # Skip certain files - GSM has different naming pattern
        if any(skip in filename for skip in ['DP', 'comparison_gsm.csv']):
            continue
            
        # Extract model and modification from filename
        # GSM format: {model}-{model}-0shot-{modification}_100.csv
        parts = filename.split('-')
        if len(parts) < 4:
            continue
            
        model = parts[0]
        
        # Map model names
        model = _map_model_from_filename(filename)
            
        modification = parts[-1].replace('_100.csv', '')
        
        # Check if corresponding comparison file exists
        compare_file = (SCRIPT_DIR / f'../../data/modified_data/gsm/{modification}_100.jsonl')
        if not compare_file.exists():
            continue
            
        # Read the CSV file
        try:
            df = pd.read_csv(results_file)
        except Exception as e:
            print(f'Error reading {results_file}: {e}')
            continue
        # Overwrite predictions in-memory (and optionally on-disk) using strict extraction
        try:
            df_fixed, changed = _rewrite_predictions_in_df(df, model)
            if changed:
                print(f"[fix] {os.path.basename(results_file)}: corrected {changed} prediction fields")
            if rewrite_predictions and changed:
                backup = results_file + '.backup'
                try:
                    if not os.path.exists(backup):
                        os.rename(results_file, backup)
                except Exception:
                    pass
                df_fixed.to_csv(results_file, index=False)
                df = pd.read_csv(results_file)  # reload to ensure consistency
            else:
                df = df_fixed
        except Exception as _e:
            print(f"[warn] could not rewrite preds for {os.path.basename(results_file)}: {_e}")
            
        # Load comparison data
        try:
            compare_df = load_jsonl_file(compare_file)
        except Exception as e:
            print(f'Error reading {compare_file}: {e}')
            continue
        
        if len(compare_df) != len(df):
            if str(modification).startswith('negation'):
                print(f'[WARN] Length mismatch (negation allowed): {modification}, {model} - dataset {len(compare_df)} vs csv {len(df)}; proceeding with index-based join')
            else:
                print(f'Length mismatch: {modification}, {model} - {len(compare_df)} vs {len(df)}')
                continue
            
        # Get labels and predictions
        # Precompute dataset map by index for safer joins (negation needs strict mapping)
        ds_by_index = {}
        if modification.startswith('negation'):
            try:
                for obj in compare_df:
                    idx = obj.get('index')
                    if idx is None:
                        continue
                    try:
                        idx = int(idx)
                    except Exception:
                        continue
                    oa_val = obj.get('original_answer', obj.get('short_answer'))
                    subtype_val = str(obj.get('negation_subtype', obj.get('type', ''))).lower()
                    ds_by_index[idx] = (oa_val, subtype_val)
            except Exception:
                ds_by_index = {}
        
        if modification.startswith('negation'):
            # Do not rely on positional alignment; we'll look up by index row-by-row.
            ori_labels = []
            mod_labels = []
            # Placeholder arrays to keep shape; not used for correctness directly
            for _ in range(len(compare_df)):
                ori_labels.append(0)
                mod_labels.append(0)
        else:
            # For non-negation GSM, keep prior behavior (use CSV labels when present)
            if 'original_answer' in df.columns:
                ori_labels = df['original_answer'].values
                mod_labels = df['modified_answer'].values
            else:
                # Fallback to dataset JSONL short_answer (answers are unchanged for non-negation GSM)
                ori_labels = [item.get('short_answer') for item in compare_df]
                mod_labels = ori_labels
        # Ensure negation subtype is available for consistent evaluation across models
        types_series = None
        if modification.startswith('negation'):
            # We'll fetch subtype per-row via ds_by_index; keep an empty placeholder list for debug
            types_series = []
            
        # Predictions presence check (we will prefer parsed columns when available for non-DeepSeek/LLaMA)
        if 'original_pred' not in df.columns or 'modified_pred' not in df.columns:
            print(f'No prediction columns found in {filename}')
            continue
        
        # Convert to numerical values for comparison (non-negation path uses prefilled labels; negation uses per-row lookups)
        ori_labels_num = [extract_numerical_answer(x) for x in ori_labels]
        mod_labels_num = [extract_numerical_answer(x) for x in mod_labels]
        
        # Calculate accuracy scores
        ori_correct = 0
        mod_correct = 0
        total = 0
        
        ori_correct_list = []
        mod_correct_list = []
        
        for i in range(len(df)):
            # Compute correctness flags for each row
            if modification.startswith('negation'):
                try:
                    idx_val = df.iloc[i]['index'] if 'index' in df.columns else None
                    idx_int = int(idx_val) if idx_val is not None else None
                except Exception:
                    idx_int = None
                oa_raw, subtype = (None, '')
                if idx_int is not None and idx_int in ds_by_index:
                    oa_raw, subtype = ds_by_index.get(idx_int, (None, ''))
                oa_num = extract_numerical_answer(oa_raw)
                # Prefer CoT/raw with LAST #### for all models; fallback to preds/parsed/LLM
                op_num = _final_num_from_row(df.iloc[i], 'original', model)
                mp_num = _final_num_from_row(df.iloc[i], 'modified', model)
                if op_num is None:
                    op_num = extract_numerical_answer(df.iloc[i].get('original_pred', ''))
                if mp_num is None:
                    mp_num = extract_numerical_answer(df.iloc[i].get('modified_pred', ''))
                if op_num is None and 'parsed_original_pred' in df.columns:
                    op_num = extract_numerical_answer(df.iloc[i].get('parsed_original_pred', ''))
                if mp_num is None and 'parsed_modified_pred' in df.columns:
                    mp_num = extract_numerical_answer(df.iloc[i].get('parsed_modified_pred', ''))
                if op_num is None:
                    op_num = _llm_extract_number_from_texts([
                        df.iloc[i].get('original_step_by_step_reasoning', ''),
                        df.iloc[i].get('original_raw_output', ''),
                        df.iloc[i].get('original_reasoning', ''),
                    ]) or extract_numerical_answer(df.iloc[i].get('original_pred', ''))
                if mp_num is None:
                    mp_num = _llm_extract_number_from_texts([
                        df.iloc[i].get('modified_step_by_step_reasoning', ''),
                        df.iloc[i].get('step_by_step_reasoning', ''),
                        df.iloc[i].get('raw_output', ''),
                        df.iloc[i].get('modified_reasoning', ''),
                        df.iloc[i].get('reasoning', ''),
                    ]) or extract_numerical_answer(df.iloc[i].get('modified_pred', ''))
                if oa_num is None or op_num is None or mp_num is None:
                    continue
                ori_match = abs(oa_num - op_num) < 1e-6
                eq = abs(oa_num - mp_num) < 1e-6
                st = (subtype or '').lower()
                if ('approximate' in st) or ('double' in st):
                    mod_match = eq
                else:
                    mod_match = not eq
            else:
                # Prefer CoT/raw with LAST #### for all models; fallback to preds/parsed/LLM
                op_num = _final_num_from_row(df.iloc[i], 'original', model)
                mp_num = _final_num_from_row(df.iloc[i], 'modified', model)
                if op_num is None:
                    op_num = extract_numerical_answer(df.iloc[i].get('original_pred', ''))
                if mp_num is None:
                    mp_num = extract_numerical_answer(df.iloc[i].get('modified_pred', ''))
                if op_num is None and 'parsed_original_pred' in df.columns:
                    op_num = extract_numerical_answer(df.iloc[i].get('parsed_original_pred', ''))
                if mp_num is None and 'parsed_modified_pred' in df.columns:
                    mp_num = extract_numerical_answer(df.iloc[i].get('parsed_modified_pred', ''))
                if op_num is None:
                    op_num = _llm_extract_number_from_texts([
                        df.iloc[i].get('original_step_by_step_reasoning', ''),
                        df.iloc[i].get('original_raw_output', ''),
                        df.iloc[i].get('original_reasoning', ''),
                    ]) or extract_numerical_answer(df.iloc[i].get('original_pred', ''))
                if mp_num is None:
                    mp_num = _llm_extract_number_from_texts([
                        df.iloc[i].get('modified_step_by_step_reasoning', ''),
                        df.iloc[i].get('step_by_step_reasoning', ''),
                        df.iloc[i].get('raw_output', ''),
                        df.iloc[i].get('modified_reasoning', ''),
                        df.iloc[i].get('reasoning', ''),
                    ]) or extract_numerical_answer(df.iloc[i].get('modified_pred', ''))
                # Skip row if labels or predictions are not parseable
                if (ori_labels_num[i] is None or op_num is None or
                    mod_labels_num[i] is None or mp_num is None):
                    continue
                # Use precomputed label numbers for non-negation
                ori_match = abs(ori_labels_num[i] - op_num) < 1e-6
                mod_match = abs(mod_labels_num[i] - mp_num) < 1e-6

            # Update aggregates for both branches
            if ori_match:
                ori_correct += 1
            if mod_match:
                mod_correct += 1
            ori_correct_list.append(1 if ori_match else 0)
            mod_correct_list.append(1 if mod_match else 0)
            total += 1

        # Optional debug for negation GSM: sample a few rows to validate scoring
        if modification.startswith('negation') and os.getenv('DEBUG_GSM_NEGATION', '0') == '1':
            try:
                # Compute quick stats
                ori_binary_dbg = np.array(ori_correct_list)
                mod_binary_dbg = np.array(mod_correct_list)
                flips_dbg = np.abs(ori_binary_dbg - mod_binary_dbg)
                print(f"[DEBUG] GSM Negation — {model} {modification}: N={total}, OriAcc={ori_binary_dbg.mean()*100:.1f}%, ModAcc(inv)={mod_binary_dbg.mean()*100:.1f}%, Flips={flips_dbg.mean()*100:.1f}%")
                # Show up to 5 sample rows with a flip
                shown = 0
                tol = 1e-6
                for j in range(len(df)):
                    if shown >= 5:
                        break
                    # Fetch dataset gold and subtype by index
                    try:
                        idx_val = df.iloc[j].get('index', j)
                        idx_int = int(idx_val) if idx_val is not None else None
                    except Exception:
                        idx_int = None
                    oa_raw, subtype = (None, '')
                    if idx_int is not None and idx_int in ds_by_index:
                        oa_raw, subtype = ds_by_index.get(idx_int, (None, ''))
                    oa_num = extract_numerical_answer(oa_raw)
                    # Parse predictions with fallbacks
                    op_num = extract_numerical_answer(df.iloc[j].get('original_pred', ''))
                    mp_num = extract_numerical_answer(df.iloc[j].get('modified_pred', ''))
                    if op_num is None:
                        tmp1 = extract_numerical_answer(df.iloc[j].get('original_step_by_step_reasoning', ''))
                        if tmp1 is None:
                            tmp1 = extract_numerical_answer(df.iloc[j].get('original_raw_output', ''))
                        op_num = tmp1
                    if mp_num is None:
                        tmp2 = extract_numerical_answer(df.iloc[j].get('step_by_step_reasoning', ''))
                        if tmp2 is None:
                            tmp2 = extract_numerical_answer(df.iloc[j].get('raw_output', ''))
                        mp_num = tmp2
                    if oa_num is None or op_num is None or mp_num is None:
                        continue
                    # Compute matches
                    ori_match_j = abs(oa_num - op_num) < tol
                    st = (subtype or '').lower()
                    eqm = abs(oa_num - mp_num) < tol
                    mod_match_j = eqm if (('approximate' in st) or ('double' in st)) else (not eqm)
                    if (ori_match_j != mod_match_j):
                        print(f"  [flip] idx={idx_val} subtype={st} oa={oa_num} op={op_num} mp={mp_num} -> ori={ori_match_j} mod={mod_match_j}")
                        shown += 1
            except Exception as _e:
                print(f"[DEBUG] GSM Negation debug failed: {_e}")
        
        if total == 0:
            continue
            
        ori_acc = ori_correct / total * 100
        mod_acc = mod_correct / total * 100
        
        # Calculate weighted delta and absolute change
        weighted_delta = (mod_acc - ori_acc) * np.log10(ori_acc) / np.log10(100) if ori_acc > 0 else 0
        absolute_change = abs(mod_acc - ori_acc)  # Absolute change in accuracy
        rate_of_change = absolute_change  # Keep for backward compatibility
        
        # Convert to numpy arrays for statistical tests
        ori_binary = np.array(ori_correct_list)
        mod_binary = np.array(mod_correct_list)
        # Unrobustness: discordant rate (any correctness flip) across all rows
        flips = np.abs(ori_binary - mod_binary)
        unrobustness = flips.mean() * 100.0
        
        # Statistical tests for directional change (paired)
        # Wilcoxon signed-rank on paired binary correctness
        try:
            _, p_value_w = safe_wilcoxon(ori_binary, mod_binary)
        except Exception:
            p_value_w = 1.0
        # McNemar's test (exact binomial on discordant pairs)
        n01 = int(((ori_binary == 1) & (mod_binary == 0)).sum())  # orig better
        n10 = int(((ori_binary == 0) & (mod_binary == 1)).sum())  # mod better
        n_discordant = n01 + n10
        if n_discordant > 0:
            try:
                k = min(n01, n10)
                p_value_mcn = stats.binomtest(k, n_discordant, 0.5, alternative='two-sided').pvalue
            except Exception:
                p_value_mcn = 1.0
        else:
            p_value_mcn = 1.0
        p_value = min(p_value_w, p_value_mcn)

        # Absolute change is descriptive; do not compute separate significance
        abs_p_value = np.nan
        abs_significance = 'n/a'
        
        # Determine significance levels
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        elif p_value < 0.1:
            significance = "."
        else:
            significance = "ns"
        
        # Store main results
        results_data.append({
            'model': model,
            'modification': modification,
            'original_acc': ori_acc,
            'modified_acc': mod_acc,
            'weighted_delta': weighted_delta,
            'absolute_change': absolute_change,
            'rate_of_change': rate_of_change,  # Keep for backward compatibility
            'unrobustness': unrobustness,
            'p_value': p_value,
            'significance': significance,
            'abs_p_value': abs_p_value,
            'abs_significance': abs_significance,
            'total_samples': total
        })
    
    return results_data


def save_llm_results(llm_results):
    """Save LLM results to CSV files"""
    
    # Save main results
    results_df = pd.DataFrame(llm_results)
    results_df.to_csv('gsm_modification_results_llm.csv', index=False)
    
    print("LLM results saved:")
    print("- gsm_modification_results_llm.csv")
    
    return results_df


def generate_latex_table(df, output_file):
    """Generate LaTeX table from results DataFrame"""
    # Guard against empty or malformed DataFrame
    if df is None or df.empty or 'model' not in df.columns or 'modification' not in df.columns:
        print(f"Skipping LaTeX generation for {output_file}: no data or missing columns")
        return
    # Avoid mutating caller's DataFrame; compact views rely on raw keys
    df = df.copy()
    
    # Define mappings
    mod_mapping = {
        'temporal_bias': ('Bias', 'Temporal'),
        'geographical_bias': ('Bias', 'Geographical'), 
        'length_bias': ('Bias', 'Length'),
        'typo_bias': ('Orthography', 'Spelling'),
        'capitalization': ('Orthography', 'Capitalization'),
        'punctuation': ('Orthography', 'Punctuation'),
        'concept_replacement': ('Semantics', 'Concept'),
        'negation': ('Semantics', 'Negation'),
        'sentiment': ('Discourse', 'Appraisal'),
        'casual': ('Varieties', 'Style'),
        'dialectal': ('Varieties', 'Dialect'),
        'coordinating_conjunction': ('Syntax', 'Conjunction'),
        'active_to_passive': ('Syntax', 'Voice')
    }
    
    model_order = ['GPT-4o', 'Claude-3.5', 'Llama 3.1', 'GPT-5','DS R1', 'GPT-5 (w. context)']
    model_map = {
        'gpt4o': 'GPT-4o', 
        'claude': 'Claude-3.5', 
        'llama': 'Llama 3.1', 
        'gpt-5-standard': 'GPT-5', 
        'deepseek-r1-deepseek': 'DS R1',
        'deepseek-r1': 'DS R1',
        'gpt-5-standard-context-aware': 'GPT-5 (w. context)'
    }
    
    df['model'] = df['model'].replace(model_map)
    df['category'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    df['modification'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    
    # Create pivot tables (aggregate duplicates by mean/first)
    pivot_df = df.pivot_table(index=['category', 'modification'], columns='model', values='weighted_delta', aggfunc='mean')
    significance = df.pivot_table(index=['category', 'modification'], columns='model', values='significance', aggfunc='first')
    unrob_df = df.pivot_table(index=['category', 'modification'], columns='model', values='unrobustness', aggfunc='mean')

    # Write separate Δ and U tables with colored cells
    ordered_models = [m for m in model_order if m in pivot_df.columns and m != 'GPT-5 (w. context)']

    def delta_cell(val, sig):
        if pd.isna(val):
            return ''
        intensity = min(abs(float(val))/10.0, 1.0)
        color = 'green' if float(val) > 0 else 'red'
        txt = f"{float(val):+.1f}"
        if isinstance(sig, str):
            if sig == '.':
                txt = f"\\textbf{{{txt}}}"
            elif sig == '*':
                txt = f"\\textbf{{{txt}}}*"
            elif sig == '**':
                txt = f"\\textbf{{{txt}}}**"
            elif sig == '***':
                txt = f"\\textbf{{{txt}}}***"
        return f"\\cellcolor{{{color}!{int(intensity*20)}}} {txt}"

    def unrob_cell(u):
        if pd.isna(u):
            return ''
        try:
            val = float(u)
        except Exception:
            return ''
        inten = unrob_intensity(val, _U_MIN, _U_MAX)
        txt = f"{val:.1f}"
        if inten >= 45:
            txt = f"\\textcolor{{white}}{{{txt}}}"
        return f"\\cellcolor{{blue!{inten}}} {txt}"

    # Delta table (colored) with Avg column and Avg row
    total = len(ordered_models)
    latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll' + 'r'*total + 'r}\n'
    latex += '\\toprule\n'
    latex += 'Category & Modification & ' + ' & '.join([f'\\textbf{{{m}}}' for m in ordered_models]) + ' & \\textbf{Avg} \\\\\n'
    latex += '\\midrule\n'
    row_avgs = []
    for (cat, mod), _ in pivot_df.iterrows():
        vals = []
        cells = []
        for m in ordered_models:
            v = pivot_df.loc[(cat, mod), m] if m in pivot_df.columns else float('nan')
            vals.append(v)
            s = significance.loc[(cat, mod), m] if ((cat, mod) in significance.index and m in significance.columns) else 'ns'
            cells.append(delta_cell(v, s))
        ravg = float(np.nanmean(vals)) if len(vals) else float('nan')
        row_avgs.append(ravg)
        latex += f"{cat} & {mod} & " + ' & '.join(cells) + f" & {'' if np.isnan(ravg) else f'{ravg:+.1f}'} \\\\\n"
    # Average row per model
    col_avgs = []
    for m in ordered_models:
        series = pivot_df.xs(m, axis=1, drop_level=False)[m]
        col_avgs.append(float(series.mean()))
    overall_avg = float(np.nanmean(col_avgs)) if col_avgs else float('nan')
    latex += '\\midrule\n'
    latex += '\\textbf{Average} &  ' + ' & '.join([f"{v:+.1f}" if not np.isnan(v) else '' for v in col_avgs]) + f" & {'' if np.isnan(overall_avg) else f'{overall_avg:+.1f}'} \\\\\n"
    latex += '\\bottomrule\n\\end{tabular}}\n\\caption{GSM: Weighted $\\Delta$ by model and modification}\\label{tab:gsm_delta}\n\\end{table}'
    latex = latex.replace('\\n', '\n')
    with open('gsm_results_table_delta.tex','w') as f:
        f.write(latex)

    # Unrobustness table (colored) with Avg column and Avg row
    total_u = len(ordered_models)
    latex_u = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll' + 'r'*total_u + 'r}\n'
    latex_u += '\\toprule\n'
    latex_u += 'Category & Modification & ' + ' & '.join([f'\\textbf{{{m}}}' for m in ordered_models]) + ' & \\textbf{Avg} \\\\\n'
    latex_u += '\\midrule\n'
    for (cat, mod), _ in unrob_df.iterrows():
        vals=[]
        cells = []
        for m in ordered_models:
            u = unrob_df.loc[(cat, mod), m] if m in unrob_df.columns else float('nan')
            vals.append(u)
            cells.append(unrob_cell(u))
        ravg = float(np.nanmean(vals)) if len(vals) else float('nan')
        latex_u += f"{cat} & {mod} & " + ' & '.join(cells) + f" & {'' if np.isnan(ravg) else unrob_cell(ravg)} \\\\\n"
    # Average row per model
    col_u_avgs=[]
    for m in ordered_models:
        series = unrob_df.xs(m, axis=1, drop_level=False)[m]
        col_u_avgs.append(float(series.mean()))
    overall_u = float(np.nanmean(col_u_avgs)) if col_u_avgs else float('nan')
    latex_u += '\\midrule\n'
    latex_u += '\\textbf{Average} &  ' + ' & '.join([unrob_cell(v) if not np.isnan(v) else '' for v in col_u_avgs]) + f" & {'' if np.isnan(overall_u) else unrob_cell(overall_u)} \\\\\n"
    latex_u += '\\bottomrule\n\\end{tabular}}\n\\caption{GSM: Unrobustness (U, \\%) by model and modification}\\label{tab:gsm_unrob}\n\\end{table}'
    latex_u = latex_u.replace('\\n', '\n')
    with open('gsm_results_table_unrobustness.tex','w') as f:
        f.write(latex_u)

    print("LaTeX tables saved: gsm_results_table_delta.tex, gsm_results_table_unrobustness.tex")
    return

    # Only LLM models in GSM; order by present columns
    ordered_models = [m for m in model_order if m in pivot_df.columns and m != 'GPT-5 (w. context)']
    
    def get_color(val, sig):
        """Generate LaTeX color formatting based on value and significance"""
        if np.isnan(val):
            return ''
        elif val > 0:
            intensity = min(abs(val)/10, 1)
            val_str = f'+{val:.2f}'
        else:
            intensity = min(abs(val)/10, 1)
            val_str = f'{val:.2f}'
        
        # Add significance markers
        if sig == '.':
            val_str = f'\\textbf{{{val_str}}}'
        elif sig == '*':
            val_str = f'\\textbf{{{val_str}}}*'
        elif sig == '**':
            val_str = f'\\textbf{{{val_str}}}**'
        elif sig == '***':
            val_str = f'\\textbf{{{val_str}}}***'
        
        # Color based on weighted delta magnitude (green for positive, red for negative)
        color = 'green' if val > 0 else 'red'
        return f'\\cellcolor{{{color}!{int(intensity*20)}}} {val_str}'

    def get_unrob_cell(u):
        if np.isnan(u):
            return ''
        inten = unrob_intensity(float(u), _U_MIN, _U_MAX)
        txt = f"{u:.1f}"
        if inten >= 45:
            txt = f"\\textcolor{{white}}{{{txt}}}"
        return f'\\cellcolor{{blue!{inten}}} {txt}'
    
    # Generate LaTeX table
    total_models = len(ordered_models)
    latex_table = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll' + 'rr'*total_models + '}\n'
    latex_table += '\\toprule\n'
    # Single LLM group header
    latex_table += 'Category & Modification & ' + f'\\multicolumn{{{2*total_models}}}{{c}}{{\\textbf{{LLM}}}} \\\\\n'
    # Model header and subheader
    model_spans = [f'\\multicolumn{{2}}{{c}}{{\\textbf{{{col}}}}}' for col in ordered_models]
    latex_table += ' &  & ' + ' & '.join(model_spans) + ' \\\\\n'
    sub_headers = []
    for _ in ordered_models:
        sub_headers += ['\\textbf{$\\Delta$}', '\\textbf{U}']
    latex_table += ' &  & ' + ' & '.join(sub_headers) + ' \\\\\n'
    latex_table += '\\midrule\n'
    
    categories_seen = []
    for mod_key in mod_mapping:
        category, modification = mod_mapping[mod_key]
        if (category, modification) not in pivot_df.index:
            continue
            
        if category not in categories_seen:
            if categories_seen:
                latex_table += '\\midrule\n'
            categories_seen.append(category)
            row_start = f'\\textbf{{{category}}}'
        else:
            row_start = ' '
        
        latex_table += f'{row_start} & \\textbf{{{modification}}} & '
        row_values = []
        for col in ordered_models:
            # Delta cell
            if (category, modification) in pivot_df.index and col in pivot_df.columns:
                delta_val = get_color(pivot_df.loc[(category, modification), col], 
                                      significance.loc[(category, modification), col] if (category, modification) in significance.index and col in significance.columns else 'ns')
            else:
                delta_val = ''
            row_values.append(delta_val)
            # Unrobustness cell
            if (category, modification) in unrob_df.index and col in unrob_df.columns:
                u = unrob_df.loc[(category, modification), col]
                unrob_val = get_unrob_cell(u)
            else:
                unrob_val = ''
            row_values.append(unrob_val)
        latex_table += ' & '.join(row_values) + ' \\\\\n'
    
    latex_table += '\\bottomrule\n\\end{tabular}}\n'
    latex_table += '\\caption{Weighted Delta ($\\Delta$) and Unrobustness (U) by Model and Modification Type (GSM Math)}\n'
    latex_table += '\\label{tab:gsm_results}\n\\end{table}'
    
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to {output_file}")


def generate_summary_plot(df, output_file):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import numpy.ma as ma
    except Exception as e:
        print(f"Skipping plot generation: matplotlib not available ({e})")
        return

    if df is None or df.empty or 'model' not in df.columns or 'modification' not in df.columns:
        print(f"Skipping plot generation for {output_file}: no data or missing columns")
        return

    mod_mapping = {
        'temporal_bias': ('Bias', 'Temporal'),
        'geographical_bias': ('Bias', 'Geographical'), 
        'length_bias': ('Bias', 'Length'),
        'typo_bias': ('Orthographic', 'Spelling'),
        'capitalization': ('Orthographic', 'Capitalization'),
        'punctuation': ('Orthographic', 'Punctuation'),
        'concept_replacement': ('Semantic', 'Concept'),
        'negation': ('Semantic', 'Negation'),
        'sentiment': ('Pragmatic', 'Sentiment'),
        'casual': ('Genre', 'Casual'),
        'dialectal': ('Genre', 'Dialectal'),
        'coordinating_conjunction': ('Syntactic', 'Conjunction'),
        'active_to_passive': ('Syntactic', 'Voice')
    }
    model_order = ['GPT-4o', 'Claude-3.5', 'Llama 3.1', 'GPT-5','DS R1', 'GPT-5 (w. context)']
    model_map = {
        'gpt4o': 'GPT-4o', 'claude': 'Claude-3.5', 'llama': 'Llama 3.1', 'gpt-5-standard': 'GPT-5', 'deepseek-r1-deepseek': 'DS R1', 'gpt-5-standard-context-aware': 'GPT-5 (w. context)'
    }

    df = df.copy()
    df['model'] = df['model'].replace(model_map)
    df['category'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    df['modification_disp'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    ordered_mods = [(cat, name) for key, (cat, name) in mod_mapping.items()]
    present = set(zip(df['category'], df['modification_disp']))
    ordered_mods = [m for m in ordered_mods if m in present]
    pivot_delta = df.pivot_table(index=['category', 'modification_disp'], columns='model', values='weighted_delta', aggfunc='mean')
    pivot_sig = df.pivot_table(index=['category', 'modification_disp'], columns='model', values='significance', aggfunc='first')
    cols = [m for m in model_order if m in pivot_delta.columns]
    if not cols or not ordered_mods:
        print("Skipping plot: insufficient data after pivoting")
        return
    pivot_delta = pivot_delta.reindex(index=ordered_mods, columns=cols)
    pivot_sig = pivot_sig.reindex(index=ordered_mods, columns=cols)
    data = pivot_delta.values.astype(float)
    data_masked = ma.masked_invalid(data)
    h = max(3, 0.45 * data_masked.shape[0] + 1.5)
    w = max(6, 0.6 * data_masked.shape[1] + 2.0)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(data_masked, cmap='RdYlGn', vmin=-10, vmax=10, aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weighted Δ (positive is better)', rotation=90)
    ylabels = [f"{cat} · {mod}" for (cat, mod) in pivot_delta.index]
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha='right')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            s = pivot_sig.iat[i, j] if (pivot_sig is not None) else ''
            sig = s if isinstance(s, str) and s in {'.','*','**','***'} else ''
            txt = f"{val:+.1f}{sig if sig!='.' else ''}"
            ax.text(j, i, txt, ha='center', va='center', fontsize=8, color='black')
    fig.tight_layout()
    try:
        fig.savefig(output_file, dpi=200)
        print(f"Summary plot saved to {output_file}")
    finally:
        plt.close(fig)


def generate_unrobustness_plot(df, output_file):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import numpy.ma as ma
    except Exception as e:
        print(f"Skipping unrobustness plot: matplotlib not available ({e})")
        return

    needed = {'model','modification','unrobustness'}
    if df is None or df.empty or not needed.issubset(df.columns):
        print(f"Skipping unrobustness plot for {output_file}: missing data/columns")
        return

    mod_mapping = {
        'temporal_bias': ('Bias', 'Temporal'),
        'geographical_bias': ('Bias', 'Geographical'), 
        'length_bias': ('Bias', 'Length'),
        'typo_bias': ('Orthographic', 'Spelling'),
        'capitalization': ('Orthographic', 'Capitalization'),
        'punctuation': ('Orthographic', 'Punctuation'),
        'concept_replacement': ('Semantic', 'Concept'),
        'negation': ('Semantic', 'Negation'),
        'sentiment': ('Pragmatic', 'Sentiment'),
        'casual': ('Genre', 'Casual'),
        'dialectal': ('Genre', 'Dialectal'),
        'coordinating_conjunction': ('Syntactic', 'Conjunction'),
        'active_to_passive': ('Syntactic', 'Voice')
    }
    model_order = ['GPT-4o', 'Claude-3.5', 'Llama 3.1', 'GPT-5','DS R1', 'GPT-5 (w. context)']
    model_map = {
        'gpt4o': 'GPT-4o', 'claude': 'Claude-3.5', 'llama': 'Llama 3.1', 'gpt-5-standard': 'GPT-5', 'deepseek-r1-deepseek': 'DS R1', 'gpt-5-standard-context-aware': 'GPT-5 (w. context)'
    }
    df = df.copy()
    df['model'] = df['model'].replace(model_map)
    df['category'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    df['modification_disp'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    ordered_mods = [(cat, name) for key, (cat, name) in mod_mapping.items()]
    present = set(zip(df['category'], df['modification_disp']))
    ordered_mods = [m for m in ordered_mods if m in present]
    pivot_u = df.pivot_table(index=['category', 'modification_disp'], columns='model', values='unrobustness', aggfunc='mean')
    cols = [m for m in model_order if m in pivot_u.columns]
    if pivot_u.empty or not cols or not ordered_mods:
        print("Skipping unrobustness plot: insufficient data after pivoting")
        return
    pivot_u = pivot_u.reindex(index=ordered_mods, columns=cols)
    data = pivot_u.values.astype(float)
    data_masked = ma.masked_invalid(data)
    h = max(3, 0.45 * data_masked.shape[0] + 1.5)
    w = max(6, 0.6 * data_masked.shape[1] + 2.0)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(data_masked, cmap='Blues', vmin=_U_MIN, vmax=_U_MAX, aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Unrobustness U (flip \%)', rotation=90)
    ylabels = [f"{cat} · {mod}" for (cat, mod) in pivot_u.index]
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha='right')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=8, color='black')
    fig.tight_layout()
    try:
        fig.savefig(output_file, dpi=200)
        print(f"Unrobustness plot saved to {output_file}")
    finally:
        plt.close(fig)


def generate_compact_summary_plot(df, output_file):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import numpy.ma as ma
    except Exception as e:
        print(f"Skipping compact plot: matplotlib not available ({e})")
        return

    if df is None or df.empty or 'model' not in df.columns or 'modification' not in df.columns:
        print(f"Skipping compact plot {output_file}: no data or missing columns")
        return

    mod_to_cat = {
        'temporal_bias': 'Bias', 'geographical_bias': 'Bias', 'length_bias': 'Bias',
        'typo_bias': 'Orthographic','capitalization': 'Orthographic','punctuation': 'Orthographic',
        'concept_replacement': 'Semantic','negation': 'Semantic',
        'sentiment': 'Pragmatic','casual': 'Genre','dialectal': 'Genre',
        'coordinating_conjunction': 'Syntactic','active_to_passive': 'Syntactic'
    }
    model_order = ['GPT-4o','Claude-3.5','Llama 3.1','GPT-5','DS R1','GPT-5 (w. context)']
    model_map = {'gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','deepseek-r1':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    df = df.copy(); df['model'] = df['model'].replace(model_map); df['category'] = df['modification'].map(lambda x: mod_to_cat.get(x,'Other'))
    agg = df.groupby(['category','model']).agg(weighted_delta=('weighted_delta','mean')).reset_index()
    overall = df.groupby(['model']).agg(weighted_delta=('weighted_delta','mean')).reset_index(); overall.insert(0,'category','Overall')
    agg = pd.concat([agg, overall], ignore_index=True)
    pivot = agg.pivot_table(index='category', columns='model', values='weighted_delta', aggfunc='mean')
    cols = [m for m in model_order if m in pivot.columns and m != 'GPT-5 (w. context)']; rows = [*sorted([r for r in pivot.index if r!='Overall']), 'Overall'] if 'Overall' in pivot.index else sorted(list(pivot.index))
    if not cols or not rows: print("Skipping compact plot: insufficient data after aggregation"); return
    pivot = pivot.reindex(index=rows, columns=cols)
    data = pivot.values.astype(float); data_masked = ma.masked_invalid(data)
    h = max(3, 0.6*data_masked.shape[0]+1.0); w = max(6, 0.6*data_masked.shape[1]+2.0)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(data_masked, cmap='RdYlGn', vmin=-10, vmax=10, aspect='auto')
    plt.colorbar(im, ax=ax).set_label('Weighted Δ (avg by category)')
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha='right')
    ax.set_title(''); ax.set_xlabel(''); ax.set_ylabel('')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i,j]
            if np.isnan(v):
                continue
            ax.text(j,i,f"{v:+.1f}", ha='center', va='center', fontsize=8)
    fig.tight_layout();
    try:
        fig.savefig(output_file, dpi=200); print(f"Compact summary plot saved to {output_file}")
    finally:
        plt.close(fig)


def generate_compact_latex_table(df, output_file):
    if df is None or df.empty or 'model' not in df.columns or 'modification' not in df.columns:
        print(f"Skipping compact LaTeX for {output_file}: no data or missing columns"); return
    mod_to_cat = {
        'temporal_bias': 'Bias', 'geographical_bias': 'Bias', 'length_bias': 'Bias',
        'typo_bias': 'Orthographic','capitalization': 'Orthographic','punctuation': 'Orthographic',
        'concept_replacement': 'Semantic','negation': 'Semantic',
        'sentiment': 'Pragmatic','casual': 'Genre','dialectal': 'Genre',
        'coordinating_conjunction': 'Syntactic','active_to_passive': 'Syntactic'
    }
    model_order = ['GPT-4o','Claude-3.5','Llama 3.1','GPT-5','DS R1','GPT-5 (w. context)']
    model_map = {'gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','deepseek-r1':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    df = df.copy(); df['model'] = df['model'].replace(model_map); df['category'] = df['modification'].map(lambda x: mod_to_cat.get(x,'Other'))
    agg = df.groupby(['category','model']).agg(weighted_delta=('weighted_delta','mean'), unrobustness=('unrobustness','mean')).reset_index()
    overall = df.groupby(['model']).agg(weighted_delta=('weighted_delta','mean'), unrobustness=('unrobustness','mean')).reset_index(); overall.insert(0,'category','Overall')
    agg = pd.concat([agg, overall], ignore_index=True)
    p_delta = agg.pivot_table(index='category', columns='model', values='weighted_delta', aggfunc='mean')
    p_unrob = agg.pivot_table(index='category', columns='model', values='unrobustness', aggfunc='mean')
    cols = [m for m in model_order if m in p_delta.columns and m != 'GPT-5 (w. context)']; rows = [*sorted([r for r in p_delta.index if r!='Overall']), 'Overall'] if 'Overall' in p_delta.index else sorted(list(p_delta.index))
    if not cols or not rows: print(f"Skipping compact LaTeX for {output_file}: insufficient data"); return
    def fcell(d,u):
        sd = '' if pd.isna(d) else f"{d:+.1f}"
        su = '' if pd.isna(u) else f"{u:.1f}"
        if sd=='' and su=='':
            return ''
        return f"{sd} | {su}"
    latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{l' + 'r'*len(cols) + '}\n'
    latex += '\\hline\nCategory & ' + ' & '.join([f'\\textbf{{{c}}}' for c in cols]) + ' \\\\\n'
    latex += '\\hline\n'
    for cat in rows:
        cells=[]
        for c in cols:
            d = p_delta.loc[cat, c] if (cat in p_delta.index and c in p_delta.columns) else np.nan
            u = p_unrob.loc[cat, c] if (cat in p_unrob.index and c in p_unrob.columns) else np.nan
            cells.append(fcell(d,u))
        latex += f"{cat} & " + ' & '.join(cells) + ' \\\\\n'
    latex += '\\hline\n\\end{tabular}}\n'
    latex += '\\caption{Compact category-level averages (GSM): $\\Delta$ | U per model}\n'
    latex += '\\label{tab:gsm_compact}\n\\end{table}'
    with open(output_file,'w') as f: f.write(latex)
    print(f"Compact LaTeX table saved to {output_file}")


def generate_compact_unrobustness_plot(df, output_file):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import numpy.ma as ma
    except Exception as e:
        print(f"Skipping compact U plot: matplotlib not available ({e})")
        return

    if df is None or df.empty or 'model' not in df.columns or 'modification' not in df.columns:
        print(f"Skipping compact U plot {output_file}: no data or missing columns")
        return

    mod_to_cat = {
        'temporal_bias': 'Bias', 'geographical_bias': 'Bias', 'length_bias': 'Bias',
        'typo_bias': 'Orthographic','capitalization': 'Orthographic','punctuation': 'Orthographic',
        'concept_replacement': 'Semantic','negation': 'Semantic',
        'sentiment': 'Pragmatic','casual': 'Genre','dialectal': 'Genre',
        'coordinating_conjunction': 'Syntactic','active_to_passive': 'Syntactic'
    }
    model_order = ['GPT-4o','Claude','Llama 3.1','GPT-5','Deepseek R1','GPT-5-Context']
    model_map = {'gpt4o':'GPT-4o','claude':'Claude','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'Deepseek R1','deepseek-r1':'Deepseek R1','gpt-5-standard-context-aware':'GPT-5-Context'}
    df = df.copy(); df['model'] = df['model'].replace(model_map); df['category'] = df['modification'].map(lambda x: mod_to_cat.get(x,'Other'))
    agg = df.groupby(['category','model']).agg(unrobustness=('unrobustness','mean')).reset_index()
    overall = df.groupby(['model']).agg(unrobustness=('unrobustness','mean')).reset_index(); overall.insert(0,'category','Overall')
    agg = pd.concat([agg, overall], ignore_index=True)
    pivot = agg.pivot_table(index='category', columns='model', values='unrobustness', aggfunc='mean')
    cols = [m for m in model_order if m in pivot.columns and m != 'GPT-5 (w. context)']; rows = [*sorted([r for r in pivot.index if r!='Overall']), 'Overall'] if 'Overall' in pivot.index else sorted(list(pivot.index))
    if not cols or not rows: print("Skipping compact U plot: insufficient data after aggregation"); return
    pivot = pivot.reindex(index=rows, columns=cols)
    data = pivot.values.astype(float); data_masked = ma.masked_invalid(data)
    h = max(3, 0.6*data_masked.shape[0]+1.0); w = max(6, 0.6*data_masked.shape[1]+2.0)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(data_masked, cmap='Blues', vmin=_U_MIN, vmax=_U_MAX, aspect='auto')
    plt.colorbar(im, ax=ax).set_label('Unrobustness U (avg by category, %)')
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha='right')
    ax.set_title(''); ax.set_xlabel(''); ax.set_ylabel('')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i,j]
            if np.isnan(v):
                continue
            ax.text(j,i,f"{v:.1f}", ha='center', va='center', fontsize=8)
    fig.tight_layout();
    try:
        fig.savefig(output_file, dpi=200); print(f"Compact unrobustness plot saved to {output_file}")
    finally:
        plt.close(fig)

def generate_latex_table_combined(df, output_file):
    """Generate a single GSM LaTeX table with one cell per model: "Δ | U".
    Excludes GPT-5 (w. context) from main columns.
    """
    need = {'model','modification','weighted_delta','unrobustness'}
    if df is None or df.empty or not need.issubset(df.columns):
        print(f"Skipping combined LaTeX for {output_file}: missing data/columns"); return
    df = df.copy()
    mod_mapping = {
        'temporal_bias': ('Bias', 'Temporal'),
        'geographical_bias': ('Bias', 'Geographical'), 
        'length_bias': ('Bias', 'Length'),
        'typo_bias': ('Orthography', 'Spelling'),
        'capitalization': ('Orthography', 'Capitalization'),
        'punctuation': ('Orthography', 'Punctuation'),
        'concept_replacement': ('Semantics', 'Concept'),
        'negation': ('Semantics', 'Negation'),
        'sentiment': ('Discourse', 'Appraisal'),
        'casual': ('Varieties', 'Style'),
        'dialectal': ('Varieties', 'Dialect'),
        'coordinating_conjunction': ('Syntax', 'Conjunction'),
        'active_to_passive': ('Syntax', 'Voice')
    }
    model_order = ['GPT-4o','Claude-3.5','Llama 3.1','GPT-5','DS R1','GPT-5 (w. context)']
    model_map = {'gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','deepseek-r1':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    df['model'] = df['model'].replace(model_map)
    df['category'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    df['modification'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    p_delta = df.pivot_table(index=['category','modification'], columns='model', values='weighted_delta', aggfunc='mean')
    p_u = df.pivot_table(index=['category','modification'], columns='model', values='unrobustness', aggfunc='mean')
    cols = [m for m in model_order if m in p_delta.columns and m != 'GPT-5 (w. context)']
    if not cols:
        print(f"Skipping combined LaTeX for {output_file}: no columns after filtering"); return
    tot = len(cols)
    latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll'+'r'*tot+'}\n'
    latex += '\\toprule\n'
    latex += 'Category & Modification & ' + ' & '.join([f'\\textbf{{{c}}}' for c in cols]) + ' \\\\\n'
    latex += '\\midrule\n'
    cats=[]
    for key,(cat,mod) in mod_mapping.items():
        if (cat,mod) not in set(p_delta.index):
            continue
        if cat not in cats:
            if cats: latex += '\\midrule\n'
            cats.append(cat)
            lead = f'\\textbf{{{cat}}}'
        else:
            lead = ' '
        vals=[]
        for c in cols:
            d = p_delta.loc[(cat,mod), c] if (c in p_delta.columns and (cat,mod) in p_delta.index) else np.nan
            u = p_u.loc[(cat,mod), c] if (c in p_u.columns and (cat,mod) in p_u.index) else np.nan
            sd = '' if pd.isna(d) else f"{d:+.1f}"
            su = '' if pd.isna(u) else f"{u:.1f}"
            vals.append('' if (sd=='' and su=='') else f"{sd} | {su}")
        latex += f"{lead} & \\textbf{{{mod}}} & " + ' & '.join(vals) + ' \\\\\n'
    latex += '\\bottomrule\n\\end{tabular}}\n'
    latex += '\\caption{GSM: Weighted $\\Delta$ | U by model and modification}\\label{tab:gsm_combined}\n\\end{table}'
    with open(output_file,'w') as f: f.write(latex)
    print(f"Combined LaTeX table saved to {output_file}")

def generate_latex_table_dual(df, output_file):
    """GSM dual table: left block Δ, right block U (LLMs only)."""
    need = {'model','modification','weighted_delta','unrobustness','significance'}
    if df is None or df.empty or not need.issubset(df.columns):
        print(f"Skipping dual LaTeX for {output_file}: missing data/columns"); return
    df = df.copy()
    mod_mapping = {
        'temporal_bias': ('Bias', 'Temporal'),
        'geographical_bias': ('Bias', 'Geographical'), 
        'length_bias': ('Bias', 'Length'),
        'typo_bias': ('Orthography', 'Spelling'),
        'capitalization': ('Orthography', 'Capitalization'),
        'punctuation': ('Orthography', 'Punctuation'),
        'concept_replacement': ('Semantics', 'Concept'),
        'negation': ('Semantics', 'Negation'),
        'sentiment': ('Discourse', 'Appraisal'),
        'casual': ('Varieties', 'Style'),
        'dialectal': ('Varieties', 'Dialect'),
        'coordinating_conjunction': ('Syntax', 'Conjunction'),
        'active_to_passive': ('Syntax', 'Voice')
    }
    model_order = ['GPT-4o','Claude-3.5','Llama 3.1','GPT-5','DS R1','GPT-5 (w. context)']
    model_map = {'gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','deepseek-r1':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    df['model'] = df['model'].replace(model_map)
    df['category'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    df['modification'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    p_delta = df.pivot_table(index=['category','modification'], columns='model', values='weighted_delta', aggfunc='mean')
    p_sig = df.pivot_table(index=['category','modification'], columns='model', values='significance', aggfunc='first')
    p_u = df.pivot_table(index=['category','modification'], columns='model', values='unrobustness', aggfunc='mean')
    cols = [m for m in model_order if m in p_delta.columns and m != 'GPT-5 (w. context)']
    if not cols:
        print(f"Skipping dual LaTeX for {output_file}: no columns after filtering"); return
    def fd_color(v, sig):
        if pd.isna(v): return ''
        try: val = float(v)
        except Exception: return ''
        intensity = int(min(abs(val)/10.0, 1.0) * 20)
        color = 'green' if val > 0 else 'red'
        s = f"{val:+.1f}"
        if isinstance(sig, str):
            if sig == '.': s = f"\\textbf{{{s}}}"
            elif sig in {'*','**','***'}: s = f"\\textbf{{{s}}}{sig}"
        return f"\\cellcolor{{{color}!{intensity}}} {s}"
    def fu_color(v):
        if pd.isna(v): return ''
        try: val = float(v)
        except Exception: return ''
        intensity = int(min(max(val, 0.0)/5.0, 20.0))
        return f"\\cellcolor{{blue!{intensity}}} {val:.1f}"
    tot = len(cols)
    latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll'+'r'*tot+'r'*tot+'}\n'
    latex += '\\toprule\n'
    latex += f"Category & Modification & \\multicolumn{{{tot}}}{{c}}{{\\textbf{{Δ (Weighted)}}}} & \\multicolumn{{{tot}}}{{c}}{{\\textbf{{U (flip \%)}}}} \\\\\n"
    latex += ' &  & ' + ' & '.join([f'\\textbf{{{c}}}' for c in cols]) + ' & ' + ' & '.join([f'\\textbf{{{c}}}' for c in cols]) + ' \\\\\n'
    latex += '\\midrule\n'
    cats=[]
    for key,(cat,mod) in mod_mapping.items():
        if (cat,mod) not in set(p_delta.index):
            continue
        if cat not in cats:
            if cats: latex += '\\midrule\n'
            cats.append(cat)
            lead = f'\\textbf{{{cat}}}'
        else:
            lead = ' '
        deltas=[]; us=[]
        for c in cols:
            sig = p_sig.loc[(cat,mod), c] if ((cat,mod) in p_sig.index and c in p_sig.columns) else ''
            d = p_delta.loc[(cat,mod), c] if (c in p_delta.columns and (cat,mod) in p_delta.index) else np.nan
            u = p_u.loc[(cat,mod), c] if (c in p_u.columns and (cat,mod) in p_u.index) else np.nan
            deltas.append(fd_color(d, sig if isinstance(sig,str) else ''))
            us.append(fu_color(u))
        latex += f"{lead} & \\textbf{{{mod}}} & " + ' & '.join(deltas) + ' & ' + ' & '.join(us) + ' \\\\\n'
    latex += '\\bottomrule\n\\end{tabular}}\n'
    latex += '\\caption{GSM: Left = $\\Delta$, Right = U (one block per metric)}\\label{tab:gsm_dual}\n\\end{table}'
    with open(output_file,'w') as f: f.write(latex)
    print(f"Dual LaTeX table saved to {output_file}")

def generate_latex_table_combined_cells(df, output_file):
    """Combined table with two cells per model (Δ then U), colored (GSM)."""
    need = {'model','modification','weighted_delta','unrobustness','significance'}
    if df is None or df.empty or not need.issubset(df.columns):
        print(f"Skipping combined-cells LaTeX for {output_file}: missing data/columns"); return
    df = df.copy()
    mod_mapping = {
        'temporal_bias': ('Bias', 'Temporal'),
        'geographical_bias': ('Bias', 'Geographical'), 
        'length_bias': ('Bias', 'Length'),
        'typo_bias': ('Orthography', 'Spelling'),
        'capitalization': ('Orthography', 'Capitalization'),
        'punctuation': ('Orthography', 'Punctuation'),
        'concept_replacement': ('Semantics', 'Concept'),
        'negation': ('Semantics', 'Negation'),
        'sentiment': ('Discourse', 'Appraisal'),
        'casual': ('Varieties', 'Style'),
        'dialectal': ('Varieties', 'Dialect'),
        'coordinating_conjunction': ('Syntax', 'Conjunction'),
        'active_to_passive': ('Syntax', 'Voice')
    }
    model_order = ['GPT-4o','Claude-3.5','Llama 3.1','GPT-5','DS R1','GPT-5 (w. context)']
    model_map = {'gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','deepseek-r1':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    df['model'] = df['model'].replace(model_map)
    df['category'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    df['modification'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    p_delta = df.pivot_table(index=['category','modification'], columns='model', values='weighted_delta', aggfunc='mean')
    p_sig = df.pivot_table(index=['category','modification'], columns='model', values='significance', aggfunc='first')
    p_u = df.pivot_table(index=['category','modification'], columns='model', values='unrobustness', aggfunc='mean')
    cols = [m for m in model_order if m in p_delta.columns and m != 'GPT-5 (w. context)']
    if not cols:
        print(f"Skipping combined-cells LaTeX for {output_file}: no columns after filtering"); return
    def fd_color(v, sig):
        if pd.isna(v): return ''
        try: val = float(v)
        except Exception: return ''
        intensity = int(min(abs(val)/10.0, 1.0) * 20)
        color = 'green' if val > 0 else 'red'
        s = f"{val:+.1f}"
        if isinstance(sig, str):
            if sig == '.': s = f"\\textbf{{{s}}}"
            elif sig in {'*','**','***'}: s = f"\\textbf{{{s}}}{sig}"
        return f"\\cellcolor{{{color}!{intensity}}} {s}"
    def fu_color(v):
        if pd.isna(v): return ''
        try: val = float(v)
        except Exception: return ''
        intensity = int(min(max(val, 0.0), 100.0) * 0.8)
        return f"\\cellcolor{{blue!{intensity}}} {val:.1f}"
    tot = len(cols)
    latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll'+'rr'*tot+'}\n'
    latex += '\\toprule\n'
    latex += 'Category & Modification & ' + ' & '.join([f'\\multicolumn{{2}}{{c}}{{\\textbf{{{c}}}}}' for c in cols]) + ' \\\\\n'
    latex += ' &  & ' + ' & '.join(['$\\Delta$ & U' for _ in cols]) + ' \\\\\n'
    latex += '\\midrule\n'
    cats=[]
    for key,(cat,mod) in mod_mapping.items():
        if (cat,mod) not in set(p_delta.index):
            continue
        if cat not in cats:
            if cats: latex += '\\midrule\n'
            cats.append(cat)
            lead = f'\\textbf{{{cat}}}'
        else:
            lead = ' '
        cells=[]
        for c in cols:
            sig = p_sig.loc[(cat,mod), c] if ((cat,mod) in p_sig.index and c in p_sig.columns) else ''
            d = p_delta.loc[(cat,mod), c] if (c in p_delta.columns and (cat,mod) in p_delta.index) else np.nan
            u = p_u.loc[(cat,mod), c] if (c in p_u.columns and (cat,mod) in p_u.index) else np.nan
            cells.append(fd_color(d, sig if isinstance(sig,str) else ''))
            cells.append(fu_color(u))
        latex += f"{lead} & \\textbf{{{mod}}} & " + ' & '.join(cells) + ' \\\\\n'
    latex += '\\bottomrule\n\\end{tabular}}\n'
    latex += '\\caption{GSM: Two cells per model — $\\Delta$ and U}\\label{tab:gsm_combined_cells}\n\\end{table}'
    with open(output_file,'w') as f: f.write(latex)
    print(f"Combined-cells LaTeX table saved to {output_file}")

def generate_compact_latex_table_combined_cells(df, output_file):
    """Compact category-level combined-cells table for GSM (LLMs)."""
    if df is None or df.empty or 'model' not in df.columns or 'modification' not in df.columns:
        print(f"Skipping compact combined-cells: missing data/columns"); return
    mod_to_cat = {
        'temporal_bias': 'Bias', 'geographical_bias': 'Bias', 'length_bias': 'Bias',
        'typo_bias': 'Orthographic','capitalization': 'Orthographic','punctuation': 'Orthographic',
        'concept_replacement': 'Semantic','negation': 'Semantic',
        'sentiment': 'Pragmatic','casual': 'Genre','dialectal': 'Genre',
        'coordinating_conjunction': 'Syntactic','active_to_passive': 'Syntactic'
    }
    model_order = ['GPT-4o','Claude-3.5','Llama 3.1','GPT-5','DS R1','GPT-5 (w. context)']
    model_map = {'gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','deepseek-r1':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    dfx=df.copy(); dfx['model']=dfx['model'].replace(model_map); dfx['category']=dfx['modification'].map(lambda x: mod_to_cat.get(x,'Other'))
    agg = dfx.groupby(['category','model']).agg(weighted_delta=('weighted_delta','mean'), unrobustness=('unrobustness','mean')).reset_index()
    overall = dfx.groupby(['model']).agg(weighted_delta=('weighted_delta','mean'), unrobustness=('unrobustness','mean')).reset_index(); overall.insert(0,'category','Overall')
    agg = pd.concat([agg, overall], ignore_index=True)
    p_delta = agg.pivot_table(index='category', columns='model', values='weighted_delta', aggfunc='mean')
    p_u = agg.pivot_table(index='category', columns='model', values='unrobustness', aggfunc='mean')
    cols=[m for m in model_order if m in p_delta.columns and m != 'GPT-5 (w. context)']
    if not cols: print(f"Skipping compact combined-cells: no columns"); return
    def fd_color(v):
        if pd.isna(v): return ''
        try: val=float(v)
        except Exception: return ''
        inten=int(min(abs(val)/10.0,1.0)*20); col='green' if val>0 else 'red'
        return f"\\cellcolor{{{col}!{inten}}} {val:+.1f}"
    def fu_color(v):
        if pd.isna(v): return ''
        try: val=float(v)
        except Exception: return ''
        inten = unrob_intensity(val, _U_MIN, _U_MAX)
        txt = f"{val:.1f}"
        if inten >= 45:
            txt = f"\\textcolor{{white}}{{{txt}}}"
        return f"\\cellcolor{{blue!{inten}}} {txt}"
    tot=len(cols)
    latex='\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{l'+'rr'*tot+'}\n'
    latex+='\\toprule\nCategory & ' + ' & '.join([f'\\multicolumn{{2}}{{c}}{{\\textbf{{{c}}}}}' for c in cols]) + ' \\\\\n'
    latex+=' ' + ' & ' + ' & '.join(['$\\Delta$ & U' for _ in cols]) + ' \\\\\n'
    latex+='\\midrule\n'
    rows=[*sorted([r for r in p_delta.index if r!='Overall']), 'Overall'] if 'Overall' in p_delta.index else list(p_delta.index)
    for cat in rows:
        cells=[]
        for c in cols:
            d=p_delta.loc[cat,c] if (cat in p_delta.index and c in p_delta.columns) else np.nan
            u=p_u.loc[cat,c] if (cat in p_u.index and c in p_u.columns) else np.nan
            cells += [fd_color(d), fu_color(u)]
        latex += f"{cat} & " + ' & '.join(cells) + ' \\\\\n'
    latex+='\\bottomrule\n\\end{tabular}}\n'
    latex+='\\caption{GSM (compact): Two cells per model — $\\Delta$ and U}\\label{tab:gsm_compact_combined_cells}\n\\end{table}'
    with open(output_file,'w') as f: f.write(latex)
    print(f"Compact combined-cells LaTeX table saved to {output_file}")


def generate_gpt5_context_table(df, output_file):
    if df is None or df.empty:
        return
    model_map = {'gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','deepseek-r1':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    df = df.copy(); df['model'] = df['model'].replace(model_map)
    df = df[df['model'].isin(['GPT-5','GPT-5 (w. context)'])]
    if df.empty:
        raise ValueError('No GPT-5 context data')
    mod_to_cat = {
        'temporal_bias': 'Bias', 'geographical_bias': 'Bias', 'length_bias': 'Bias',
        'typo_bias': 'Orthographic','capitalization': 'Orthographic','punctuation': 'Orthographic',
        'concept_replacement': 'Semantic','negation': 'Semantic',
        'sentiment': 'Pragmatic','casual': 'Genre','dialectal': 'Genre',
        'coordinating_conjunction': 'Syntactic','active_to_passive': 'Syntactic'
    }
    df['category'] = df['modification'].map(lambda x: mod_to_cat.get(x,'Other'))
    agg = df.groupby(['category','model']).agg(weighted_delta=('weighted_delta','mean'), unrobustness=('unrobustness','mean')).reset_index()
    overall = df.groupby(['model']).agg(weighted_delta=('weighted_delta','mean'), unrobustness=('unrobustness','mean')).reset_index(); overall.insert(0,'category','Overall')
    agg = pd.concat([agg, overall], ignore_index=True)
    p_delta = agg.pivot_table(index='category', columns='model', values='weighted_delta', aggfunc='mean')
    p_unrob = agg.pivot_table(index='category', columns='model', values='unrobustness', aggfunc='mean')
    cols = [c for c in ['GPT-5','GPT-5 (w. context)'] if (c in p_delta.columns) or (c in p_unrob.columns)]
    rows = [*sorted([r for r in p_delta.index if r!='Overall']), 'Overall'] if 'Overall' in p_delta.index else sorted(list(p_delta.index))
    def fd(v): return '' if pd.isna(v) else f"{v:+.1f}"
    def fu(v): return '' if pd.isna(v) else f"{v:.1f}"
    latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{l' + 'rr'*len(cols) + '}\n'
    latex += '\\toprule\nCategory & ' + ' & '.join([f'\\multicolumn{{2}}{{c}}{{\\textbf{{{c}}}}}' for c in cols]) + ' \\\\\n'
    latex += ' & ' + ' & '.join(['$\\Delta$ & U' for _ in cols]) + ' \\\\\n'
    latex += '\\midrule\n'
    for cat in rows:
        cells=[]
        for c in cols:
            d = p_delta.loc[cat, c] if (cat in p_delta.index and c in p_delta.columns) else np.nan
            u = p_unrob.loc[cat, c] if (cat in p_unrob.index and c in p_unrob.columns) else np.nan
            cells += [fd(d), fu(u)]
        latex += f"{cat} & " + ' & '.join(cells) + ' \\\\\n'
    latex += '\\bottomrule\n\\end{tabular}}\n'
    latex += '\\caption{GSM: GPT-5 vs GPT-5 (w. context) — category-level $\\Delta$ and U}\\label{tab:gsm_gpt5_context}\n\\end{table}'
    with open(output_file,'w') as f: f.write(latex)

def generate_latex_table_unrob_gsm(df, output_file):
    """GSM U-only LaTeX table with colored cells, Avg column and Avg row (LLMs only)."""
    need = {'model','modification','unrobustness'}
    if df is None or df.empty or not need.issubset(df.columns):
        print(f"Skipping U-only LaTeX for {output_file}: missing data/columns"); return
    mod_mapping = {
        'temporal_bias': ('Bias', 'Temporal'),
        'geographical_bias': ('Bias', 'Geographical'), 
        'length_bias': ('Bias', 'Length'),
        'typo_bias': ('Orthographic', 'Spelling'),
        'capitalization': ('Orthographic', 'Capitalization'),
        'punctuation': ('Orthographic', 'Punctuation'),
        'concept_replacement': ('Semantic', 'Concept'),
        'negation': ('Semantic', 'Negation'),
        'sentiment': ('Discourse', 'Appraisal'),
        'casual': ('Varieties', 'Style'),
        'dialectal': ('Varieties', 'Dialect'),
        'coordinating_conjunction': ('Syntactic', 'Conjunction'),
        'active_to_passive': ('Syntactic', 'Voice')
    }
    model_order = ['GPT-4o','Claude-3.5','Llama 3.1','GPT-5','DS R1','GPT-5 (w. context)']
    model_map = {'gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','deepseek-r1':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    def _normalize_mod(m):
        s = str(m)
        # Drop trailing _### (e.g., _100)
        if s.endswith('_100'):
            s = s[:-4]
        # Map any negation variant to key 'negation'
        if s.startswith('negation'):
            s = 'negation'
        return s
    dfx = df.copy()
    dfx['model'] = dfx['model'].replace(model_map)
    dfx['mod_key'] = dfx['modification'].apply(_normalize_mod)
    dfx['category'] = dfx['mod_key'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    dfx['modification'] = dfx['mod_key'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    p_u = dfx.pivot_table(index=['category','modification'], columns='model', values='unrobustness', aggfunc='mean')
    cols = [m for m in model_order if m in p_u.columns and m != 'GPT-5 (w. context)']
    if not cols:
        print(f"Skipping U-only LaTeX for {output_file}: no columns after filtering"); return
    def fu_color(v):
        if pd.isna(v): return ''
        try: val = float(v)
        except Exception: return ''
        inten = unrob_intensity(val, _U_MIN, _U_MAX)
        txt = f"{val:.1f}"
        if inten >= 45:
            txt = f"\\textcolor{{white}}{{{txt}}}"
        return f"\\cellcolor{{blue!{inten}}} {txt}"
    tot = len(cols)
    latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll' + 'r'*tot + 'r}\n'
    latex += '\\toprule\n'
    latex += 'Category & Modification & ' + ' & '.join([f'\\textbf{{{c}}}' for c in cols]) + ' & \\textbf{Avg} \\\\\n'
    latex += '\\midrule\n'
    cats=[]
    for key,(cat,mod) in mod_mapping.items():
        idx = (cat,mod)
        if idx not in p_u.index: continue
        if cat not in cats:
            if cats: latex += '\\midrule\n'
            cats.append(cat)
            lead = f'\\textbf{{{cat}}}'
        else:
            lead = ' '
        vals=[]; row_vals=[]
        for c in cols:
            if idx in p_u.index and c in p_u.columns:
                v = p_u.loc[idx, c]
                row_vals.append(v)
                vals.append(fu_color(v))
            else:
                vals.append('')
        ravg = float(pd.to_numeric(pd.Series(row_vals), errors='coerce').mean()) if row_vals else float('nan')
        ravg_cell = '' if np.isnan(ravg) else fu_color(ravg)
        latex += f"{lead} & \\textbf{{{mod}}} & " + ' & '.join(vals) + f" & {ravg_cell} \\\\\n"
    col_means=[]
    for c in cols:
        series = p_u[c] if c in p_u.columns else pd.Series(dtype=float)
        col_means.append(float(pd.to_numeric(series, errors='coerce').mean()))
    overall = float(np.nanmean(col_means)) if col_means else float('nan')
    latex += '\\midrule\n'
    avg_cells = ['' if np.isnan(v) else fu_color(v) for v in col_means]
    overall_cell = '' if np.isnan(overall) else fu_color(overall)
    # Leave the Modification column empty to align model averages under their headers
    latex += '\\textbf{Average} &  & ' + ' & '.join(avg_cells) + f" & {overall_cell} \\\\\n" 
    latex += '\\bottomrule\n\\end{tabular}}\n'
    latex += '\\caption{GSM: Unrobustness (U, \\%) by model and modification}\\label{tab:gsm_unrob}\n\\end{table}'
    latex = latex.replace('\\n', '\n')
    with open(output_file,'w') as f: f.write(latex)
    print(f"U-only LaTeX table saved to {output_file}")


def generate_latex_table_robust_gsm(df, output_file):
    need = {'model','modification','unrobustness'}
    if df is None or df.empty or not need.issubset(df.columns):
        print(f"Skipping R-only LaTeX for {output_file}: missing data/columns"); return
    mod_mapping = {
        'temporal_bias': ('Bias', 'Temporal'),
        'geographical_bias': ('Bias', 'Geographical'), 
        'length_bias': ('Bias', 'Length'),
        'typo_bias': ('Orthographic', 'Spelling'),
        'capitalization': ('Orthographic', 'Capitalization'),
        'punctuation': ('Orthographic', 'Punctuation'),
        'concept_replacement': ('Semantic', 'Concept'),
        'negation': ('Semantic', 'Negation'),
        'sentiment': ('Discourse', 'Appraisal'),
        'casual': ('Varieties', 'Style'),
        'dialectal': ('Varieties', 'Dialect'),
        'coordinating_conjunction': ('Syntactic', 'Conjunction'),
        'active_to_passive': ('Syntactic', 'Voice')
    }
    model_order = ['GPT-4o','Claude-3.5','Llama 3.1','GPT-5','DS R1','GPT-5 (w. context)']
    model_map = {'gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','deepseek-r1':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    dfx = df.copy(); dfx['model'] = dfx['model'].replace(model_map)
    def _normalize_mod(m):
        s = str(m)
        if s.endswith('_100'): s = s[:-4]
        if s.startswith('negation'): s = 'negation'
        return s
    dfx['mod_key'] = dfx['modification'].apply(_normalize_mod)
    dfx['category'] = dfx['mod_key'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    dfx['modification'] = dfx['mod_key'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    p_u = dfx.pivot_table(index=['category','modification'], columns='model', values='unrobustness', aggfunc='mean')
    p_r = 100.0 - p_u
    cols = [m for m in model_order if m in p_r.columns and m != 'GPT-5 (w. context)']
    if not cols:
        print(f"Skipping R-only LaTeX for {output_file}: no columns after filtering"); return
    def fr(v): return '' if pd.isna(v) else f"{float(v):.1f}"
    tot = len(cols)
    latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll' + 'r'*tot + 'r}\n'
    latex += '\\toprule\n'
    latex += 'Category & Modification & ' + ' & '.join([f'\\textbf{{{c}}}' for c in cols]) + ' & \\textbf{Avg} \\\\\n'
    latex += '\\midrule\n'
    cats=[]
    for key,(cat,mod) in mod_mapping.items():
        idx = (cat,mod)
        if idx not in p_r.index: continue
        if cat not in cats:
            if cats: latex += '\\midrule\n'
            cats.append(cat)
            lead = f'\\textbf{{{cat}}}'
        else:
            lead = ' '
        vals=[]; row_vals=[]
        for c in cols:
            v = p_r.loc[idx, c] if (idx in p_r.index and c in p_r.columns) else float('nan')
            row_vals.append(v)
            vals.append(fr(v))
        ravg = float(pd.to_numeric(pd.Series(row_vals), errors='coerce').mean()) if row_vals else float('nan')
        latex += f"{lead} & \\textbf{{{mod}}} & " + ' & '.join(vals) + f" & {'' if np.isnan(ravg) else fr(ravg)} \\\\\n"
    col_means=[]
    for c in cols:
        series = p_r[c] if c in p_r.columns else pd.Series(dtype=float)
        col_means.append(float(pd.to_numeric(series, errors='coerce').mean()))
    overall = float(np.nanmean(col_means)) if col_means else float('nan')
    latex += '\\midrule\n'
    avg_cells = ['' if np.isnan(v) else fr(v) for v in col_means]
    overall_cell = '' if np.isnan(overall) else fr(overall)
    latex += '\\textbf{Average} &  ' + ' & '.join(avg_cells) + f" & {overall_cell} \\\\\n"
    latex += '\\bottomrule\n\\end{tabular}}\n'
    latex += '\\caption{GSM: Robustness (R, \\%) by model and modification}\\label{tab:gsm_rob}\n\\end{table}'
    latex = latex.replace('\\n', '\n')
    with open(output_file,'w') as f: f.write(latex)
    print(f"R-only LaTeX table saved to {output_file}")

def main():
    """Main function to run the analysis"""
    ap = argparse.ArgumentParser(description='GSM analysis (LLM) with optional prediction correction')
    ap.add_argument('--rewrite_predictions', action='store_true', help='Overwrite original_pred/modified_pred in results CSVs using strict #### extraction (backs up .backup)')
    args = ap.parse_args()
    # Configuration
    llm_results_dir = str((SCRIPT_DIR / '../LLM/results/gsm').resolve())
    
    # Process LLM results
    llm_results = process_llm_results(llm_results_dir, rewrite_predictions=args.rewrite_predictions)
    
    if not llm_results:
        print("No results found to process!")
        return
    
    # Save LLM results
    results_df = save_llm_results(llm_results)
    
    # Generate plots and LaTeX table
    plot_path = str((SCRIPT_DIR / 'gsm_results_heatmap.png').resolve())
    u_plot_path = str((SCRIPT_DIR / 'gsm_unrobustness_heatmap.png').resolve())
    generate_summary_plot(results_df, plot_path)
    generate_unrobustness_plot(results_df, u_plot_path)
    # U-only LaTeX table (LLMs)
    try:
        generate_latex_table_unrob_gsm(results_df, 'gsm_results_table_unrobustness.tex')
    except Exception as e:
        print(f"U-only table skipped (GSM): {e}")
    # R-only LaTeX table (LLMs)
    try:
        generate_latex_table_robust_gsm(results_df, 'gsm_results_table_robustness.tex')
    except Exception as e:
        print(f"R-only table skipped (GSM): {e}")
    # Full combined table (Δ | U per model)
    try:
        generate_latex_table_combined(results_df, 'gsm_results_table.tex')
    except Exception as e:
        print(f"Combined table skipped (GSM): {e}")
    # Compact outputs
    compact_plot = str((SCRIPT_DIR / 'gsm_results_heatmap_compact.png').resolve())
    generate_compact_summary_plot(results_df, compact_plot)
    generate_compact_latex_table(results_df, 'gsm_results_table_compact.tex')
    generate_compact_latex_table_combined_cells(results_df, 'gsm_results_table_compact_combined_cells.tex')
    u_compact_plot = str((SCRIPT_DIR / 'gsm_unrobustness_heatmap_compact.png').resolve())
    generate_compact_unrobustness_plot(results_df, u_compact_plot)
    # Separate GPT-5 vs GPT-5 (w. context)
    try:
        generate_gpt5_context_table(results_df, 'gsm_gpt5_context_table.tex')
    except Exception as e:
        print(f"GPT-5 context table skipped (GSM): {e}")
    # Combined per-model two-cell table
    try:
        generate_latex_table_combined_cells(results_df, 'gsm_results_table_combined_cells.tex')
    except Exception as e:
        print(f"Combined-cells table skipped (GSM): {e}")
    # Dual combined table (left Δ, right U)
    try:
        generate_latex_table_dual(results_df, 'gsm_results_table_dual.tex')
    except Exception as e:
        print(f"Dual table skipped (GSM): {e}")
    
    print("\\nAnalysis complete!")


if __name__ == "__main__":
    main()
