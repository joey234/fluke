#!/usr/bin/env python3
"""
Create per-sample comparison CSV for IFEval from combined outputs CSVs.

Inputs: ../results/ifeval/<MODEL>-0shot-<mod>_100.csv
  Columns: key, original_text, text, original_raw_output, raw_output, original_reasoning, reasoning

Outputs: <model>_comparison_ifeval.csv with columns aligning to the centralized viewer expectations.
"""

import re
import json
import glob
from pathlib import Path
import pandas as pd

from ifeval_checkers import check_constraint
import re


def extract_model_and_mod(filename: str):
    patterns = [
        (r'gpt-5-standard-context-aware-0shot-(.+)\.csv', 'gpt-5-standard-context-aware'),
        # Accept alt naming without 'standard' for context-aware and standard GPT-5
        (r'gpt-5-context-aware-0shot-(.+)\.csv', 'gpt-5-standard-context-aware'),
        (r'gpt-5-standard-0shot-(.+)\.csv', 'gpt-5-standard'),
        # Accept plain gpt-5 naming too, map to our canonical 'gpt-5-standard'
        (r'gpt-5-0shot-(.+)\.csv', 'gpt-5-standard'),
        (r'claude-3-5-sonnet-0shot-(.+)\.csv', 'claude-3-5-sonnet'),
        (r'claude-0shot-(.+)\.csv', 'claude-3-5-sonnet'),
        # Support both older and newer DeepSeek R1 filename variants
        (r'deepseek-r1-deepseek-0shot-(.+)\.csv', 'deepseek-r1'),
        (r'deepseek-r1-0shot-(.+)\.csv', 'deepseek-r1'),
        (r'gpt4o-0shot-(.+)\.csv', 'gpt4o'),
        (r'llama-0shot-(.+)\.csv', 'llama'),
        (r'mixtral-0shot-(.+)\.csv', 'mixtral'),
        (r'gpt4o-gpt4o-0shot-(.+)\.csv', 'gpt4o'),
    ]
    for pattern, model in patterns:
        m = re.match(pattern, filename)
        if m:
            mod = m.group(1)
            if mod.endswith('_100_new'):
                mod = mod[:-4]
            if mod.startswith('singlish'):
                mod = mod.replace('singlish', 'dialectal', 1)
            return model, mod
    return None, None

# Keys to ignore per modification
IGNORED_KEYS = {
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


def load_ifeval_dataset(path: Path):
    data = {}
    dec = json.JSONDecoder()
    with path.open('r', encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                try:
                    obj, end = dec.raw_decode(s)
                    if any(ch.strip() for ch in s[end:]):
                        print(f"Warning: trailing data ignored in {path.name} at line {ln}")
                except Exception:
                    print(f"Warning: failed to parse {path.name} line {ln}: {e}")
                    continue
            data[obj.get('key')] = {
                'instruction_ids': obj.get('instruction_id_list', []) or [],
                'kwargs': obj.get('kwargs', []) or [],
                # capture subtype if present to mirror GSM handling and viewer display
                'type': obj.get('type', ''),
            }
    return data


def score_output(text: str, instruction_ids, kwargs_list):
    """Return (num_satisfied, num_constraints, compliance_rate, strict_success)."""
    if not instruction_ids or not kwargs_list or len(instruction_ids) != len(kwargs_list):
        return 0, 0, 0.0, False
    passed = 0
    for cid, params in zip(instruction_ids, kwargs_list):
        ok = check_constraint(cid, params or {}, text or '')
        passed += 1 if ok else 0
    num_constraints = len(instruction_ids)
    rate = 100.0 * passed / num_constraints if num_constraints > 0 else 0.0
    strict = (passed == num_constraints and num_constraints > 0)
    return passed, num_constraints, rate, strict


def extract_eval_text(text: str) -> str:
    """Extract main prediction segment; prefer fenced code blocks.
    If multiple, choose the longest; fallback to full text.
    """
    if not text:
        return ""
    blocks = re.findall(r"```(?:\w+)?\n([\s\S]*?)```", text)
    if blocks:
        return max((b.strip() for b in blocks), key=len, default=text)
    return text


def main():
    script_dir = Path(__file__).resolve().parent
    results_dir = (script_dir / '../results/ifeval').resolve()

    files = list(results_dir.glob('*.csv'))
    if not files:
        print(f"No IFEval outputs found in {results_dir}")
        return

    rows = []
    for fp in files:
        fname = fp.name
        model, mod = extract_model_and_mod(fname)
        if not model or not mod:
            print(f"Skip unrecognized file: {fname}")
            continue

        # Resolve dataset path for this modification
        dataset_path = script_dir / f"../../../data/modified_data/ifeval/{mod}.jsonl"
        if not dataset_path.exists():
            # also try without _100 suffix
            alt = script_dir / f"../../../data/modified_data/ifeval/{mod.replace('_100','')}_100.jsonl"
            dataset_path = alt if alt.exists() else dataset_path
        if not dataset_path.exists():
            print(f"Dataset not found for {mod}: {dataset_path}")
            continue

        ds = load_ifeval_dataset(dataset_path)
        df = pd.read_csv(fp)
        print(f"Processing {fname} with {len(df)} rows")
        for idx, r in df.iterrows():
            key = r.get('key')
            # Skip ignored keys for this modification
            try:
                mod_key = _norm_mod_key(mod)
                if key is not None:
                    k_int = int(key)
                    if mod_key in IGNORED_KEYS and k_int in IGNORED_KEYS[mod_key]:
                        continue
            except Exception:
                pass
            item = ds.get(key)
            if not item:
                # Skip if key not found in dataset
                continue
            instruction_ids = item['instruction_ids']
            kwargs_list = item['kwargs']
            negation_subtype = item.get('type', '')

            original_text = str(r.get('original_text', ''))
            modified_text = str(r.get('text', ''))
            original_raw = str(r.get('original_raw_output', ''))
            modified_raw = str(r.get('raw_output', ''))
            # Evaluate only on the extracted prediction segment (e.g., code block)
            original_pred = extract_eval_text(original_raw)
            modified_pred = extract_eval_text(modified_raw)
            original_reasoning = str(r.get('original_reasoning', ''))
            modified_reasoning = str(r.get('reasoning', ''))

            o_pass, n_cons, o_rate, o_strict = score_output(original_pred, instruction_ids, kwargs_list)
            m_pass, _, m_rate, m_strict = score_output(modified_pred, instruction_ids, kwargs_list)
            # Apply GSM-like negation subtype semantics on modified side for IFEVAL
            # For verbal/lexical/absolute => treat "correct" as any violation (invert strict)
            # For approximate/double => treat "correct" as satisfy all constraints (strict as-is)
            neg_type_raw = str(negation_subtype or '')
            neg_type = neg_type_raw.replace('negation_', '').lower()
            if isinstance(mod, str) and mod.startswith('negation') and neg_type in {'verbal','lexical','absolute','approximate','double'}:
                m_correct_eff = (0 if m_strict else 1) if neg_type in {'verbal','lexical','absolute'} else (1 if m_strict else 0)
            else:
                m_correct_eff = 1 if m_strict else 0
            # Per-constraint detail
            detail = []
            for cid, params in zip(instruction_ids, kwargs_list):
                p = params or {}
                ok_o = check_constraint(cid, p, original_pred)
                ok_m = check_constraint(cid, p, modified_pred)
                detail.append({"id": cid, "params": p, "orig": bool(ok_o), "mod": bool(ok_m)})

            if o_strict and m_correct_eff:
                performance = 'both_correct'
            elif o_strict and not m_correct_eff:
                performance = 'original_better'
            elif (not o_strict) and m_correct_eff:
                performance = 'modified_better'
            else:
                performance = 'both_wrong'

            rows.append({
                'task': 'ifeval',
                'model': model,
                'modification': mod,
                'original_text': original_text,
                'modified_text': modified_text,
                'original_pred': original_pred,
                'modified_pred': modified_pred,
                'original_correct': str(o_strict),
                'modified_correct': str(bool(m_correct_eff)),
                'num_constraints': n_cons,
                'original_num_satisfied': o_pass,
                'modified_num_satisfied': m_pass,
                'original_compliance_rate': round(o_rate, 2),
                'modified_compliance_rate': round(m_rate, 2),
                'performance': performance,
                'original_reasoning': original_reasoning,
                'modified_reasoning': modified_reasoning,
                'constraints_detail': json.dumps(detail, ensure_ascii=False),
                # For negation variants, expose subtype for viewer badges
                # Store a clean subtype label (without the 'negation_' prefix) for negation mods
                'negation_subtype': (neg_type if str(mod).startswith('negation') else ''),
                'file': fname,
                'row_index': idx,
            })

    if not rows:
        print("No IFEval comparisons generated.")
        return

    out = pd.DataFrame(rows)
    # Write one file per model to match viewer expectations
    for model, dfm in out.groupby('model'):
        out_file = script_dir / f"{model}_comparison_ifeval.csv"
        dfm.to_csv(out_file, index=False)
        print(f"Saved IFEval comparison for {model}: {out_file} ({len(dfm)} rows)")


if __name__ == '__main__':
    main()
