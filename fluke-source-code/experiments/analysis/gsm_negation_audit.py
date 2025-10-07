#!/usr/bin/env python3
"""
GSM Negation Audit Helper

Inspect a negation results CSV alongside the dataset JSONL and verify scoring:
- Loads original_answer from JSONL for both sides
- Applies subtype-based inversion for modified correctness
- Prints summary stats and writes a small audit CSV with sample rows

Usage:
  python gsm_negation_audit.py \
    --csv ../LLM/results/gsm/claude-claude-0shot-negation_100.csv \
    --jsonl ../../data/modified_data/gsm/negation_100.jsonl \
    --out audit_claude_negation.csv \
    --n 25
"""

import argparse
import json
import re
from pathlib import Path
import pandas as pd


def extract_number(s):
    if s is None:
        return None
    t = str(s).strip()
    if t == '' or t.lower() == 'nan':
        return None
    t = t.replace(',', '')
    # Prefer #### <number>
    m = re.search(r'####\s*[$€£¥₹₽]?\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?|[+-]?\d+\s*/\s*\d+)', t)
    if m:
        t = m.group(1)
    # Fraction
    mf = re.fullmatch(r'\s*([+-]?\d+)\s*/\s*(\d+)\s*', t)
    if mf:
        a = float(mf.group(1)); b = float(mf.group(2))
        return a / b if b != 0 else None
    try:
        return float(t)
    except Exception:
        return None


def load_jsonl(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='Path to negation_100 results CSV for a model')
    ap.add_argument('--jsonl', required=True, help='Path to negation_100.jsonl dataset')
    ap.add_argument('--out', default='gsm_negation_audit.csv', help='Where to write audit CSV')
    ap.add_argument('--n', type=int, default=20, help='Number of sample rows to include')
    args = ap.parse_args()

    csv_path = Path(args.csv)
    ds_path = Path(args.jsonl)

    df = pd.read_csv(csv_path)
    data = load_jsonl(ds_path)
    by_index = {int(obj['index']): obj for obj in data if 'index' in obj}

    tol = 1e-6
    records = []
    total = 0
    ori_hits = 0
    mod_hits = 0
    flips = 0

    for _, row in df.iterrows():
        try:
            idx = int(row.get('index'))
        except Exception:
            continue
        obj = by_index.get(idx)
        if not obj:
            continue
        subtype = str(obj.get('negation_subtype', obj.get('type', ''))).lower()
        oa_raw = obj.get('original_answer', obj.get('short_answer'))
        oa = extract_number(oa_raw)
        op = extract_number(row.get('original_pred'))
        mp = extract_number(row.get('modified_pred'))

        ori_ok = (op is not None and oa is not None and abs(op - oa) < tol)
        eq_mod = (mp is not None and oa is not None and abs(mp - oa) < tol)
        if ('approximate' in subtype) or ('double' in subtype):
            mod_ok = eq_mod
        else:
            # verbal/lexical/absolute default: flip relative to original answer
            mod_ok = not eq_mod if (mp is not None and oa is not None) else (str(row.get('modified_pred', '')).strip() != str(oa_raw).strip())

        total += 1
        ori_hits += int(ori_ok)
        mod_hits += int(mod_ok)
        flips += int(ori_ok != mod_ok)

        if len(records) < args.n:
            records.append({
                'index': idx,
                'negation_subtype': subtype,
                'original_answer': oa_raw,
                'original_pred_num': op,
                'modified_pred_num': mp,
                'original_correct': ori_ok,
                'modified_correct_inversion': mod_ok,
                'flip': ori_ok != mod_ok,
                'original_pred': str(row.get('original_pred', '')),
                'modified_pred': str(row.get('modified_pred', '')),
            })

    if total == 0:
        print('No rows joined by index; check inputs.')
        return

    print(f'N={total} | OriAcc={ori_hits/total*100:.1f}% | ModAcc(inv)={mod_hits/total*100:.1f}% | Flips={flips/total*100:.1f}%')
    out_path = Path(args.out)
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f'Wrote sample audit rows to {out_path}')


if __name__ == '__main__':
    main()

