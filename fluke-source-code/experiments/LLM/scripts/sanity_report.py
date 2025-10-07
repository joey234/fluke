#!/usr/bin/env python3
"""
Sanity report: summarize per-task, per-model, per-modification coverage and metrics.

Outputs one CSV per task under ../results/<task>/sanity_summary_<task>.csv and
prints a concise summary listing models and modifications processed.
"""

import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# For NER F1
try:
    from fluke_reasoning_utils import calculate_f1_ent, convert_string_to_entities
except Exception:
    calculate_f1_ent = None
    convert_string_to_entities = None


def detect_model_and_mod(filename: str) -> Tuple[str, str]:
    patterns = [
        (r'deepseek-r1-deepseek-0shot-(.+)\.csv', 'deepseek-r1'),
        (r'gpt4o-0shot-(.+)\.csv', 'gpt4o'),
        (r'gpt4o-gpt4o-0shot-(.+)\.csv', 'gpt4o'),
        (r'claude-3-5-sonnet-0shot-(.+)\.csv', 'claude-3-5-sonnet'),
        (r'claude-0shot-(.+)\.csv', 'claude-3-5-sonnet'),
        (r'llama-0shot-(.+)\.csv', 'llama'),
        (r'mixtral-0shot-(.+)\.csv', 'mixtral'),
        (r'gpt-5-standard-context-aware-0shot-(.+)\.csv', 'gpt-5-standard-context-aware'),
        (r'gpt-5-standard-0shot-(.+)\.csv', 'gpt-5-standard'),
    ]
    for pat, model in patterns:
        m = re.match(pat, filename)
        if m:
            return model, m.group(1)
    return 'unknown', 'unknown'


def summarize_sa_coref_dialogue(df: pd.DataFrame) -> Tuple[int, float]:
    # Accuracy on modified side
    if 'modified_label' not in df.columns or 'modified_pred' not in df.columns:
        return len(df), float('nan')
    corr = 0
    n = 0
    for _, r in df.iterrows():
        mp = str(r.get('modified_pred', '')).strip().lower()
        ml = str(r.get('modified_label', '')).strip().lower()
        if mp != '' and ml != '':
            n += 1
            corr += 1 if mp == ml else 0
    return n, (corr / n if n else float('nan'))


def summarize_gsm(df: pd.DataFrame) -> Tuple[int, float, float]:
    # Returns (n, acc_standard, acc_negation_flip)
    if 'modified_pred' not in df.columns or 'modified_answer' not in df.columns:
        return len(df), float('nan'), float('nan')
    n = 0
    acc = 0
    flip_accs: List[int] = []
    for _, r in df.iterrows():
        mp = str(r.get('modified_pred', '')).replace(',', '').strip()
        ma = str(r.get('modified_answer', '')).replace(',', '').strip()
        if mp == '' or ma == '':
            continue
        n += 1
        # standard
        try:
            ok = float(mp) == float(ma)
        except Exception:
            ok = (mp == ma)
        acc += 1 if ok else 0
        # flip/no flip
        if 'negation_flip_correct' in df.columns:
            v = r.get('negation_flip_correct')
            if pd.notna(v) and v != '':
                flip_accs.append(1 if bool(v) in (True, 'True', 'true', 1, '1') else 0)
        else:
            t = str(r.get('type', ''))
            expected_flip = (t.startswith('negation_') and (not any(s in t for s in ['approximate', 'double'])))
            # correctness relative to original answer
            oa = str(r.get('original_answer', ma)).replace(',', '').strip()
            try:
                neq = (float(mp) != float(oa))
                eq = (float(mp) == float(oa))
            except Exception:
                neq = (mp != oa)
                eq = (mp == oa)
            flip_accs.append(1 if (neq if expected_flip else eq) else 0)
    neg_flip = (sum(flip_accs) / len(flip_accs)) if flip_accs else float('nan')
    return n, (acc / n if n else float('nan')), neg_flip


def summarize_ner(df: pd.DataFrame) -> Tuple[int, float]:
    if calculate_f1_ent is None or convert_string_to_entities is None:
        return len(df), float('nan')
    f1s = []
    for _, r in df.iterrows():
        gold = r.get('modified_label', r.get('label', ''))
        pred = r.get('modified_pred', r.get('pred', ''))
        # Convert string to entities
        ge = convert_string_to_entities(gold)
        pe = convert_string_to_entities(pred)
        p, rec, f1 = calculate_f1_ent(ge, pe)
        f1s.append(f1)
    return len(f1s), (sum(f1s) / len(f1s)) if f1s else float('nan')


def main():
    base = Path(__file__).resolve().parent / '../results'
    tasks = ['sa', 'coref', 'dialogue', 'ner', 'gsm']
    for task in tasks:
        task_dir = (base / task).resolve()
        if not task_dir.exists():
            print(f"Skip missing task dir: {task_dir}")
            continue
        files = sorted(glob.glob(str(task_dir / '*.csv')))
        if not files:
            print(f"No CSVs for task {task}")
            continue
        rows = []
        coverage: Dict[str, set] = {}
        for fp in files:
            fname = os.path.basename(fp)
            model, mod = detect_model_and_mod(fname)
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                print(f"Failed reading {fname}: {e}")
                continue
            n = len(df)
            metric = None
            metric2 = None
            if task in ('sa', 'coref', 'dialogue'):
                n_eff, acc = summarize_sa_coref_dialogue(df)
                metric = acc
            elif task == 'gsm':
                n_eff, acc, neg_flip = summarize_gsm(df)
                metric = acc
                metric2 = neg_flip
            elif task == 'ner':
                n_eff, f1 = summarize_ner(df)
                metric = f1
            rows.append({
                'task': task,
                'model': model,
                'modification': mod,
                'rows': n,
                'effective_rows': n_eff,
                'metric': metric,
                'metric2': metric2 if metric2 is not None else ''
            })
            coverage.setdefault(model, set()).add(mod)
        if not rows:
            print(f"No rows summarized for task {task}")
            continue
        out = pd.DataFrame(rows)
        out_path = task_dir / f"sanity_summary_{task}.csv"
        out.to_csv(out_path, index=False)
        print(f"\nSanity report saved: {out_path}")
        for model, mods in coverage.items():
            print(f" - {task} | {model}: {len(mods)} modifications -> {', '.join(sorted(mods))}")


if __name__ == '__main__':
    main()

