#!/usr/bin/env python3
"""
Build a U-only negation subtype table across tasks (SA, Coref, Dialogue, NER, GSM, IFEVAL)
for either a chosen model or all models (default: all), with per-subtype averages across tasks.

GSM/IFEVAL negation subtype mapping is derived from modified_data negation_100.jsonl.

Outputs:
- Console table with canonical subtype names as rows, tasks as columns, and an AVG column.
- LaTeX table with category "Negation" and colored cells (blue intensity) similar to other U tables.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from utils import get_global_unrobustness_range, unrob_intensity
import math
import json

SCRIPT_DIR = Path(__file__).parent

NEG_FILES = {
    'sa': SCRIPT_DIR / 'sa_negation_type_results_llm.csv',
    'coref': SCRIPT_DIR / 'coref_negation_type_results_llm.csv',
    'dialogue': SCRIPT_DIR / 'dialogue_negation_type_results_llm.csv',
    'ner': SCRIPT_DIR / 'ner_negation_type_results_llm.csv',
    # gsm and ifeval subtype computed from per-sample results + modified_data
}

ALIASES = {
    'gpt-4o': 'gpt4o', 'gpt4o': 'gpt4o', 'gpt4o-gpt4o': 'gpt4o',
    'gpt-5-standard': 'gpt-5-standard', 'gpt-5': 'gpt-5-standard',
    'gpt-5-standard-context-aware': 'gpt-5-standard-context-aware', 'gpt-5-context-aware': 'gpt-5-standard-context-aware',
    'claude': 'claude', 'claude-3-5-sonnet': 'claude', 'claude-claude': 'claude',
    'llama': 'llama', 'llama-llama': 'llama',
    'deepseek': 'deepseek-r1-deepseek', 'deepseek-deepseek': 'deepseek-r1-deepseek',
    'deepseek-r1': 'deepseek-r1-deepseek', 'deepseek-r1-deepseek': 'deepseek-r1-deepseek',
}

CANON_NAMES = {
    'verbal': 'Verbal',
    'lexical': 'Lexical',
    'double': 'Double',
    'approximate': 'Approximate',
    'absolute': 'Absolute',
}

def norm_model(name: str) -> str:
    key = str(name).strip().lower()
    return ALIASES.get(key, key)

def load_task_neg(task: str, model_key: str, exclude_context: bool = False) -> pd.Series:
    p = NEG_FILES.get(task)
    if not p or not p.exists():
        return pd.Series(dtype=float)
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.Series(dtype=float)
    if not {'model','negation_type','unrobustness'}.issubset(df.columns):
        return pd.Series(dtype=float)
    if model_key == 'all':
        # Optionally exclude GPT-5 (context-aware); macro-average across models
        if 'model' in df.columns:
            df['_model'] = df['model'].apply(norm_model)
            if exclude_context:
                df = df[df['_model'] != 'gpt-5-standard-context-aware']
            per_model = df.groupby(['_model','negation_type'])['unrobustness'].mean().reset_index()
            s = per_model.groupby('negation_type')['unrobustness'].mean()
        else:
            s = df.groupby('negation_type')['unrobustness'].mean()
    else:
        df = df[df['model'].apply(norm_model) == model_key]
        if df.empty:
            return pd.Series(dtype=float)
        s = df.groupby('negation_type')['unrobustness'].mean()
    s.index = s.index.map(lambda x: CANON_NAMES.get(x, str(x)))
    return s

def csv_task_counts(task: str, model_key: str, exclude_context: bool = False) -> pd.Series:
    p = NEG_FILES.get(task)
    if not p or not p.exists():
        return pd.Series(dtype=int)
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.Series(dtype=int)
    if 'negation_type' not in df.columns:
        return pd.Series(dtype=int)
    # restrict models
    if 'model' in df.columns:
        df['_model'] = df['model'].apply(norm_model)
        if model_key == 'all':
            # Keep all models; optionally drop GPT-5 (w. context)
            if exclude_context:
                df = df[df['_model'] != 'gpt-5-standard-context-aware']
        else:
            df = df[df['_model'] == norm_model(model_key)]
    # Prefer sample_size if present (do not sum across models; take first per subtype)
    if 'sample_size' in df.columns:
        first = df.groupby('negation_type')['sample_size'].first()
        return first.rename(index=lambda x: CANON_NAMES.get(x, str(x))).astype(int)
    # Fallback to count rows per subtype (per model)
    cnt = df.groupby('negation_type').size()
    return cnt.rename(index=lambda x: CANON_NAMES.get(x, str(x))).astype(int)

def csv_task_models(task: str, model_key: str, exclude_context: bool = False) -> list:
    p = NEG_FILES.get(task)
    if not p or not p.exists():
        return []
    try:
        df = pd.read_csv(p)
    except Exception:
        return []
    if 'model' not in df.columns:
        return []
    df['_model'] = df['model'].apply(norm_model)
    if model_key == 'all':
        if exclude_context:
            df = df[df['_model'] != 'gpt-5-standard-context-aware']
    else:
        df = df[df['_model'] == norm_model(model_key)]
    return sorted(df['_model'].dropna().unique().tolist())

def load_gsm_neg(model_key: str, exclude_context: bool = False) -> pd.Series:
    root = (SCRIPT_DIR / '../LLM/results/gsm').resolve()
    if not root.exists():
        return pd.Series(dtype=float)
    # Load subtype mapping from modified_data (index -> subtype)
    subtype_map = {}
    try:
        with open((SCRIPT_DIR / '../../data/modified_data/gsm/negation_100.jsonl').resolve(), 'r') as f:
            for ln in f:
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                idx = obj.get('index')
                typ = str(obj.get('type','')).replace('negation_','')
                if idx is not None:
                    subtype_map[int(idx)] = typ
    except Exception:
        subtype_map = {}
    rows = []
    for csv in root.glob('*-0shot-negation_100.csv'):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        # infer model
        model_raw = csv.name.split('-0shot-')[0]
        model = norm_model(model_raw.split('/')[-1])
        if model_key == 'all':
            # Optionally exclude GPT-5 (context-aware)
            if exclude_context and model == 'gpt-5-standard-context-aware':
                continue
        elif model != model_key:
            continue
        need = {'original_answer','modified_answer','original_pred','modified_pred'}
        if not need.issubset(df.columns):
            continue
        # subtype: prefer modified_data mapping via index
        if 'index' in df.columns and subtype_map:
            try:
                sub = df['index'].apply(lambda x: subtype_map.get(int(x), 'Unknown'))
            except Exception:
                sub = pd.Series(['Unknown']*len(df))
        else:
            # fallback to CSV type column if present
            if 'type' in df.columns:
                sub = df['type'].astype(str).str.replace('negation_','', regex=False)
            else:
                sub = pd.Series(['Unknown']*len(df))
        def _norm_num(x):
            try:
                s = str(x).strip().replace(',', '')
                if s.endswith('%'):
                    return float(s[:-1])
                return float(s)
            except Exception:
                return None
        def _eq(a, b):
            na, nb = _norm_num(a), _norm_num(b)
            if na is not None and nb is not None:
                return 1 if math.isclose(na, nb, rel_tol=1e-9, abs_tol=1e-9) else 0
            # fallback to normalized string compare
            sa = str(a).strip().lower()
            sb = str(b).strip().lower()
            return 1 if sa == sb else 0
        orig_ok = df.apply(lambda r: _eq(r.get('original_pred'), r.get('original_answer')), axis=1).astype(int)
        # GSM negation subtype rule for modified correctness:
        # - For 'approximate' and 'double': correct if modified_pred equals modified_answer
        # - For other negation subtypes: correct if modified_pred does NOT equal modified_answer
        mod_eq = df.apply(lambda r: _eq(r.get('modified_pred'), r.get('modified_answer')), axis=1).astype(int)
        sub_str = sub.astype(str).str.lower()
        is_approx_or_double = sub_str.str.contains('approximate') | sub_str.str.contains('double')
        mod_ok = np.where(is_approx_or_double.values, mod_eq.values, 1 - mod_eq.values)
        # U-only: only count degradations (orig correct -> modified incorrect)
        u_only = ((orig_ok == 1) & (pd.Series(mod_ok, index=df.index) == 0)).astype(int) * 100.0
        tmp = pd.DataFrame({'subtype': sub, 'flip': u_only})
        tmp['model'] = model
        rows.append(tmp)
    if not rows:
        return pd.Series(dtype=float)
    all_df = pd.concat(rows, ignore_index=True)
    if model_key == 'all':
        # macro-average across models
        per_model = all_df.groupby(['model','subtype'])['flip'].mean().reset_index()
        s = per_model.groupby('subtype')['flip'].mean()
    else:
        s = all_df.groupby('subtype')['flip'].mean()
    s.index = s.index.map(lambda x: CANON_NAMES.get(x, str(x)))
    return s

def gsm_counts(model_key: str, exclude_context: bool = False) -> pd.Series:
    """Count GSM negation subtypes from modified_data only (single dataset of ~100 items).
    This avoids double-counting across multiple model outputs."""
    path = (SCRIPT_DIR / '../../data/modified_data/gsm/negation_100.jsonl').resolve()
    if not path.exists():
        return pd.Series(dtype=int)
    cnt = {}
    try:
        with open(path, 'r') as f:
            for ln in f:
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                typ = str(obj.get('type','')).replace('negation_','')
                name = CANON_NAMES.get(typ, typ)
                cnt[name] = cnt.get(name, 0) + 1
    except Exception:
        return pd.Series(dtype=int)
    return pd.Series(cnt, dtype=int)

def gsm_models(model_key: str, exclude_context: bool = False) -> list:
    root = (SCRIPT_DIR / '../LLM/results/gsm').resolve()
    if not root.exists():
        return []
    models = set()
    for csv in root.glob('*-0shot-negation_100.csv'):
        model_raw = csv.name.split('-0shot-')[0]
        model = norm_model(model_raw.split('/')[-1])
        if model_key == 'all':
            if exclude_context and model == 'gpt-5-standard-context-aware':
                continue
        elif model != norm_model(model_key):
            continue
        models.add(model)
    return sorted(models)

def load_ifeval_neg(model_key: str, exclude_context: bool = False) -> pd.Series:
    # Compute strict flip% by applying IFEval checkers
    results_root = (SCRIPT_DIR / '../LLM/results/ifeval').resolve()
    data_root = (SCRIPT_DIR / '../../data/modified_data/ifeval').resolve()
    scripts_dir = (SCRIPT_DIR / '../LLM/scripts').resolve()
    import sys
    if str(scripts_dir) not in sys.path:
        sys.path.append(str(scripts_dir))
    try:
        from ifeval_checkers import check_constraint  # type: ignore
    except Exception:
        return pd.Series(dtype=float)
    if not results_root.exists() or not data_root.exists():
        return pd.Series(dtype=float)
    # Load subtype map from modified_data
    subtype_map = {}
    try:
        with open(data_root / 'negation_100.jsonl','r') as f:
            for ln in f:
                obj = json.loads(ln)
                subtype_map[obj.get('key')] = str(obj.get('type','')).replace('negation_','')
    except Exception:
        subtype_map = {}
    rows = []
    for csv in results_root.glob('*-0shot-negation_100.csv'):
        model_raw = csv.name.split('-0shot-')[0]
        model = norm_model(model_raw)
        if model_key == 'all':
            if exclude_context and model == 'gpt-5-standard-context-aware':
                continue
        elif model != model_key:
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if not {'key','original_raw_output','raw_output'}.issubset(df.columns):
            continue
        # For each row, use instruction_id_list from modified_data
        # Load id lists lazily
        # We don't have instruction_id_list here; but strict success is whether all constraints satisfied.
        # We cannot easily reconstruct kwargs; rely on checkers that take (id, kwargs, text)
        # Extract args from modified_data isn't trivial; use fallback: compare original vs modified success parity via approximated method? Skip if missing.
        # Here we approximate by using presence of 'not' might be fragile; prefer skipping if no checkers context.
        # For accuracy, we reuse approach from ifeval_analysis.load_from_raw_results; however that required reading dataset files with instruction ids.
        # We will attempt to join to modified_data for required kwargs.
        # Build key -> (ids, kwargs)
        key_to_ck = {}
        try:
            with open(data_root / 'negation_100.jsonl','r') as f:
                for ln in f:
                    try:
                        o = json.loads(ln)
                    except Exception:
                        continue
                    key = o.get('key')
                    ids = o.get('instruction_id_list') or []
                    kwargs = o.get('kwargs') or []
                    if key is not None:
                        key_to_ck[key] = (ids, kwargs)
        except Exception:
            key_to_ck = {}
        flips = []
        subs = []
        for _, r in df.iterrows():
            k = r.get('key')
            ck = key_to_ck.get(k)
            if not ck:
                continue
            ids, kwargs_list = ck
            text_o = str(r.get('original_raw_output',''))
            text_m = str(r.get('raw_output',''))
            so = 0; sm = 0; n = len(ids)
            for cid, kw in zip(ids, kwargs_list):
                try:
                    if check_constraint(cid, kw or {}, text_o):
                        so += 1
                    if check_constraint(cid, kw or {}, text_m):
                        sm += 1
                except Exception:
                    continue
            so_strict = 1 if (n>0 and so==n) else 0
            sm_strict = 1 if (n>0 and sm==n) else 0
            # Apply GSM-like semantics on modified side for IFEVAL negation subtypes
            st = str(subtype_map.get(k, '')).lower()
            if st in {'verbal','lexical','absolute'}:
                sm_eff = 0 if sm_strict == 1 else 1
            else:
                sm_eff = sm_strict
            # U-only: only count degradations (orig correct -> modified incorrect)
            flips.append((1 if (so_strict == 1 and sm_eff == 0) else 0) * 100.0)
            subs.append(CANON_NAMES.get(subtype_map.get(k,''), subtype_map.get(k,'')))
        if flips:
            tmp = pd.DataFrame({'subtype': subs, 'flip': flips})
            tmp['model'] = model
            rows.append(tmp)
    if not rows:
        return pd.Series(dtype=float)
    all_df = pd.concat(rows, ignore_index=True)
    if model_key == 'all':
        per_model = all_df.groupby(['model','subtype'])['flip'].mean().reset_index()
        s = per_model.groupby('subtype')['flip'].mean()
    else:
        s = all_df.groupby('subtype')['flip'].mean()
    return s

def ifeval_counts(model_key: str, exclude_context: bool = False) -> pd.Series:
    """Count IFEVAL negation subtypes from modified_data only (single dataset of ~100 items)."""
    path = (SCRIPT_DIR / '../../data/modified_data/ifeval/negation_100.jsonl').resolve()
    if not path.exists():
        return pd.Series(dtype=int)
    cnt = {}
    try:
        with open(path, 'r') as f:
            for ln in f:
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                typ = str(obj.get('type','')).replace('negation_','')
                name = CANON_NAMES.get(typ, typ)
                cnt[name] = cnt.get(name, 0) + 1
    except Exception:
        return pd.Series(dtype=int)
    return pd.Series(cnt, dtype=int)

def ifeval_models(model_key: str, exclude_context: bool = False) -> list:
    root = (SCRIPT_DIR / '../LLM/results/ifeval').resolve()
    if not root.exists():
        return []
    models = set()
    for csv in root.glob('*-0shot-negation_100.csv'):
        model_raw = csv.name.split('-0shot-')[0]
        model = norm_model(model_raw)
        if model_key == 'all':
            if exclude_context and model == 'gpt-5-standard-context-aware':
                continue
        elif model != norm_model(model_key):
            continue
        models.add(model)
    return sorted(models)

def main(model: str = 'all', exclude_context: bool = False):
    model_key = norm_model(model)
    tasks = ['sa','coref','dialogue','ner','gsm','ifeval']
    cols = {}
    for t in tasks:
        if t in NEG_FILES:
            s = load_task_neg(t, model_key, exclude_context)
        elif t == 'gsm':
            s = load_gsm_neg(model_key, exclude_context)
        elif t == 'ifeval':
            s = load_ifeval_neg(model_key, exclude_context)
        else:
            s = pd.Series(dtype=float)
        if not s.empty:
            cols[t] = s
    if not cols:
        print('No negation subtype data found for', ('all models' if model_key=='all' else model))
        return
    df = pd.concat(cols, axis=1)
    order = [CANON_NAMES[k] for k in ['verbal','lexical','double','approximate','absolute']]
    df = df.reindex(index=order, fill_value=np.nan)
    df['avg'] = df.mean(axis=1, skipna=True)
    # Console: include an Average row
    avg_row = {t: pd.to_numeric(df[t], errors='coerce').mean() for t in cols.keys()}
    avg_row['avg'] = pd.to_numeric(df['avg'], errors='coerce').mean()
    avg_series = pd.Series(avg_row, name='Average')
    df_print = pd.concat([df, avg_series.to_frame().T])
    print('Negation subtype U by task (', ('all models' if model_key=='all' else model), ')')
    print(df_print.to_string(float_format=lambda x: f'{x:.1f}' if pd.notna(x) else 'NA'))

    # Sanity counts per task
    print('\nSanity counts (rows per subtype) + models loaded:')
    for t in cols.keys():
        if t in NEG_FILES:
            c = csv_task_counts(t, model_key, exclude_context).reindex(order)
            models = csv_task_models(t, model_key, exclude_context)
        elif t == 'gsm':
            c = gsm_counts(model_key, exclude_context).reindex(order)
            models = gsm_models(model_key, exclude_context)
        elif t == 'ifeval':
            c = ifeval_counts(model_key, exclude_context).reindex(order)
            models = ifeval_models(model_key, exclude_context)
        else:
            c = pd.Series(dtype=int)
            models = []
        total = int(pd.to_numeric(c, errors='coerce').sum()) if c is not None else 0
        print(f'- {t}: total={total} models={models}', (' ' + str({k: int(v) for k, v in c.fillna(0).items()})) if c is not None else '')

    # LaTeX
    umin, umax = get_global_unrobustness_range()
    header = ['Category','Subtype'] + [t.upper() for t in cols.keys()] + ['AVG']
    lines = []
    lines.append('\\begin{table}[!tbp]')
    lines.append('\\centering')
    lines.append('\\footnotesize')
    lines.append('\\begin{adjustbox}{max width=\\linewidth}')
    lines.append('\\begin{tabular}{ll' + 'r'* (len(cols.keys())+1) + '}')
    lines.append('\\toprule')
    lines.append(' & '.join([f'\\textbf{{{h}}}' for h in header]) + ' \\\\')
    lines.append('\\midrule')
    first_cat = True
    for subtype, row in df.iterrows():
        cat_cell = '\\textbf{Negation}' if first_cat else ''
        first_cat = False
        cells = []
        for t in list(cols.keys()) + ['avg']:
            v = row.get(t)
            if pd.isna(v):
                cells.append('NA')
            else:
                inten = unrob_intensity(v, umin, umax)
                txt = f'{v:.1f}'
                cell = f"\\cellcolor{{blue!{inten}}} {txt}" if inten < 45 else f"\\cellcolor{{blue!{inten}}} \\textcolor{{white}}{{{txt}}}"
                cells.append(cell)
        lines.append(' & '.join([cat_cell, f'\\textbf{{{subtype}}}'] + cells) + ' \\\\')
    # Average row (column-wise means)
    lines.append('\\midrule')
    avg_cells = []
    for t in list(cols.keys()) + ['avg']:
        m = pd.to_numeric(df[t], errors='coerce').mean()
        if pd.isna(m):
            avg_cells.append('NA')
        else:
            inten = unrob_intensity(m, umin, umax)
            txt = f'{m:.1f}'
            avg_cells.append(f"\\cellcolor{{blue!{inten}}} {txt}" if inten < 45 else f"\\cellcolor{{blue!{inten}}} \\textcolor{{white}}{{{txt}}}")
    lines.append(' & '.join(['\\textbf{Average}', '\\textbf{Average}'] + avg_cells) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{adjustbox}')
    lines.append('\\caption{Negation subtype unrobustness U (flip \%) across tasks — ' + model + '}')
    lines.append('\\label{tab:negation_subtype_unrob}')
    lines.append('\\end{table}')
    out_tex = SCRIPT_DIR / 'negation_subtype_unrobustness_table.tex'
    out_tex.write_text('\n'.join(lines))
    print('Wrote LaTeX table to', out_tex)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='all', help='Model name or "all" to aggregate across models')
    ap.add_argument('--exclude-context', action='store_true', help='Exclude GPT-5 (w. context) when aggregating all models')
    args = ap.parse_args()
    main(args.model, exclude_context=args.exclude_context)
