#!/usr/bin/env python3
"""
IFEval Analysis (LLM-only)
Aggregates per-model, per-modification results from ../LLM/results/ifeval_aggregates
and produces concise heatmaps for main content, plus an optional LaTeX table for appendix.
"""

from pathlib import Path
import sys
import re
import pandas as pd
from utils import get_global_unrobustness_range, unrob_intensity
import numpy as np

SCRIPT_DIR = Path(__file__).parent
_U_MIN, _U_MAX = get_global_unrobustness_range()


def load_aggregates(root: Path) -> pd.DataFrame:
    rows = []
    if not root.exists():
        return pd.DataFrame()
    # Each subfolder is a modification; inside are <model>_comparison.csv files
    for mod_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        mod = mod_dir.name
        for csv_file in sorted(mod_dir.glob('*.csv')):
            try:
                df = pd.read_csv(csv_file)
            except Exception:
                continue
            if df.empty:
                continue
            # Expected columns: task, mod, model, n, A, B, delta, absolute_change, U_frac, U_strict, p_value, significance
            rec = df.iloc[0].to_dict()
            rows.append({
                'task': 'ifeval',
                'modification': rec.get('mod', mod),
                'model': rec.get('model', csv_file.stem.replace('_comparison','')),
                'A': rec.get('A', np.nan),
                'B': rec.get('B', np.nan),
                'weighted_delta': rec.get('delta', np.nan),
                'absolute_change': rec.get('absolute_change', np.nan),
                'unrobustness': rec.get('U_strict', np.nan),  # % flip on strict success
                'p_value': rec.get('p_value', np.nan),
                'significance': rec.get('significance', ''),
                'n': rec.get('n', np.nan),
                'source': 'agg',
            })
    return pd.DataFrame(rows)


def _normalize_model_key(name: str) -> str:
    """Map various model name variants to internal canonical keys used upstream."""
    if not isinstance(name, str):
        return str(name)
    n = name.strip()
    if n.startswith('gpt-5-standard-context-aware') or n.startswith('gpt-5-context-aware'):
        return 'gpt-5-standard-context-aware'
    if n.startswith('gpt-5'):
        return 'gpt-5-standard'
    if n.startswith('gpt4o') or n.startswith('gpt-4o'):
        return 'gpt4o'
    if n.startswith('claude'):
        return 'claude'
    if n.startswith('deepseek'):
        return 'deepseek-r1-deepseek'
    if n.startswith('llama'):
        return 'llama'
    return n


def load_from_raw_results(results_root: Path, data_root: Path) -> pd.DataFrame:
    """Build per-model, per-mod rows directly from individual results CSVs.
    Computes compliance_rate and strict_success per sample using IFEval checkers.
    """
    # Import IFEval checkers
    scripts_dir = (SCRIPT_DIR / '../LLM/scripts').resolve()
    if str(scripts_dir) not in sys.path:
        sys.path.append(str(scripts_dir))
    try:
        from ifeval_checkers import check_constraint  # type: ignore
    except Exception:
        print('Warning: could not import ifeval_checkers; skipping raw IFEval loading')
        return pd.DataFrame()

    def extract_eval_text(text: str) -> str:
        if not isinstance(text, str):
            text = '' if text is None else str(text)
        blocks = re.findall(r"```(?:\\w+)?\\n([\\s\\S]*?)```", text)
        if blocks:
            return max((b.strip() for b in blocks), key=len, default=text)
        return text

    # Indices to ignore per modification
    IGNORED_KEYS = {
        'capitalization': {349},
        'concept_replacement': {3369},
        'punctuation': {2337},
        'sentiment': {2337, 1129},
        'temporal_bias': {3633},
        'typo_bias': {2337, 1129, 349},
    }

    def _norm_mod_key(m: str) -> str:
        s = str(m or '').strip()
        if s.endswith('_100'):
            s = s[:-4]
        if s.startswith('negation'):
            return 'negation'
        return s

    rows = []
    if not results_root.exists():
        return pd.DataFrame()
    for csv_file in sorted(results_root.glob('*.csv')):
        name = csv_file.name
        # Expect pattern: <model>-0shot-<mod>_100.csv or with context-aware
        if '_100.csv' not in name:
            continue
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue
        # Derive mod and model
        try:
            base = name.replace('.csv','')
            mod = base.split('_100')[0].split('-')[-1]
            model_raw = base.split('-0shot-')[0]
        except Exception:
            continue

        # Map model names to display keys consistent with other analyses
        if model_raw.startswith('gpt-5-context-aware') or model_raw == 'gpt-5-standard-context-aware':
            model = 'gpt-5-standard-context-aware'
        elif model_raw.startswith('gpt-5'):
            model = 'gpt-5-standard'
        elif model_raw.startswith('gpt4o') or model_raw.startswith('gpt-4o'):
            model = 'gpt4o'
        elif model_raw.startswith('claude'):
            model = 'claude'
        elif model_raw.startswith('deepseek'):
            model = 'deepseek-r1-deepseek'
        elif model_raw.startswith('llama'):
            model = 'llama'
        else:
            model = model_raw

        # Load dataset JSONL for constraints
        ds_path = (data_root / f'{mod}_100.jsonl')
        if not ds_path.exists():
            # Try without _100
            ds_path = (data_root / f'{mod}.jsonl')
        if not ds_path.exists():
            print(f'Missing dataset for {mod}; skipping {name}')
            continue
        # Read dataset
        items = []
        with ds_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(pd.json.loads(line))
                except Exception:
                    try:
                        items.append(__import__('json').loads(line))
                    except Exception:
                        continue
        # Fallback simple JSON parser
        if not items:
            import json as _json
            with ds_path.open('r', encoding='utf-8') as f:
                for ln in f:
                    s = ln.strip()
                    if not s:
                        continue
                    try:
                        items.append(_json.loads(s))
                    except Exception:
                        continue
        if not items:
            continue

        # Build key->(ids, kwargs, type) map and ordered list for index-based fallback
        key_to_ck = {}
        key_to_type = {}
        ordered_ck = []
        ordered_types = []
        for obj in items:
            key = obj.get('key')
            ids = obj.get('instruction_id_list') or []
            kwargs = obj.get('kwargs') or []
            typ = str(obj.get('type',''))
            ordered_ck.append((ids, kwargs))
            ordered_types.append(typ)
            if key is not None:
                key_to_ck[key] = (ids, kwargs)
                key_to_type[key] = typ

        # Extract outputs
        has_key = 'key' in df.columns
        orig_col = 'original_raw_output' if 'original_raw_output' in df.columns else 'original_output'
        mod_col = 'raw_output' if 'raw_output' in df.columns else 'modified_output'
        if orig_col not in df.columns or mod_col not in df.columns:
            continue

        comp_o = []
        comp_m = []
        strict_o = []
        strict_m = []
        row_types = []  # capture subtype per row when available (for negation logic)
        # Debug: sanity-check a specific sample
        DEBUG_KEY = 2250
        def _first_word_of_nth_par(text: str, n: int) -> str:
            try:
                parts = str(text).split("\n\n")
                if len(parts) >= n:
                    para = parts[n-1].strip()
                    # grab first token-like word (letters/digits/_)
                    import re
                    m = re.search(r"[A-Za-z0-9'_\-]+", para)
                    return (m.group(0) if m else '').strip().lower()
            except Exception:
                pass
            return ''
        for idx, row in df.iterrows():
            # Skip specific keys for listed modifications
            try:
                mod_key = _norm_mod_key(mod)
                k = row.get('key')
                if k is not None:
                    k_int = int(k)
                    if mod_key in IGNORED_KEYS and k_int in IGNORED_KEYS[mod_key]:
                        continue
            except Exception:
                pass
            if has_key:
                ck = key_to_ck.get(row.get('key'))
                r_typ = key_to_type.get(row.get('key'))
            else:
                ck = ordered_ck[idx] if idx < len(ordered_ck) else None
                r_typ = ordered_types[idx] if idx < len(ordered_types) else None
            if not ck or not isinstance(ck, tuple) or len(ck) != 2:
                continue
            ids, kwargs_list = ck
            # Score original
            num = len(ids)
            sat_o = 0
            text_o = extract_eval_text(str(row.get(orig_col, '')))
            for cid, p in zip(ids, kwargs_list):
                if check_constraint(cid, p or {}, text_o):
                    sat_o += 1
            rate_o = 100.0 * sat_o / num if num > 0 else 0.0
            strict_ok_o = 1 if num > 0 and sat_o == num else 0
            comp_o.append(rate_o)
            strict_o.append(strict_ok_o)
            # Score modified
            sat_m = 0
            text_m = extract_eval_text(str(row.get(mod_col, '')))
            for cid, p in zip(ids, kwargs_list):
                if check_constraint(cid, p or {}, text_m):
                    sat_m += 1
            rate_m = 100.0 * sat_m / num if num > 0 else 0.0
            strict_ok_m = 1 if num > 0 and sat_m == num else 0
            comp_m.append(rate_m)
            strict_m.append(strict_ok_m)
            row_types.append(str(r_typ or ''))

            # Debug print for the specific sanity-check sample (key 2250)
            try:
                this_key = int(row.get('key')) if has_key else None
            except Exception:
                this_key = None
            if this_key == DEBUG_KEY:
                st_raw = str(r_typ or '').lower()
                st = st_raw.replace('negation_','')
                neg_invert = (isinstance(mod, str) and str(mod).startswith('negation') and st in {'verbal','lexical','absolute'})
                strict_m_eff = (0 if strict_ok_m == 1 else 1) if neg_invert else strict_ok_m
                # Try to extract first word of 4th paragraph for quick human check
                orig_first = _first_word_of_nth_par(text_o, 4)
                mod_first = _first_word_of_nth_par(text_m, 4)
                print(
                    f"IFEval DEBUG key={DEBUG_KEY} model={model} mod={mod} subtype={st_raw} | "
                    f"strict_o={strict_ok_o} strict_m_raw={strict_ok_m} strict_m_eff={strict_m_eff} | "
                    f"orig_p4_first='{orig_first}' mod_p4_first='{mod_first}'"
                )

        if not comp_o or not comp_m:
            continue

        # Strict accuracy: 0 if any constraint violated, else 1
        so_arr = np.array(strict_o, dtype=int)
        sm_arr_raw = np.array(strict_m, dtype=int)
        # Negation subtype semantics (GSM-like) for modified side:
        # For verbal, lexical, absolute: treat "correct" as violating at least one constraint.
        # For approximate/double: treat "correct" as satisfying all constraints.
        if isinstance(mod, str) and mod.startswith('negation') and len(row_types) == len(sm_arr_raw):
            sm_eff = []
            for t, v in zip(row_types, sm_arr_raw.tolist()):
                st = str(t).replace('negation_','').lower()
                if st in {'verbal','lexical','absolute'}:
                    sm_eff.append(0 if v == 1 else 1)
                else:
                    sm_eff.append(int(v))
            sm_arr = np.array(sm_eff, dtype=int)
        else:
            sm_arr = sm_arr_raw
        A = 100.0 * float(np.mean(so_arr))
        B = 100.0 * float(np.mean(sm_arr))
        a_safe = max(A, 1e-6)
        delta = (B - A) * (np.log10(a_safe) / np.log10(100.0))
        abs_change = abs(B - A)
        U_strict = 100.0 * float(np.mean(np.abs(sm_arr - so_arr)))
        # U-only (degradations only): orig correct -> modified incorrect
        U_only = 100.0 * float(np.mean((so_arr == 1) & (sm_arr == 0)))

        # Significance (paired) using strict arrays
        from scipy.stats import wilcoxon, binomtest  # type: ignore
        pvals = []
        try:
            stat = wilcoxon(so_arr, sm_arr, zero_method='wilcox', alternative='two-sided')
            pvals.append(stat.pvalue)
        except Exception:
            pass
        try:
            so = pd.Series(so_arr).astype(int)
            sm_eff = pd.Series(sm_arr).astype(int)
            n01 = int(((so == 0) & (sm_eff == 1)).sum())
            n10 = int(((so == 1) & (sm_eff == 0)).sum())
            n = n01 + n10
            if n > 0:
                pv = binomtest(k=min(n01, n10), n=n, p=0.5).pvalue
                pvals.append(pv)
        except Exception:
            pass
        p = min(pvals) if pvals else float('nan')
        sig = 'n/a'
        if p == p:
            if p < 0.001: sig = '***'
            elif p < 0.01: sig = '**'
            elif p < 0.05: sig = '*'
            elif p < 0.1: sig = '.'
            else: sig = 'ns'

        rows.append({
            'task': 'ifeval',
            'modification': mod,
            'model': model,
            'A': A,
            'B': B,
            'weighted_delta': delta,
            'absolute_change': abs_change,
            'unrobustness': U_strict,
            'u_only': U_only,
            'p_value': p,
            'significance': sig,
            'source': 'raw',
        })

    return pd.DataFrame(rows)


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
        'capitalization': ('Orthography', 'Capitalization'),
        'punctuation': ('Orthography', 'Punctuation'),
        'typo_bias': ('Orthography', 'Spelling'),
        'coordinating_conjunction': ('Syntax', 'Conjunction'),
        'active_to_passive': ('Syntax', 'Voice'),
        'concept_replacement': ('Semantics', 'Concept'),
        'negation': ('Semantics', 'Negation'),
        'sentiment': ('Discourse', 'Appraisal'),
        'casual': ('Varieties', 'Style'),
        'dialectal': ('Varieties', 'Dialect'),
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
    cols = [m for m in model_order if m in pivot_delta.columns and m != 'GPT-5 (w. context)']
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
        fig.savefig(str(output_file), dpi=200)
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
        'capitalization': ('Orthography', 'Capitalization'),
        'punctuation': ('Orthography', 'Punctuation'),
        'typo_bias': ('Orthography', 'Spelling'),
        'coordinating_conjunction': ('Syntax', 'Conjunction'),
        'active_to_passive': ('Syntax', 'Voice'),
        'concept_replacement': ('Semantics', 'Concept'),
        'negation': ('Semantics', 'Negation'),
        'sentiment': ('Discourse', 'Appraisal'),
        'casual': ('Varieties', 'Style'),
        'dialectal': ('Varieties', 'Dialect'),
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
    cols = [m for m in model_order if m in pivot_u.columns and m != 'GPT-5 (w. context)']
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
    cbar.set_label('Unrobustness U (strict flip %)', rotation=90)
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
        fig.savefig(str(output_file), dpi=200)
        print(f"Unrobustness plot saved to {output_file}")
    finally:
        plt.close(fig)


def generate_compact_unrobustness_plot(df, output_file):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import numpy.ma as ma
    except Exception as e:
        print(f"Skipping compact U plot: matplotlib not available ({e})")
        return

    needed = {'model','modification','unrobustness'}
    if df is None or df.empty or not needed.issubset(df.columns):
        print(f"Skipping compact U plot for {output_file}: missing data/columns")
        return

    mod_to_cat = {
        'temporal_bias': 'Bias', 'geographical_bias': 'Bias', 'length_bias': 'Bias',
        'capitalization': 'Orthography', 'punctuation': 'Orthography', 'typo_bias': 'Orthography',
        'coordinating_conjunction': 'Syntax', 'active_to_passive': 'Syntax',
        'concept_replacement': 'Semantics', 'negation': 'Semantics',
        'sentiment': 'Discourse',
        'casual': 'Varieties', 'dialectal': 'Varieties'
    }
    model_map = {'gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}

    dfc = df.copy(); dfc['model'] = dfc['model'].map(_normalize_model_key).replace(model_map)
    dfc['category'] = dfc['modification'].map(lambda x: mod_to_cat.get(x, 'Other'))
    agg = dfc.groupby(['category','model']).agg(unrobustness=('unrobustness','mean')).reset_index()
    overall = dfc.groupby(['model']).agg(unrobustness=('unrobustness','mean')).reset_index(); overall.insert(0,'category','Overall')
    agg = pd.concat([agg, overall], ignore_index=True)

    p_u = agg.pivot_table(index='category', columns='model', values='unrobustness', aggfunc='mean')
    cols = list(p_u.columns)
    rows = [*sorted([r for r in p_u.index if r!='Overall']), 'Overall'] if 'Overall' in p_u.index else list(p_u.index)
    if not cols or not rows:
        print("Skipping compact U plot: insufficient data after aggregation")
        return
    p_u = p_u.reindex(index=rows, columns=cols)

    data = p_u.values.astype(float); data_masked = ma.masked_invalid(data)
    h = max(3, 0.6*data_masked.shape[0]+1.0); w = max(6, 0.6*data_masked.shape[1]+2.0)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(data_masked, cmap='Blues', vmin=_U_MIN, vmax=_U_MAX, aspect='auto')
    cbar = plt.colorbar(im, ax=ax); cbar.set_label('Unrobustness U (avg by category, %)', rotation=90)
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
        fig.savefig(str(output_file), dpi=200); print(f"Compact unrobustness plot saved to {output_file}")
    finally:
        plt.close(fig)

def generate_latex_table_combined(df, output_file):
    """Generate a single IFEval LaTeX table with one cell per model: "Δ | U".
    Excludes GPT-5 (w. context) from main columns.
    """
    need = {'model','modification','weighted_delta','unrobustness'}
    if df is None or df.empty or not need.issubset(df.columns):
        print(f"Skipping combined LaTeX for {output_file}: missing data/columns"); return
    mod_mapping = {
        'temporal_bias': ('Bias', 'Temporal'),
        'geographical_bias': ('Bias', 'Geographical'),
        'length_bias': ('Bias', 'Length'),
        'capitalization': ('Orthography', 'Capitalization'),
        'punctuation': ('Orthography', 'Punctuation'),
        'typo_bias': ('Orthography', 'Spelling'),
        'coordinating_conjunction': ('Syntax', 'Conjunction'),
        'active_to_passive': ('Syntax', 'Voice'),
        'concept_replacement': ('Semantics', 'Concept'),
        'negation': ('Semantics', 'Negation'),
        'sentiment': ('Discourse', 'Appraisal'),
        'casual': ('Varieties', 'Style'),
        'dialectal': ('Varieties', 'Dialect'),
    }
    model_order = ['GPT-4o', 'Claude-3.5', 'Llama 3.1', 'GPT-5', 'DS R1', 'GPT-5 (w. context)']
    model_map = {'gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    dfc = df.copy(); dfc['model'] = dfc['model'].map(_normalize_model_key).replace(model_map)
    dfc['category'] = dfc['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    dfc['modification_disp'] = dfc['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    p_delta = dfc.pivot_table(index=['category','modification_disp'], columns='model', values='weighted_delta', aggfunc='mean')
    p_u = dfc.pivot_table(index=['category','modification_disp'], columns='model', values='unrobustness', aggfunc='mean')
    cols = [m for m in model_order if m in p_delta.columns and m != 'GPT-5 (w. context)']
    if not cols:
        print(f"Skipping combined LaTeX for {output_file}: no columns after filtering"); return
    tot = len(cols)
    latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll'+'r'*tot+'}\n'
    latex += '\\toprule\nCategory & Modification & ' + ' & '.join([f'\\textbf{{{c}}}' for c in cols]) + ' \\\\\n'
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
    latex += '\\caption{IFEval: Weighted $\\Delta$ | U by model and modification}\\label{tab:ifeval_combined}\n\\end{table}'
    latex = latex.replace('\\n', '\n')
    with open(output_file,'w') as f: f.write(latex)
    print(f"Combined LaTeX table saved to {output_file}")

def generate_latex_table_unrob(df, output_file):
    """IFEval U-only table (discordant flip %, both directions) with colored cells, Avg column and Avg row."""
    need = {'model','modification','unrobustness'}
    if df is None or df.empty or not need.issubset(df.columns):
        print(f"Skipping U-only LaTeX for {output_file}: missing data/columns"); return
    # Map mods to categories and model display names
    mod_mapping = {
        'temporal_bias': ('Bias', 'Temporal'),
        'geographical_bias': ('Bias', 'Geographical'),
        'length_bias': ('Bias', 'Length'),
        'capitalization': ('Orthography', 'Capitalization'),
        'punctuation': ('Orthography', 'Punctuation'),
        'typo_bias': ('Orthography', 'Spelling'),
        'coordinating_conjunction': ('Syntax', 'Conjunction'),
        'active_to_passive': ('Syntax', 'Voice'),
        'concept_replacement': ('Semantics', 'Concept'),
        'negation': ('Semantics', 'Negation'),
        'sentiment': ('Discourse', 'Appraisal'),
        'casual': ('Varieties', 'Style'),
        'dialectal': ('Varieties', 'Dialect'),
    }
    model_order = ['GPT-4o', 'Claude-3.5', 'Llama 3.1', 'GPT-5', 'DS R1', 'GPT-5 (w. context)']
    model_map = {'gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    dfx = df.copy(); dfx['model'] = dfx['model'].map(_normalize_model_key).replace(model_map)
    dfx['category'] = dfx['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    dfx['modification_disp'] = dfx['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    p_u = dfx.pivot_table(index=['category','modification_disp'], columns='model', values='unrobustness', aggfunc='mean')
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
    # Build LaTeX with Avg column and Avg row
    tot = len(cols)
    latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll'+'r'*tot+'r}\n'
    latex += '\\toprule\n'
    latex += 'Category & Modification & ' + ' & '.join([f'\\textbf{{{c}}}' for c in cols]) + ' & \\textbf{Avg} \\\\\n'
    latex += '\\midrule\n'
    cats=[]
    for key,(cat,mod) in mod_mapping.items():
        idx = (cat, mod)
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
    # Average row per model
    col_means=[]
    for c in cols:
        series = p_u[c] if c in p_u.columns else pd.Series(dtype=float)
        col_means.append(float(pd.to_numeric(series, errors='coerce').mean()))
    overall = float(np.nanmean(col_means)) if col_means else float('nan')
    latex += '\\midrule\n'
    avg_cells = ['' if np.isnan(v) else fu_color(v) for v in col_means]
    overall_cell = '' if np.isnan(overall) else fu_color(overall)
    latex += '\\textbf{Average} &  ' + ' & '.join(avg_cells) + f" & {overall_cell} \\\\\n"
    latex += '\\bottomrule\n\\end{tabular}}\n'
    latex += '\\caption{IFEval: Unrobustness (U, \\%) by model and modification}\\label{tab:ifeval_unrob}\n\\end{table}'
    latex = latex.replace('\\n', '\n')
    with open(output_file,'w') as f: f.write(latex)
    print(f"U-only LaTeX table saved to {output_file}")

def generate_latex_table_rob(df, output_file):
    needed = {'model','modification','unrobustness'}
    if df is None or df.empty or not needed.issubset(df.columns):
        print(f"Skipping R-only LaTeX for {output_file}: missing data/columns"); return
    dfx = df.copy(); dfx['model'] = dfx['model'].map(_normalize_model_key).replace({'gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'})
    mod_mapping = {
        'temporal_bias': ('Bias', 'Temporal'),
        'geographical_bias': ('Bias', 'Geographical'),
        'length_bias': ('Bias', 'Length'),
        'capitalization': ('Orthography', 'Capitalization'),
        'punctuation': ('Orthography', 'Punctuation'),
        'typo_bias': ('Orthography', 'Spelling'),
        'coordinating_conjunction': ('Syntax', 'Conjunction'),
        'active_to_passive': ('Syntax', 'Voice'),
        'concept_replacement': ('Semantics', 'Concept'),
        'negation': ('Semantics', 'Negation'),
        'sentiment': ('Discourse', 'Appraisal'),
        'casual': ('Varieties', 'Style'),
        'dialectal': ('Varieties', 'Dialect'),
    }
    dfx['category'] = dfx['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    dfx['modification_disp'] = dfx['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    p_u = dfx.pivot_table(index=['category','modification_disp'], columns='model', values='unrobustness', aggfunc='mean')
    p_r = 100.0 - p_u
    cols = [c for c in ['GPT-4o','Claude-3.5','Llama 3.1','GPT-5','DS R1','GPT-5 (w. context)'] if c in p_r.columns]
    if not cols:
        print(f"Skipping R-only LaTeX for {output_file}: no columns after filtering"); return
    def fmt(v):
        try:
            return '' if pd.isna(v) else f"{float(v):.1f}"
        except Exception:
            return ''
    latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll'+'r'*len(cols)+'r}\n'
    latex += '\\toprule\n'
    latex += 'Category & Modification & ' + ' & '.join([f'\\textbf{{{c}}}' for c in cols]) + ' & \\textbf{Avg} \\\\\n'
    latex += '\\midrule\n'
    cats=[]
    for key,(cat,mod) in mod_mapping.items():
        idx=(cat,mod)
        if idx not in p_r.index: continue
        if cat not in cats:
            if cats: latex += '\\midrule\n'
            cats.append(cat)
            lead=f'\\textbf{{{cat}}}'
        else:
            lead=' '
        vals=[]; row_vals=[]
        for c in cols:
            v = p_r.loc[idx, c] if (idx in p_r.index and c in p_r.columns) else float('nan')
            row_vals.append(v)
            vals.append(fmt(v))
        ravg=float(pd.to_numeric(pd.Series(row_vals), errors='coerce').mean()) if row_vals else float('nan')
        latex += f"{lead} & \\textbf{{{mod}}} & " + ' & '.join(vals) + f" & {'' if np.isnan(ravg) else fmt(ravg)} \\\\\n"
    # Average row per model
    col_means=[]
    for c in cols:
        series = p_r[c] if c in p_r.columns else pd.Series(dtype=float)
        col_means.append(float(pd.to_numeric(series, errors='coerce').mean()))
    overall=float(np.nanmean(col_means)) if col_means else float('nan')
    latex += '\\midrule\n'
    latex += '\\textbf{Average} &  ' + ' & '.join([('' if np.isnan(v) else fmt(v)) for v in col_means]) + f" & {'' if np.isnan(overall) else fmt(overall)} \\\\\n"
    latex += '\\bottomrule\n\\end{tabular}}\n'
    latex += '\\caption{IFEval: Robustness (R, \\%) by model and modification}\\label{tab:ifeval_rob}\n\\end{table}'
    latex = latex.replace('\\n', '\n')
    with open(output_file,'w') as f: f.write(latex)
    print(f"R-only LaTeX table saved to {output_file}")

def generate_latex_table_dual(df, output_file):
    """IFEval dual table: left Δ block, right U block."""
    need = {'model','modification','weighted_delta','unrobustness','significance'}
    if df is None or df.empty or not need.issubset(df.columns):
        print(f"Skipping dual LaTeX for {output_file}: missing data/columns"); return
    mod_mapping = {
        'temporal_bias': ('Bias', 'Temporal'),
        'geographical_bias': ('Bias', 'Geographical'),
        'length_bias': ('Bias', 'Length'),
        'capitalization': ('Orthography', 'Capitalization'),
        'punctuation': ('Orthography', 'Punctuation'),
        'typo_bias': ('Orthography', 'Spelling'),
        'coordinating_conjunction': ('Syntax', 'Conjunction'),
        'active_to_passive': ('Syntax', 'Voice'),
        'concept_replacement': ('Semantics', 'Concept'),
        'negation': ('Semantics', 'Negation'),
        'sentiment': ('Discourse', 'Appraisal'),
        'casual': ('Varieties', 'Style'),
        'dialectal': ('Varieties', 'Dialect'),
    }
    model_order = ['GPT-4o', 'Claude-3.5', 'Llama 3.1', 'GPT-5', 'DS R1', 'GPT-5 (w. context)']
    model_map = {'gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    dfc = df.copy(); dfc['model'] = dfc['model'].map(_normalize_model_key).replace(model_map)
    dfc['category'] = dfc['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    dfc['modification_disp'] = dfc['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    p_delta = dfc.pivot_table(index=['category','modification_disp'], columns='model', values='weighted_delta', aggfunc='mean')
    p_sig = dfc.pivot_table(index=['category','modification_disp'], columns='model', values='significance', aggfunc='first')
    p_u = dfc.pivot_table(index=['category','modification_disp'], columns='model', values='unrobustness', aggfunc='mean')
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
        inten = unrob_intensity(val, _U_MIN, _U_MAX)
        txt = f"{val:.1f}"
        if inten >= 45:
            txt = f"\\textcolor{{white}}{{{txt}}}"
        return f"\\cellcolor{{blue!{inten}}} {txt}"
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
    latex += '\\caption{IFEval: Left = $\\Delta$, Right = U (one block per metric)}\\label{tab:ifeval_dual}\n\\end{table}'
    latex = latex.replace('\\n', '\n')
    with open(output_file,'w') as f: f.write(latex)
    print(f"Dual LaTeX table saved to {output_file}")

def generate_latex_table_combined_cells(df, output_file):
    """Combined table with two cells per model (Δ then U), colored (IFEval)."""
    need = {'model','modification','weighted_delta','unrobustness','significance'}
    if df is None or df.empty or not need.issubset(df.columns):
        print(f"Skipping combined-cells LaTeX for {output_file}: missing data/columns"); return
    mod_mapping = {
        'temporal_bias': ('Bias', 'Temporal'),
        'geographical_bias': ('Bias', 'Geographical'),
        'length_bias': ('Bias', 'Length'),
        'capitalization': ('Orthography', 'Capitalization'),
        'punctuation': ('Orthography', 'Punctuation'),
        'typo_bias': ('Orthography', 'Spelling'),
        'coordinating_conjunction': ('Syntax', 'Conjunction'),
        'active_to_passive': ('Syntax', 'Voice'),
        'concept_replacement': ('Semantics', 'Concept'),
        'negation': ('Semantics', 'Negation'),
        'sentiment': ('Discourse', 'Appraisal'),
        'casual': ('Varieties', 'Style'),
        'dialectal': ('Varieties', 'Dialect'),
    }
    model_order = ['GPT-4o', 'Claude-3.5', 'Llama 3.1', 'GPT-5', 'DS R1', 'GPT-5 (w. context)']
    model_map = {'gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    dfx = df.copy(); dfx['model'] = dfx['model'].map(_normalize_model_key).replace(model_map)
    dfx['category'] = dfx['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    dfx['modification_disp'] = dfx['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    p_delta = dfx.pivot_table(index=['category','modification_disp'], columns='model', values='weighted_delta', aggfunc='mean')
    p_sig = dfx.pivot_table(index=['category','modification_disp'], columns='model', values='significance', aggfunc='first')
    p_u = dfx.pivot_table(index=['category','modification_disp'], columns='model', values='unrobustness', aggfunc='mean')
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
        inten = unrob_intensity(val, _U_MIN, _U_MAX)
        txt = f"{val:.1f}"
        if inten >= 45:
            txt = f"\\textcolor{{white}}{{{txt}}}"
        return f"\\cellcolor{{blue!{inten}}} {txt}"
    tot = len(cols)
    latex = '\\begin{table}[h]\n\\centering\\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll'+'rr'*tot+'}\n'
    latex += '\\toprule\n'
    latex += 'Category & Modification & ' + ' & '.join([f'\\multicolumn{{2}}{{c}}{{\\textbf{{{c}}}}}' for c in cols]) + ' \\\\\n'
    latex += ' &  & ' + ' & '.join(['$\\Delta$ & U' for _ in cols]) + ' \\\\\n'
    latex += '\\midrule\n'
    # Group by category with a bold header once per category, consistent with other tasks
    cats = []
    # Build an ordered list of (category, modification) pairs present in pivots
    present = set(p_delta.index)
    for key, (cat, mod) in mod_mapping.items():
        if (cat, mod) not in present:
            continue
        if cat not in cats:
            if cats:
                latex += '\\midrule\n'
            cats.append(cat)
            lead = f'\\textbf{{{cat}}}'
        else:
            lead = ' '
        cells = []
        for c in cols:
            sig = p_sig.loc[(cat,mod), c] if ((cat,mod) in p_sig.index and c in p_sig.columns) else ''
            d = p_delta.loc[(cat,mod), c] if (c in p_delta.columns) else np.nan
            u = p_u.loc[(cat,mod), c] if (c in p_u.columns) else np.nan
            cells.append(fd_color(d, sig if isinstance(sig,str) else ''))
            cells.append(fu_color(u))
        latex += f"{lead} & \\textbf{{{mod}}} & " + ' & '.join(cells) + ' \\\\\n'
    latex += '\\bottomrule\n\\end{tabular}}\n'
    latex += '\\caption{IFEval: Two cells per model — $\\Delta$ and U}\\label{tab:ifeval_combined_cells}\n\\end{table}'
    latex = latex.replace('\\n', '\n')
    with open(output_file,'w') as f: f.write(latex)
    print(f"Combined-cells LaTeX table saved to {output_file}")

def generate_compact_latex_table_combined_cells(df, output_file):
    """Compact category-level combined-cells table for IFEval."""
    if df is None or df.empty or 'model' not in df.columns or 'modification' not in df.columns:
        print(f"Skipping compact combined-cells: missing data/columns"); return
    mod_to_cat = {
        'temporal_bias': 'Bias', 'geographical_bias': 'Bias', 'length_bias': 'Bias',
        'capitalization': 'Orthographic', 'punctuation': 'Orthographic', 'typo_bias': 'Orthographic',
        'coordinating_conjunction': 'Syntactic', 'active_to_passive': 'Syntactic',
        'concept_replacement': 'Semantic', 'negation': 'Semantic',
        'sentiment': 'Pragmatic', 'casual': 'Genre', 'dialectal': 'Genre'
    }
    model_order = ['GPT-4o', 'Claude-3.5', 'Llama 3.1', 'GPT-5', 'DS R1', 'GPT-5 (w. context)']
    model_map = {'gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    dfx=df.copy(); dfx['model']=dfx['model'].map(_normalize_model_key).replace(model_map); dfx['category']=dfx['modification'].map(lambda x: mod_to_cat.get(x,'Other'))
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
    latex+='\\caption{IFEval (compact): Two cells per model — $\\Delta$ and U}\\label{tab:ifeval_compact_combined_cells}\n\\end{table}'
    latex = latex.replace('\\n', '\n')
    with open(output_file,'w') as f: f.write(latex)
    print(f"Compact combined-cells LaTeX table saved to {output_file}")


def generate_compact_summary_plot(df, output_file):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import numpy.ma as ma
    except Exception as e:
        print(f"Skipping compact Δ plot: matplotlib not available ({e})")
        return

    if df is None or df.empty or 'model' not in df.columns or 'modification' not in df.columns:
        print(f"Skipping compact Δ plot for {output_file}: no data or missing columns")
        return

    # Canonical model display
    model_order = ['GPT-4o', 'Claude-3.5', 'Llama 3.1', 'GPT-5','DS R1', 'GPT-5 (w. context)']
    model_map = {
        'gpt4o': 'GPT-4o', 'claude': 'Claude-3.5', 'llama': 'Llama 3.1',
        'gpt-5-standard': 'GPT-5', 'deepseek-r1-deepseek': 'DS R1', 'gpt-5-standard-context-aware': 'GPT-5 (w. context)'
    }
    mod_to_cat = {
        'temporal_bias': 'Bias', 'geographical_bias': 'Bias', 'length_bias': 'Bias',
        'capitalization': 'Orthographic', 'punctuation': 'Orthographic', 'typo_bias': 'Orthographic',
        'coordinating_conjunction': 'Syntactic', 'active_to_passive': 'Syntactic',
        'concept_replacement': 'Semantic', 'negation': 'Semantic',
        'sentiment': 'Pragmatic', 'casual': 'Genre', 'dialectal': 'Genre'
    }

    dfc = df.copy()
    dfc['model'] = dfc['model'].map(_normalize_model_key).replace(model_map)
    dfc['category'] = dfc['modification'].map(lambda x: mod_to_cat.get(x, 'Other'))
    agg = dfc.groupby(['category','model']).agg(weighted_delta=('weighted_delta','mean')).reset_index()
    overall = dfc.groupby(['model']).agg(weighted_delta=('weighted_delta','mean')).reset_index(); overall.insert(0,'category','Overall')
    agg = pd.concat([agg, overall], ignore_index=True)

    p_delta = agg.pivot_table(index='category', columns='model', values='weighted_delta', aggfunc='mean')
    # Columns present in data only, in canonical order
    cols = [m for m in model_order if m in getattr(p_delta, 'columns', [])]
    if not cols:
        print('No models with Δ data found for compact plot; skipping')
        return
    # Rows present; drop categories fully empty
    rows_all = list(getattr(p_delta, 'index', []))
    rows = []
    for cat in rows_all:
        has_val = any(not pd.isna(p_delta.loc[cat, c]) for c in cols if cat in p_delta.index)
        if has_val:
            rows.append(cat)
    if not rows:
        print('No categories with Δ data found for compact plot; skipping')
        return
    p_delta = p_delta.reindex(index=rows, columns=cols)

    data = p_delta.values.astype(float)
    data_masked = ma.masked_invalid(data)
    h = max(3, 0.6*data_masked.shape[0]+1.0)
    w = max(6, 0.6*data_masked.shape[1]+2.0)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(data_masked, cmap='RdYlGn', vmin=-10, vmax=10, aspect='auto')
    cbar = plt.colorbar(im, ax=ax); cbar.set_label('Weighted Δ (avg by category)', rotation=90)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha='right')
    ax.set_title(''); ax.set_xlabel(''); ax.set_ylabel('')
    # Annotate with values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:+.1f}", ha='center', va='center', fontsize=8)
    fig.tight_layout()
    try:
        fig.savefig(str(output_file), dpi=200); print(f"Compact Δ plot saved to {output_file}")
    finally:
        plt.close(fig)


def main():
    aggregates_root = (SCRIPT_DIR / '../LLM/results/ifeval_aggregates')
    df_agg = load_aggregates(aggregates_root)
    # If aggregates are missing/incomplete, build from raw results as a fallback
    results_root = (SCRIPT_DIR / '../LLM/results/ifeval')
    data_root = (SCRIPT_DIR / '../../data/modified_data/ifeval')
    raw_df = load_from_raw_results(results_root, data_root)

    # Sanity checks: counts per model
    if not raw_df.empty:
        raw_counts = raw_df.groupby('model')['modification'].nunique().sort_index()
        print("IFEval raw results loaded (mods per model):")
        for model, cnt in raw_counts.items():
            print(f"  - {model}: {cnt}")
    if not df_agg.empty:
        agg_counts = df_agg.groupby('model')['modification'].nunique().sort_index()
        print("IFEval aggregates loaded (mods per model):")
        for model, cnt in agg_counts.items():
            print(f"  - {model}: {cnt}")

    # Prefer raw rows; drop aggregate rows that duplicate a raw (model, modification) after normalization
    if not raw_df.empty and not df_agg.empty:
        try:
            raw_df = raw_df.copy(); df_agg = df_agg.copy()
            raw_df['_mkey'] = raw_df['model'].map(_normalize_model_key)
            df_agg['_mkey'] = df_agg['model'].map(_normalize_model_key)
            keys_raw = set(zip(raw_df['_mkey'], raw_df['modification']))
            mask = [ (m, mod) not in keys_raw for m, mod in zip(df_agg['_mkey'], df_agg['modification']) ]
            df_agg = df_agg[mask]
        except Exception:
            pass
    df = pd.concat([d for d in [raw_df.drop(columns=['_mkey'], errors='ignore'), df_agg.drop(columns=['_mkey'], errors='ignore')] if (isinstance(d, pd.DataFrame) and not d.empty)], ignore_index=True) if (not raw_df.empty or not df_agg.empty) else pd.DataFrame()
    # Ensure negation rows come only from RAW computations (agg U may not reflect negation semantics)
    try:
        if not df.empty and 'source' in df.columns and 'modification' in df.columns:
            df = pd.concat([
                df[(df['modification'] == 'negation') & (df['source'] == 'raw')],
                df[df['modification'] != 'negation']
            ], ignore_index=True)
    except Exception:
        pass
    # Normalize model names to canonical keys prior to diagnostics and pivots
    if not df.empty and 'model' in df.columns:
        df['model'] = df['model'].map(_normalize_model_key)

    if df.empty:
        print('No IFEval aggregates or raw results found.')
        return
    # Coverage diagnostics (which mods/categories per model)
    try:
        cov_mods = df.groupby('model')['modification'].nunique().sort_index()
        all_mods = set(df['modification'].unique())
        print('IFEval coverage by model (unique modifications):')
        for m in sorted(df['model'].unique()):
            mods_m = set(df[df['model']==m]['modification'].unique())
            missing = sorted(all_mods - mods_m)
            print(f"  - {m}: {len(mods_m)} mods; missing: {', '.join(missing) if missing else 'none'}")
        # By category
        mod_to_cat_diag = {
            'temporal_bias': 'Bias','geographical_bias': 'Bias','length_bias': 'Bias',
            'capitalization': 'Orthographic','punctuation': 'Orthographic','typo_bias': 'Orthographic',
            'coordinating_conjunction': 'Syntactic','active_to_passive': 'Syntactic',
            'concept_replacement': 'Semantic','negation': 'Semantic',
            'sentiment': 'Pragmatic','casual': 'Genre','dialectal': 'Genre'
        }
        df['category'] = df['modification'].map(lambda x: mod_to_cat_diag.get(x, 'Other'))
        cov_cats = df.groupby('model')['category'].nunique().sort_index()
        all_cats = set(df['category'].unique())
        print('IFEval coverage by model (categories):')
        for m in sorted(df['model'].unique()):
            cats_m = set(df[df['model']==m]['category'].unique())
            missing_c = sorted(all_cats - cats_m)
            print(f"  - {m}: {len(cats_m)} cats; missing: {', '.join(missing_c) if missing_c else 'none'}")
    except Exception:
        pass

    plot_path = (SCRIPT_DIR / 'ifeval_results_heatmap.png').resolve()
    u_plot_path = (SCRIPT_DIR / 'ifeval_unrobustness_heatmap.png').resolve()
    generate_summary_plot(df, plot_path)
    generate_unrobustness_plot(df, u_plot_path)
    # Compact Δ and U heatmaps
    compact_path = (SCRIPT_DIR / 'ifeval_results_heatmap_compact.png').resolve()
    u_compact_path = (SCRIPT_DIR / 'ifeval_unrobustness_heatmap_compact.png').resolve()
    generate_compact_summary_plot(df, compact_path)
    generate_compact_unrobustness_plot(df, u_compact_path)
    # Full combined table (Δ | U per model)
    try:
        generate_latex_table_combined(df, SCRIPT_DIR / 'ifeval_results_table.tex')
    except Exception as e:
        print(f"Combined table skipped (IFEval): {e}")
    # U-only table
    try:
        generate_latex_table_unrob(df, SCRIPT_DIR / 'ifeval_results_table_unrobustness.tex')
    except Exception as e:
        print(f"U-only table skipped (IFEval): {e}")
    # R-only table
    try:
        generate_latex_table_rob(df, SCRIPT_DIR / 'ifeval_results_table_robustness.tex')
    except Exception as e:
        print(f"R-only table skipped (IFEval): {e}")
    # Compact category + overall
    # Map modifications to categories consistent with IFEval set
    mod_to_cat = {
        'temporal_bias': 'Bias', 'geographical_bias': 'Bias', 'length_bias': 'Bias',
        'capitalization': 'Orthographic', 'punctuation': 'Orthographic', 'typo_bias': 'Orthographic',
        'coordinating_conjunction': 'Syntactic', 'active_to_passive': 'Syntactic',
        'concept_replacement': 'Semantic', 'negation': 'Semantic',
        'sentiment': 'Pragmatic',
        'casual': 'Genre', 'dialectal': 'Genre'
    }
    model_map = {
        'gpt4o': 'GPT-4o', 'claude': 'Claude-3.5', 'llama': 'Llama 3.1', 'gpt-5-standard': 'GPT-5', 'deepseek-r1-deepseek': 'DS R1', 'gpt-5-standard-context-aware': 'GPT-5 (w. context)'
    }
    dfc = df.copy(); dfc['model'] = dfc['model'].map(_normalize_model_key).replace(model_map)
    dfc['category'] = dfc['modification'].map(lambda x: mod_to_cat.get(x, 'Other'))
    agg = dfc.groupby(['category','model']).agg(weighted_delta=('weighted_delta','mean'), unrobustness=('unrobustness','mean')).reset_index()
    overall = dfc.groupby(['model']).agg(weighted_delta=('weighted_delta','mean'), unrobustness=('unrobustness','mean')).reset_index(); overall.insert(0,'category','Overall')
    agg = pd.concat([agg, overall], ignore_index=True)
    # Save compact LaTeX table
    p_delta = agg.pivot_table(index='category', columns='model', values='weighted_delta', aggfunc='mean')
    p_unrob = agg.pivot_table(index='category', columns='model', values='unrobustness', aggfunc='mean')
    # Use a stable model order with canonical display names; include models present in either pivot
    model_order = ['GPT-4o', 'Claude-3.5', 'Llama 3.1', 'GPT-5', 'DS R1', 'GPT-5 (w. context)']
    cols = [m for m in model_order if ((m in getattr(p_delta, 'columns', [])) or (m in getattr(p_unrob, 'columns', []))) and m != 'GPT-5 (w. context)']
    print('IFEval compact table models included:', ', '.join(cols))
    # Diagnostics: non-null counts
    for m in cols:
        nn_d = int(p_delta[m].notna().sum()) if m in getattr(p_delta, 'columns', []) else 0
        nn_u = int(p_unrob[m].notna().sum()) if m in getattr(p_unrob, 'columns', []) else 0
        print(f"  {m}: Δ={nn_d}, U={nn_u} non-null")
    if not cols:
        print('No models with data found for compact table; skipping LaTeX generation')
        return
    # Build row set from either pivot and drop categories that are entirely empty
    row_set = set(getattr(p_delta, 'index', [])) | set(getattr(p_unrob, 'index', []))
    rows_all = sorted([r for r in row_set if r != 'Overall']) + (['Overall'] if 'Overall' in row_set else [])
    rows = []
    for cat in rows_all:
        has_val = False
        for c in cols:
            d = p_delta.loc[cat, c] if (cat in getattr(p_delta, 'index', []) and c in getattr(p_delta, 'columns', [])) else np.nan
            u = p_unrob.loc[cat, c] if (cat in getattr(p_unrob, 'index', []) and c in getattr(p_unrob, 'columns', [])) else np.nan
            if not pd.isna(d) or not pd.isna(u):
                has_val = True
                break
        if has_val:
            rows.append(cat)
    if not rows:
        print('No categories with data found for compact table; skipping LaTeX generation')
        return
    def fcell(d,u):
        sd = '' if pd.isna(d) else f"{d:+.1f}"
        su = '' if pd.isna(u) else f"{u:.1f}"
        if sd=='' and su=='':
            return ''
        return f"{sd} | {su}"
    # Booktabs table
    latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{l' + 'r'*len(cols) + '}\n'
    latex += '\\toprule\nCategory & ' + ' & '.join([f'\\textbf{{{c}}}' for c in cols]) + ' \\\\\n'
    latex += '\\midrule\n'
    for cat in rows:
        cells=[]
        for c in cols:
            d = p_delta.loc[cat, c] if (cat in p_delta.index and c in p_delta.columns) else np.nan
            u = p_unrob.loc[cat, c] if (cat in p_unrob.index and c in p_unrob.columns) else np.nan
            cells.append(fcell(d,u))
        latex += f"{cat} & " + ' & '.join(cells) + ' \\\\\n'
    latex += '\\bottomrule\n\\end{tabular}}\n'
    latex += '\\caption{IFEval: Compact category-level averages — $\\Delta$ | U per model}\n'
    latex += '\\label{tab:ifeval_compact}\n\\end{table}'
    with open(SCRIPT_DIR / 'ifeval_results_table_compact.tex', 'w') as f:
        f.write(latex)
    # Optional: produce LaTeX if desired later

    # Separate GPT-5 vs GPT-5 (w. context) compact table
    try:
        generate_gpt5_context_table(dfc, SCRIPT_DIR / 'ifeval_gpt5_context_table.tex')
    except Exception as e:
        print(f"GPT-5 context table skipped (IFEval): {e}")
    # Combined per-model two-cell table
    try:
        generate_latex_table_combined_cells(df, SCRIPT_DIR / 'ifeval_results_table_combined_cells.tex')
    except Exception as e:
        print(f"Combined-cells table skipped (IFEval): {e}")
    # Compact combined-cells table
    try:
        generate_compact_latex_table_combined_cells(df, SCRIPT_DIR / 'ifeval_results_table_compact_combined_cells.tex')
    except Exception as e:
        print(f"Compact combined-cells table skipped (IFEval): {e}")
    # Dual combined table (left Δ, right U)
    try:
        generate_latex_table_dual(df, SCRIPT_DIR / 'ifeval_results_table_dual.tex')
    except Exception as e:
        print(f"Dual table skipped (IFEval): {e}")

def generate_gpt5_context_table(df, output_file):
    if df is None or df.empty:
        return
    # df here is already normalized and display-mapped in caller (dfc)
    dfx = df.copy()
    dfx = dfx[dfx['model'].isin(['GPT-5','GPT-5 (w. context)'])]
    if dfx.empty:
        raise ValueError('No GPT-5 context data')
    mod_to_cat = {
        'temporal_bias': 'Bias', 'geographical_bias': 'Bias', 'length_bias': 'Bias',
        'capitalization': 'Orthographic', 'punctuation': 'Orthographic', 'typo_bias': 'Orthographic',
        'coordinating_conjunction': 'Syntactic', 'active_to_passive': 'Syntactic',
        'concept_replacement': 'Semantic', 'negation': 'Semantic',
        'sentiment': 'Pragmatic', 'casual': 'Genre', 'dialectal': 'Genre'
    }
    dfx['category'] = dfx['modification'].map(lambda x: mod_to_cat.get(x, 'Other'))
    agg = dfx.groupby(['category','model']).agg(weighted_delta=('weighted_delta','mean'), unrobustness=('unrobustness','mean')).reset_index()
    overall = dfx.groupby(['model']).agg(weighted_delta=('weighted_delta','mean'), unrobustness=('unrobustness','mean')).reset_index(); overall.insert(0,'category','Overall')
    agg = pd.concat([agg, overall], ignore_index=True)
    p_delta = agg.pivot_table(index='category', columns='model', values='weighted_delta', aggfunc='mean')
    p_unrob = agg.pivot_table(index='category', columns='model', values='unrobustness', aggfunc='mean')
    cols = [c for c in ['GPT-5','GPT-5 (w. context)'] if (c in p_delta.columns) or (c in p_unrob.columns)]
    if not cols:
        raise ValueError('No GPT-5 columns present')
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
    latex += '\\caption{IFEval: GPT-5 vs GPT-5 (w. context) — category-level $\\Delta$ and U}\\label{tab:ifeval_gpt5_context}\n\\end{table}'
    latex = latex.replace('\\n', '\n')
    with open(output_file,'w') as f: f.write(latex)


if __name__ == '__main__':
    main()
