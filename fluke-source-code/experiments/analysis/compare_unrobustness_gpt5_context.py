#!/usr/bin/env python3
"""
Compare unrobustness between GPT-5 Standard and GPT-5 Context-Aware across tasks.

Outputs a wide table with one row per modification (canonical names) and
per-task columns:
  <task>_std, <task>_ctx, <task>_diff (ctx - std)

Tasks considered: sa, coref, dialogue, ner, gsm, ifeval (ifeval from aggregates).

Also prints sanity checks: tasks loaded and number of overlapping modifications.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import re
from utils import get_global_unrobustness_range, unrob_intensity

SCRIPT_DIR = Path(__file__).parent

# Canonical display ↔ internal key mapping (same as correlation script)
DISPLAY_TO_KEY = {
    'Temporal': 'temporal_bias',
    'Geographical': 'geographical_bias',
    'Length': 'length_bias',
    # Orthography
    'Capitalization': 'capitalization',
    'Punctuation': 'punctuation',
    'Spelling': 'typo_bias',
    'Derivation': 'derivation',
    'Compound': 'compound_word',
    'Voice': 'active_to_passive',
    'Grammar': 'grammatical_role',
    'Conjunction': 'coordinating_conjunction',
    'Concept': 'concept_replacement',
    'Negation': 'negation',
    'Disc. markers': 'discourse',
    'Appraisal': 'sentiment',
    'Style': 'casual',
    'Dialect': 'dialectal',
}
KEY_TO_DISPLAY = {v: k for k, v in DISPLAY_TO_KEY.items()}

# Category mapping for canonical modifications (by internal key)
KEY_TO_CATEGORY = {
    # Bias
    'temporal_bias': 'Bias',
    'geographical_bias': 'Bias',
    'length_bias': 'Bias',
    # Orthography
    'capitalization': 'Orthography',
    'punctuation': 'Orthography',
    'typo_bias': 'Orthography',
    # Morphology
    'derivation': 'Morphology',
    'compound_word': 'Morphology',
    # Syntax
    'active_to_passive': 'Syntax',
    'grammatical_role': 'Syntax',
    'coordinating_conjunction': 'Syntax',
    # Semantics
    'concept_replacement': 'Semantics',
    'negation': 'Semantics',
    # Discourse/Pragmatic
    'sentiment': 'Discourse',
    'discourse': 'Discourse',
    # Varieties
    'casual': 'Varieties',
    'dialectal': 'Varieties',
}


def _normalize_model(name: str) -> str:
    if name is None:
        return ""
    return name.strip().lower().replace(" ", "").replace("-", "")


STD_ALIASES = {
    _normalize_model(x)
    for x in (
        'gpt-5-standard',
        'gpt5-standard',
        'gpt5',
        'gpt-5',
        # common display variants
        'GPT-5',
    )
}
CTX_ALIASES = {
    _normalize_model(x)
    for x in (
        'gpt-5-standard-context-aware',
        'gpt5-standard-context-aware',
        'gpt-5-context-aware',
        'gpt5-context-aware',
        # common display variants seen in tables/viewers
        'GPT-5 (w. context)',
        'GPT5 (w. context)',
        'GPT-5 w. context',
        'GPT5 w. context',
        'GPT-5 with context',
        'GPT5 with context',
    )
}


def load_task_un(task_csv: Path):
    """Load per-modification unrobustness for both GPT-5 variants from a task CSV.
    Returns two Series indexed by modification: (std, ctx). Missing returns None.
    """
    if not task_csv.exists():
        return None, None
    try:
        df = pd.read_csv(task_csv)
    except Exception:
        return None, None
    if 'modification' not in df.columns or 'unrobustness' not in df.columns:
        return None, None
    if 'model' not in df.columns:
        return None, None
    df['_norm_model'] = df['model'].astype(str).apply(_normalize_model)
    df_std = df[df['_norm_model'].isin(STD_ALIASES)]
    df_ctx = df[df['_norm_model'].isin(CTX_ALIASES)]
    s_std = df_std.groupby('modification')['unrobustness'].mean() if not df_std.empty else None
    s_ctx = df_ctx.groupby('modification')['unrobustness'].mean() if not df_ctx.empty else None
    return s_std, s_ctx


def load_ifeval_un(_: Path):
    """Load per-modification unrobustness for IFEval from scripts-level CSVs only.
    Returns two Series (std, ctx) indexed by modification using U_frac.
    """
    return _load_ifeval_from_scripts()


def _load_ifeval_from_scripts():
    """Build per-modification U (strict flip %) from scripts-level comparison CSVs.
    These CSVs contain per-sample original_correct/modified_correct; we compute
    U = 100 * mean(abs(modified_correct - original_correct)) grouped by modification.
    """
    scripts_dir = (SCRIPT_DIR / '../LLM/scripts').resolve()
    std_path = scripts_dir / 'gpt-5-standard_comparison_ifeval.csv'
    ctx_path = scripts_dir / 'gpt-5-standard-context-aware_comparison_ifeval.csv'
    s_std = s_ctx = None
    try:
        if std_path.exists():
            df = pd.read_csv(std_path)
            # Build a normalized modification key (strip trailing _100)
            mod_col = 'mod' if 'mod' in df.columns else ('modification' if 'modification' in df.columns else None)
            if mod_col is not None:
                df['_mod_key'] = df[mod_col].astype(str).str.replace(r'_100$', '', regex=True)
            # Prefer aggregated if available; else derive from per-sample flags
            if 'U_frac' in df.columns and mod_col is not None:
                s_std = df.groupby('_mod_key')['U_frac'].mean()
            elif {'original_correct', 'modified_correct'}.issubset(df.columns) and mod_col is not None:
                flips = (df['modified_correct'].astype(int) - df['original_correct'].astype(int)).abs()
                s_std = (100.0 * flips.groupby(df['_mod_key']).mean())
    except Exception:
        pass
    try:
        if ctx_path.exists():
            df = pd.read_csv(ctx_path)
            mod_col = 'mod' if 'mod' in df.columns else ('modification' if 'modification' in df.columns else None)
            if mod_col is not None:
                df['_mod_key'] = df[mod_col].astype(str).str.replace(r'_100$', '', regex=True)
            if 'U_frac' in df.columns and mod_col is not None:
                s_ctx = df.groupby('_mod_key')['U_frac'].mean()
            elif {'original_correct', 'modified_correct'}.issubset(df.columns) and mod_col is not None:
                flips = (df['modified_correct'].astype(int) - df['original_correct'].astype(int)).abs()
                s_ctx = (100.0 * flips.groupby(df['_mod_key']).mean())
    except Exception:
        pass
    return s_std, s_ctx


def _load_gsm_from_scripts():
    """Fallback: Build per-modification U (flip %) for GSM from scripts-level comparison CSVs.
    Uses precomputed per-sample correctness flags when available. If absent, attempts
    a conservative recompute that respects negation semantics by comparing predictions
    to the dataset's original_answer mapped by index.
    """
    scripts_dir = (SCRIPT_DIR / '../LLM/scripts').resolve()
    std_path = scripts_dir / 'gpt-5-standard_comparison_gsm.csv'
    ctx_path = scripts_dir / 'gpt-5-standard-context-aware_comparison_gsm.csv'

    def _series_from(df: pd.DataFrame) -> pd.Series | None:
        if df is None or df.empty:
            return None
        # Normalize modification key: strip trailing _100
        if 'modification' not in df.columns:
            return None
        df = df.copy()
        df['_mod_key'] = df['modification'].astype(str).str.replace(r'_100$', '', regex=True)
        # Prefer using correctness flags if present (already negation-aware)
        if {'original_correct', 'modified_correct'}.issubset(df.columns):
            flips = (df['modified_correct'].astype(int) - df['original_correct'].astype(int)).abs()
            return 100.0 * flips.groupby(df['_mod_key']).mean()
        # Minimal fallback: attempt to recompute flips for negation only
        if not {'original_pred', 'modified_pred', 'index'}.issubset(df.columns):
            return None
        try:
            # Build dataset mapping per modification as needed
            data_root = (SCRIPT_DIR / '../data/modified_data/gsm').resolve()
            u_vals = {}
            for mod, group in df.groupby('_mod_key'):
                if not str(mod).startswith('negation'):
                    continue  # skip non-negation without flags
                # Load JSONL to map index -> original_answer and subtype
                import json, os
                p = data_root / f'{mod}_100.jsonl'
                if not p.exists():
                    p = data_root / f'{mod}.jsonl'
                ans_by_idx = {}
                sub_by_idx = {}
                if p.exists():
                    with open(p, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                obj = json.loads(line.strip())
                            except Exception:
                                continue
                            idx = obj.get('index')
                            if idx is None:
                                continue
                            ans_by_idx[int(idx)] = str(obj.get('original_answer', obj.get('short_answer', '')))
                            st = obj.get('negation_subtype', obj.get('type', ''))
                            if st:
                                sub_by_idx[int(idx)] = str(st)
                # Compute flips
                flips = []
                for _, r in group.iterrows():
                    try:
                        idx = int(r.get('index'))
                    except Exception:
                        continue
                    gold = ans_by_idx.get(idx)
                    if gold is None:
                        continue
                    # simple numeric normalization
                    def _norm(s):
                        return str(s).replace(',', '').strip()
                    mp = _norm(r.get('modified_pred', ''))
                    op = _norm(r.get('original_pred', ''))
                    oa = _norm(gold)
                    # parse to float if possible
                    def _flt(x):
                        try:
                            return float(x)
                        except Exception:
                            return None
                    mpf, opf, oaf = _flt(mp), _flt(op), _flt(oa)
                    tol = 1e-6
                    ori_ok = (opf is not None and oaf is not None and abs(opf - oaf) < tol) or (op == oa)
                    st = str(sub_by_idx.get(idx, '')).lower()
                    eq = (mpf is not None and oaf is not None and abs(mpf - oaf) < tol) or (mp == oa)
                    if ('approximate' in st) or ('double' in st):
                        mod_ok = eq
                    else:
                        mod_ok = not eq
                    flips.append(int(ori_ok) ^ int(mod_ok))
                if flips:
                    u_vals[mod] = 100.0 * (sum(flips) / len(flips))
            if not u_vals:
                return None
            return pd.Series(u_vals)
        except Exception:
            return None

    s_std = s_ctx = None
    try:
        if std_path.exists():
            s_std = _series_from(pd.read_csv(std_path))
    except Exception:
        s_std = None
    try:
        if ctx_path.exists():
            s_ctx = _series_from(pd.read_csv(ctx_path))
    except Exception:
        s_ctx = None
    return s_std, s_ctx


def main():
    task_files = {
        'sa': SCRIPT_DIR / 'sa_modification_results_llm.csv',
        'coref': SCRIPT_DIR / 'coref_modification_results_llm.csv',
        'dialogue': SCRIPT_DIR / 'dialogue_modification_results_llm.csv',
        'ner': SCRIPT_DIR / 'ner_modification_results_llm.csv',
        'gsm': SCRIPT_DIR / 'gsm_modification_results_llm.csv',
    }
    # Load tasks
    std_map, ctx_map = {}, {}
    for task, path in task_files.items():
        s_std, s_ctx = load_task_un(path)
        if s_std is not None:
            std_map[task] = s_std
        if s_ctx is not None:
            ctx_map[task] = s_ctx
    # IFEval
    agg_root = SCRIPT_DIR / '../LLM/results/ifeval_aggregates'
    s_std, s_ctx = load_ifeval_un(agg_root)
    if s_std is not None:
        std_map['ifeval'] = s_std
    if s_ctx is not None:
        ctx_map['ifeval'] = s_ctx

    # GSM fallback: if gsm not present for either std or ctx, try building from scripts-level comparison CSVs
    if 'gsm' not in std_map or 'gsm' not in ctx_map or std_map.get('gsm') is None or ctx_map.get('gsm') is None:
        try:
            s_std_gsm, s_ctx_gsm = _load_gsm_from_scripts()
            if s_std_gsm is not None:
                std_map['gsm'] = s_std_gsm
            if s_ctx_gsm is not None:
                ctx_map['gsm'] = s_ctx_gsm
        except Exception:
            pass

    tasks = ['sa', 'coref', 'dialogue', 'ner', 'gsm', 'ifeval']
    std_loaded = [t for t in tasks if t in std_map]
    ctx_loaded = [t for t in tasks if t in ctx_map]
    print(f'Tasks loaded (std): {len(std_loaded)}/{len(tasks)} -> {", ".join(std_loaded) if std_loaded else "none"}')
    print(f'Tasks loaded (ctx): {len(ctx_loaded)}/{len(tasks)} -> {", ".join(ctx_loaded) if ctx_loaded else "none"}')

    # Build table index: intersection of modifications present in either
    mods = set()
    for s in list(std_map.values()) + list(ctx_map.values()):
        mods.update(s.index.tolist())
    if not mods:
        print('No modifications found for either model.')
        return

    # Assemble DataFrame
    rows = []
    for mod in sorted(mods):
        row = {
            'modification': mod,
            'display': KEY_TO_DISPLAY.get(mod, mod),
            'category': KEY_TO_CATEGORY.get(mod, 'Other'),
        }
        for task in tasks:
            std_v = std_map.get(task, pd.Series(dtype=float)).get(mod, np.nan)
            ctx_v = ctx_map.get(task, pd.Series(dtype=float)).get(mod, np.nan)
            row[f'{task}_std'] = std_v
            row[f'{task}_ctx'] = ctx_v
        rows.append(row)
    df = pd.DataFrame(rows)

    # Sort by display name for readability
    df = df.sort_values('display')
    print(f'Unique modifications compared: {df.shape[0]}')

    # Print a compact U-only table with both columns per task (std vs ctx)
    cols = []
    for t in tasks:
        cols.extend([f'{t}_std', f'{t}_ctx'])
    printable = df[['display'] + cols]
    # Friendly column names
    rename_map = {}
    for t in tasks:
        rename_map[f'{t}_std'] = f'{t} (std)'
        rename_map[f'{t}_ctx'] = f'{t} (ctx)'
    printable = printable.rename(columns=rename_map)
    print('\nU by task (std vs ctx):')
    print(printable.to_string(index=False, float_format=lambda x: f'{x:.3f}' if pd.notna(x) else 'NA'))

    # Also write a CSV with std/ctx per task
    out_csv = SCRIPT_DIR / 'gpt5_ctx_unrobustness_comparison.csv'
    try:
        df.to_csv(out_csv, index=False)
        print(f'Wrote detailed comparison to {out_csv}')
    except Exception as e:
        print(f'Skipped writing CSV: {e}')

    # Write LaTeX table (ΔU-only = ctx - std), with category column and red/green coloring
    try:
        out_tex = SCRIPT_DIR / 'gpt5_ctx_unrobustness_table.tex'
        # Build LaTeX header
        header_cols = ['Category', 'Modification']
        # one column per task: ΔU (ctx - std)
        for t in tasks:
            header_cols.append(f"{t.upper()} (ΔU)")
        # Append average Δ column
        header_cols.append('AVG (ΔU)')
        # Prepare values without mutating display strings
        def fmt_num(v):
            return 'NA' if pd.isna(v) else f"{float(v):.1f}"
        # Red/Green intensity scaling for diffs
        def cell_rg(diff_val: float) -> str:
            # negative (ctx < std) = improvement -> green; positive -> red
            if pd.isna(diff_val):
                return 'NA'
            mag = min(10.0, abs(float(diff_val)))
            inten = int(round((mag / 10.0) * 20))  # 0..20
            txt = fmt_num(diff_val)
            if diff_val < 0:
                # green
                return f"\\cellcolor{{green!{inten}}} {txt}"
            elif diff_val > 0:
                return f"\\cellcolor{{red!{inten}}} {txt}"
            else:
                return txt
        # Emit LaTeX using booktabs and adjustbox
        lines = []
        lines.append('\\begin{table}[!tbp]')
        lines.append('\\centering')
        lines.append('\\footnotesize')
        lines.append('\\begin{adjustbox}{max width=\\linewidth}')
        cols_spec = 'll' + 'r' * (len(header_cols) - 2)
        lines.append(f'\\begin{{tabular}}{{{cols_spec}}}')
        lines.append('\\toprule')
        lines.append(' & '.join([f'\\textbf{{{h}}}' for h in header_cols]) + ' \\\\')
        lines.append('\\midrule')
        # Order rows by category then display using a canonical order
        cat_order = ['Bias', 'Orthography', 'Morphology', 'Syntax', 'Semantics', 'Discourse', 'Varieties', 'Other']
        def sort_key(series: pd.Series) -> pd.Series:
            if series.name == 'category':
                order_map = {c: i for i, c in enumerate(cat_order)}
                return series.map(lambda x: order_map.get(x, len(order_map)))
            return series
        df_sorted = df.sort_values(['category', 'display'], key=sort_key)
        current_cat = None
        for _, r in df_sorted.iterrows():
            cat = str(r['category'])
            disp = str(r['display'])
            cat_cell = f"\\textbf{{{cat}}}" if cat != current_cat else ''
            current_cat = cat
            # Build ΔU cells per task (ctx - std), with red/green coloring
            diff_cells = []
            diffs_for_avg = []
            for t in tasks:
                std_v = r.get(f'{t}_std')
                ctx_v = r.get(f'{t}_ctx')
                diff = (ctx_v - std_v) if (pd.notna(std_v) and pd.notna(ctx_v)) else np.nan
                diffs_for_avg.append(diff)
                diff_cells.append(cell_rg(diff))
            # Per-row average ΔU
            diffs_arr = np.array([d for d in diffs_for_avg if pd.notna(d)])
            row_avg = float(diffs_arr.mean()) if diffs_arr.size else np.nan
            diff_cells.append(cell_rg(row_avg))
            lines.append(' & '.join([cat_cell, f"\\textbf{{{disp}}}"] + diff_cells) + ' \\\\')
        # Average row across modifications (column-wise ΔU means)
        avg_cells = []
        for t in tasks:
            diffs = []
            for _, r in df_sorted.iterrows():
                std_v = r.get(f'{t}_std')
                ctx_v = r.get(f'{t}_ctx')
                diffs.append((ctx_v - std_v) if (pd.notna(std_v) and pd.notna(ctx_v)) else np.nan)
            col_mean = float(pd.Series(diffs).mean(skipna=True)) if diffs else np.nan
            avg_cells.append(cell_rg(col_mean))
        # Average of row averages
        row_avgs = []
        for _, r in df_sorted.iterrows():
            diffs = []
            for t in tasks:
                std_v = r.get(f'{t}_std')
                ctx_v = r.get(f'{t}_ctx')
                diffs.append((ctx_v - std_v) if (pd.notna(std_v) and pd.notna(ctx_v)) else np.nan)
            row_avgs.append(float(pd.Series(diffs).mean(skipna=True)))
        overall_mean = float(pd.Series(row_avgs).mean(skipna=True)) if row_avgs else np.nan
        avg_cells.append(cell_rg(overall_mean))
        lines.append(' \\midrule')
        lines.append(' & '.join(["\\textbf{Average}", "\\textbf{Average}"] + avg_cells) + ' \\\\')
        lines.append('\\bottomrule')
        lines.append('\\end{tabular}')
        lines.append('\\end{adjustbox}')
        lines.append('\\caption{GPT-5 vs GPT-5 (w. context): ΔU = U(ctx) - U(std) (flip \%) by task and modification}')
        lines.append('\\label{tab:gpt5_ctx_unrob}')
        lines.append('\\end{table}')
        out_tex.write_text('\n'.join(lines))
        print(f'Wrote LaTeX table to {out_tex}')
    except Exception as e:
        print(f'Skipped writing LaTeX table: {e}')


if __name__ == '__main__':
    main()
