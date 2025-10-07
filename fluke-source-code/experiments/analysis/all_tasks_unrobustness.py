#!/usr/bin/env python3
"""
Generate a cross-task Unrobustness (U) LaTeX table averaged across all tasks.

Reads all_models_comparison_{ner,dialogue,sa,coref,gsm,ifeval}.csv from
fluke-source-code/experiments/LLM/scripts and aggregates flip% (U) by
category, modification, and model across tasks. Outputs a LaTeX table
all_tasks_results_table_unrobustness.tex beside this script.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from utils import get_global_unrobustness_range, unrob_intensity

SCRIPT_DIR = Path(__file__).parent
_U_MIN, _U_MAX = get_global_unrobustness_range()


def load_all_comparisons() -> pd.DataFrame:
    base = (SCRIPT_DIR / '../LLM/scripts').resolve()
    tasks = ['ner', 'dialogue', 'sa', 'coref', 'gsm', 'ifeval']
    frames = []
    for t in tasks:
        fp = base / f'all_models_comparison_{t}.csv'
        if not fp.exists():
            continue
        try:
            df = pd.read_csv(fp)
            df['task'] = t
            frames.append(df)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Normalize booleans
    def _to_bool(v):
        s = str(v).strip().lower()
        return True if s in {'1', 'true', 't', 'yes'} else False
    df = df.copy()
    if 'original_correct' in df.columns:
        df['original_correct'] = df['original_correct'].map(_to_bool)
    if 'modified_correct' in df.columns:
        df['modified_correct'] = df['modified_correct'].map(_to_bool)
    df['flip'] = (df['original_correct'] ^ df['modified_correct']).astype(int)
    # Normalize model names
    model_map = {
        # PLM
        'bert': 'BERT', 'gpt2': 'GPT-2', 't5': 'T5',
        # LLM
        'gpt4o': 'GPT-4o', 'gpt-4o': 'GPT-4o',
        'claude': 'Claude-3.5', 'claude-3-5-sonnet': 'Claude-3.5',
        'llama': 'Llama 3.1', 'llama-3': 'Llama 3.1',
        'gpt-5-standard': 'GPT-5', 'gpt-5': 'GPT-5',
        'deepseek-r1': 'DS R1', 'deepseek-r1-deepseek': 'DS R1',
        'gpt-5-standard-context-aware': 'GPT-5 (w. context)'
    }
    if 'model' in df.columns:
        df['model'] = df['model'].replace(model_map)
    # Normalize mods -> categories
    def _norm_mod(m):
        s = str(m or '')
        if s.endswith('_100'):
            s = s[:-4]
        if s.startswith('negation'):
            s = 'negation'
        return s
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
    df['mod_key'] = df['modification'].map(_norm_mod)
    df['category'] = df['mod_key'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    df['modification_disp'] = df['mod_key'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    return df


def build_unrobustness_table(df: pd.DataFrame) -> str:
    # Compute per-task unrobustness first, then macro-average across tasks
    task_agg = (
        df.groupby(['task', 'category', 'modification_disp', 'model'])['flip']
          .mean()
          .reset_index(name='U_task')
    )
    macro = (
        task_agg.groupby(['category', 'modification_disp', 'model'])['U_task']
                .mean()
                .reset_index(name='unrobustness')
    )
    # Convert to %
    macro['unrobustness'] = macro['unrobustness'] * 100.0
    pivot = macro.pivot_table(index=['category', 'modification_disp'], columns='model', values='unrobustness', aggfunc='mean')
    cols = [c for c in ['BERT','GPT-2','T5','GPT-4o', 'Claude-3.5', 'Llama 3.1', 'GPT-5', 'DS R1'] if c in pivot.columns]
    if not cols:
        return ''
    def fu_color(v):
        if pd.isna(v):
            return ''
        try:
            val = float(v)
        except Exception:
            return ''
        inten = unrob_intensity(val, _U_MIN, _U_MAX)
        txt = f"{val:.1f}"
        if inten >= 45:
            txt = f"\\textcolor{{white}}{{{txt}}}"
        return f"\\cellcolor{{blue!{inten}}} {txt}"
    # Emit LaTeX
    lines = []
    lines.append('\\begin{table}[h]')
    lines.append('\\centering')
    lines.append('\\resizebox{\\linewidth}{!}{')
    lines.append('\\begin{tabular}{ll' + 'r'*len(cols) + 'r}')
    lines.append('\\toprule')
    lines.append('Category & Modification & ' + ' & '.join([f'\\textbf{{{c}}}' for c in cols]) + ' & \\textbf{Avg} \\\\')
    lines.append('\\midrule')
    # Use same mapping order
    mod_order = [
        ('Bias', 'Temporal'), ('Bias', 'Geographical'), ('Bias', 'Length'),
        ('Orthographic', 'Spelling'), ('Orthographic', 'Capitalization'), ('Orthographic', 'Punctuation'),
        ('Semantic', 'Concept'), ('Semantic', 'Negation'),
        ('Discourse', 'Appraisal'),
        ('Varieties', 'Style'), ('Varieties', 'Dialect'),
        ('Syntactic', 'Conjunction'), ('Syntactic', 'Voice')
    ]
    cats = []
    printed_rows = []  # keep track of exactly which (cat,mod) we rendered
    for cat, mod in mod_order:
        idx = (cat, mod)
        if idx not in pivot.index:
            continue
        if cat not in cats:
            if cats:
                lines.append('\\midrule')
            cats.append(cat)
            lead = f'\\textbf{{{cat}}}'
        else:
            lead = ' '
        row_vals = []
        cells = []
        for c in cols:
            v = pivot.loc[idx, c] if (idx in pivot.index and c in pivot.columns) else float('nan')
            row_vals.append(v)
            cells.append(fu_color(v))
        ravg = float(pd.to_numeric(pd.Series(row_vals), errors='coerce').mean()) if row_vals else float('nan')
        lines.append(f"{lead} & \\textbf{{{mod}}} & " + ' & '.join(cells) + f" & {'' if np.isnan(ravg) else fu_color(ravg)} \\\\ ")
        printed_rows.append(idx)
    # Column means and overall (only across printed rows)
    col_means = []
    for c in cols:
        if printed_rows and c in pivot.columns:
            sub = pivot.loc[printed_rows, c]
        else:
            sub = pivot[c] if c in pivot.columns else pd.Series(dtype=float)
        col_means.append(float(pd.to_numeric(sub, errors='coerce').mean()))
    overall = float(np.nanmean(col_means)) if col_means else float('nan')
    lines.append('\\midrule')
    avg_cells = ['' if np.isnan(v) else fu_color(v) for v in col_means]
    overall_cell = '' if np.isnan(overall) else fu_color(overall)
    lines.append('\\textbf{Average} &  ' + ' & '.join(avg_cells) + f" & {overall_cell} \\\\ ")
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}}')
    lines.append('\\caption{All Tasks: Unrobustness (U, \\%) by model and modification (averaged across tasks)}\\label{tab:all_tasks_unrob}')
    lines.append('\\end{table}')
    return '\n'.join(lines).replace('\\n', '\n')


# Fallback/simple path: build from existing per-task LaTeX U tables
def parse_models_block(lines: list[str], start_idx: int) -> list[str]:
    end = start_idx
    for i in range(start_idx, min(start_idx+5, len(lines))):
        if '\\midrule' in lines[i]:
            end = i
            break
    header_line = lines[end-1] if end-1 > start_idx else lines[start_idx]
    names = re.findall(r"\\textbf\{([^}]+)\}", header_line)
    return [n.strip() for n in names if n.strip().lower() not in {'avg','plm','llm','models'}]


def parse_task_table(tex_path: Path) -> dict:
    if not tex_path.exists():
        return {}
    text = tex_path.read_text(encoding='utf-8')
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header_idx = None
    models: list[str] = []
    for i, ln in enumerate(lines):
        if 'Category & Modification &' in ln:
            header_idx = i
            models = parse_models_block(lines, i)
            break
    if header_idx is None or not models:
        return {}
    out: dict[tuple[str, str, str], float] = {}
    last_cat = ''
    for ln in lines[header_idx+1:]:
        if '\\bottomrule' in ln:
            break
        if '\\midrule' in ln:
            continue
        if ln.startswith('\\textbf{Average}'):
            continue
        parts = [p.strip() for p in ln.split('&')]
        if len(parts) < 3:
            continue
        cat_raw = parts[0]
        if cat_raw and not cat_raw.startswith('\\textbf'):
            last_cat = cat_raw
        elif cat_raw.startswith('\\textbf'):
            mcat = re.search(r"\\textbf\{([^}]+)\}", cat_raw)
            last_cat = (mcat.group(1).strip() if mcat else cat_raw)
        cat = last_cat
        mmod = re.search(r"\\textbf\{([^}]+)\}", parts[1])
        mod = mmod.group(1).strip() if mmod else parts[1]
        model_cells = parts[2:2+len(models)]
        for mname, cell in zip(models, model_cells):
            m = re.findall(r"[0-9]+(?:\.[0-9]+)?", cell)
            if not m:
                continue
            try:
                val = float(m[-1])
            except Exception:
                continue
            out[(cat, mod, mname)] = val
    return out


def build_from_task_tex() -> str | None:
    task_files = [
        'dialogue_results_table_unrobustness.tex',
        'coref_results_table_unrobustness.tex',
        'ner_results_table_unrobustness.tex',
        'sa_results_table_unrobustness.tex',
        'gsm_results_table_unrobustness.tex',
        'ifeval_results_table_unrobustness.tex',
    ]
    per_task = [parse_task_table(SCRIPT_DIR / f) for f in task_files]
    per_task = [t for t in per_task if t]
    if not per_task:
        return None
    # Macro average across tasks for each (cat, mod, model)
    buckets: dict[tuple[str, str, str], list[float]] = {}
    for table in per_task:
        for k, v in table.items():
            buckets.setdefault(k, []).append(v)
    macro: dict[tuple[str, str, str], float] = {k: (sum(v)/len(v)) for k, v in buckets.items() if v}
    # Build a pivot-like dict structure for rendering
    cols = ['BERT','GPT-2','T5','GPT-4o','Claude-3.5','Llama 3.1','GPT-5','DS R1']
    mod_order = [
        ('Bias', 'Temporal'), ('Bias', 'Geographical'), ('Bias', 'Length'),
        ('Orthographic', 'Spelling'), ('Orthographic', 'Capitalization'), ('Orthographic', 'Punctuation'),
        ('Semantic', 'Concept'), ('Semantic', 'Negation'),
        ('Discourse', 'Appraisal'),
        ('Varieties', 'Style'), ('Varieties', 'Dialect'),
        ('Syntactic', 'Conjunction'), ('Syntactic', 'Voice')
    ]
    # Render LaTeX
    def fu_color(v):
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return ''
        val = float(v)
        inten = unrob_intensity(val, _U_MIN, _U_MAX)
        txt = f"{val:.1f}"
        if inten >= 45:
            txt = f"\\textcolor{{white}}{{{txt}}}"
        return f"\\cellcolor{{blue!{inten}}} {txt}"
    lines = []
    lines.append('\\begin{table}[h]')
    lines.append('\\centering')
    lines.append('\\resizebox{\\linewidth}{!}{')
    lines.append('\\begin{tabular}{ll' + 'r'*len(cols) + 'r}')
    lines.append('\\toprule')
    lines.append('Category & Modification & ' + ' & '.join([f'\\textbf{{{c}}}' for c in cols]) + ' & \\textbf{Avg} \\\\')
    lines.append('\\midrule')
    cats=[]; printed_rows=[]
    for cat, mod in mod_order:
        if not any((cat,mod,model) in macro for model in cols):
            continue
        if cat not in cats:
            if cats: lines.append('\\midrule')
            cats.append(cat)
            lead = f'\\textbf{{{cat}}}'
        else:
            lead = ' '
        row_vals=[]; cells=[]
        for m in cols:
            v = macro.get((cat,mod,m))
            row_vals.append(v if v is not None else float('nan'))
            cells.append(fu_color(v) if v is not None else '')
        ravg = float(np.nanmean([float(x) for x in row_vals if x is not None])) if row_vals else float('nan')
        lines.append(f"{lead} & \\textbf{{{mod}}} & " + ' & '.join(cells) + f" & {'' if np.isnan(ravg) else fu_color(ravg)} \\\\ ")
        printed_rows.append((cat,mod))
    # Averages per model across printed rows
    col_means=[]
    for m in cols:
        vals = [macro.get((cat,mod,m)) for (cat,mod) in printed_rows]
        vals = [float(x) for x in vals if x is not None]
        col_means.append(float(np.nanmean(vals)) if vals else float('nan'))
    overall = float(np.nanmean([v for v in col_means if not np.isnan(v)])) if col_means else float('nan')
    lines.append('\\midrule')
    avg_cells = ['' if np.isnan(v) else fu_color(v) for v in col_means]
    overall_cell = '' if np.isnan(overall) else fu_color(overall)
    lines.append('\\textbf{Average} &  ' + ' & '.join(avg_cells) + f" & {overall_cell} \\\\ ")
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}}')
    lines.append('\\caption{All Tasks: Unrobustness (U, \\%) by model and modification (averaged across tasks)}\\label{tab:all_tasks_unrob}')
    lines.append('\\end{table}')
    return '\n'.join(lines).replace('\\n', '\n')


def main():
    # Prefer building from existing per-task LaTeX tables for exact parity
    tex = build_from_task_tex()
    if not tex:
        # Fallback to CSV-based macro aggregation
        df = load_all_comparisons()
        if df.empty:
            print('No all_models_comparison_*.csv found; cannot generate cross-task table.')
            return
        df = normalize(df)
        tex = build_unrobustness_table(df)
    if not tex:
        print('No data available after pivot; table not generated.')
        return
    out = SCRIPT_DIR / 'all_tasks_results_table_unrobustness.tex'
    out.write_text(tex, encoding='utf-8')
    print(f'Cross-task U table saved to {out}')


if __name__ == '__main__':
    main()
