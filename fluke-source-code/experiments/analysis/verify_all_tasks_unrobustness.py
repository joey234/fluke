#!/usr/bin/env python3
"""
Verify cross-task unrobustness by averaging existing per-task LaTeX tables,
and compare against the values in all_tasks_results_table_unrobustness.tex.

This script parses task-level U tables:
  - dialogue_results_table_unrobustness.tex
  - coref_results_table_unrobustness.tex
  - ner_results_table_unrobustness.tex
  - sa_results_table_unrobustness.tex
  - gsm_results_table_unrobustness.tex
  - ifeval_results_table_unrobustness.tex

It extracts U values per (category, modification, model), macro-averages
them across tasks (simple mean of the available task values), and compares
to the corresponding entries in:
  - all_tasks_results_table_unrobustness.tex

Reports mismatches > tolerance and prints a summary.
"""

from pathlib import Path
import re
import math
from utils import get_global_unrobustness_range, unrob_intensity

SCRIPT_DIR = Path(__file__).parent
_U_MIN, _U_MAX = get_global_unrobustness_range()


def parse_models_block(lines: list[str], start_idx: int) -> list[str]:
    """Given lines and index at 'Category & Modification' header, find the line with actual model names.
    Strategy: scan forward until \\midrule; select the last line before \\midrule, then extract textbf tokens.
    """
    end = start_idx
    for i in range(start_idx, min(start_idx+5, len(lines))):
        if '\\midrule' in lines[i]:
            end = i
            break
    # pick the last line before midrule
    header_line = lines[end-1] if end-1 > start_idx else lines[start_idx]
    names = re.findall(r"\\textbf\{([^}]+)\}", header_line)
    return [n.strip() for n in names if n.strip().lower() not in {'avg','plm','llm','models'}]


def parse_task_table(tex_path: Path) -> dict:
    """Return {(category, modification, model): value} parsed from a U LaTeX table."""
    if not tex_path.exists():
        return {}
    text = tex_path.read_text(encoding='utf-8')
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Find the header that lists models
    header_idx = None
    models = []
    for i, ln in enumerate(lines):
        if 'Category & Modification &' in ln:
            header_idx = i
            models = parse_models_block(lines, i)
            break
    if header_idx is None or not models:
        return {}
    # Parse rows until we hit bottomrule
    out: dict[tuple[str, str, str], float] = {}
    for ln in lines[header_idx+1:]:
        if '\\bottomrule' in ln:
            break
        # Skip midrule and Average row
        if '\\midrule' in ln:
            continue
        if ln.startswith('\\textbf{Average}'):
            continue
        # Expect: <cat> & \textbf{<mod>} & m1 & m2 ... & Avg
        parts = [p.strip() for p in ln.split('&')]
        if len(parts) < 3:
            continue
        cat = re.sub(r"^\\textbf\{|\}$", '', parts[0]).strip() if parts[0].startswith('\\textbf') else parts[0]
        # Find the \textbf{mod} part (second column)
        mod_match = re.search(r"\\textbf\{([^}]+)\}", parts[1])
        if not mod_match:
            # If for some reason the mod is not bolded, use raw
            mod = parts[1]
        else:
            mod = mod_match.group(1).strip()
        # Model value cells start at parts[2] .. parts[2+len(models)-1]
        model_cells = parts[2:2+len(models)]
        for mname, cell in zip(models, model_cells):
            # Extract numeric value from cell (ignore \cellcolor, textcolor, etc.)
            m = re.findall(r"[0-9]+(?:\.[0-9]+)?", cell)
            if not m:
                continue
            try:
                val = float(m[-1])
            except Exception:
                continue
            out[(cat, mod, mname)] = val
    return out


def macro_average_across_tasks(task_tables: list[dict]) -> dict:
    # Macro average = mean across tasks of available values
    buckets: dict[tuple[str, str, str], list[float]] = {}
    for table in task_tables:
        for key, val in table.items():
            buckets.setdefault(key, []).append(val)
    return {k: (sum(v)/len(v) if v else math.nan) for k, v in buckets.items()}


def render_latex_from_macro(macro: dict, out_path: Path) -> None:
    cols = ['BERT','GPT-2','T5','GPT-4o','Claude-3.5','Llama 3.1','GPT-5','DS R1']
    mod_order = [
        ('Bias', 'Temporal'), ('Bias', 'Geographical'), ('Bias', 'Length'),
        ('Orthographic', 'Spelling'), ('Orthographic', 'Capitalization'), ('Orthographic', 'Punctuation'),
        ('Semantic', 'Concept'), ('Semantic', 'Negation'),
        ('Discourse', 'Appraisal'),
        ('Varieties', 'Style'), ('Varieties', 'Dialect'),
        ('Syntactic', 'Conjunction'), ('Syntactic', 'Voice')
    ]
    def fmt(v):
        try:
            return '' if v is None or (isinstance(v,float) and math.isnan(v)) else f"{float(v):.1f}"
        except Exception:
            return ''
    def fu_color(v):
        if v is None or (isinstance(v,float) and math.isnan(v)):
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
        # Only include rows where we have at least one value across models
        if not any((cat,mod,model) in macro for model in cols):
            continue
        if cat not in cats:
            if cats: lines.append('\\midrule')
            cats.append(cat)
            lead=f'\\textbf{{{cat}}}'
        else:
            lead=' '
        row_vals=[]; cells=[]
        for m in cols:
            v = macro.get((cat,mod,m))
            cells.append(fu_color(v))
            row_vals.append(v if v is not None else float('nan'))
        ravg = (sum([x for x in row_vals if not (isinstance(x,float) and math.isnan(x))])/len([x for x in row_vals if not (isinstance(x,float) and math.isnan(x))])) if any(not (isinstance(x,float) and math.isnan(x)) for x in row_vals) else float('nan')
        lines.append(f"{lead} & \\textbf{{{mod}}} & " + ' & '.join(cells) + f" & {'' if isinstance(ravg,float) and math.isnan(ravg) else fu_color(ravg)} \\\\ ")
        printed_rows.append((cat,mod))
    # Column means across printed rows
    col_means=[]
    for m in cols:
        vals = [macro.get((cat,mod,m)) for (cat,mod) in printed_rows]
        vals = [float(x) for x in vals if x is not None]
        col_means.append(sum(vals)/len(vals) if vals else float('nan'))
    overall = sum([v for v in col_means if not (isinstance(v,float) and math.isnan(v))]) / (len([v for v in col_means if not (isinstance(v,float) and math.isnan(v))]) or 1)
    lines.append('\\midrule')
    avg_cells = [fu_color(v) if not (isinstance(v,float) and math.isnan(v)) else '' for v in col_means]
    overall_cell = fu_color(overall) if not (isinstance(overall,float) and math.isnan(overall)) else ''
    lines.append('\\textbf{Average} &  ' + ' & '.join(avg_cells) + f" & {overall_cell} \\\\ ")
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}}')
    lines.append('\\caption{All Tasks: Unrobustness (U, \\%) by model and modification (averaged across tasks)}\\label{tab:all_tasks_unrob_from_tasks}')
    lines.append('\\end{table}')
    out_path.write_text('\n'.join(lines).replace('\\n','\n'), encoding='utf-8')


def parse_all_tasks_table(tex_path: Path) -> dict:
    return parse_task_table(tex_path)


def main():
    task_files = [
        'dialogue_results_table_unrobustness.tex',
        'coref_results_table_unrobustness.tex',
        'ner_results_table_unrobustness.tex',
        'sa_results_table_unrobustness.tex',
        'gsm_results_table_unrobustness.tex',
        'ifeval_results_table_unrobustness.tex',
    ]
    task_tables = [parse_task_table(SCRIPT_DIR / f) for f in task_files]
    task_tables = [t for t in task_tables if t]
    if not task_tables:
        print('No per-task U LaTeX tables found to verify against.')
        return
    macro = macro_average_across_tasks(task_tables)
    # Load our all-tasks table
    all_tasks_path = SCRIPT_DIR / 'all_tasks_results_table_unrobustness.tex'
    all_table = parse_all_tasks_table(all_tasks_path)
    if not all_table:
        print('No all_tasks_results_table_unrobustness.tex found or could not parse.')
        return
    # Compare on intersection of keys
    keys = set(macro.keys()) & set(all_table.keys())
    tol = 0.05  # percentage points tolerance
    mismatches = []
    for k in sorted(keys):
        v_macro = macro[k]
        v_all = all_table[k]
        if (v_macro is None) or (v_all is None):
            continue
        if abs(v_macro - v_all) > tol:
            mismatches.append((k, v_macro, v_all))
    print(f'Compared {len(keys)} entries (intersection). Mismatches > {tol} pp: {len(mismatches)}')
    for (cat, mod, model), v_macro, v_all in mismatches[:50]:
        print(f'- {model} | {cat} · {mod}: macro={v_macro:.2f}, ours={v_all:.2f}, diff={v_all - v_macro:+.2f}')
    if not mismatches:
        print('All values match within tolerance.')
    # Always generate a table from the per-task macros for reference
    out_tex = SCRIPT_DIR / 'all_tasks_results_table_unrobustness_from_tasks.tex'
    render_latex_from_macro(macro, out_tex)
    print(f'Wrote macro-averaged table to {out_tex}')


if __name__ == '__main__':
    main()
