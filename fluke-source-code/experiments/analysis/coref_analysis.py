#!/usr/bin/env python3
"""
Clean Coreference Analysis Script
Processes LLM coreference results using accuracy metrics
"""

import json
import numpy as np
from scipy import stats
import pandas as pd
import os
import glob
from pathlib import Path
from tqdm import tqdm
import re

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).parent

# Global unrobustness (U) color scaling across tasks
def _global_u_range() -> tuple:
    """Compute global min/max for U across all tasks using saved LLM results CSVs.
    Falls back to (0.0, 100.0) if files are missing.
    """
    csvs = [
        SCRIPT_DIR / 'gsm_modification_results_llm.csv',
        SCRIPT_DIR / 'sa_modification_results_llm.csv',
        SCRIPT_DIR / 'ner_modification_results_llm.csv',
        SCRIPT_DIR / 'dialogue_modification_results_llm.csv',
        SCRIPT_DIR / 'coref_modification_results_llm.csv',
    ]
    vals = []
    for p in csvs:
        try:
            if p.exists():
                df = pd.read_csv(p)
                if 'unrobustness' in df.columns:
                    vals.extend(pd.to_numeric(df['unrobustness'], errors='coerce').dropna().tolist())
        except Exception:
            continue
    if not vals:
        return (0.0, 100.0)
    vmin = float(np.nanmin(vals)) if len(vals) else 0.0
    vmax = float(np.nanmax(vals)) if len(vals) else 100.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return (0.0, 100.0)
    return (vmin, vmax)

_U_MIN, _U_MAX = _global_u_range()
def _u_intensity(u: float) -> int:
    try:
        u = float(u)
    except Exception:
        return 0
    # Map [_U_MIN, _U_MAX] -> [0, 80] for a readable blue scale
    lo, hi = _U_MIN, _U_MAX
    if not np.isfinite(u) or not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return 0
    t = (u - lo) / (hi - lo)
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    return int(round(80 * t))

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
        return 'deepseek-r1-deepseek'
    if base.startswith('llama-'):
        return 'llama'
    parts = base.split('-')
    model = parts[0]
    if model == 'gpt':
        return 'gpt-5-standard-context-aware' if 'context' in base else 'gpt-5-standard'
    return model


def process_llm_results(results_dir):
    """Process LLM results from CSV files"""
    results_files = glob.glob(os.path.join(results_dir, '*.csv'))
    
    results_data = []
    negation_results_data = []
    
    print("Processing LLM results...")
    
    for results_file in tqdm(results_files, desc="Processing files"):
        filename = os.path.basename(results_file)
        
        # Skip certain files
        if any(skip in filename for skip in ['DP', 'coref.csv', 'compare']):
            continue
            
        # Extract model and modification from filename
        model = _map_model_from_filename(filename)
        parts = filename.split('-')
            
        modification = parts[-1].replace('_100.csv', '')
        # Normalize modification aliases
        if modification == 'singlish':
            modification = 'dialectal'
        
        # Check if corresponding comparison file exists
        compare_file = (SCRIPT_DIR / f'../../data/modified_data/coref/{modification}_100.json')
        if not compare_file.exists():
            continue
            
        # Read the CSV file
        df = pd.read_csv(results_file)
        compare_df = json.load(open(compare_file))
        
        if len(compare_df) != len(df):
            print(f'Length mismatch: {modification}, {model} - {len(compare_df)} vs {len(df)}')
            continue
            
        # Get labels and predictions
        ori_labels = df['original_label'].values
        mod_labels = df['modified_label'].values
        ori_preds = df['original_pred'].values
        mod_preds = df['modified_pred'].values

        # DeepSeek R1 coref: override preds by parsing the final digit from raw outputs
        if model == 'deepseek-r1-deepseek':
            try:
                import re as _re
                def _last_digit(val):
                    s = '' if pd.isna(val) else str(val)
                    m = _re.findall(r'(\d)', s)
                    return int(m[-1]) if m else None
                if 'original_raw_output' in df.columns:
                    op = df['original_raw_output'].map(_last_digit)
                    # Use parsed pred when available
                    ori_preds = np.where(op.notna(), op.astype(object).values, ori_preds)
                if 'raw_output' in df.columns:
                    mp = df['raw_output'].map(_last_digit)
                    mod_preds = np.where(mp.notna(), mp.astype(object).values, mod_preds)
            except Exception:
                pass

        # Normalize predictions to match allowed label set
        def build_canonical_map(classes):
            cmap = {}
            for c in classes:
                cl = str(c).strip()
                key = re.sub(r"[^a-z0-9]+", " ", cl.lower()).strip()
                cmap[key] = cl
            # common coref aliases
            aliases = {
                'coref': ['coref','coreferent','anaphora','yes','y','same'],
                'not coref': ['not coref','non coref','no coref','different','no','n','not']
            }
            lower_to_canon = {re.sub(r"[^a-z0-9]+"," ", k.lower()).strip(): v for k,v in cmap.items()}
            for canon, arr in aliases.items():
                if canon in lower_to_canon:
                    for a in arr:
                        lower_to_canon[a] = lower_to_canon[canon]
            return lower_to_canon

        def all_numeric(arr):
            try:
                _ = pd.to_numeric(arr)
                return True
            except Exception:
                return False
        allowed = pd.unique(pd.concat([pd.Series(ori_labels).astype(str), pd.Series(mod_labels).astype(str)], ignore_index=True))
        cmap = build_canonical_map(allowed)

        def extract(pred: str):
            s = str(pred)
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    for k in ['label','answer','prediction','result']:
                        v = obj.get(k)
                        if v is not None:
                            s = str(v)
                            break
            except Exception:
                pass
            sl = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
            if sl in cmap:
                return cmap[sl]
            for key, canon in cmap.items():
                if key and re.search(rf"\b{re.escape(key)}\b", sl):
                    return canon
            return s.strip()

        if not (all_numeric(ori_preds) and all_numeric(mod_preds)):
            ori_preds = np.array([extract(str(x)) for x in ori_preds], dtype=object)
            mod_preds = np.array([extract(str(x)) for x in mod_preds], dtype=object)
        
        # Calculate accuracy scores (not F1 like NER)
        ori_correct = sum(ori_preds == ori_labels)
        mod_correct = sum(mod_preds == mod_labels)
        total = len(df)
        
        ori_acc = ori_correct / total * 100 if total > 0 else 0
        mod_acc = mod_correct / total * 100 if total > 0 else 0
        
        # Calculate weighted delta and absolute change
        weighted_delta = (mod_acc - ori_acc) * np.log10(ori_acc) / np.log10(100) if ori_acc > 0 else 0
        absolute_change = abs(mod_acc - ori_acc)  # Absolute change in accuracy
        rate_of_change = absolute_change  # Keep for backward compatibility
        
        # Convert to binary accuracy arrays for statistical tests
        ori_binary = (ori_preds == ori_labels).astype(int)
        mod_binary = (mod_preds == mod_labels).astype(int)
        # Unrobustness: discordant rate
        flips = np.abs(ori_binary - mod_binary)
        unrobustness = flips.mean() * 100.0
        
        # Statistical tests for directional change (paired)
        try:
            _, p_value_w = safe_wilcoxon(ori_binary, mod_binary)
        except Exception:
            p_value_w = 1.0
        n01 = int(((ori_binary == 1) & (mod_binary == 0)).sum())
        n10 = int(((ori_binary == 0) & (mod_binary == 1)).sum())
        n_discordant = n01 + n10
        if n_discordant > 0:
            try:
                p_value_mcn = stats.binomtest(n10, n_discordant, 0.5, alternative='two-sided').pvalue
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
            'abs_significance': abs_significance
        })
        
        # Process negation subtypes if applicable
        if modification == 'negation' and 'type' in df.columns:
            for subtype in ['verbal', 'lexical', 'double', 'approximate', 'absolute']:
                subtype_df = df[df['type'] == subtype]
                if len(subtype_df) == 0:
                    continue
                    
                # Calculate accuracy scores for subtype
                subtype_ori_correct = sum(subtype_df['original_pred'] == subtype_df['original_label'])
                subtype_mod_correct = sum(subtype_df['modified_pred'] == subtype_df['modified_label'])
                subtype_total = len(subtype_df)
                
                subtype_ori_acc = subtype_ori_correct / subtype_total * 100 if subtype_total > 0 else 0
                subtype_mod_acc = subtype_mod_correct / subtype_total * 100 if subtype_total > 0 else 0
                
                # Calculate weighted delta and absolute change
                subtype_weighted_delta = (subtype_mod_acc - subtype_ori_acc) * np.log10(subtype_ori_acc) / np.log10(100) if subtype_ori_acc > 0 else 0
                subtype_absolute_change = abs(subtype_mod_acc - subtype_ori_acc)  # Absolute change
                subtype_rate_of_change = subtype_absolute_change  # Keep for backward compatibility
                
                # Convert to binary arrays
                subtype_ori_binary = (subtype_df['original_pred'] == subtype_df['original_label']).astype(int)
                subtype_mod_binary = (subtype_df['modified_pred'] == subtype_df['modified_label']).astype(int)
                # Unrobustness for subtype
                s_flips = np.abs(subtype_ori_binary - subtype_mod_binary)
                s_unrob = s_flips.mean() * 100.0
                
                # Statistical tests for directional change (paired)
                try:
                    _, p_w = safe_wilcoxon(subtype_ori_binary, subtype_mod_binary)
                except Exception:
                    p_w = 1.0
                s_n01 = int(((subtype_ori_binary == 1) & (subtype_mod_binary == 0)).sum())
                s_n10 = int(((subtype_ori_binary == 0) & (subtype_mod_binary == 1)).sum())
                s_disc = s_n01 + s_n10
                if s_disc > 0:
                    try:
                        p_mcn = stats.binomtest(s_n10, s_disc, 0.5, alternative='two-sided').pvalue
                    except Exception:
                        p_mcn = 1.0
                else:
                    p_mcn = 1.0
                p_val = min(p_w, p_mcn)

                # Absolute change significance not computed
                abs_p_val = np.nan
                
                # Determine significance
                if p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < 0.05:
                    sig = '*'
                elif p_val < 0.1:
                    sig = '.'
                else:
                    sig = 'ns'
                
                abs_sig = 'n/a'
                
                negation_results_data.append({
                    'model': model,
                    'negation_type': subtype,
                    'original_acc': subtype_ori_acc,
                    'modified_acc': subtype_mod_acc,
                    'weighted_delta': subtype_weighted_delta,
                    'absolute_change': subtype_absolute_change,
                    'rate_of_change': subtype_rate_of_change,  # Keep for backward compatibility
                    'unrobustness': s_unrob,
                    'sample_size': subtype_total,
                    'p_value': p_val,
                    'significance': sig,
                    'abs_p_value': abs_p_val,
                    'abs_significance': abs_sig
                })
    
    return results_data, negation_results_data


def process_plm_results():
    """Process PLM results (bert, gpt2, t5) for coref task."""
    base_dir = (SCRIPT_DIR / '../PLM/coreference_resolution')
    model_dirs = {
        'bert': base_dir / 'coref_bert',
        'gpt2': base_dir / 'coref_gpt2',
        't5': base_dir / 'coref_t5'
    }
    results_data = []

    for model, mdir in model_dirs.items():
        if not mdir.exists():
            continue
        files = list(mdir.glob('*_predictions.csv'))
        mods_present = set(f.stem.replace('_predictions', '') for f in files)
        for csv_file in files:
            modification = csv_file.stem.replace('_predictions', '')
            # Prefer singlish over dialectal: if both exist, skip dialectal
            if modification == 'dialectal' and 'singlish' in mods_present:
                continue
            if modification == 'singlish':
                modification = 'dialectal'

            try:
                df = pd.read_csv(csv_file)
            except Exception:
                continue

            # Expect columns: original_label, modified_label, original_pred, modified_pred
            if not set(['original_label','modified_label','original_pred','modified_pred']).issubset(df.columns):
                continue

            ori_labels = df['original_label'].values
            ori_preds = df['original_pred'].values
            mod_labels = df['modified_label'].values
            mod_preds = df['modified_pred'].values

            ori_correct = sum(ori_preds == ori_labels)
            mod_correct = sum(mod_preds == mod_labels)
            total = len(df)

            if total == 0:
                continue

            ori_acc = ori_correct / total * 100
            mod_acc = mod_correct / total * 100
            weighted_delta = (mod_acc - ori_acc) * np.log10(ori_acc) / np.log10(100) if ori_acc > 0 else 0
            absolute_change = abs(mod_acc - ori_acc)
            rate_of_change = absolute_change

            ori_binary = (ori_preds == ori_labels).astype(int)
            mod_binary = (mod_preds == mod_labels).astype(int)
            flips = np.abs(ori_binary - mod_binary)
            unrobustness = flips.mean() * 100.0

            # Significance: Wilcoxon + McNemar
            try:
                _, p_w = safe_wilcoxon(ori_binary, mod_binary)
            except Exception:
                p_w = 1.0
            n01 = int(((ori_binary == 1) & (mod_binary == 0)).sum())  # orig better
            n10 = int(((ori_binary == 0) & (mod_binary == 1)).sum())  # mod better
            disc = n01 + n10
            if disc > 0:
                try:
                    k = min(n01, n10)
                    p_mcn = stats.binomtest(k, disc, 0.5, alternative='two-sided').pvalue
                except Exception:
                    p_mcn = 1.0
            else:
                p_mcn = 1.0
            p_value = min(p_w, p_mcn)

            if p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'
            elif p_value < 0.1:
                significance = '.'
            else:
                significance = 'ns'

            results_data.append({
                'model': model,
                'modification': modification,
                'original_acc': ori_acc,
                'modified_acc': mod_acc,
                'weighted_delta': weighted_delta,
                'absolute_change': absolute_change,
                'rate_of_change': rate_of_change,
                'unrobustness': unrobustness,
                'p_value': p_value,
                'significance': significance
            })

    return results_data


def save_llm_results(llm_results, llm_negation):
    """Save LLM results to CSV files"""
    
    # Save main results
    results_df = pd.DataFrame(llm_results)
    results_df.to_csv('coref_modification_results_llm.csv', index=False)
    
    # Save negation results
    negation_df = pd.DataFrame(llm_negation)
    negation_df.to_csv('coref_negation_type_results_llm.csv', index=False)
    
    print("LLM results saved:")
    print("- coref_modification_results_llm.csv")
    print("- coref_negation_type_results_llm.csv")
    
    return results_df, negation_df


def generate_latex_table(df, output_file):
    """Generate LaTeX table from results DataFrame"""
    # Guard against empty or malformed DataFrame
    if df is None or df.empty or 'model' not in df.columns or 'modification' not in df.columns:
        print(f"Skipping LaTeX generation for {output_file}: no data or missing columns")
        return
    # Work on a copy to avoid mutating caller's DataFrame (used later for compact views)
    df = df.copy()
    
    # Define mappings
    mod_mapping = {
        'temporal_bias': ('Bias', 'Temporal'),
        'geographical_bias': ('Bias', 'Geographical'), 
        'length_bias': ('Bias', 'Length'),
        'typo_bias': ('Orthography', 'Spelling'),
        'capitalization': ('Orthography', 'Capitalization'),
        'punctuation': ('Orthography', 'Punctuation'),
        'derivation': ('Morphology', 'Derivation'),
        'compound_word': ('Morphology', 'Compound'),
        'active_to_passive': ('Syntax', 'Voice'),
        'grammatical_role': ('Syntax', 'Grammar'),
        'coordinating_conjunction': ('Syntax', 'Conjunction'),
        'concept_replacement': ('Semantics', 'Concept'),
        'negation': ('Semantics', 'Negation'),
        'discourse': ('Discourse', 'Disc. markers'),
        'sentiment': ('Discourse', 'Appraisal'),
        'casual': ('Varieties', 'Style'),
        'dialectal': ('Varieties', 'Dialect'),
    }
    
    model_order = ['BERT', 'GPT-2', 'T5', 'GPT-4o', 'Claude-3.5', 'Llama 3.1', 'GPT-5','DS R1', 'GPT-5 (w. context)']
    model_map = {
        'bert': 'BERT',
        'gpt2': 'GPT-2',
        't5': 'T5',
        'gpt4o': 'GPT-4o', 
        'claude': 'Claude-3.5', 
        'llama': 'Llama 3.1', 
        'gpt-5-standard': 'GPT-5', 
        'deepseek-r1-deepseek': 'DS R1',
        'gpt-5-standard-context-aware': 'GPT-5 (w. context)'
    }
    
    df['model'] = df['model'].replace(model_map)
    df['category'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    df['modification'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    
    # Create pivot tables
    pivot_df = df.pivot_table(index=['category', 'modification'], columns='model', values='weighted_delta', aggfunc='mean')
    significance = df.pivot_table(index=['category', 'modification'], columns='model', values='significance', aggfunc='first')
    unrob_df = df.pivot_table(index=['category', 'modification'], columns='model', values='unrobustness', aggfunc='mean')

    # Split models into PLM and LLM for grouped headers
    plm_models_all = ['BERT', 'GPT-2', 'T5']
    plm_models = [m for m in model_order if m in plm_models_all and m in pivot_df.columns]
    llm_models = [m for m in model_order if (m not in plm_models_all) and (m in pivot_df.columns) and m != 'GPT-5 (w. context)']
    ordered_models = plm_models + llm_models

    # Write separate Δ and U tables (do not combine)
    def _fmt_delta(val, sig):
        if pd.isna(val):
            return ''
        try:
            v = float(val)
        except Exception:
            return ''
        intensity = int(min(abs(v)/10.0, 1.0) * 20)
        color = 'green' if v > 0 else 'red'
        s = f"{v:+.1f}"
        if sig == '.':
            s = f"\\textbf{{{s}}}"
        elif isinstance(sig, str) and sig in {'*','**','***'}:
            s = f"\\textbf{{{s}}}{sig}"
        return f"\\cellcolor{{{color}!{intensity}}} {s}"
    def _fmt_u(val):
        if pd.isna(val):
            return ''
        v = float(val)
        inten = _u_intensity(v)
        txt = f"{v:.1f}"
        if inten >= 45:
            txt = f"\\textcolor{{white}}{{{txt}}}"
        return f"\\cellcolor{{blue!{inten}}} {txt}"

    def _write_table(pivot, fmt_fn, caption, fname):
        total = len(ordered_models)
        add_avg = (fmt_fn is _fmt_u)
        latex = '\\begin{table}[h]\\n\\centering\\n\\resizebox{\\linewidth}{!}{\\n\\begin{tabular}{ll' + 'r'*total + ('r' if add_avg else '') + '}\\n'
        latex += '\\toprule\\n'
        if len(plm_models) and len(llm_models):
            latex += 'Category & Modification & ' + f'\\multicolumn{{{len(plm_models)}}}{{c}}{{\\textbf{{PLM}}}} & ' + f'\\multicolumn{{{len(llm_models)}}}{{c}}{{\\textbf{{LLM}}}}' + (' & \\textbf{Avg}' if add_avg else '') + ' \\\\\\n'
        else:
            latex += 'Category & Modification & ' + f'\\multicolumn{{{total}}}{{c}}{{\\textbf{{Models}}}}' + (' & \\textbf{Avg}' if add_avg else '') + ' \\\\\\n'
        latex += ' &  & ' + ' & '.join([f'\\textbf{{{m}}}' for m in ordered_models]) + ('' if not add_avg else ' & \\textbf{Avg}') + ' \\\\\\n'
        latex += '\\midrule\\n'
        cats = []
        for key,(cat,mod) in mod_mapping.items():
            if cat not in cats:
                if cats: latex += '\\midrule\\n'
                cats.append(cat)
                lead = f'\\textbf{{{cat}}}'
            else:
                lead = ' '
            vals = []; row_vals = []
            for m in ordered_models:
                if (cat,mod) in pivot.index and m in pivot.columns:
                    if fmt_fn is _fmt_delta:
                        sig = significance.loc[(cat,mod), m] if ((cat,mod) in significance.index and m in significance.columns) else ''
                        vals.append(fmt_fn(pivot.loc[(cat,mod), m], sig))
                    else:
                        v = pivot.loc[(cat,mod), m]
                        row_vals.append(v)
                        vals.append(fmt_fn(v))
                else:
                    vals.append('')
            if add_avg:
                ravg = float(pd.to_numeric(pd.Series(row_vals), errors='coerce').mean()) if row_vals else float('nan')
                avg_cell = '' if np.isnan(ravg) else _fmt_u(ravg)
                latex += f"{lead} & \\textbf{{{mod}}} & " + ' & '.join(vals) + f" & {avg_cell} \\\\\n"
            else:
                latex += f"{lead} & \\textbf{{{mod}}} & " + ' & '.join(vals) + ' \\\\\n'
        # Average row if U table (within same table)
        if add_avg and isinstance(pivot, pd.DataFrame):
            col_means=[]
            for m in ordered_models:
                if m in pivot.columns:
                    col_means.append(float(pd.to_numeric(pivot[m], errors='coerce').mean()))
                else:
                    col_means.append(float('nan'))
            overall = float(np.nanmean(col_means)) if col_means else float('nan')
            latex += '\\midrule\\n'
            avg_cells = ['' if np.isnan(v) else _fmt_u(v) for v in col_means]
            overall_cell = '' if np.isnan(overall) else _fmt_u(overall)
            latex += '\\textbf{Average} &  ' + ' & '.join(avg_cells) + f" & {overall_cell} \\\\\n"
        latex += '\\bottomrule\\n\\end{tabular}}\\n'
        latex += caption + '\\n\\end{table}'
        latex = latex.replace('\\n', '\n')
        with open(fname,'w') as f: f.write(latex)

    _write_table(pivot_df, _fmt_delta, '\\caption{Coreference: Weighted $\\Delta$ by model and modification}\\label{tab:coref_delta}', 'coref_results_table_delta.tex')
    _write_table(unrob_df, _fmt_u, '\\caption{Coreference: Unrobustness (U, \%) by model and modification}\\label{tab:coref_unrob}', 'coref_results_table_unrobustness.tex')
    try:
        rob_df = 100.0 - unrob_df
        _write_table(rob_df, _fmt_r, '\\caption{Coreference: Robustness (R, \\%) by model and modification}\\label{tab:coref_rob}', 'coref_results_table_robustness.tex')
    except Exception:
        pass
    print("LaTeX tables saved: coref_results_table_delta.tex, coref_results_table_unrobustness.tex, coref_results_table_robustness.tex")
    return
    
    def get_color(val, sig):
        """Generate LaTeX color formatting based on value and significance"""
        if np.isnan(val):
            return ''
        elif val > 0:
            intensity = min(abs(val)/10, 1)
            val_str = f'+{val:.1f}'
        else:
            intensity = min(abs(val)/10, 1)
            val_str = f'{val:.1f}'
        
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
        inten = _u_intensity(float(u))
        return f'\\cellcolor{{blue!{inten}}} {u:.1f}'
    
    # Generate LaTeX table
    total_models = len(ordered_models)
    latex_table = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll' + 'rr'*total_models + '}\n'
    latex_table += '\\toprule\n'
    # Group header
    if len(plm_models) and len(llm_models):
        latex_table += 'Category & Modification & ' + f'\\multicolumn{{{2*len(plm_models)}}}{{c}}{{\\textbf{{PLM}}}} & ' + f'\\multicolumn{{{2*len(llm_models)}}}{{c}}{{\\textbf{{LLM}}}} \\\\\n'
    else:
        latex_table += 'Category & Modification & ' + f'\\multicolumn{{{2*total_models}}}{{c}}{{\\textbf{{Models}}}} \\\\\n'
    # Model header
    model_spans = [f'\\multicolumn{{2}}{{c}}{{\\textbf{{{col}}}}}' for col in ordered_models]
    latex_table += ' &  & ' + ' & '.join(model_spans) + ' \\\\\n'
    # Subheader for Delta and Unrobustness
    sub_headers = []
    for _ in ordered_models:
        sub_headers += ['\\textbf{$\\Delta$}', '\\textbf{U}']
    latex_table += ' &  & ' + ' & '.join(sub_headers) + ' \\\\\n'
    latex_table += '\\midrule\n'
    
    categories_seen = []
    for mod, (category, modification) in mod_mapping.items():
        if category not in categories_seen:
            if categories_seen:
                latex_table += '\\midrule\\n'
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
    latex_table += '\\caption{Weighted Delta ($\\Delta$) and Unrobustness (U) by Model and Modification Type (Coreference)}\n'
    latex_table += '\\label{tab:coref_results}\n\\end{table}'
    
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
        'derivation': ('Morphological', 'Derivation'),
        'compound_word': ('Morphological', 'Compound'),
        'active_to_passive': ('Syntactic', 'Voice'),
        'grammatical_role': ('Syntactic', 'Grammar'),
        'coordinating_conjunction': ('Syntactic', 'Conjunction'),
        'concept_replacement': ('Semantic', 'Concept'),
        'negation': ('Semantic', 'Negation'),
        'discourse': ('Pragmatic', 'Discourse'),
        'sentiment': ('Pragmatic', 'Sentiment'),
        'casual': ('Genre', 'Casual'),
        'dialectal': ('Genre', 'Dialectal'),
    }

    model_order = ['BERT', 'GPT-2', 'T5', 'GPT-4o', 'Claude-3.5-Sonnet', 'Llama 3.1 405B', 'GPT-5','DeepSeek R1', 'GPT-5 (w. context)']
    model_map = {
        'bert': 'BERT',
        'gpt2': 'GPT-2',
        't5': 'T5',
        'gpt4o': 'GPT-4o', 
        'claude': 'Claude-3.5-Sonnet', 
        'llama': 'Llama 3.1 405B', 
        'gpt-5-standard': 'GPT-5', 
        'deepseek-r1-deepseek': 'DeepSeek R1',
        'gpt-5-standard-context-aware': 'GPT-5 (w. context)'
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
        'derivation': ('Morphological', 'Derivation'),
        'compound_word': ('Morphological', 'Compound'),
        'active_to_passive': ('Syntactic', 'Voice'),
        'grammatical_role': ('Syntactic', 'Grammar'),
        'coordinating_conjunction': ('Syntactic', 'Conjunction'),
        'concept_replacement': ('Semantic', 'Concept'),
        'negation': ('Semantic', 'Negation'),
        'discourse': ('Pragmatic', 'Discourse'),
        'sentiment': ('Pragmatic', 'Sentiment'),
        'casual': ('Genre', 'Casual'),
        'dialectal': ('Genre', 'Dialectal'),
    }
    model_order = ['BERT', 'GPT-2', 'T5', 'GPT-4o', 'Claude-3.5-Sonnet', 'Llama 3.1 405B', 'GPT-5','DeepSeek R1', 'GPT-5 (w. context)']
    model_map = {
        'bert': 'BERT', 'gpt2': 'GPT-2', 't5': 'T5',
        'gpt4o': 'GPT-4o','claude': 'Claude-3.5-Sonnet','llama': 'Llama 3.1 405B',
        'gpt-5-standard': 'GPT-5','deepseek-r1-deepseek': 'DeepSeek R1',
        'gpt-5-standard-context-aware': 'GPT-5 (w. context)'
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
        'derivation': 'Morphological','compound_word': 'Morphological',
        'active_to_passive': 'Syntactic','grammatical_role': 'Syntactic','coordinating_conjunction': 'Syntactic',
        'concept_replacement': 'Semantic','negation': 'Semantic',
        'discourse': 'Pragmatic','sentiment': 'Pragmatic',
        'casual': 'Genre','dialectal': 'Genre'
    }
    model_order = ['BERT', 'GPT-2', 'T5', 'GPT-4o', 'Claude-3.5-Sonnet', 'Llama 3.1 405B', 'GPT-5','DeepSeek R1', 'GPT-5 (w. context)']
    model_map = {'bert':'BERT','gpt2':'GPT-2','t5':'T5','gpt4o':'GPT-4o','claude':'Claude-3.5-Sonnet','llama':'Llama 3.1 405B','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DeepSeek R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    df = df.copy()
    df['model'] = df['model'].replace(model_map)
    df['category'] = df['modification'].map(lambda x: mod_to_cat.get(x, 'Other'))
    agg = df.groupby(['category','model']).agg(weighted_delta=('weighted_delta','mean')).reset_index()
    overall = df.groupby(['model']).agg(weighted_delta=('weighted_delta','mean')).reset_index(); overall.insert(0,'category','Overall')
    agg = pd.concat([agg, overall], ignore_index=True)
    pivot = agg.pivot_table(index='category', columns='model', values='weighted_delta', aggfunc='mean')
    cols = [m for m in model_order if m in pivot.columns and m != 'GPT-5 (w. context)']
    rows = [*sorted([r for r in pivot.index if r!='Overall']), 'Overall'] if 'Overall' in pivot.index else sorted(list(pivot.index))
    if not cols or not rows:
        print("Skipping compact plot: insufficient data after aggregation")
        return
    pivot = pivot.reindex(index=rows, columns=cols)
    data = pivot.values.astype(float)
    data_masked = ma.masked_invalid(data)
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
        'derivation': 'Morphological','compound_word': 'Morphological',
        'active_to_passive': 'Syntactic','grammatical_role': 'Syntactic','coordinating_conjunction': 'Syntactic',
        'concept_replacement': 'Semantic','negation': 'Semantic',
        'discourse': 'Pragmatic','sentiment': 'Pragmatic',
        'casual': 'Genre','dialectal': 'Genre'
    }
    model_order = ['BERT','GPT-2','T5','GPT-4o','Claude-3.5','Llama 3.1','GPT-5','DS R1','GPT-5 (w. context)']
    model_map = {'bert':'BERT','gpt2':'GPT-2','t5':'T5','gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
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
    latex += '\\caption{Compact category-level averages (Coreference): $\\Delta$ | U per model}\n'
    latex += '\\label{tab:coref_compact}\n\\end{table}'
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
        'derivation': 'Morphological','compound_word': 'Morphological',
        'active_to_passive': 'Syntactic','grammatical_role': 'Syntactic','coordinating_conjunction': 'Syntactic',
        'concept_replacement': 'Semantic','negation': 'Semantic',
        'discourse': 'Pragmatic','sentiment': 'Pragmatic',
        'casual': 'Genre','dialectal': 'Genre'
    }
    model_order = ['BERT','GPT-2','T5','GPT-4o','Claude-3.5-Sonnet','Llama 3.1 405B','GPT-5','DeepSeek R1','GPT-5 (w. context)']
    model_map = {'bert':'BERT','gpt2':'GPT-2','t5':'T5','gpt4o':'GPT-4o','claude':'Claude-3.5-Sonnet','llama':'Llama 3.1 405B','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DeepSeek R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
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
    im = ax.imshow(data_masked, cmap='Blues', vmin=0, vmax=100, aspect='auto')
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
    """Generate a single Coref LaTeX table with one cell per model: "Δ | U".
    Excludes GPT-5 (w. context) from main columns.
    """
    need = {'model','modification','weighted_delta','unrobustness'}
    if df is None or df.empty or not need.issubset(df.columns):
        print(f"Skipping combined LaTeX for {output_file}: missing data/columns"); return
    df = df.copy()
    mod_mapping = {
        'temporal_bias': ('Bias', 'Temporal'),'geographical_bias': ('Bias', 'Geographical'),'length_bias': ('Bias', 'Length'),
        'typo_bias': ('Orthography', 'Spelling'),'capitalization': ('Orthography', 'Capitalization'),'punctuation': ('Orthography', 'Punctuation'),
        'derivation': ('Morphology', 'Derivation'),'compound_word': ('Morphology', 'Compound'),
        'active_to_passive': ('Syntax', 'Voice'),'grammatical_role': ('Syntax', 'Grammar'),'coordinating_conjunction': ('Syntax', 'Conjunction'),
        'concept_replacement': ('Semantics', 'Concept'),'negation': ('Semantics', 'Negation'),
        'discourse': ('Discourse', 'Disc. markers'),'sentiment': ('Discourse', 'Appraisal'),
        'casual': ('Varieties', 'Style'),'dialectal': ('Varieties', 'Dialect'),
    }
    model_order = ['BERT','GPT-2','T5','GPT-4o','Claude-3.5','Llama 3.1','GPT-5','DS R1','GPT-5 (w. context)']
    model_map = {'bert':'BERT','gpt2':'GPT-2','t5':'T5','gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    df['model'] = df['model'].replace(model_map)
    df['category'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    df['modification'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    p_delta = df.pivot_table(index=['category','modification'], columns='model', values='weighted_delta', aggfunc='mean')
    p_u = df.pivot_table(index=['category','modification'], columns='model', values='unrobustness', aggfunc='mean')
    plm = [m for m in model_order if m in ['BERT','GPT-2','T5'] and m in p_delta.columns]
    llm = [m for m in model_order if m not in ['BERT','GPT-2','T5'] and m in p_delta.columns and m != 'GPT-5 (w. context)']
    cols = plm + llm
    if not cols:
        print(f"Skipping combined LaTeX for {output_file}: no columns after filtering"); return
    tot = len(cols)
    latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll'+'r'*tot+'}\n'
    latex += '\\toprule\n'
    if plm and llm:
        latex += 'Category & Modification & ' + f'\\multicolumn{{{len(plm)}}}{{c}}{{\\textbf{{PLM}}}} & ' + f'\\multicolumn{{{len(llm)}}}{{c}}{{\\textbf{{LLM}}}} \\\\\n'
    else:
        latex += 'Category & Modification & ' + f'\\multicolumn{{{tot}}}{{c}}{{\\textbf{{Models}}}} \\\\\n'
    latex += ' &  & ' + ' & '.join([f'\\textbf{{{c}}}' for c in cols]) + ' \\\\\n'
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
    latex += '\\caption{Coreference: Weighted $\\Delta$ | U by model and modification}\\label{tab:coref_combined}\n\\end{table}'
    with open(output_file,'w') as f: f.write(latex)
    print(f"Combined LaTeX table saved to {output_file}")

def generate_latex_table_dual(df, output_file):
    """Generate a single table with left block = Δ, right block = U (Coref)."""
    need = {'model','modification','weighted_delta','unrobustness','significance'}
    if df is None or df.empty or not need.issubset(df.columns):
        print(f"Skipping dual LaTeX for {output_file}: missing data/columns"); return
    df = df.copy()
    mod_mapping = {
        'temporal_bias': ('Bias', 'Temporal'),'geographical_bias': ('Bias', 'Geographical'),'length_bias': ('Bias', 'Length'),
        'typo_bias': ('Orthography', 'Spelling'),'capitalization': ('Orthography', 'Capitalization'),'punctuation': ('Orthography', 'Punctuation'),
        'derivation': ('Morphology', 'Derivation'),'compound_word': ('Morphology', 'Compound'),
        'active_to_passive': ('Syntax', 'Voice'),'grammatical_role': ('Syntax', 'Grammar'),'coordinating_conjunction': ('Syntax', 'Conjunction'),
        'concept_replacement': ('Semantics', 'Concept'),'negation': ('Semantics', 'Negation'),
        'discourse': ('Discourse', 'Disc. markers'),'sentiment': ('Discourse', 'Appraisal'),
        'casual': ('Varieties', 'Style'),'dialectal': ('Varieties', 'Dialect'),
    }
    model_order = ['BERT','GPT-2','T5','GPT-4o','Claude-3.5','Llama 3.1','GPT-5','DS R1','GPT-5 (w. context)']
    model_map = {'bert':'BERT','gpt2':'GPT-2','t5':'T5','gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    df['model'] = df['model'].replace(model_map)
    df['category'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    df['modification'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    p_delta = df.pivot_table(index=['category','modification'], columns='model', values='weighted_delta', aggfunc='mean')
    p_sig = df.pivot_table(index=['category','modification'], columns='model', values='significance', aggfunc='first')
    p_u = df.pivot_table(index=['category','modification'], columns='model', values='unrobustness', aggfunc='mean')
    plm = [m for m in model_order if m in ['BERT','GPT-2','T5'] and m in p_delta.columns]
    llm = [m for m in model_order if m not in ['BERT','GPT-2','T5'] and m in p_delta.columns and m != 'GPT-5 (w. context)']
    cols = plm + llm
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
    latex += '\\caption{Coreference: Left = $\\Delta$, Right = U (one block per metric)}\\label{tab:coref_dual}\n\\end{table}'
    with open(output_file,'w') as f: f.write(latex)
    print(f"Dual LaTeX table saved to {output_file}")

def generate_latex_table_combined_cells(df, output_file):
    """Combined table with two cells per model (Δ then U), colored (Coref)."""
    need = {'model','modification','weighted_delta','unrobustness','significance'}
    if df is None or df.empty or not need.issubset(df.columns):
        print(f"Skipping combined-cells LaTeX for {output_file}: missing data/columns"); return
    df = df.copy()
    mod_mapping = {
        'temporal_bias': ('Bias', 'Temporal'),'geographical_bias': ('Bias', 'Geographical'),'length_bias': ('Bias', 'Length'),
        'typo_bias': ('Orthography', 'Spelling'),'capitalization': ('Orthography', 'Capitalization'),'punctuation': ('Orthography', 'Punctuation'),
        'derivation': ('Morphology', 'Derivation'),'compound_word': ('Morphology', 'Compound'),
        'active_to_passive': ('Syntax', 'Voice'),'grammatical_role': ('Syntax', 'Grammar'),'coordinating_conjunction': ('Syntax', 'Conjunction'),
        'concept_replacement': ('Semantics', 'Concept'),'negation': ('Semantics', 'Negation'),
        'discourse': ('Discourse', 'Disc. markers'),'sentiment': ('Discourse', 'Appraisal'),
        'casual': ('Varieties', 'Style'),'dialectal': ('Varieties', 'Dialect'),
    }
    model_order = ['BERT','GPT-2','T5','GPT-4o','Claude-3.5','Llama 3.1','GPT-5','DS R1','GPT-5 (w. context)']
    model_map = {'bert':'BERT','gpt2':'GPT-2','t5':'T5','gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    df['model'] = df['model'].replace(model_map)
    df['category'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    df['modification'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    p_delta = df.pivot_table(index=['category','modification'], columns='model', values='weighted_delta', aggfunc='mean')
    p_sig = df.pivot_table(index=['category','modification'], columns='model', values='significance', aggfunc='first')
    p_u = df.pivot_table(index=['category','modification'], columns='model', values='unrobustness', aggfunc='mean')
    plm = [m for m in model_order if m in ['BERT','GPT-2','T5'] and m in p_delta.columns]
    llm = [m for m in model_order if m not in ['BERT','GPT-2','T5'] and m in p_delta.columns and m != 'GPT-5 (w. context)']
    cols = plm + llm
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
    latex = '\\begin{table}[h]\n\\centering\\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll'+'rr'*tot+'}\n'
    latex += '\\toprule\n'
    if plm and llm:
        latex += 'Category & Modification & ' + f'\\multicolumn{{{2*len(plm)}}}{{c}}{{\\textbf{{PLM}}}} & ' + f'\\multicolumn{{{2*len(llm)}}}{{c}}{{\\textbf{{LLM}}}} \\\\\n'
    else:
        latex += 'Category & Modification & ' + f'\\multicolumn{{{2*tot}}}{{c}}{{\\textbf{{Models}}}} \\\\\n'
    latex += ' &  & ' + ' & '.join([f'\\multicolumn{{2}}{{c}}{{\\textbf{{{c}}}}}' for c in cols]) + ' \\\\\n'
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
    latex += '\\caption{Coreference: Two cells per model — $\\Delta$ and U}\\label{tab:coref_combined_cells}\n\\end{table}'
    with open(output_file,'w') as f: f.write(latex)
    print(f"Combined-cells LaTeX table saved to {output_file}")

def generate_compact_latex_table_combined_cells(df, output_file):
    """Compact category-level combined-cells table for Coref."""
    if df is None or df.empty or 'model' not in df.columns or 'modification' not in df.columns:
        print(f"Skipping compact combined-cells: missing data/columns"); return
    mod_to_cat = {
        'temporal_bias': 'Bias','geographical_bias': 'Bias','length_bias': 'Bias',
        'typo_bias': 'Orthography','capitalization': 'Orthography','punctuation': 'Orthography',
        'derivation': 'Morphology','compound_word': 'Morphology',
        'active_to_passive': 'Syntax','grammatical_role': 'Syntax','coordinating_conjunction': 'Syntax',
        'concept_replacement': 'Semantics','negation': 'Semantics',
        'discourse': 'Discourse','sentiment': 'Discourse',
        'casual': 'Varieties','dialectal': 'Varieties'
    }
    model_order = ['BERT','GPT-2','T5','GPT-4o','Claude-3.5','Llama 3.1','GPT-5','DS R1','GPT-5 (w. context)']
    model_map = {'bert':'BERT','gpt2':'GPT-2','t5':'T5','gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    dfx = df.copy(); dfx['model']=dfx['model'].replace(model_map); dfx['category']=dfx['modification'].map(lambda x: mod_to_cat.get(x,'Other'))
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
        inten=int(min(max(val,0.0),100.0)*0.8)
        return f"\\cellcolor{{blue!{inten}}} {val:.1f}"
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
    latex+='\\caption{Coreference (compact): Two cells per model — $\\Delta$ and U}\\label{tab:coref_compact_combined_cells}\n\\end{table}'
    with open(output_file,'w') as f: f.write(latex)
    print(f"Compact combined-cells LaTeX table saved to {output_file}")


def main():
    """Main function to run the analysis"""
    # Configuration
    llm_results_dir = str((SCRIPT_DIR / '../LLM/results/coref').resolve())
    
    # Process LLM results
    llm_results, llm_negation = process_llm_results(llm_results_dir)
    # Process PLM results
    plm_results = process_plm_results()
    
    # Save LLM results
    # Combine for LaTeX (negation subtype remains LLM-only)
    combined_df = pd.DataFrame(llm_results + plm_results)
    results_df, negation_df = save_llm_results(llm_results, llm_negation)
    
    # Generate plots and LaTeX table
    plot_path = str((SCRIPT_DIR / 'coref_results_heatmap.png').resolve())
    u_plot_path = str((SCRIPT_DIR / 'coref_unrobustness_heatmap.png').resolve())
    generate_summary_plot(combined_df, plot_path)
    generate_unrobustness_plot(combined_df, u_plot_path)
    generate_latex_table(combined_df, 'coref_results_table.tex')
    # Compact outputs
    compact_plot = str((SCRIPT_DIR / 'coref_results_heatmap_compact.png').resolve())
    generate_compact_summary_plot(combined_df, compact_plot)
    generate_compact_latex_table(combined_df, 'coref_results_table_compact.tex')
    generate_compact_latex_table_combined_cells(combined_df, 'coref_results_table_compact_combined_cells.tex')
    # Full combined table (Δ | U per model)
    try:
        generate_latex_table_combined(combined_df, 'coref_results_table.tex')
    except Exception as e:
        print(f"Combined table skipped (Coref): {e}")
    try:
        generate_gpt5_context_table(combined_df, 'coref_gpt5_context_table.tex')
    except Exception as e:
        print(f"GPT-5 context table skipped (Coref): {e}")
    # Dual combined table (left Δ, right U)
    try:
        generate_latex_table_dual(combined_df, 'coref_results_table_dual.tex')
    except Exception as e:
        print(f"Dual table skipped (Coref): {e}")
    # Combined per-model two-cell table
    try:
        generate_latex_table_combined_cells(combined_df, 'coref_results_table_combined_cells.tex')
    except Exception as e:
        print(f"Combined-cells table skipped (Coref): {e}")

def generate_gpt5_context_table(df, output_file):
    if df is None or df.empty:
        return
    model_map = {'bert':'BERT','gpt2':'GPT-2','t5':'T5','gpt4o':'GPT-4o','claude':'Claude-3.5-Sonnet','llama':'Llama 3.1 405B','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DeepSeek R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    df = df.copy(); df['model'] = df['model'].replace(model_map)
    df = df[df['model'].isin(['GPT-5','GPT-5 (w. context)'])]
    if df.empty:
        raise ValueError('No GPT-5 context data')
    mod_to_cat = {
        'temporal_bias': 'Bias','geographical_bias': 'Bias','length_bias': 'Bias',
        'typo_bias': 'Orthographic','capitalization': 'Orthographic','punctuation': 'Orthographic',
        'derivation': 'Morphological','compound_word': 'Morphological',
        'active_to_passive': 'Syntactic','grammatical_role': 'Syntactic','coordinating_conjunction': 'Syntactic',
        'concept_replacement': 'Semantic','negation': 'Semantic','discourse': 'Pragmatic','sentiment': 'Pragmatic','casual': 'Genre','dialectal': 'Genre'
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
    latex += '\\caption{Coreference: GPT-5 vs GPT-5 (w. context) — category-level $\\Delta$ and U}\\label{tab:coref_gpt5_context}\n\\end{table}'
    with open(output_file,'w') as f: f.write(latex)
    u_compact_plot = str((SCRIPT_DIR / 'coref_unrobustness_heatmap_compact.png').resolve())
    generate_compact_unrobustness_plot(combined_df, u_compact_plot)
    
    print("\\nAnalysis complete!")


if __name__ == "__main__":
    main()
