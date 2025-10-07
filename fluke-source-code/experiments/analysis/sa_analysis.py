#!/usr/bin/env python3
"""
Clean Sentiment Analysis Script
Processes LLM sentiment analysis results using accuracy metrics
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
from utils import get_global_unrobustness_range, unrob_intensity

# Resolve paths relative to this script instead of CWD
SCRIPT_DIR = Path(__file__).parent
_U_MIN, _U_MAX = get_global_unrobustness_range()


def safe_wilcoxon(x, y, **kwargs):
    """A robust wrapper around scipy.stats.wilcoxon that avoids RuntimeWarnings
    and ValueErrors in degenerate cases.

    - Filters NaNs on either side
    - Returns (0.0, 1.0) when arrays are empty, mismatched, or all differences are zero
    - Uses zero_method='wilcox' by default
    """
    import numpy as _np
    from scipy.stats import wilcoxon as _wilcoxon

    xa = _np.asarray(x)
    ya = _np.asarray(y)

    # Align and drop NaNs
    mask = _np.isfinite(xa) & _np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]

    # Must be non-empty and same length
    if xa.size == 0 or ya.size == 0 or xa.size != ya.size:
        return 0.0, 1.0

    diff = xa - ya
    # Degenerate case: all differences are zero
    if _np.all(diff == 0):
        return 0.0, 1.0

    try:
        return _wilcoxon(xa, ya, zero_method=kwargs.get('zero_method', 'wilcox'),
                         alternative=kwargs.get('alternative', 'two-sided'))
    except Exception:
        return 0.0, 1.0


def _map_model_from_filename(filename: str) -> str:
    """Map raw filename prefix to our internal model key.
    Handles GPT-5 vs GPT-5-Context explicitly.
    """
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
    # Fallback to legacy split-based mapping
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
        if any(skip in filename for skip in ['DP', 'sst2.csv', 'compare']):
            continue
            
        # Extract model and modification from filename
        model = _map_model_from_filename(filename)
            
        # Handle both standard and legacy suffixes
        parts = filename.split('-')
        last = parts[-1]
        if last.endswith('_100_new.csv'):
            modification = last.replace('_100_new.csv', '')
        else:
            modification = last.replace('_100.csv', '')
        # Normalize aliases
        if modification == 'singlish':
            modification = 'dialectal'
        
        # Check if corresponding comparison file exists
        compare_file = (SCRIPT_DIR / f'../../data/modified_data/sa/{modification}_100.json')
        if not compare_file.exists():
            continue
            
        # Read the CSV file
        df = pd.read_csv(results_file)
        compare_df = json.load(open(compare_file))
        
        if len(compare_df) != len(df):
            print(f'Length mismatch: {modification}, {model} - {len(compare_df)} vs {len(df)}')
            continue
            
        # Get labels and predictions (prefer direct numeric preds if present)
        ori_labels = df['original_label'].values
        mod_labels = df['modified_label'].values
        ori_preds = df['original_pred'].values
        mod_preds = df['modified_pred'].values

        # If predictions are already numeric (0/1 or ints as strings), coerce and skip text normalization
        def all_numeric(arr):
            try:
                _ = pd.to_numeric(arr)
                return True
            except Exception:
                return False
        if not (all_numeric(ori_preds) and all_numeric(mod_preds)):
            # Fallback: normalize predictions to allowed labels when preds are textual
            ori_labels_str = pd.Series(ori_labels).astype(str)
            mod_labels_str = pd.Series(mod_labels).astype(str)
            allowed = pd.unique(pd.concat([ori_labels_str, mod_labels_str], ignore_index=True))

            def build_canonical_map(classes):
                cmap = {}
                for c in classes:
                    cl = str(c).strip()
                    key = re.sub(r"[^a-z0-9]+", " ", cl.lower()).strip()
                    cmap[key] = cl
                aliases = {
                    'positive': ['pos','positive','positif','favorable','good','great','glad'],
                    'negative': ['neg','negative','negatif','unfavorable','bad','poor','sad'],
                    'neutral': ['neu','neutral','neutre']
                }
                lower = {re.sub(r"[^a-z0-9]+"," ", k.lower()).strip(): v for k,v in cmap.items()}
                for canon, arr in aliases.items():
                    if canon in lower:
                        for a in arr:
                            lower[a] = lower[canon]
                return lower

            cmap = build_canonical_map(allowed)
            def extract(pred: str):
                s = str(pred)
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict):
                        for k in ['label','sentiment','answer','prediction']:
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

            ori_preds = np.array([extract(x) for x in pd.Series(ori_preds).astype(str)], dtype=object)
            mod_preds = np.array([extract(x) for x in pd.Series(mod_preds).astype(str)], dtype=object)
        
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
        # Robust Wilcoxon: guard NaNs and zero-variance differences
        try:
            _, p_value_w = safe_wilcoxon(ori_binary, mod_binary)
        except Exception:
            p_value_w = 1.0
        n01 = int(((ori_binary == 1) & (mod_binary == 0)).sum())  # orig better
        n10 = int(((ori_binary == 0) & (mod_binary == 1)).sum())  # mod better
        n_disc = n01 + n10
        if n_disc > 0:
            try:
                k = min(n01, n10)
                p_value_mcn = stats.binomtest(k, n_disc, 0.5, alternative='two-sided').pvalue
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
                s_n01 = int(((subtype_ori_binary == 1) & (subtype_mod_binary == 0)).sum())  # orig better
                s_n10 = int(((subtype_ori_binary == 0) & (subtype_mod_binary == 1)).sum())  # mod better
                s_disc = s_n01 + s_n10
                if s_disc > 0:
                    try:
                        k = min(s_n01, s_n10)
                        p_mcn = stats.binomtest(k, s_disc, 0.5, alternative='two-sided').pvalue
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
    """Process PLM results (bert, gpt2, t5) for sentiment analysis.

    Loads all "*_comparison.csv" per model and returns one record per
    modification, matching the LLM pipeline. Prefers tmp/singlish_comparison.csv
    to represent the dialectal variant if present, and skips the dialectal
    file in sa_plm_compare when tmp/singlish exists to avoid double counting.
    """
    base_dir = (SCRIPT_DIR / '../PLM/sentiment_analysis/sa_plm_compare')
    tmp_base = (SCRIPT_DIR / '../PLM/sentiment_analysis/tmp')
    model_dirs = {
        'bert': base_dir / 'bert',
        'gpt2': base_dir / 'gpt2',
        't5': base_dir / 't5',
    }

    results_data = []

    for model, mdir in model_dirs.items():
        if not mdir.exists():
            continue

        # Track whether dialectal was loaded from tmp (singlish)
        dialectal_done = False
        singlish_file = (tmp_base / model / 'singlish_comparison.csv')
        if singlish_file.exists():
            try:
                df = pd.read_csv(singlish_file)
                modification = 'dialectal'
                dialectal_done = True
            except Exception:
                df = None
            else:
                if df is not None and set(['original_label','modified_label','original_pred','modified_pred']).issubset(df.columns):
                    # Compute metrics and append
                    ori_labels = df['original_label'].values
                    ori_preds = df['original_pred'].values
                    mod_labels = df['modified_label'].values
                    mod_preds = df['modified_pred'].values
                    total = len(df)
                    if total > 0:
                        ori_acc = (ori_preds == ori_labels).sum() / total * 100
                        mod_acc = (mod_preds == mod_labels).sum() / total * 100
                        weighted_delta = (mod_acc - ori_acc) * np.log10(ori_acc) / np.log10(100) if ori_acc > 0 else 0
                        absolute_change = abs(mod_acc - ori_acc)
                        rate_of_change = absolute_change
                        ori_binary = (ori_preds == ori_labels).astype(int)
                        mod_binary = (mod_preds == mod_labels).astype(int)
                        flips = np.abs(ori_binary - mod_binary)
                        unrobustness = flips.mean() * 100.0
                        try:
                            _, p_w = safe_wilcoxon(ori_binary, mod_binary)
                        except Exception:
                            p_w = 1.0
                        n01 = int(((ori_binary == 1) & (mod_binary == 0)).sum())
                        n10 = int(((ori_binary == 0) & (mod_binary == 1)).sum())
                        disc = n01 + n10
                        if disc > 0:
                            try:
                                p_mcn = stats.binomtest(n10, disc, 0.5, alternative='two-sided').pvalue
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
                            'significance': significance,
                        })

        # Iterate all comparison files; skip dialectal if already done via tmp
        for csv_file in sorted(mdir.glob('*_comparison.csv')):
            mod_raw = csv_file.stem.replace('_comparison', '')
            if mod_raw == 'dialectal' and dialectal_done:
                continue
            try:
                df = pd.read_csv(csv_file)
            except Exception:
                continue
            modification = 'dialectal' if mod_raw == 'dialectal' else mod_raw
            if not set(['original_label','modified_label','original_pred','modified_pred']).issubset(df.columns):
                continue
            total = len(df)
            if total == 0:
                continue
            ori_labels = df['original_label'].values
            ori_preds = df['original_pred'].values
            mod_labels = df['modified_label'].values
            mod_preds = df['modified_pred'].values
            ori_acc = (ori_preds == ori_labels).sum() / total * 100
            mod_acc = (mod_preds == mod_labels).sum() / total * 100
            weighted_delta = (mod_acc - ori_acc) * np.log10(ori_acc) / np.log10(100) if ori_acc > 0 else 0
            absolute_change = abs(mod_acc - ori_acc)
            rate_of_change = absolute_change
            ori_binary = (ori_preds == ori_labels).astype(int)
            mod_binary = (mod_preds == mod_labels).astype(int)
            flips = np.abs(ori_binary - mod_binary)
            unrobustness = flips.mean() * 100.0
            try:
                _, p_w = safe_wilcoxon(ori_binary, mod_binary)
            except Exception:
                p_w = 1.0
            n01 = int(((ori_binary == 1) & (mod_binary == 0)).sum())
            n10 = int(((ori_binary == 0) & (mod_binary == 1)).sum())
            disc = n01 + n10
            if disc > 0:
                try:
                    p_mcn = stats.binomtest(n10, disc, 0.5, alternative='two-sided').pvalue
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
                'significance': significance,
            })

    return results_data


def save_llm_results(llm_results, llm_negation):
    """Save LLM results to CSV files"""
    
    # Save main results
    results_df = pd.DataFrame(llm_results)
    results_df.to_csv('sa_modification_results_llm.csv', index=False)
    
    # Save negation results
    negation_df = pd.DataFrame(llm_negation)
    negation_df.to_csv('sa_negation_type_results_llm.csv', index=False)
    
    print("LLM results saved:")
    print("- sa_modification_results_llm.csv")
    print("- sa_negation_type_results_llm.csv")
    
    return results_df, negation_df


def generate_latex_table(df, output_file):
    """Generate LaTeX table from results DataFrame"""
    # Guard against empty or malformed DataFrame
    if df is None or df.empty or 'model' not in df.columns or 'modification' not in df.columns:
        print(f"Skipping LaTeX generation for {output_file}: no data or missing columns")
        return
    # Avoid mutating the caller's DataFrame used after this
    df = df.copy()
    
    # Define mappings
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

    # Split models into PLM and LLM
    plm_models_all = ['BERT', 'GPT-2', 'T5']
    plm_models = [m for m in model_order if m in plm_models_all and m in pivot_df.columns]
    llm_models = [m for m in model_order if (m not in plm_models_all) and (m in pivot_df.columns) and m != 'GPT-5 (w. context)']
    ordered_models = plm_models + llm_models
    
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
        inten = unrob_intensity(float(u), _U_MIN, _U_MAX)
        return f'\\cellcolor{{blue!{inten}}} {u:.1f}'
    
    # Generate LaTeX table
    total_models = len(ordered_models)
    latex_table = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll' + 'rr'*total_models + '}\n'
    latex_table += '\\toprule\n'
    if len(plm_models) and len(llm_models):
        latex_table += 'Category & Modification & ' + f'\\multicolumn{{{2*len(plm_models)}}}{{c}}{{\\textbf{{PLM}}}} & ' + f'\\multicolumn{{{2*len(llm_models)}}}{{c}}{{\\textbf{{LLM}}}} \\\\\n'
    else:
        latex_table += 'Category & Modification & ' + f'\\multicolumn{{{2*total_models}}}{{c}}{{\\textbf{{Models}}}} \\\\\n'
    model_spans = [f'\\multicolumn{{2}}{{c}}{{\\textbf{{{col}}}}}' for col in ordered_models]
    latex_table += ' &  & ' + ' & '.join(model_spans) + ' \\\\\n'
    sub_headers = []
    for _ in ordered_models:
        sub_headers += ['\\textbf{$\\Delta$}', '\\textbf{U}']
    latex_table += ' &  & ' + ' & '.join(sub_headers) + ' \\\\\n'
    latex_table += '\\midrule\n'
    
    categories_seen = []
    for mod, (category, modification) in mod_mapping.items():
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
    latex_table += '\\caption{Weighted Delta ($\\Delta$) and Unrobustness (U) by Model and Modification Type (Sentiment Analysis)}\n'
    latex_table += '\\label{tab:sa_results}\n\\end{table}'
    
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to {output_file}")


def generate_latex_table_delta(df, output_file):
    if df is None or df.empty or 'model' not in df.columns or 'modification' not in df.columns:
        print(f"Skipping Δ LaTeX generation for {output_file}: no data or missing columns"); return
    df = df.copy()
    mod_mapping = {
        'temporal_bias': ('Bias', 'Temporal'),'geographical_bias': ('Bias', 'Geographical'),'length_bias': ('Bias', 'Length'),
        'typo_bias': ('Orthographic', 'Spelling'),'capitalization': ('Orthographic', 'Capitalization'),'punctuation': ('Orthographic', 'Punctuation'),
        'derivation': ('Morphological', 'Derivation'),'compound_word': ('Morphological', 'Compound'),
        'active_to_passive': ('Syntactic', 'Voice'),'grammatical_role': ('Syntactic', 'Grammar'),'coordinating_conjunction': ('Syntactic', 'Conjunction'),
        'concept_replacement': ('Semantic', 'Concept'),'negation': ('Semantic', 'Negation'),
        'discourse': ('Pragmatic', 'Discourse'),'sentiment': ('Pragmatic', 'Sentiment'),'casual': ('Genre', 'Casual'),'dialectal': ('Genre', 'Dialectal'),
    }
    model_order = ['BERT','GPT-2','T5','GPT-4o','Claude-3.5','Llama 3.1','GPT-5','DS R1','GPT-5 (w. context)']
    model_map = {'bert':'BERT','gpt2':'GPT-2','t5':'T5','gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    df['model'] = df['model'].replace(model_map)
    df['category'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    df['modification'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    p_delta = df.pivot_table(index=['category','modification'], columns='model', values='weighted_delta', aggfunc='mean')
    p_sig = df.pivot_table(index=['category','modification'], columns='model', values='significance', aggfunc='first')
    plm = [m for m in model_order if m in ['BERT','GPT-2','T5'] and m in p_delta.columns]
    llm = [m for m in model_order if m not in ['BERT','GPT-2','T5'] and m in p_delta.columns and m != 'GPT-5 (w. context)']
    cols = plm + llm
    def fd_color(v, sig):
        if pd.isna(v):
            return ''
        try:
            val = float(v)
        except Exception:
            return ''
        intensity = int(min(abs(val)/10.0, 1.0) * 20)
        color = 'green' if val > 0 else 'red'
        s = f"{val:+.1f}"
        if sig == '.':
            s = f"\\textbf{{{s}}}"
        elif sig in {'*','**','***'}:
            s = f"\\textbf{{{s}}}{sig}"
        return f"\\cellcolor{{{color}!{intensity}}} {s}"
    tot = len(cols)
    latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll'+'r'*tot+'r}\n'
    latex += '\\toprule\n'
    if plm and llm:
        latex += 'Category & Modification & ' + f'\\multicolumn{{{len(plm)}}}{{c}}{{\\textbf{{PLM}}}} & ' + f'\\multicolumn{{{len(llm)}}}{{c}}{{\\textbf{{LLM}}}} \\\\\n'
    else:
        latex += 'Category & Modification & ' + f'\\multicolumn{{{tot}}}{{c}}{{\\textbf{{Models}}}} \\\\\n'
    latex += ' &  & ' + ' & '.join([f'\\textbf{{{c}}}' for c in cols]) + ' & \\textbf{Avg} \\\\\n'
    latex += '\\midrule\n'
    cats = []
    for key,(cat,mod) in mod_mapping.items():
        if cat not in cats:
            if cats: latex += '\\midrule\n'
            cats.append(cat)
            row0 = f'\\textbf{{{cat}}}'
        else:
            row0 = ' '
        vals=[]; row_vals=[]
        for c in cols:
            if (cat,mod) in p_delta.index and c in p_delta.columns:
                sig = p_sig.loc[(cat,mod), c] if ((cat,mod) in p_sig.index and c in p_sig.columns) else ''
                v = p_delta.loc[(cat,mod), c]
                row_vals.append(v)
                vals.append(fd_color(v, sig if isinstance(sig,str) else ''))
            else:
                vals.append('')
        ravg = float(np.nanmean(row_vals)) if row_vals else float('nan')
        latex += f"{row0} & \\textbf{{{mod}}} & " + ' & '.join(vals) + f" & {'' if np.isnan(ravg) else f'{ravg:+.1f}'} \\\\\n"
    # Average row per model
    col_means=[]
    for c in cols:
        series = p_delta[c] if c in p_delta.columns else pd.Series(dtype=float)
        col_means.append(float(series.mean()))
    overall = float(np.nanmean(col_means)) if col_means else float('nan')
    latex += '\\midrule\n'
    latex += '\\textbf{Average} &  ' + ' & '.join([('' if np.isnan(v) else f"{v:+.1f}") for v in col_means]) + f" & {'' if np.isnan(overall) else f'{overall:+.1f}'} \\\\\n"
    latex += '\\bottomrule\n\\end{tabular}}\n'
    latex += '\\caption{Sentiment Analysis: Weighted $\\Delta$ by model and modification}\\label{tab:sa_delta}\n\\end{table}'
    latex = latex.replace('\\n', '\n')
    with open(output_file,'w') as f: f.write(latex)
    print(f"Δ LaTeX table saved to {output_file}")

def generate_latex_table_unrob(df, output_file):
    if df is None or df.empty or 'model' not in df.columns or 'modification' not in df.columns:
        print(f"Skipping U LaTeX generation for {output_file}: no data or missing columns"); return
    df = df.copy()
    mod_mapping = {
        'temporal_bias': ('Bias', 'Temporal'),'geographical_bias': ('Bias', 'Geographical'),'length_bias': ('Bias', 'Length'),
        'typo_bias': ('Orthographic', 'Spelling'),'capitalization': ('Orthographic', 'Capitalization'),'punctuation': ('Orthographic', 'Punctuation'),
        'derivation': ('Morphological', 'Derivation'),'compound_word': ('Morphological', 'Compound'),
        'active_to_passive': ('Syntactic', 'Voice'),'grammatical_role': ('Syntactic', 'Grammar'),'coordinating_conjunction': ('Syntactic', 'Conjunction'),
        'concept_replacement': ('Semantic', 'Concept'),'negation': ('Semantic', 'Negation'),
        'discourse': ('Pragmatic', 'Discourse'),'sentiment': ('Pragmatic', 'Sentiment'),'casual': ('Genre', 'Casual'),'dialectal': ('Genre', 'Dialectal'),
    }
    model_order = ['BERT','GPT-2','T5','GPT-4o','Claude-3.5','Llama 3.1','GPT-5','DS R1','GPT-5 (w. context)']
    model_map = {'bert':'BERT','gpt2':'GPT-2','t5':'T5','gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    df['model'] = df['model'].replace(model_map)
    df['category'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    df['modification'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    p_u = df.pivot_table(index=['category','modification'], columns='model', values='unrobustness', aggfunc='mean')
    plm = [m for m in model_order if m in ['BERT','GPT-2','T5'] and m in p_u.columns]
    llm = [m for m in model_order if m not in ['BERT','GPT-2','T5'] and m in p_u.columns and m != 'GPT-5 (w. context)']
    cols = plm + llm
    def fu(v):
        if pd.isna(v): return ''
        try:
            fv = float(v)
        except Exception:
            return ''
        inten = unrob_intensity(fv, _U_MIN, _U_MAX)
        txt = f"{fv:.1f}"
        if inten >= 45:
            txt = f"\\textcolor{{white}}{{{txt}}}"
        return f"\\cellcolor{{blue!{inten}}} {txt}"
    tot = len(cols)
    latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll'+'r'*tot+'r}\n'
    latex += '\\toprule\n'
    if plm and llm:
        latex += 'Category & Modification & ' + f'\\multicolumn{{{len(plm)}}}{{c}}{{\\textbf{{PLM}}}} & ' + f'\\multicolumn{{{len(llm)}}}{{c}}{{\\textbf{{LLM}}}} \\\\\n'
    else:
        latex += 'Category & Modification & ' + f'\\multicolumn{{{tot}}}{{c}}{{\\textbf{{Models}}}} \\\\\n'
    latex += ' &  & ' + ' & '.join([f'\\textbf{{{c}}}' for c in cols]) + ' & \\textbf{Avg} \\\\\n'
    latex += '\\midrule\n'
    cats = []
    for key,(cat,mod) in mod_mapping.items():
        if cat not in cats:
            if cats: latex += '\\midrule\n'
            cats.append(cat)
            row0 = f'\\textbf{{{cat}}}'
        else:
            row0 = ' '
        vals=[]; row_vals=[]
        for c in cols:
            if (cat,mod) in p_u.index and c in p_u.columns:
                v=p_u.loc[(cat,mod), c]
                row_vals.append(v)
                vals.append(fu(v))
            else:
                vals.append('')
        ravg=float(np.nanmean(row_vals)) if row_vals else float('nan')
        latex += f"{row0} & \\textbf{{{mod}}} & " + ' & '.join(vals) + f" & {'' if np.isnan(ravg) else fu(ravg)} \\\\\n"
    # Average row per model
    col_means=[]
    for c in cols:
        series = p_u[c] if c in p_u.columns else pd.Series(dtype=float)
        col_means.append(float(series.mean()))
    overall=float(np.nanmean(col_means)) if col_means else float('nan')
    latex += '\\midrule\n'
    latex += '\\textbf{Average} &  ' + ' & '.join([('' if np.isnan(v) else fu(v)) for v in col_means]) + f" & {'' if np.isnan(overall) else fu(overall)} \\\\\n"
    latex += '\\bottomrule\n\\end{tabular}}\n'
    latex += '\\caption{Sentiment Analysis: Unrobustness (U, \%) by model and modification}\\label{tab:sa_unrob}\n\\end{table}'
    latex = latex.replace('\\n', '\n')
    with open(output_file,'w') as f: f.write(latex)
    print(f"U LaTeX table saved to {output_file}")

def generate_latex_table_robustness(df, output_file):
    if df is None or df.empty or 'model' not in df.columns or 'modification' not in df.columns:
        print(f"Skipping R LaTeX generation for {output_file}: no data or missing columns"); return
    df = df.copy()
    # Reuse unrobustness pipeline, but transform to robustness
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
    df['model'] = df['model'].replace(model_map)
    df['category'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    df['modification'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    p_u = df.pivot_table(index=['category','modification'], columns='model', values='unrobustness', aggfunc='mean')
    p_r = 100.0 - p_u
    cols = [m for m in model_order if m in p_r.columns]
    tot = len(cols)
    def fmt(v): return '' if pd.isna(v) else f"{float(v):.1f}"
    latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll' + 'r'*tot + 'r}\n'
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
    latex += '\\caption{Sentiment Analysis: Robustness (R, \\%) by model and modification}\\label{tab:sa_rob}\n\\end{table}'
    latex = latex.replace('\\n', '\n')
    with open(output_file,'w') as f: f.write(latex)
    print(f"R LaTeX table saved to {output_file}")


def generate_summary_plot(df, output_file):
    """Generate a heatmap figure for weighted deltas by model and modification.

    Produces a compact visual for the main content; the LaTeX table can be
    placed in the appendix. Falls back gracefully if matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy.ma as ma
    except Exception as e:
        print(f"Skipping plot generation: matplotlib not available ({e})")
        return

    # Guard against empty or malformed DataFrame
    if df is None or df.empty or 'model' not in df.columns or 'modification' not in df.columns:
        print(f"Skipping plot generation for {output_file}: no data or missing columns")
        return

    # Map models and group modifications by category
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

    model_order = ['BERT', 'GPT-2', 'T5', 'GPT-4o', 'Claude-3.5', 'Llama 3.1', 'GPT-5', 'DS R1', 'GPT-5 (w. context)']
    model_map = {
        'bert': 'BERT',
        'gpt2': 'GPT-2',
        't5': 'T5',
        'gpt4o': 'GPT-4o',
        'claude': 'Claude-3.5',
        'llama': 'Llama 3.1',
        'gpt-5-standard': 'GPT-5',
        'deepseek-r1-deepseek': 'DS R1',
        'gpt-5-standard-context-aware': 'GPT-5 (w. context)',
    }

    df = df.copy()
    df['model'] = df['model'].replace(model_map)
    df['category'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    df['modification_disp'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])

    # Build ordered index of (category, modification)
    ordered_mods = [(cat, name) for key, (cat, name) in mod_mapping.items()]
    # Keep only rows present in data
    present = set(zip(df['category'], df['modification_disp']))
    ordered_mods = [m for m in ordered_mods if m in present]

    # Pivot weighted delta and significance
    pivot_delta = df.pivot_table(index=['category', 'modification_disp'], columns='model', values='weighted_delta', aggfunc='mean')
    pivot_sig = df.pivot_table(index=['category', 'modification_disp'], columns='model', values='significance', aggfunc='first')

    # Order rows and columns
    cols = [m for m in model_order if m in pivot_delta.columns and m != 'GPT-5 (w. context)']
    if not cols or not ordered_mods:
        print(f"Skipping plot: insufficient data after pivoting")
        return
    pivot_delta = pivot_delta.reindex(index=ordered_mods, columns=cols)
    pivot_sig = pivot_sig.reindex(index=ordered_mods, columns=cols)

    # Prepare data for imshow; mask NaNs
    data = pivot_delta.values.astype(float)
    data_masked = ma.masked_invalid(data)

    # Dynamic figure size
    h = max(3, 0.45 * data_masked.shape[0] + 1.5)
    w = max(6, 0.6 * data_masked.shape[1] + 2.0)
    fig, ax = plt.subplots(figsize=(w, h))

    im = ax.imshow(data_masked, cmap='RdYlGn', vmin=-10, vmax=10, aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weighted Δ (positive is better)', rotation=90)

    # Y tick labels: show Category · Modification
    ylabels = [f"{cat} · {mod}" for (cat, mod) in pivot_delta.index]
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)

    # X tick labels: model names
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha='right')

    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Annotate cells with value and significance markers
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            sig = ''
            if pivot_sig is not None and (i < pivot_sig.shape[0]) and (j < pivot_sig.shape[1]):
                s = pivot_sig.iat[i, j]
                if isinstance(s, str) and s in {'.', '*', '**', '***'}:
                    sig = s
            txt = f"{val:+.1f}{sig if sig!='.' else ''}"  # omit dot marker for cleanliness
            ax.text(j, i, txt, ha='center', va='center', fontsize=8, color='black')

    fig.tight_layout()
    try:
        fig.savefig(output_file, dpi=200)
        print(f"Summary plot saved to {output_file}")
    finally:
        plt.close(fig)

def generate_latex_table_combined(df, output_file):
    """Generate a single SA LaTeX table with one cell per model: "Δ | U".
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
        'casual': ('Varieties', 'Style'),'dialectal': ('Varieties', 'Dialectal'),
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
    latex += '\\caption{Sentiment Analysis: Weighted $\\Delta$ | U by model and modification}\\label{tab:sa_combined}\n\\end{table}'
    latex = latex.replace('\\n', '\n')
    with open(output_file,'w') as f: f.write(latex)
    print(f"Combined LaTeX table saved to {output_file}")

def generate_latex_table_dual(df, output_file):
    """Generate a single table with left block = Δ, right block = U (SA)."""
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
        'casual': ('Varieties', 'Style'),'dialectal': ('Varieties', 'Dialectal'),
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
    latex += '\\caption{Sentiment Analysis: Left = $\\Delta$, Right = U (one block per metric)}\\label{tab:sa_dual}\n\\end{table}'
    latex = latex.replace('\\n', '\n')
    with open(output_file,'w') as f: f.write(latex)
    print(f"Dual LaTeX table saved to {output_file}")

def generate_latex_table_combined_cells(df, output_file):
    """Combined table with two cells per model (Δ then U), colored (SA)."""
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
        'casual': ('Varieties', 'Style'),'dialectal': ('Varieties', 'Dialectal'),
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
        intensity = int(min(max(val, 0.0), 100.0) * 0.8)  # stronger blue scale
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
    latex += '\\caption{Sentiment Analysis: Two cells per model — $\\Delta$ and U}\\label{tab:sa_combined_cells}\n\\end{table}'
    latex = latex.replace('\\n', '\n')
    with open(output_file,'w') as f: f.write(latex)
    print(f"Combined-cells LaTeX table saved to {output_file}")

def generate_compact_latex_table_combined_cells(df, output_file):
    """Compact category-level combined-cells table for SA."""
    if df is None or df.empty or 'model' not in df.columns or 'modification' not in df.columns:
        print(f"Skipping compact combined-cells for {output_file}: missing data/columns"); return
    mod_to_cat = {
        'temporal_bias': 'Bias','geographical_bias': 'Bias','length_bias': 'Bias',
        'typo_bias': 'Orthographic','capitalization': 'Orthographic','punctuation': 'Orthographic',
        'derivation': 'Morphological','compound_word': 'Morphological',
        'active_to_passive': 'Syntactic','grammatical_role': 'Syntactic','coordinating_conjunction': 'Syntactic',
        'concept_replacement': 'Semantic','negation': 'Semantic',
        'discourse': 'Pragmatic','sentiment': 'Pragmatic',
        'casual': 'Genre','dialectal': 'Genre'
    }
    model_order = ['BERT','GPT-2','T5','GPT-4o','Claude-3.5','Llama 3.1','GPT-5','DS R1','GPT-5 (w. context)']
    model_map = {'bert':'BERT','gpt2':'GPT-2','t5':'T5','gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    dfx=df.copy(); dfx['model']=dfx['model'].replace(model_map); dfx['category']=dfx['modification'].map(lambda x: mod_to_cat.get(x,'Other'))
    agg = dfx.groupby(['category','model']).agg(weighted_delta=('weighted_delta','mean'), unrobustness=('unrobustness','mean')).reset_index()
    overall = dfx.groupby(['model']).agg(weighted_delta=('weighted_delta','mean'), unrobustness=('unrobustness','mean')).reset_index(); overall.insert(0,'category','Overall')
    agg = pd.concat([agg, overall], ignore_index=True)
    p_delta = agg.pivot_table(index='category', columns='model', values='weighted_delta', aggfunc='mean')
    p_u = agg.pivot_table(index='category', columns='model', values='unrobustness', aggfunc='mean')
    cols=[m for m in model_order if m in p_delta.columns and m != 'GPT-5 (w. context)']
    if not cols: print(f"Skipping compact combined-cells for {output_file}: no columns"); return
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
    latex+='\\caption{Sentiment (compact): Two cells per model — $\\Delta$ and U}\\label{tab:sa_compact_combined_cells}\n\\end{table}'
    latex = latex.replace('\\n', '\n')
    with open(output_file,'w') as f: f.write(latex)
    print(f"Compact combined-cells LaTeX table saved to {output_file}")


def generate_compact_summary_plot(df, output_file):
    """Compact heatmap aggregated by category plus Overall row."""
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
    model_order = ['BERT', 'GPT-2', 'T5', 'GPT-4o', 'Claude-3.5', 'Llama 3.1', 'GPT-5','DS R1', 'GPT-5 (w. context)']
    model_map = {
        'bert': 'BERT', 'gpt2': 'GPT-2', 't5': 'T5', 'gpt4o': 'GPT-4o', 'claude': 'Claude-3.5', 'llama': 'Llama 3.1', 'gpt-5-standard': 'GPT-5', 'deepseek-r1-deepseek': 'DS R1', 'gpt-5-standard-context-aware': 'GPT-5 (w. context)'
    }

    df = df.copy()
    df['model'] = df['model'].replace(model_map)
    df['category'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])

    # Aggregate by category and model
    agg = df.groupby(['category', 'model']).agg(weighted_delta=('weighted_delta','mean'), unrobustness=('unrobustness','mean')).reset_index()
    # Add Overall row
    overall = df.groupby(['model']).agg(weighted_delta=('weighted_delta','mean'), unrobustness=('unrobustness','mean')).reset_index()
    overall.insert(0, 'category', 'Overall')
    agg = pd.concat([agg, overall], ignore_index=True)

    # Pivot for delta
    pivot_delta = agg.pivot_table(index='category', columns='model', values='weighted_delta', aggfunc='mean')
    cols = [m for m in model_order if m in pivot_delta.columns and m != 'GPT-5 (w. context)']
    rows = [*sorted([r for r in pivot_delta.index if r != 'Overall']), 'Overall'] if 'Overall' in pivot_delta.index else sorted(list(pivot_delta.index))
    if not cols or not rows:
        print("Skipping compact plot: insufficient data after aggregation")
        return
    pivot_delta = pivot_delta.reindex(index=rows, columns=cols)

    data = pivot_delta.values.astype(float)
    data_masked = ma.masked_invalid(data)

    h = max(3, 0.6 * data_masked.shape[0] + 1.0)
    w = max(6, 0.6 * data_masked.shape[1] + 2.0)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(data_masked, cmap='RdYlGn', vmin=-10, vmax=10, aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weighted Δ (avg by category)', rotation=90)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha='right')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    # Annotate values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:+.1f}", ha='center', va='center', fontsize=8, color='black')
    fig.tight_layout()
    try:
        fig.savefig(output_file, dpi=200)
        print(f"Compact summary plot saved to {output_file}")
    finally:
        plt.close(fig)


def generate_compact_latex_table(df, output_file):
    """Compact LaTeX table with category averages and an Overall row."""
    if df is None or df.empty or 'model' not in df.columns or 'modification' not in df.columns:
        print(f"Skipping compact LaTeX for {output_file}: no data or missing columns")
        return
    mod_mapping = {
        'temporal_bias': 'Bias',
        'geographical_bias': 'Bias', 
        'length_bias': 'Bias',
        'typo_bias': 'Orthographic',
        'capitalization': 'Orthographic',
        'punctuation': 'Orthographic',
        'derivation': 'Morphological',
        'compound_word': 'Morphological',
        'active_to_passive': 'Syntactic',
        'grammatical_role': 'Syntactic',
        'coordinating_conjunction': 'Syntactic',
        'concept_replacement': 'Semantic',
        'negation': 'Semantic',
        'discourse': 'Pragmatic',
        'sentiment': 'Pragmatic',
        'casual': 'Genre',
        'dialectal': 'Genre',
    }
    model_order = ['BERT', 'GPT-2', 'T5', 'GPT-4o', 'Claude-3.5', 'Llama 3.1', 'GPT-5','DS R1', 'GPT-5 (w. context)']
    model_map = {
        'bert': 'BERT','gpt2': 'GPT-2','t5': 'T5','gpt4o': 'GPT-4o','claude': 'Claude-3.5','llama': 'Llama 3.1','gpt-5-standard': 'GPT-5','deepseek-r1-deepseek': 'DS R1','gpt-5-standard-context-aware': 'GPT-5 (w. context)'
    }
    df = df.copy()
    df['model'] = df['model'].replace(model_map)
    df['category'] = df['modification'].map(lambda x: mod_mapping.get(x, 'Other'))
    agg = df.groupby(['category','model']).agg(weighted_delta=('weighted_delta','mean'), unrobustness=('unrobustness','mean')).reset_index()
    overall = df.groupby(['model']).agg(weighted_delta=('weighted_delta','mean'), unrobustness=('unrobustness','mean')).reset_index()
    overall.insert(0, 'category', 'Overall')
    agg = pd.concat([agg, overall], ignore_index=True)
    pivot_delta = agg.pivot_table(index='category', columns='model', values='weighted_delta', aggfunc='mean')
    pivot_unrob = agg.pivot_table(index='category', columns='model', values='unrobustness', aggfunc='mean')
    cols = [m for m in model_order if m in pivot_delta.columns and m != 'GPT-5 (w. context)']
    rows = [*sorted([r for r in pivot_delta.index if r != 'Overall']), 'Overall'] if 'Overall' in pivot_delta.index else sorted(list(pivot_delta.index))
    if not cols or not rows:
        print(f"Skipping compact LaTeX for {output_file}: insufficient data")
        return

    def fmt_cell(d, u):
        sd = '' if pd.isna(d) else f"{d:+.1f}"
        su = '' if pd.isna(u) else f"{u:.1f}"
        if sd == '' and su == '':
            return ''
        return f"{sd} | {su}"

    total_models = len(cols)
    latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{l' + 'r'*total_models + '}\n'
    latex += '\\hline\nCategory & ' + ' & '.join([f'\\textbf{{{c}}}' for c in cols]) + ' \\\\\n'
    latex += '\\hline\n'
    for cat in rows:
        row_vals = []
        for c in cols:
            d = pivot_delta.loc[cat, c] if (cat in pivot_delta.index and c in pivot_delta.columns) else np.nan
            u = pivot_unrob.loc[cat, c] if (cat in pivot_unrob.index and c in pivot_unrob.columns) else np.nan
            row_vals.append(fmt_cell(d, u))
        latex += f"{cat} & " + ' & '.join(row_vals) + ' \\\\\n'
    latex += '\\hline\n\\end{tabular}}\n'
    latex += '\\caption{Compact category-level averages (Sentiment): $\\Delta$ | U per model}\n'
    latex += '\\label{tab:sa_compact}\n\\end{table}'
    with open(output_file, 'w') as f:
        f.write(latex)
    print(f"Compact LaTeX table saved to {output_file}")


def generate_compact_unrobustness_plot(df, output_file):
    """Compact unrobustness heatmap aggregated by category plus Overall row."""
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

    mod_mapping = {
        'temporal_bias': 'Bias', 'geographical_bias': 'Bias', 'length_bias': 'Bias',
        'typo_bias': 'Orthographic', 'capitalization': 'Orthographic', 'punctuation': 'Orthographic',
        'derivation': 'Morphological', 'compound_word': 'Morphological',
        'active_to_passive': 'Syntactic', 'grammatical_role': 'Syntactic', 'coordinating_conjunction': 'Syntactic',
        'concept_replacement': 'Semantic', 'negation': 'Semantic',
        'discourse': 'Pragmatic', 'sentiment': 'Pragmatic',
        'casual': 'Genre', 'dialectal': 'Genre',
    }
    model_order = ['BERT', 'GPT-2', 'T5', 'GPT-4o', 'Claude-3.5', 'Llama 3.1', 'GPT-5','DS R1', 'GPT-5 (w. context)']
    model_map = {'bert': 'BERT','gpt2': 'GPT-2','t5': 'T5','gpt4o': 'GPT-4o','claude': 'Claude-3.5','llama': 'Llama 3.1','gpt-5-standard': 'GPT-5','deepseek-r1-deepseek': 'DS R1','gpt-5-standard-context-aware': 'GPT-5 (w. context)'}

    df = df.copy()
    df['model'] = df['model'].replace(model_map)
    df['category'] = df['modification'].map(lambda x: mod_mapping.get(x, 'Other'))

    agg = df.groupby(['category', 'model']).agg(unrobustness=('unrobustness','mean')).reset_index()
    overall = df.groupby(['model']).agg(unrobustness=('unrobustness','mean')).reset_index(); overall.insert(0,'category','Overall')
    agg = pd.concat([agg, overall], ignore_index=True)

    pivot_u = agg.pivot_table(index='category', columns='model', values='unrobustness', aggfunc='mean')
    cols = [m for m in model_order if m in pivot_u.columns and m != 'GPT-5 (w. context)']
    rows = [*sorted([r for r in pivot_u.index if r != 'Overall']), 'Overall'] if 'Overall' in pivot_u.index else sorted(list(pivot_u.index))
    if not cols or not rows:
        print("Skipping compact U plot: insufficient data after aggregation")
        return
    pivot_u = pivot_u.reindex(index=rows, columns=cols)

    data = pivot_u.values.astype(float)
    data_masked = ma.masked_invalid(data)
    h = max(3, 0.6 * data_masked.shape[0] + 1.0)
    w = max(6, 0.6 * data_masked.shape[1] + 2.0)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(data_masked, cmap='Blues', vmin=_U_MIN, vmax=_U_MAX, aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Unrobustness U (avg by category, %)', rotation=90)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
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
        print(f"Compact unrobustness plot saved to {output_file}")
    finally:
        plt.close(fig)


def generate_unrobustness_plot(df, output_file):
    """Generate a heatmap for Unrobustness (flip rate in %) by model and modification.

    Blue intensity indicates higher unrobustness. This complements the weighted
    delta heatmap and is suitable for main-text visualization.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import numpy.ma as ma
    except Exception as e:
        print(f"Skipping unrobustness plot: matplotlib not available ({e})")
        return

    # Guard
    needed_cols = {'model', 'modification', 'unrobustness'}
    if df is None or df.empty or not needed_cols.issubset(df.columns):
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

    model_order = ['BERT', 'GPT-2', 'T5', 'GPT-4o', 'Claude-3.5', 'Llama 3.1', 'GPT-5', 'DS R1', 'GPT-5 (w. context)']
    model_map = {
        'bert': 'BERT',
        'gpt2': 'GPT-2',
        't5': 'T5',
        'gpt4o': 'GPT-4o',
        'claude': 'Claude-3.5',
        'llama': 'Llama 3.1',
        'gpt-5-standard': 'GPT-5',
        'deepseek-r1-deepseek': 'DS R1',
        'gpt-5-standard-context-aware': 'GPT-5 (w. context)',
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

    # Annotate with values
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

def main():
    """Main function to run the analysis"""
    # Configuration
    llm_results_dir = str((SCRIPT_DIR / '../LLM/results/sa').resolve())
    
    # Process LLM and PLM results
    llm_results, llm_negation = process_llm_results(llm_results_dir)
    plm_results = process_plm_results()
    print(f"Loaded LLM rows: {len(llm_results)}; PLM rows: {len(plm_results)}")
    
    # Save LLM results
    combined_df = pd.DataFrame(llm_results + plm_results)
    results_df, negation_df = save_llm_results(llm_results, llm_negation)
    
    # Generate visual summary (main content) and LaTeX table (appendix)
    plot_path = str((SCRIPT_DIR / 'sa_results_heatmap.png').resolve())
    u_plot_path = str((SCRIPT_DIR / 'sa_unrobustness_heatmap.png').resolve())
    compact_plot_path = str((SCRIPT_DIR / 'sa_results_heatmap_compact.png').resolve())
    generate_summary_plot(combined_df, plot_path)
    generate_unrobustness_plot(combined_df, u_plot_path)
    # Separate LaTeX tables: Δ and U
    generate_latex_table_delta(combined_df, 'sa_results_table_delta.tex')
    generate_latex_table_unrob(combined_df, 'sa_results_table_unrobustness.tex')
    try:
        generate_latex_table_robustness(combined_df, 'sa_results_table_robustness.tex')
    except Exception:
        pass
    generate_latex_table_combined(combined_df, 'sa_results_table.tex')
    generate_latex_table_dual(combined_df, 'sa_results_table_dual.tex')
    generate_latex_table_combined_cells(combined_df, 'sa_results_table_combined_cells.tex')
    # Compact visuals and table
    generate_compact_summary_plot(combined_df, compact_plot_path)
    u_compact_plot_path = str((SCRIPT_DIR / 'sa_unrobustness_heatmap_compact.png').resolve())
    generate_compact_unrobustness_plot(combined_df, u_compact_plot_path)
    generate_compact_latex_table(combined_df, 'sa_results_table_compact.tex')
    generate_compact_latex_table_combined_cells(combined_df, 'sa_results_table_compact_combined_cells.tex')
    # Also write a separate GPT-5 vs GPT-5 (w. context) compact table
    try:
        generate_gpt5_context_table(combined_df, 'sa_gpt5_context_table.tex')
    except Exception as e:
        print(f"GPT-5 context table skipped (SA): {e}")

def generate_gpt5_context_table(df, output_file):
    """Generate a compact Booktabs table comparing only GPT-5 vs GPT-5 (w. context)."""
    if df is None or df.empty:
        return
    model_map = {
        'bert': 'BERT','gpt2': 'GPT-2','t5': 'T5','gpt4o': 'GPT-4o','claude': 'Claude-3.5','llama': 'Llama 3.1','gpt-5-standard': 'GPT-5','deepseek-r1-deepseek': 'DS R1','gpt-5-standard-context-aware': 'GPT-5 (w. context)'
    }
    df = df.copy()
    df['model'] = df['model'].replace(model_map)
    df = df[df['model'].isin(['GPT-5','GPT-5 (w. context)'])]
    if df.empty:
        raise ValueError('No GPT-5 context data')
    mod_to_cat = {
        'temporal_bias': 'Bias','geographical_bias': 'Bias','length_bias': 'Bias',
        'typo_bias': 'Orthography','capitalization': 'Orthography','punctuation': 'Orthography',
        'derivation': 'Morphology','compound_word': 'Morphology',
        'active_to_passive': 'Syntax','grammatical_role': 'Syntax','coordinating_conjunction': 'Syntax',
        'concept_replacement': 'Semantics','negation': 'Semantics',
        'discourse': 'Discourse','sentiment': 'Discourse','casual': 'Varieties','dialectal': 'Varieties'
    }
    df['category'] = df['modification'].map(lambda x: mod_to_cat.get(x, 'Other'))
    agg = df.groupby(['category','model']).agg(weighted_delta=('weighted_delta','mean'), unrobustness=('unrobustness','mean')).reset_index()
    overall = df.groupby(['model']).agg(weighted_delta=('weighted_delta','mean'), unrobustness=('unrobustness','mean')).reset_index(); overall.insert(0,'category','Overall')
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
    latex += '\\caption{Sentiment: GPT-5 vs GPT-5 (w. context) — category-level $\\Delta$ and U}\\label{tab:sa_gpt5_context}\n\\end{table}'
    with open(output_file,'w') as f: f.write(latex)
    
    print("\\nAnalysis complete!")


if __name__ == "__main__":
    main()
