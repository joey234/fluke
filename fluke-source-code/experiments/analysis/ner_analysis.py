#!/usr/bin/env python3
"""
Clean NER Analysis Script
Processes LLM NER results and combines with pre-calculated PLM results
"""

import json
import numpy as np
from scipy import stats
import pandas as pd
import os
import glob
from pathlib import Path
import ast
from tqdm import tqdm
from utils import get_global_unrobustness_range, unrob_intensity

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).parent
_U_MIN, _U_MAX = get_global_unrobustness_range()

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


def get_example_f1_and_counts_list(gold, pred):
    """Calculate F1 score and counts for a single example with list-based entities"""
    # Convert string labels to lists if needed
    if isinstance(gold, str):
        gold = ast.literal_eval(gold)
    if isinstance(pred, str):
        pred = ast.literal_eval(pred)

    # Standardize format to list of dicts with 'text' and 'value' keys
    def standardize_format(data):
        if isinstance(data, dict):
            return [{'text': k, 'value': v} for k, v in data.items()]
        elif isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                standardized = []
                for item in data:
                    if 'text' not in item:
                        for text, value in item.items():
                            standardized.append({'text': text, 'value': value})
                    else:
                        standardized.append(item)
                return standardized
            elif isinstance(data[0], str) and data[0].startswith('Entities:'):
                # Handle "Entities: [...]" format
                entities_str = data[0].replace('Entities: ', '')
                entities = ast.literal_eval(entities_str)
                return entities
        return data

    gold = standardize_format(gold)
    pred = standardize_format(pred)

    # Calculate metrics by comparing each prediction against gold
    tp = 0
    gold_matched = [False] * len(gold)
    pred_matched = [False] * len(pred)

    # First pass - find exact matches
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            if not gold_matched[j] and not pred_matched[i]:
                if p['text'] == g['text'] and p['value'] == g['value']:
                    tp += 1
                    gold_matched[j] = True
                    pred_matched[i] = True

    # Calculate false positives and false negatives
    fp = len(pred) - tp  # Predictions that didn't match any gold
    fn = len(gold) - tp  # Gold entities that weren't matched

    # Calculate F1 score for this example
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1, (tp, fp, fn)


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
        if any(skip in filename for skip in ['DP', 'ner.csv', 'compare']):
            continue
            
        # Extract model and modification from filename
        parts = filename.split('-')
        model = _map_model_from_filename(filename)
            
        # Handle both standard and legacy suffixes
        last = parts[-1]
        if last.endswith('_100_new.csv'):
            modification = last.replace('_100_new.csv', '')
        else:
            modification = last.replace('_100.csv', '')
        # Normalize aliases
        if modification == 'singlish':
            modification = 'dialectal'
        
        # Check if corresponding comparison file exists
        compare_file = (SCRIPT_DIR / f'../../data/modified_data/ner/{modification}_100.json')
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
        ori_preds = df['original_pred'].values
        mod_labels = df['modified_label'].values
        mod_preds = df['modified_pred'].values
        
        # Calculate F1 scores
        ori_f1_scores = []
        modif_f1_scores = []
        
        for l, p in zip(ori_labels, ori_preds):
            f1, _ = get_example_f1_and_counts_list(l, p)
            ori_f1_scores.append(f1 * 100)
            
        for l, p in zip(mod_labels, mod_preds):
            f1, _ = get_example_f1_and_counts_list(l, p)
            modif_f1_scores.append(f1 * 100)
        
        # Calculate statistics
        ori_mean_f1 = np.mean(ori_f1_scores)
        modif_mean_f1 = np.mean(modif_f1_scores)
        weighted_delta = (modif_mean_f1 - ori_mean_f1) * np.log10(ori_mean_f1) / np.log10(100)
        absolute_change = abs(modif_mean_f1 - ori_mean_f1)  # Absolute change in F1
        rate_of_change = absolute_change  # Keep for backward compatibility

        # Unrobustness for NER: mean absolute per-sample F1 change
        per_sample_abs_change = np.abs(np.array(modif_f1_scores) - np.array(ori_f1_scores))
        unrobustness = per_sample_abs_change.mean()
        
        # Statistical tests for directional change (paired)
        try:
            _, p_value_w = safe_wilcoxon(ori_f1_scores, modif_f1_scores)
        except Exception:
            p_value_w = 1.0
        p_value = p_value_w

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
            'original_mean_f1': ori_mean_f1,
            'modified_mean_f1': modif_mean_f1,
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
        if modification == 'negation':
            for subtype in ['verbal', 'lexical', 'double', 'approximate', 'absolute']:
                subtype_df = df[df['type'] == subtype]
                if len(subtype_df) == 0:
                    continue
                    
                # Calculate F1 scores for subtype
                ori_subtype_f1 = []
                mod_subtype_f1 = []
                
                for l, p in zip(subtype_df['original_label'], subtype_df['original_pred']):
                    f1, _ = get_example_f1_and_counts_list(l, p)
                    ori_subtype_f1.append(f1 * 100)
                    
                for l, p in zip(subtype_df['modified_label'], subtype_df['modified_pred']):
                    f1, _ = get_example_f1_and_counts_list(l, p)
                    mod_subtype_f1.append(f1 * 100)
                
                # Calculate stats
                ori_mean = np.mean(ori_subtype_f1)
                mod_mean = np.mean(mod_subtype_f1)
                weighted_delta = (mod_mean - ori_mean) * np.log10(ori_mean) / np.log10(100)
                absolute_change = abs(mod_mean - ori_mean)  # Absolute change
                rate_of_change = absolute_change  # Keep for backward compatibility

                # Unrobustness for subtype: mean absolute per-sample F1 change
                s_abs_change = np.abs(np.array(mod_subtype_f1) - np.array(ori_subtype_f1))
                s_unrob = s_abs_change.mean()
                
                # Statistical tests for directional change (paired)
                try:
                    _, p_w = safe_wilcoxon(ori_subtype_f1, mod_subtype_f1)
                except Exception:
                    p_w = 1.0
                p_val = p_w

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
                    'original_mean_f1': ori_mean,
                    'modified_mean_f1': mod_mean,
                    'weighted_delta': weighted_delta,
                    'absolute_change': absolute_change,
                    'rate_of_change': rate_of_change,  # Keep for backward compatibility
                    'unrobustness': s_unrob,
                    'sample_size': len(subtype_df),
                    'p_value': p_val,
                    'significance': sig,
                    'abs_p_value': abs_p_val,
                    'abs_significance': abs_sig
                })
    
    return results_data, negation_results_data


def process_plm_results():
    """Process PLM results (bert, gpt2, t5) for NER using parsed per-sample CSVs."""
    base_dir = (SCRIPT_DIR / '../PLM/ner')
    model_dirs = {
        'bert': base_dir / 'parsed_NER_BERT',
        'gpt2': base_dir / 'parsed_NER_GPT2',
        't5': base_dir / 'parsed_NER_T5'
    }
    results_data = []

    for model, mdir in model_dirs.items():
        if not mdir.exists():
            continue
        files = list(mdir.glob(f'{model}_ner_results_*.csv'))
        names = [f.stem.replace(f'{model}_ner_results_', '') for f in files]
        for csv_file in files:
            # modification name after prefix
            stem = csv_file.stem
            mod = stem.replace(f'{model}_ner_results_', '')
            modification = mod
            # Prefer singlish over dialectal
            if modification == 'dialectal' and 'singlish' in names:
                continue
            if modification == 'singlish':
                modification = 'dialectal'
            try:
                df = pd.read_csv(csv_file)
            except Exception:
                continue

            # Expect columns original_gold, modified_gold, original_pred, modified_pred
            if not set(['original_gold','modified_gold','original_pred','modified_pred']).issubset(df.columns):
                continue

            ori_f1_scores = []
            modif_f1_scores = []
            for l, p in zip(df['original_gold'], df['original_pred']):
                f1, _ = get_example_f1_and_counts_list(l, p)
                ori_f1_scores.append(f1 * 100)
            for l, p in zip(df['modified_gold'], df['modified_pred']):
                f1, _ = get_example_f1_and_counts_list(l, p)
                modif_f1_scores.append(f1 * 100)

            if len(ori_f1_scores) == 0:
                continue

            ori_mean_f1 = np.mean(ori_f1_scores)
            modif_mean_f1 = np.mean(modif_f1_scores)
            weighted_delta = (modif_mean_f1 - ori_mean_f1) * np.log10(ori_mean_f1) / np.log10(100) if ori_mean_f1 > 0 else 0
            absolute_change = abs(modif_mean_f1 - ori_mean_f1)
            rate_of_change = absolute_change

            # Unrobustness: mean abs per-sample F1 change
            per_sample_abs_change = np.abs(np.array(modif_f1_scores) - np.array(ori_f1_scores))
            unrobustness = per_sample_abs_change.mean()

            try:
                _, p_value_w = safe_wilcoxon(ori_f1_scores, modif_f1_scores)
            except Exception:
                p_value_w = 1.0
            p_value = p_value_w

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
                'original_mean_f1': ori_mean_f1,
                'modified_mean_f1': modif_mean_f1,
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
    results_df.to_csv('ner_modification_results_llm.csv', index=False)
    
    # Save negation results
    negation_df = pd.DataFrame(llm_negation)
    negation_df.to_csv('ner_negation_type_results_llm.csv', index=False)
    
    print("LLM results saved:")
    print("- ner_modification_results_llm.csv")
    print("- ner_negation_type_results_llm.csv")
    
    return results_df, negation_df


def generate_latex_table(df, output_file):
    """Generate LaTeX table from results DataFrame"""
    # Guard against empty or malformed DataFrame
    if df is None or df.empty or 'model' not in df.columns or 'modification' not in df.columns:
        print(f"Skipping LaTeX generation for {output_file}: no data or missing columns")
        return
    # Avoid mutating the caller's DataFrame used by compact plots
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
    
    model_order = ['BERT', 'GPT-2', 'T5', 'GPT-4o', 'Claude-3.5-Sonnet', 'Llama 3.1 405B', 'GPT-5','DeepSeek R1', 'GPT-5 (w. context)']
    model_map = {
        'bert': 'BERT',
        'gpt2': 'GPT-2',
        't5': 'T5',
        'gpt4o': 'GPT-4o', 
        'claude': 'Claude', 
        'llama': 'Llama 3.1 405B', 
        'gpt-5-standard': 'GPT-5',
        'deepseek-r1-deepseek': 'DeepSeek R1',
        'gpt-5-standard-context-aware': 'GPT-5 (w. context)'
    }
    
    df['model'] = df['model'].replace(model_map)
    df['category'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[0])
    df['modification'] = df['modification'].map(lambda x: mod_mapping.get(x, ('Other', x))[1])
    
    # Create pivot tables (aggregate duplicates by mean/first)
    pivot_df = df.pivot_table(index=['category', 'modification'], columns='model', values='weighted_delta', aggfunc='mean')
    significance = df.pivot_table(index=['category', 'modification'], columns='model', values='significance', aggfunc='first')
    unrob_df = df.pivot_table(index=['category', 'modification'], columns='model', values='unrobustness', aggfunc='mean')
    # Write separate Δ and U tables and return (do not combine)
    plm_models_all = ['BERT','GPT-2','T5']
    plm_models = [m for m in ['BERT','GPT-2','T5'] if m in pivot_df.columns]
    llm_models = [m for m in ['GPT-4o','Claude-3.5-Sonnet','Llama 3.1 405B','GPT-5','DeepSeek R1'] if m in pivot_df.columns]
    ordered_models = plm_models + llm_models
    def write_simple(pivot, caption, fname):
        total = len(ordered_models)
        # add Avg column and Avg row
        latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll' + 'r'*total + 'r}\n'
        latex += 'Category & Modification & ' + ' & '.join([f'\\textbf{{{m}}}' for m in ordered_models]) + ' & \\textbf{Avg} \\\\\n'
        latex += '\\midrule\n'
        row_means=[]
        for (cat,mod), _ in pivot.iterrows():
            row_vals=[]
            cells=[]
            for m in ordered_models:
                v = pivot.loc[(cat,mod), m] if (m in pivot.columns) else float('nan')
                row_vals.append(v)
                cells.append('' if pd.isna(v) else (f"{v:+.1f}" if 'delta' in fname else f"{v:.1f}"))
            ravg=float(np.nanmean(row_vals)) if row_vals else float('nan')
            latex += f"{cat} & {mod} & " + ' & '.join(cells) + f" & {'' if np.isnan(ravg) else (f'{ravg:+.1f}' if 'delta' in fname else f'{ravg:.1f}')} \\\\\n"
        # Average row per model
        col_means=[]
        for m in ordered_models:
            series = pivot.xs(m, axis=1, drop_level=False)[m]
            col_means.append(float(series.mean()))
        overall=float(np.nanmean(col_means)) if col_means else float('nan')
        latex += '\\midrule\n'
        latex += '\\textbf{Average} &  ' + ' & '.join([('' if np.isnan(v) else (f"{v:+.1f}" if 'delta' in fname else f"{v:.1f}")) for v in col_means]) + f" & {'' if np.isnan(overall) else (f'{overall:+.1f}' if 'delta' in fname else f'{overall:.1f}')} \\\\\n"
        latex += '\\end{tabular}}\n' + caption + '\\n\\end{table}'
        with open(fname,'w') as f: f.write(latex)
    write_simple(pivot_df, '\caption{NER: Weighted $\Delta$ by model and modification}\label{tab:ner_delta}', 'ner_results_table_delta.tex')
    write_simple(unrob_df, '\caption{NER: Unrobustness (U, \%) by model and modification}\label{tab:ner_unrob}', 'ner_results_table_unrobustness.tex')
    try:
        rob_df = 100.0 - unrob_df
        write_simple(rob_df, '\caption{NER: Robustness (R, \%) by model and modification}\label{tab:ner_rob}', 'ner_results_table_robustness.tex')
    except Exception:
        pass
    print("LaTeX tables saved: ner_results_table_delta.tex, ner_results_table_unrobustness.tex, ner_results_table_robustness.tex")
    return

def generate_latex_table_delta(df, output_file):
    if df is None or df.empty or 'model' not in df.columns or 'modification' not in df.columns:
        print(f"Skipping Δ LaTeX generation for {output_file}: no data or missing columns"); return
    df = df.copy()
    mod_mapping = {
        'temporal_bias': ('Bias', 'Temporal'),'geographical_bias': ('Bias', 'Geographical'),'length_bias': ('Bias', 'Length'),
        'typo_bias': ('Orthography', 'Spelling'),'capitalization': ('Orthography', 'Capitalization'),'punctuation': ('Orthography', 'Punctuation'),
        'derivation': ('Morphology', 'Derivation'),'compound_word': ('Morphology', 'Compound'),
        'active_to_passive': ('Syntax', 'Voice'),'grammatical_role': ('Syntax', 'Grammar'),'coordinating_conjunction': ('Syntax', 'Conjunction'),
        'concept_replacement': ('Semantics', 'Concept'),'negation': ('Semantics', 'Negation'),
        'discourse': ('Discourse', 'Disc. markers'),'sentiment': ('Discourse', 'Appraisal'),'casual': ('Varieties', 'Style'),'dialectal': ('Varieties', 'Dialect'),
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
        # intensity scaled by |Δ| (cap -> 0–20)
        intensity = int(min(abs(val)/10.0, 1.0) * 20)
        color = 'green' if val > 0 else 'red'
        s = f"{val:+.1f}"
        if isinstance(sig, str):
            if sig == '.':
                s = f"\\textbf{{{s}}}"
            elif sig == '*':
                s = f"\\textbf{{{s}}}*"
            elif sig == '**':
                s = f"\\textbf{{{s}}}**"
            elif sig == '***':
                s = f"\\textbf{{{s}}}***"
        return f"\\cellcolor{{{color}!{intensity}}} {s}"
    tot = len(cols)
    latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll'+'r'*tot+'r}\n'
    latex += '\\toprule\n'
    if plm and llm:
        latex += 'Category & Modification & ' + f'\\multicolumn{{{len(plm)}}}{{c}}{{\\textbf{{PLM}}}} & ' + f'\\multicolumn{{{len(llm)}}}{{c}}{{\\textbf{{LLM}}}} \\\\\n'
    else:
        latex += 'Category & Modification & ' + f'\\multicolumn{{{tot}}}{{c}}{{\\textbf{{Models}}}} & \\textbf{Avg} \\\\\n'
    latex += ' &  & ' + ' & '.join([f'\\textbf{{{c}}}' for c in cols]) + ' & \\textbf{Avg} \\\\\n'
    latex += '\\midrule\n'
    cats=[]
    for key,(cat,mod) in mod_mapping.items():
        if cat not in cats:
            if cats: latex += '\\midrule\n'
            cats.append(cat)
            lead = f'\\textbf{{{cat}}}'
        else:
            lead = ' '
        vals=[]
        for c in cols:
            if (cat,mod) in p_delta.index and c in p_delta.columns:
                sig = p_sig.loc[(cat,mod), c] if ((cat,mod) in p_sig.index and c in p_sig.columns) else ''
                vals.append(fd_color(p_delta.loc[(cat,mod), c], sig if isinstance(sig,str) else ''))
            else:
                vals.append('')
        latex += f"{lead} & \\textbf{{{mod}}} & " + ' & '.join(vals) + ' \\\\\n'
    latex += '\\bottomrule\n\\end{tabular}}\n'
    latex += '\\caption{NER: Weighted $\\Delta$ by model and modification}\\label{tab:ner_delta}\n\\end{table}'
    latex = latex.replace('\\n', '\n')
    with open(output_file,'w') as f: f.write(latex)

def generate_latex_table_unrob(df, output_file):
    if df is None or df.empty or 'model' not in df.columns or 'modification' not in df.columns:
        print(f"Skipping U LaTeX generation for {output_file}: no data or missing columns"); return
    df = df.copy()
    mod_mapping = {
        'temporal_bias': ('Bias', 'Temporal'),'geographical_bias': ('Bias', 'Geographical'),'length_bias': ('Bias', 'Length'),
        'typo_bias': ('Orthography', 'Spelling'),'capitalization': ('Orthography', 'Capitalization'),'punctuation': ('Orthography', 'Punctuation'),
        'derivation': ('Morphology', 'Derivation'),'compound_word': ('Morphology', 'Compound'),
        'active_to_passive': ('Syntax', 'Voice'),'grammatical_role': ('Syntax', 'Grammar'),'coordinating_conjunction': ('Syntax', 'Conjunction'),
        'concept_replacement': ('Semantics', 'Concept'),'negation': ('Semantics', 'Negation'),
        'discourse': ('Discourse', 'Disc. markers'),'sentiment': ('Discourse', 'Appraisal'),'casual': ('Varieties', 'Style'),'dialectal': ('Varieties', 'Dialectal'),
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
    def fu_color(v):
        if pd.isna(v):
            return ''
        try:
            iv = float(v)
        except Exception:
            return ''
        inten = unrob_intensity(iv, _U_MIN, _U_MAX)
        txt = f"{iv:.1f}"
        if inten >= 45:
            txt = f"\\textcolor{{white}}{{{txt}}}"
        return f"\\cellcolor{{blue!{inten}}} {txt}"
    tot = len(cols)
    latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll'+'r'*tot+'r}\n'
    latex += '\\toprule\n'
    if plm and llm:
        latex += 'Category & Modification & ' + f'\\multicolumn{{{len(plm)}}}{{c}}{{\\textbf{{PLM}}}} & ' + f'\\multicolumn{{{len(llm)}}}{{c}}{{\\textbf{{LLM}}}} \\\\\n'
    else:
        latex += 'Category & Modification & ' + f'\\multicolumn{{{tot}}}{{c}}{{\\textbf{{Models}}}} & \\textbf{Avg} \\\\\n'
    latex += ' &  & ' + ' & '.join([f'\\textbf{{{c}}}' for c in cols]) + ' & \\textbf{Avg} \\\\\n'
    latex += '\\midrule\n'
    cats=[]
    for key,(cat,mod) in mod_mapping.items():
        if cat not in cats:
            if cats: latex += '\\midrule\n'
            cats.append(cat)
            lead = f'\\textbf{{{cat}}}'
        else:
            lead = ' '
        vals=[]; row_vals=[]
        for c in cols:
            if (cat,mod) in p_u.index and c in p_u.columns:
                v = p_u.loc[(cat,mod), c]
                row_vals.append(v)
                vals.append(fu_color(v))
            else:
                vals.append('')
        ravg = float(np.nanmean(row_vals)) if row_vals else float('nan')
        latex += f"{lead} & \\textbf{{{mod}}} & " + ' & '.join(vals) + f" & {'' if np.isnan(ravg) else fu_color(ravg)} \\\\\n"
    # Average row per model
    col_means=[]
    for c in cols:
        series = p_u[c] if c in p_u.columns else pd.Series(dtype=float)
        col_means.append(float(series.mean()))
    overall = float(np.nanmean(col_means)) if col_means else float('nan')
    latex += '\\midrule\n'
    latex += '\\textbf{Average} &  ' + ' & '.join([('' if np.isnan(v) else fu_color(v)) for v in col_means]) + f" & {'' if np.isnan(overall) else fu_color(overall)} \\\\\n"
    latex += '\\bottomrule\n\\end{tabular}}\n'
    latex += '\\caption{NER: Unrobustness (U, \%) by model and modification}\\label{tab:ner_unrob}\\n\\end{table}'
    latex = latex.replace('\\n', '\n')
    with open(output_file,'w') as f: f.write(latex)
    print(f"U LaTeX table saved to {output_file}")
    return

def generate_latex_table_dual(df, output_file):
    """Generate a single table with left block = Δ, right block = U.
    Excludes GPT-5 (w. context) from main columns. Rounds to 1 decimal.
    """
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
        'discourse': ('Discourse', 'Disc. markers'),'sentiment': ('Discourse', 'Appraisal'),'casual': ('Varieties', 'Style'),'dialectal': ('Varieties', 'Dialectal'),
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
    # Formatters with color
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
        return f"\\cellcolor{{blue!{inten}}} {val:.1f}"
    # Build table: left Δ, right U
    tot = len(cols)
    latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll'+'r'*tot+'r'*tot+'}\n'
    latex += '\\toprule\n'
    # Group headers
    head_left = f'\\multicolumn{{{tot}}}{{c}}{{\\textbf{{Δ (Weighted)}}}}'
    head_right = f'\\multicolumn{{{tot}}}{{c}}{{\\textbf{{U (flip \%)}}}}'
    latex += f"Category & Modification & {head_left} & {head_right} \\\\\n"
    # Model names row
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
            if c in p_delta.columns and (cat,mod) in p_delta.index:
                sig = p_sig.loc[(cat,mod), c] if ((cat,mod) in p_sig.index and c in p_sig.columns) else ''
                deltas.append(fd_color(p_delta.loc[(cat,mod), c], sig if isinstance(sig,str) else ''))
            else:
                deltas.append('')
            if c in p_u.columns and (cat,mod) in p_u.index:
                us.append(fu_color(p_u.loc[(cat,mod), c]))
            else:
                us.append('')
        latex += f"{lead} & \\textbf{{{mod}}} & " + ' & '.join(deltas) + ' & ' + ' & '.join(us) + ' \\\\\n'
    latex += '\\bottomrule\n\\end{tabular}}\n'
    latex += '\\caption{NER: Left = $\\Delta$, Right = U (one block per metric)}\\label{tab:ner_dual}\n\\end{table}'
    latex = latex.replace('\\n', '\n')
    with open(output_file,'w') as f: f.write(latex)
    print(f"Dual LaTeX table saved to {output_file}")

def generate_latex_table_combined_cells(df, output_file):
    """Combined table with two cells per model (Δ then U), colored.
    Excludes GPT-5 (w. context). Rounds to 1 decimal.
    """
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
        'discourse': ('Discourse', 'Disc. markers'),'sentiment': ('Discourse', 'Appraisal'),'casual': ('Varieties', 'Style'),'dialectal': ('Varieties', 'Dialectal'),
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
    latex = '\\begin{table}[h]\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{ll'+'rr'*tot+'}\n'
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
    latex += '\\caption{NER: Two cells per model — $\\Delta$ and U}\\label{tab:ner_combined_cells}\n\\end{table}'
    latex = latex.replace('\\n', '\n')
    with open(output_file,'w') as f: f.write(latex)
    print(f"Combined-cells LaTeX table saved to {output_file}")

def generate_compact_latex_table_combined_cells(df, output_file):
    """Compact category-level combined-cells table: two cells per model (Δ then U), colored."""
    need = {'model','modification','weighted_delta','unrobustness','significance'}
    if df is None or df.empty or not {'model','modification'}.issubset(df.columns):
        print(f"Skipping compact combined-cells for {output_file}: missing data/columns"); return
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
    dfx = df.copy(); dfx['model'] = dfx['model'].replace(model_map)
    dfx['category'] = dfx['modification'].map(lambda x: mod_to_cat.get(x,'Other'))
    agg = dfx.groupby(['category','model']).agg(weighted_delta=('weighted_delta','mean'), unrobustness=('unrobustness','mean')).reset_index()
    overall = dfx.groupby(['model']).agg(weighted_delta=('weighted_delta','mean'), unrobustness=('unrobustness','mean')).reset_index(); overall.insert(0,'category','Overall')
    agg = pd.concat([agg, overall], ignore_index=True)
    p_delta = agg.pivot_table(index='category', columns='model', values='weighted_delta', aggfunc='mean')
    p_u = agg.pivot_table(index='category', columns='model', values='unrobustness', aggfunc='mean')
    cols = [m for m in model_order if m in p_delta.columns and m != 'GPT-5 (w. context)']
    if not cols:
        print(f"Skipping compact combined-cells for {output_file}: no columns after aggregation"); return
    def fd_color(v):
        if pd.isna(v): return ''
        try: val=float(v)
        except Exception: return ''
        intensity=int(min(abs(val)/10.0,1.0)*20)
        color='green' if val>0 else 'red'
        return f"\\cellcolor{{{color}!{intensity}}} {val:+.1f}"
    def fu_color(v):
        if pd.isna(v): return ''
        try: val=float(v)
        except Exception: return ''
        inten = unrob_intensity(val, _U_MIN, _U_MAX)
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
    latex+='\\caption{NER (compact): Two cells per model — $\\Delta$ and U}\\label{tab:ner_compact_combined_cells}\n\\end{table}'
    with open(output_file,'w') as f: f.write(latex)
    print(f"Compact combined-cells LaTeX table saved to {output_file}")

def generate_latex_table_combined(df, output_file):
    """Generate a single NER LaTeX table with one cell per model: "Δ | U".
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
        'discourse': ('Discourse', 'Disc. markers'),'sentiment': ('Discourse', 'Appraisal'),'casual': ('Varieties', 'Style'),'dialectal': ('Varieties', 'Dialectal'),
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
    # Build table
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
    latex += '\\caption{NER: Weighted $\\Delta$ | U by model and modification}\\label{tab:ner_combined}\n\\end{table}'
    with open(output_file,'w') as f: f.write(latex)
    print(f"Combined LaTeX table saved to {output_file}")
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
        inten = unrob_intensity(float(u), _U_MIN, _U_MAX)
        return f'\\cellcolor{{blue!{inten}}} {u:.1f}'
    
    # Split models into PLM and LLM and build grouped headers
    plm_models_all = ['BERT', 'GPT-2', 'T5']
    plm_models = [m for m in model_order if m in plm_models_all and m in pivot_df.columns]
    llm_models = [m for m in model_order if (m not in plm_models_all) and (m in pivot_df.columns)]
    ordered_models = plm_models + llm_models

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
    latex_table += '\\caption{Weighted Delta ($\\Delta$) and Unrobustness (U) by Model and Modification Type}\n'
    latex_table += '\\label{tab:ner_results}\n\\end{table}'
    
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
        'dialectal': ('Varieties', 'Dialectal'),
    }

    model_order = ['BERT', 'GPT-2', 'T5', 'GPT-4o', 'Claude-3.5-Sonnet', 'Llama 3.1 405B', 'GPT-5','DeepSeek R1', 'GPT-5 (w. context)']
    model_map = {
        'bert': 'BERT', 'gpt2': 'GPT-2', 't5': 'T5', 'gpt4o': 'GPT-4o', 'claude': 'Claude-3.5-Sonnet', 'llama': 'Llama 3.1 405B', 'gpt-5-standard': 'GPT-5', 'deepseek-r1-deepseek': 'DeepSeek R1', 'gpt-5-standard-context-aware': 'GPT-5 (w. context)'
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
        'dialectal': ('Varieties', 'Dialectal'),
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
    cbar.set_label('Unrobustness U (mean abs F1 change)', rotation=90)
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
        'typo_bias': 'Orthography','capitalization': 'Orthography','punctuation': 'Orthography',
        'derivation': 'Morphology','compound_word': 'Morphology',
        'active_to_passive': 'Syntax','grammatical_role': 'Syntax','coordinating_conjunction': 'Syntax',
        'concept_replacement': 'Semantics','negation': 'Semantics',
        'discourse': 'Discourse','sentiment': 'Discourse',
        'casual': 'Varieties','dialectal': 'Varieties'
    }
    model_order = ['BERT','GPT-2','T5','GPT-4o','Claude-3.5','Llama 3.1','GPT-5','DS R1','GPT-5 (w. context)']
    model_map = {'bert':'BERT','gpt2':'GPT-2','t5':'T5','gpt4o':'GPT-4o','claude':'Claude-3.5','llama':'Llama 3.1','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DS R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    df = df.copy(); df['model'] = df['model'].replace(model_map); df['category'] = df['modification'].map(lambda x: mod_to_cat.get(x,'Other'))
    agg = df.groupby(['category','model']).agg(weighted_delta=('weighted_delta','mean')).reset_index()
    overall = df.groupby(['model']).agg(weighted_delta=('weighted_delta','mean')).reset_index(); overall.insert(0,'category','Overall')
    agg = pd.concat([agg, overall], ignore_index=True)
    pivot = agg.pivot_table(index='category', columns='model', values='weighted_delta', aggfunc='mean')
    cols = [m for m in model_order if m in pivot.columns and m != 'GPT-5 (w. context)']; rows = [*sorted([r for r in pivot.index if r!='Overall']), 'Overall'] if 'Overall' in pivot.index else sorted(list(pivot.index))
    if not cols or not rows: print("Skipping compact plot: insufficient data after aggregation"); return
    pivot = pivot.reindex(index=rows, columns=cols)
    data = pivot.values.astype(float); data_masked = ma.masked_invalid(data)
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
        'typo_bias': 'Orthography','capitalization': 'Orthography','punctuation': 'Orthography',
        'derivation': 'Morphology','compound_word': 'Morphology',
        'active_to_passive': 'Syntax','grammatical_role': 'Syntax','coordinating_conjunction': 'Syntax',
        'concept_replacement': 'Semantics','negation': 'Semantics',
        'discourse': 'Discourse','sentiment': 'Discourse',
        'casual': 'Varieties','dialectal': 'Varieties'
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
    latex += '\\caption{Compact category-level averages (NER): $\\Delta$ | U per model}\n'
    latex += '\\label{tab:ner_compact}\n\\end{table}'
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
        'typo_bias': 'Orthography','capitalization': 'Orthography','punctuation': 'Orthography',
        'derivation': 'Morphology','compound_word': 'Morphology',
        'active_to_passive': 'Syntax','grammatical_role': 'Syntax','coordinating_conjunction': 'Syntax',
        'concept_replacement': 'Semantics','negation': 'Semantics',
        'discourse': 'Discourse','sentiment': 'Discourse',
        'casual': 'Varieties','dialectal': 'Varieties'
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


def main():
    """Main function to run the analysis"""
    # Configuration
    llm_results_dir = str((SCRIPT_DIR / '../LLM/results/ner').resolve())
    
    # Process LLM results
    llm_results, llm_negation = process_llm_results(llm_results_dir)
    plm_results = process_plm_results()
    
    # Save LLM results
    combined_df = pd.DataFrame(llm_results + plm_results)
    results_df, negation_df = save_llm_results(llm_results, llm_negation)
    
    plot_path = str((SCRIPT_DIR / 'ner_results_heatmap.png').resolve())
    u_plot_path = str((SCRIPT_DIR / 'ner_unrobustness_heatmap.png').resolve())
    generate_summary_plot(combined_df, plot_path)
    generate_unrobustness_plot(combined_df, u_plot_path)
    generate_latex_table_delta(combined_df, 'ner_results_table_delta.tex')
    generate_latex_table_unrob(combined_df, 'ner_results_table_unrobustness.tex')
    generate_latex_table_combined(combined_df, 'ner_results_table.tex')
    generate_latex_table_dual(combined_df, 'ner_results_table_dual.tex')
    generate_latex_table_combined_cells(combined_df, 'ner_results_table_combined_cells.tex')
    # Compact outputs
    compact_plot = str((SCRIPT_DIR / 'ner_results_heatmap_compact.png').resolve())
    generate_compact_summary_plot(combined_df, compact_plot)
    generate_compact_latex_table(combined_df, 'ner_results_table_compact.tex')
    generate_compact_latex_table_combined_cells(combined_df, 'ner_results_table_compact_combined_cells.tex')
    try:
        generate_gpt5_context_table(combined_df, 'ner_gpt5_context_table.tex')
    except Exception as e:
        print(f"GPT-5 context table skipped (NER): {e}")

def generate_gpt5_context_table(df, output_file):
    if df is None or df.empty:
        return
    model_map = {'bert':'BERT','gpt2':'GPT-2','t5':'T5','gpt4o':'GPT-4o','claude':'Claude-3.5-Sonnet','llama':'Llama 3.1 405B','gpt-5-standard':'GPT-5','deepseek-r1-deepseek':'DeepSeek R1','gpt-5-standard-context-aware':'GPT-5 (w. context)'}
    df = df.copy(); df['model'] = df['model'].replace(model_map)
    df = df[df['model'].isin(['GPT-5','GPT-5 (w. context)'])]
    if df.empty: raise ValueError('No GPT-5 context data')
    mod_to_cat = {
        'temporal_bias': 'Bias','geographical_bias': 'Bias','length_bias': 'Bias',
        'typo_bias': 'Orthography','capitalization': 'Orthography','punctuation': 'Orthography',
        'derivation': 'Morphology','compound_word': 'Morphology',
        'active_to_passive': 'Syntax','grammatical_role': 'Syntax','coordinating_conjunction': 'Syntax',
        'concept_replacement': 'Semantics','negation': 'Semantics','discourse': 'Discourse','sentiment': 'Discourse','casual': 'Varieties','dialectal': 'Varieties'
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
    latex += '\\caption{NER: GPT-5 vs GPT-5 (w. context) — category-level $\\Delta$ and U}\\label{tab:ner_gpt5_context}\n\\end{table}'
    with open(output_file,'w') as f: f.write(latex)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
