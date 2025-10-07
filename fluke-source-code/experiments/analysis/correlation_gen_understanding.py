#!/usr/bin/env python3
"""
Correlations between Generation Performance (retain rate from annotation_quality.tex)
and model results for a specific model (default GPT-4o), computed per modification.

Relations:
- Retain rate vs. raw performance (avg accuracy/F1 across tasks: sa, coref, dialogue, ner, gsm, ifeval)
- Retain rate vs. robustness (100 − unrobustness; avg across the same tasks)

Outputs:
- Prints correlation stats and tables for both relations
- Saves an unrobustness table averaged across tasks to CSV
- Saves optional scatter plots if matplotlib is available
"""
from pathlib import Path
import re
import argparse
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).parent

# Map LaTeX display names to our internal modification keys
DISPLAY_TO_KEY = {
    'Temporal': 'temporal_bias',
    'Geographical': 'geographical_bias',
    'Length': 'length_bias',
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

# Invert mapping for canonical display names
KEY_TO_DISPLAY = {v: k for k, v in DISPLAY_TO_KEY.items()}


def parse_retain_rates(tex_path: Path) -> pd.DataFrame:
    if not tex_path.exists():
        raise FileNotFoundError(f"Missing {tex_path}")
    retain = {}
    pat = re.compile(r"&\\textbf\{([^}]+)\}.*?&\s*([0-9.]+)\s*\\\\")
    # We will parse lines having '& \\textbf{<Mod>} ... & <retain>'
    # The retain is the last numeric before \\
    with tex_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '& \\textbf{' not in line:
                continue
            # Extract all numbers; retain rate is last numeric in the row
            # and the display name is the first \\textbf{...} that is not a category label
            # We'll find all 'textbf{...}' tokens in the line, take the second occurrence as modification
            tfs = re.findall(r"\\textbf\{([^}]+)\}", line)
            if not tfs:
                continue
            # Choose the last textbf that is not a known category header
            # Categories: Bias, Morphology, Syntax, Semantics, Pragmatic, Varieties
            cats = {"Bias", "Morphology", "Syntax", "Semantics", "Pragmatic", "Varieties"}
            disp = None
            for tf in reversed(tfs):
                if tf not in cats:
                    disp = tf
                    break
            if disp is None:
                continue
            nums = re.findall(r"([0-9]+\.[0-9]+)", line)
            if not nums:
                continue
            retain_rate = float(nums[-1])
            retain[disp] = retain_rate
    # Map to internal keys
    rows = []
    for disp, rate in retain.items():
        key = DISPLAY_TO_KEY.get(disp)
        if key is None:
            continue
        rows.append({"display": disp, "modification": key, "retain_rate": rate})
    return pd.DataFrame(rows)


def _normalize_model(name: str) -> str:
    if name is None:
        return ""
    return name.strip().lower().replace(" ", "").replace("-", "")


def load_task_mod_averages(model_filter: str = "gpt4o") -> pd.DataFrame:
    """Load per-task, per-mod averages for a specific model (default GPT-4o).
    - raw: mean raw metric per task (original_mean_f1 if present, else original_acc; for IFEval use A)
    - unrobustness: mean unrobustness per task (for IFEval use U_frac)
    Returns a DataFrame with per-task columns for raw/unrobustness and aggregated averages:
      raw_{task}, un_{task}, raw_avg, unrobustness_avg
    """
    norm_filter = _normalize_model(model_filter)
    task_files = {
        'sa': SCRIPT_DIR / 'sa_modification_results_llm.csv',
        'coref': SCRIPT_DIR / 'coref_modification_results_llm.csv',
        'dialogue': SCRIPT_DIR / 'dialogue_modification_results_llm.csv',
        'ner': SCRIPT_DIR / 'ner_modification_results_llm.csv',
        'gsm': SCRIPT_DIR / 'gsm_modification_results_llm.csv',
    }
    per_task_raw = {}
    per_task_unrob = {}
    sig_flags = {}  # modification -> bool (any task p<0.05 for selected model)
    for task, path in task_files.items():
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if 'modification' not in df.columns:
            continue
        # Optional model filtering
        if 'model' in df.columns:
            df = df[df['model'].apply(lambda x: _normalize_model(str(x)) == norm_filter)]
        if df.empty:
            continue
        # raw metrics: prefer F1 if present; else accuracy
        raw_col = 'original_mean_f1' if 'original_mean_f1' in df.columns else ('original_acc' if 'original_acc' in df.columns else None)
        if raw_col is not None:
            per_task_raw[task] = df.groupby('modification')[raw_col].mean()
        # unrobustness
        if 'unrobustness' in df.columns:
            per_task_unrob[task] = df.groupby('modification')['unrobustness'].mean()
        # significance flags from per-task results (use only numeric p<0.05; ignore symbols like '.')
        if 'p_value' in df.columns:
            g = df.groupby('modification')['p_value'].min()
            for mod, p in g.items():
                if pd.notna(p) and p < 0.05:
                    sig_flags[mod] = True
                else:
                    sig_flags.setdefault(mod, False)

    # IFEval aggregates (optional)
    agg_root = SCRIPT_DIR / '../LLM/results/ifeval_aggregates'
    if agg_root.exists():
        raw_vals = {}
        unrob_vals = {}
        for mod_dir in sorted([p for p in agg_root.iterdir() if p.is_dir()]):
            mod = mod_dir.name
            raws, unrobs = [], []
            for csv in mod_dir.glob('*.csv'):
                try:
                    dfm = pd.read_csv(csv)
                    # Filter to the requested model if column present; otherwise
                    # fall back to filename heuristic.
                    if 'model' in dfm.columns:
                        dfm = dfm[dfm['model'].apply(lambda x: _normalize_model(str(x)) == norm_filter)]
                    else:
                        # Filename like 'gpt4o_comparison.csv'
                        if _normalize_model(csv.stem.replace('_comparison', '')) != norm_filter:
                            continue
                    if dfm.empty:
                        continue
                    # Use column A as raw performance if present
                    if 'A' in dfm.columns:
                        raws.append(float(dfm['A'].iloc[0]))
                    # Prefer U_frac for unrobustness if present
                    if 'U_frac' in dfm.columns:
                        unrobs.append(float(dfm['U_frac'].iloc[0]))
                    # significance from ifeval aggregate (only numeric p<0.05)
                    if 'p_value' in dfm.columns:
                        p = float(dfm['p_value'].iloc[0])
                        if pd.notna(p) and p < 0.05:
                            sig_flags[mod] = True
                        else:
                            sig_flags.setdefault(mod, False)
                except Exception:
                    continue
            if raws:
                raw_vals[mod] = float(np.mean(raws))
            if unrobs:
                unrob_vals[mod] = float(np.mean(unrobs))
        if raw_vals:
            per_task_raw['ifeval'] = pd.Series(raw_vals)
        if unrob_vals:
            per_task_unrob['ifeval'] = pd.Series(unrob_vals)

    # Combine raw
    full_raw = None
    for task, ser in per_task_raw.items():
        ser = ser.rename(f'raw_{task}')
        if full_raw is None:
            full_raw = ser.to_frame()
        else:
            full_raw = full_raw.join(ser, how='outer')
    if full_raw is None:
        return pd.DataFrame()
    # Ensure we average across exactly these tasks
    target_tasks = ['sa', 'coref', 'dialogue', 'ner', 'gsm', 'ifeval']
    for t in target_tasks:
        col = f'raw_{t}'
        if col not in full_raw.columns:
            full_raw[col] = np.nan
    full_raw['raw_avg'] = full_raw[[f'raw_{t}' for t in target_tasks]].mean(axis=1, skipna=True)

    # Combine unrobustness
    full_un = None
    for task, ser in per_task_unrob.items():
        ser = ser.rename(f'un_{task}')
        if full_un is None:
            full_un = ser.to_frame()
        else:
            full_un = full_un.join(ser, how='outer')
    if full_un is None:
        full_un = pd.DataFrame(index=full_raw.index)
    for t in target_tasks:
        col = f'un_{t}'
        if col not in full_un.columns:
            full_un[col] = np.nan
    full_un['unrobustness_avg'] = full_un[[f'un_{t}' for t in target_tasks]].mean(axis=1, skipna=True)

    # Merge all
    full_raw.index.name = 'modification'
    # Include per-task columns for unrobustness to support sanity checks
    un_task_cols = [c for c in full_un.columns if c.startswith('un_')]
    out = full_raw.join(full_un[un_task_cols + ['unrobustness_avg']], how='outer')
    # attach per-modification significance flag if available
    out['significant'] = out.index.map(lambda m: bool(sig_flags.get(m, False)))
    out.index.name = 'modification'
    return out.reset_index()


def main():
    parser = argparse.ArgumentParser(description='Correlate generation (retain rate) vs raw performance/robustness by modification')
    parser.add_argument('--model', default='gpt4o', help='LLM model to include (default: gpt4o). Example: gpt4o, gpt-4o, claude-3-5-sonnet, llama')
    args = parser.parse_args()
    model_name = args.model

    tex_path = SCRIPT_DIR / 'annotation_quality.tex'
    retain_df = parse_retain_rates(tex_path)
    if retain_df.empty:
        print('No retain rates parsed from annotation_quality.tex')
        return
    task_df = load_task_mod_averages(model_filter=model_name)
    if task_df.empty:
        print('No task results found to compute averages for the requested model. Please run task analyses first.')
        return

    # Sanity check: counts
    target_tasks = ['sa', 'coref', 'dialogue', 'ner', 'gsm', 'ifeval']
    raw_present = [t for t in target_tasks if f'raw_{t}' in task_df.columns and task_df[f'raw_{t}'].notna().any()]
    un_present = [t for t in target_tasks if f'un_{t}' in task_df.columns and task_df[f'un_{t}'].notna().any()]
    print(f'Tasks loaded (raw): {len(raw_present)}/{len(target_tasks)} -> {", ".join(raw_present) if raw_present else "none"}')
    print(f'Tasks loaded (unrobustness): {len(un_present)}/{len(target_tasks)} -> {", ".join(un_present) if un_present else "none"}')
    print(f'Unique modifications loaded: {task_df.shape[0]}')

    # Join retain rates with computed averages
    merged = retain_df.merge(task_df, on='modification', how='inner')
    merged = merged.dropna(subset=['retain_rate'])
    if merged.empty:
        print('No overlap between retain rates and task modifications after mapping.')
        return

    # Compute and print correlations for two relationships
    def corr_pair(df, xcol, ycol, label):
        sub = df.dropna(subset=[xcol, ycol])
        if sub.empty:
            print(f'{label}: no data')
            return (np.nan, np.nan, np.nan, np.nan, sub)
        try:
            from scipy.stats import pearsonr, spearmanr
            pr, pp = pearsonr(sub[xcol], sub[ycol])
            sr, sp = spearmanr(sub[xcol], sub[ycol])
            print(f'{label}: Pearson r={pr:.3f} (p={pp:.3g}), Spearman ρ={sr:.3f} (p={sp:.3g})')
        except Exception:
            pr = pp = sr = sp = np.nan
            print(f'{label}: scipy not available; skipping correlation stats')
        return (pr, pp, sr, sp, sub)

    print('')
    print('Correlations — model:', model_name)
    pr_r, pp_r, sr_r, sp_r, out_raw = corr_pair(merged, 'retain_rate', 'raw_avg', 'Retain vs Raw performance (avg)')
    # Use robustness (100 - U) instead of unrobustness for correlation/plots
    merged['robustness_avg'] = 100.0 - merged['unrobustness_avg']
    pr_rob, pp_rob, sr_rob, sp_rob, out_rob = corr_pair(merged, 'retain_rate', 'robustness_avg', 'Retain vs Robustness (avg)')

    
# Print summaries (use canonical display names when available)
    if not out_raw.empty:
        name_col = 'display' if 'display' in out_raw.columns else 'modification'
        name_col = 'display' if 'display' in out_raw.columns else 'modification'
        print('\nTable: Retain vs Raw performance (avg across tasks) — model:', model_name)
        print(out_raw[[name_col, 'retain_rate', 'raw_avg']].sort_values(name_col).to_string(index=False))
    if not out_rob.empty:
        name_col = 'display' if 'display' in out_rob.columns else 'modification'
        name_col = 'display' if 'display' in out_rob.columns else 'modification'
        print('\nTable: Retain vs Robustness (avg across tasks) — model:', model_name)
        print(out_rob[[name_col, 'retain_rate', 'robustness_avg']].sort_values(name_col).to_string(index=False))

    # Generate and save unrobustness table (average across all tasks)
    try:
        un_cols = [c for c in task_df.columns if c.startswith('un_')]
        keep_cols = ['modification'] + sorted(un_cols) + ['unrobustness_avg']
        tbl = task_df[keep_cols].copy()
        # Add display names if available
        tbl['display'] = tbl['modification'].map(KEY_TO_DISPLAY).fillna(tbl['modification'])
        # Reorder columns for readability
        ordered = ['display', 'modification'] + sorted(un_cols) + ['unrobustness_avg']
        tbl = tbl[ordered].sort_values('display')
        # Save CSV
        out_csv = SCRIPT_DIR / f'unrobustness_avg_table_{_normalize_model(model_name)}.csv'
        tbl.to_csv(out_csv, index=False)
        print(f"\nSaved unrobustness averages table to {out_csv}")
        # Also print a compact view
        print('\nUnrobustness (per task and avg) — model:', model_name)
        print(tbl.to_string(index=False))
    except Exception as e:
        print(f'Could not generate unrobustness table: {e}')

    
# Optional scatter plots
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        try:
            from adjustText import adjust_text  # optional, improves label overlap
        except Exception:
            adjust_text = None

        def scatter_plot(df, ycol, ylabel, suffix, pr, pp, sr, sp):
            if df.empty:
                return
            fig, ax = plt.subplots(figsize=(6,4))
            # use a single color for all points
            ax.scatter(df['retain_rate'], df[ycol], c='steelblue')
            # add labels with de-overlap
            texts = []
            pts = []  # store original point positions for leader lines
            for _, r in df.iterrows():
                base_name = r['display'] if 'display' in r else KEY_TO_DISPLAY.get(r['modification'], r['modification'])
                label = base_name  # remove significance marker in labels
                x, y = float(r['retain_rate']), float(r[ycol])
                t = ax.text(x, y, label, fontsize=7, alpha=0.95)
                texts.append(t)
                pts.append((x, y))
            # small initial alternating x/y offsets (helps symmetric overlaps)
            try:
                # helpers to convert pixels to data units
                def pix_to_data_dx(pix):
                    x0 = ax.transData.inverted().transform((0, 0))[0]
                    x1 = ax.transData.inverted().transform((pix, 0))[0]
                    return x1 - x0
                def pix_to_data_dy(pix):
                    y0 = ax.transData.inverted().transform((0, 0))[1]
                    y1 = ax.transData.inverted().transform((0, pix))[1]
                    return y1 - y0
                jx, jy = pix_to_data_dx(5), pix_to_data_dy(4)
                for i, t in enumerate(texts):
                    x0, y0 = t.get_position()
                    t.set_position((x0 + (jx if (i % 2 == 0) else -jx), y0 + (jy if (i % 3 == 0) else -jy)))
            except Exception:
                pass
            if adjust_text is not None:
                try:
                    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.3))
                except Exception:
                    pass
            else:
                # Force-directed repel in both x and y (fallback without adjustText)
                renderer = fig.canvas.get_renderer()
                def pix_to_data_dx(pix):
                    x0 = ax.transData.inverted().transform((0, 0))[0]
                    x1 = ax.transData.inverted().transform((pix, 0))[0]
                    return x1 - x0
                def pix_to_data_dy(pix):
                    y0 = ax.transData.inverted().transform((0, 0))[1]
                    y1 = ax.transData.inverted().transform((0, pix))[1]
                    return y1 - y0
                stepx = pix_to_data_dx(6)
                stepy = pix_to_data_dy(5)
                max_iter = 400
                for _ in range(max_iter):
                    moved = False
                    bboxes = [t.get_window_extent(renderer=renderer).expanded(1.05, 1.15) for t in texts]
                    for i in range(len(texts)):
                        for j in range(i+1, len(texts)):
                            if bboxes[i].overlaps(bboxes[j]):
                                xi, yi = texts[i].get_position()
                                xj, yj = texts[j].get_position()
                                # push apart diagonally
                                texts[i].set_position((xi + stepx, yi + stepy))
                                texts[j].set_position((xj - stepx, yj - stepy))
                                moved = True
                    if not moved:
                        break
                # final slight diagonal nudge to avoid near-contact
                fx, fy = pix_to_data_dx(2), pix_to_data_dy(2)
                for i, t in enumerate(texts):
                    x0, y0 = t.get_position()
                    t.set_position((x0 + (fx if i % 2 == 0 else -fx), y0 + (fy if i % 3 == 0 else -fy)))
                # Clamp maximum displacement to keep labels near their points
                max_px = 28  # limit label offset to ~28px from its point
                max_dx, max_dy = pix_to_data_dx(max_px), pix_to_data_dy(max_px)
                for (x_pt, y_pt), t in zip(pts, texts):
                    xt, yt = t.get_position()
                    dx, dy = xt - x_pt, yt - y_pt
                    # clamp per-axis displacement
                    if dx > max_dx: dx = max_dx
                    if dx < -max_dx: dx = -max_dx
                    if dy > max_dy: dy = max_dy
                    if dy < -max_dy: dy = -max_dy
                    t.set_position((x_pt + dx, y_pt + dy))
            # draw leader lines where displacement is notable
            try:
                for (x, y), t in zip(pts, texts):
                    xt, yt = t.get_position()
                    if abs(yt - y) > 1e-9 or abs(xt - x) > 1e-9:
                        ax.plot([x, xt], [y, yt], color='gray', lw=0.3, alpha=0.7)
            except Exception:
                pass
            ax.set_xlabel('Generation performance (Retain rate)')
            ax.set_ylabel(ylabel)
            # Title: report only Pearson; do not include model name
            if np.isnan(pr) or np.isnan(pp):
                ax.set_title('Pearson r=N/A')
            else:
                ax.set_title(f'Pearson r={pr:.3f} (p={pp:.3g})')
            # remove legend/significance
            fig.tight_layout()
            out_path = SCRIPT_DIR / f'correlation_scatter_{suffix}_{_normalize_model(model_name)}.png'
            fig.savefig(out_path, dpi=200)
            print(f'Saved scatter to {out_path}')

        scatter_plot(out_raw, 'raw_avg', 'Downstream performance (avg accuracy/F1)', 'raw', pr_r, pp_r, sr_r, sp_r)
        scatter_plot(out_rob, 'robustness_avg', 'Robustness (avg)', 'robustness', pr_rob, pp_rob, sr_rob, sp_rob)
    except Exception as e:
        print(f'Skipping scatter plots: {e}')


if __name__ == '__main__':
    main()
