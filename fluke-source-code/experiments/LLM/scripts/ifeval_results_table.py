#!/usr/bin/env python3
"""
Generate a LaTeX table for IFEval (LLM only) showing Δ (weighted delta) with significance and U (unrobustness).

Reads: ../results/ifeval_aggregates/length_bias/*_comparison.csv
Writes: ifeval_results_table.tex in the current directory.
"""

import glob
import os
import pandas as pd
from pathlib import Path


def load_aggregates(agg_dir: Path) -> pd.DataFrame:
    files = list((agg_dir / 'length_bias').glob('*_comparison.csv'))
    rows = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            rows.append(df)
        except Exception as e:
            print(f"Skip {fp.name}: {e}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def sig_stars(s: str) -> str:
    return s if isinstance(s, str) else ''


def color_delta(d: float, stars: str) -> str:
    # Green for positive, red for negative; include stars
    if pd.isna(d):
        return "--"
    color = 'green' if d >= 0 else 'red'
    return f"\\textcolor{{{color}}}{{{d:.2f}}}{stars}"


def shade_u(u: float) -> str:
    if pd.isna(u):
        return "--"
    # Map U (0..100) to blue shade intensity (0..!50)
    # For simplicity, use a fixed macro command; leave intensity implicit
    return f"\\cellcolor{{blue!10}} {u:.2f}"


def to_latex(df: pd.DataFrame, out_path: Path):
    # Keep only columns we need
    df2 = df[['model', 'mod', 'A', 'B', 'delta', 'U_frac', 'significance']].copy()
    # Order by model name
    df2 = df2.sort_values('model')
    # Build rows
    lines = []
    lines.append("\\begin{tabular}{lrrrrr}")
    lines.append("\\toprule")
    header = "Model & A & B & $\\Delta$ & U (pp) " + "\\\\"
    lines.append(header)
    lines.append("\\midrule")
    for _, r in df2.iterrows():
        model = r['model']
        A = r['A']
        B = r['B']
        d = color_delta(r['delta'], sig_stars(r['significance']))
        u = shade_u(r['U_frac'])
        row = f"{model} & {A:.2f} & {B:.2f} & {d} & {u} " + "\\\\"
        lines.append(row)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    out_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"Wrote LaTeX table: {out_path}")


def main():
    script_dir = Path(__file__).resolve().parent
    agg_dir = (script_dir / '../results/ifeval_aggregates').resolve()
    df = load_aggregates(agg_dir)
    if df.empty:
        print(f"No aggregates found in {agg_dir}")
        return
    out_path = script_dir / 'ifeval_results_table.tex'
    to_latex(df, out_path)


if __name__ == '__main__':
    main()
