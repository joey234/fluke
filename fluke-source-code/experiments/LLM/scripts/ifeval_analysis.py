#!/usr/bin/env python3

import argparse
import math
from pathlib import Path
from typing import Tuple
import pandas as pd

try:
    from scipy.stats import wilcoxon, binomtest
except Exception:  # pragma: no cover
    wilcoxon = None
    binomtest = None


def weighted_delta(a: float, b: float) -> float:
    # Scale by the difficulty at baseline using log10(A)/log10(100)
    a_safe = max(a, 1e-6)
    return (b - a) * (math.log10(a_safe) / math.log10(100.0))


def significance_tests(comp_o, comp_m, strict_o, strict_m) -> Tuple[float, str]:
    pvals = []
    # Wilcoxon on per-sample compliance rates (paired)
    if wilcoxon is not None:
        try:
            stat = wilcoxon(comp_o, comp_m, zero_method="wilcox", alternative="two-sided")
            pvals.append(stat.pvalue)
        except Exception:
            pass
    # McNemar exact on strict success discordant pairs
    if binomtest is not None:
        try:
            so = pd.Series(strict_o).astype(int)
            sm = pd.Series(strict_m).astype(int)
            n01 = int(((so == 0) & (sm == 1)).sum())
            n10 = int(((so == 1) & (sm == 0)).sum())
            n = n01 + n10
            if n > 0:
                # two-sided exact binomial test with p=0.5
                pv = binomtest(k=min(n01, n10), n=n, p=0.5).pvalue
                pvals.append(pv)
        except Exception:
            pass
    p = min(pvals) if pvals else float("nan")
    if not p == p:  # NaN
        return p, "n/a"
    if p < 0.001:
        return p, "***"
    if p < 0.01:
        return p, "**"
    if p < 0.05:
        return p, "*"
    return p, ""


def main():
    ap = argparse.ArgumentParser(description="Analyze IFEval original vs modified scores")
    ap.add_argument("--orig_csv", required=True, type=Path, help="Per-sample scores CSV for original side")
    ap.add_argument("--mod_csv", required=True, type=Path, help="Per-sample scores CSV for modified side")
    ap.add_argument("--model", required=True, type=str, help="Model name (for output)")
    ap.add_argument("--mod", required=True, type=str, help="Modification name, e.g., length_bias")
    ap.add_argument("--out_csv", required=True, type=Path, help="Output path for aggregate comparison CSV")
    ap.add_argument("--dataset", type=Path, help="Optional: path to the dataset JSONL to read subtype info (used for negation)")
    args = ap.parse_args()

    do = pd.read_csv(args.orig_csv)
    dm = pd.read_csv(args.mod_csv)

    # Pair by key
    merged = do[["key", "compliance_rate", "strict_success"]].merge(
        dm[["key", "compliance_rate", "strict_success"]], on="key", suffixes=("_orig", "_mod"), how="inner"
    )

    if merged.empty:
        raise SystemExit("No paired samples between original and modified.")

    comp_o = merged["compliance_rate_orig"].astype(float).tolist()
    comp_m = merged["compliance_rate_mod"].astype(float).tolist()
    strict_o = merged["strict_success_orig"].astype(int).tolist()
    strict_m = merged["strict_success_mod"].astype(int).tolist()

    A = float(pd.Series(comp_o).mean())
    B = float(pd.Series(comp_m).mean())
    d = weighted_delta(A, B)
    abs_change = abs(B - A)
    U_frac = float((pd.Series(comp_m) - pd.Series(comp_o)).abs().mean())
    U_strict = 100.0 * float((pd.Series(strict_m) - pd.Series(strict_o)).abs().mean())
    p, sig = significance_tests(comp_o, comp_m, strict_o, strict_m)

    row = {
        "task": "ifeval",
        "mod": args.mod,
        "model": args.model,
        "n": len(merged),
        "A": A,
        "B": B,
        "delta": d,
        "absolute_change": abs_change,
        "U_frac": U_frac,
        "U_strict": U_strict,
        "p_value": p,
        "significance": sig,
    }

    # Negation subtype handling: if dataset provided and modification is negation, compute flip accuracy
    if args.dataset and 'negation' in args.mod.lower():
        # Build key->subtype map
        subtype_map = {}
        try:
            import json
            with args.dataset.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    k = obj.get('key')
                    t = obj.get('type', '')
                    st = ''
                    if isinstance(t, str) and t.startswith('negation_'):
                        st = t.split('_', 1)[1]
                    subtype_map[k] = st
        except Exception:
            subtype_map = {}

        if subtype_map:
            merged['negation_subtype'] = merged['key'].map(subtype_map).fillna('')
            merged['expected_flip'] = ~merged['negation_subtype'].isin(['approximate', 'double'])
            # Correct if (mod != orig) when flip expected, else (mod == orig)
            so = merged['strict_success_orig'].astype(int)
            sm = merged['strict_success_mod'].astype(int)
            flip = merged['expected_flip']
            merged['negation_flip_correct'] = ((sm != so) & flip) | ((sm == so) & (~flip))
            row['negation_acc'] = float(merged['negation_flip_correct'].mean()) if len(merged) > 0 else float('nan')
            row['n_neg'] = int(len(merged))

    out = pd.DataFrame([row])

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved aggregate comparison to {args.out_csv}")


if __name__ == "__main__":
    main()
