#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path


DEFAULT_MODELS = [
    "gpt4o",
    "claude-3-5-sonnet",
    "llama",
    "deepseek-r1",
    "gpt-5-standard",
    "gpt-5-standard-context-aware",
]


def run(cmd):
    print("→", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=False)


def main():
    ap = argparse.ArgumentParser(description="Run IFEval scoring and analysis for LLM models")
    ap.add_argument("--mod", default="length_bias", help="Modification name (default: length_bias)")
    ap.add_argument(
        "--dataset",
        type=Path,
        help="Path to <mod>_100.jsonl; defaults to ../../../data/modified_data/ifeval/<mod>_100.jsonl (relative to this script)",
    )
    ap.add_argument(
        "--models",
        nargs="*",
        help=f"List of model names. Default: {DEFAULT_MODELS}",
    )
    ap.add_argument(
        "--outputs_root",
        type=Path,
        default=Path("../results/ifeval"),
        help="Directory containing <MODEL>-0shot-<mod>_100.csv outputs",
    )
    ap.add_argument(
        "--scores_root",
        type=Path,
        default=Path("../results/ifeval_scores"),
        help="Root to write per-side scores",
    )
    ap.add_argument(
        "--aggregates_root",
        type=Path,
        default=Path("../results/ifeval_aggregates"),
        help="Root to write aggregate comparison CSVs",
    )
    args = ap.parse_args()

    mod = args.mod
    dataset = args.dataset or Path(f"../../../data/modified_data/ifeval/{mod}_100.jsonl")
    models = args.models if args.models else DEFAULT_MODELS

    if not dataset.exists():
        raise SystemExit(f"Dataset not found: {dataset}")

    produced = []
    for model in models:
        print(f"\n=== Model: {model} | Mod: {mod} ===")
        # Expect a single combined CSV like other OpenRouter scripts
        mod_stem = f"{mod}_100"
        outputs_csv = args.outputs_root / f"{model}-0shot-{mod_stem}.csv"
        if not outputs_csv.exists():
            print(f"Skipping {model}: missing outputs CSV. Expected:\n  {outputs_csv}")
            continue

        score_dir = args.scores_root / mod
        score_dir.mkdir(parents=True, exist_ok=True)
        orig_scores = score_dir / f"{model}_original.csv"
        mod_scores = score_dir / f"{model}_modified.csv"

        # Score original from combined CSV
        rc1 = run([
            "python", "ifeval_evaluate.py",
            "--dataset", str(dataset),
            "--outputs", str(outputs_csv),
            "--side", "original",
            "--out_csv", str(orig_scores),
        ])
        # Score modified from combined CSV
        rc2 = run([
            "python", "ifeval_evaluate.py",
            "--dataset", str(dataset),
            "--outputs", str(outputs_csv),
            "--side", "modified",
            "--out_csv", str(mod_scores),
        ])

        if rc1.returncode != 0 or rc2.returncode != 0:
            print(f"Scoring failed for {model}; skipping analysis.")
            continue

        agg_dir = args.aggregates_root / mod
        agg_dir.mkdir(parents=True, exist_ok=True)
        out_csv = agg_dir / f"{model}_comparison.csv"

        rc3 = run([
            "python", "ifeval_analysis.py",
            "--orig_csv", str(orig_scores),
            "--mod_csv", str(mod_scores),
            "--model", model,
            "--mod", mod,
            "--out_csv", str(out_csv),
            "--dataset", str(dataset),
        ])
        if rc3.returncode == 0:
            produced.append(out_csv)
        else:
            print(f"Analysis failed for {model}.")

    print("\nProduced aggregate files:")
    for p in produced:
        print(" -", p)
    print(f"Done. {len(produced)} models analyzed.")


if __name__ == "__main__":
    main()
