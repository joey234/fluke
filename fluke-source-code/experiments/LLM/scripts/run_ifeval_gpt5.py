#!/usr/bin/env python3
"""
Run IFEval (instruction following) with OpenAI GPT-5 models (direct API, no DSPy).

Supports running a single modification or all modifications under
fluke-source-code/data/modified_data/ifeval. Captures reasoning when available
and writes combined CSVs compatible with ifeval_evaluate.py.
"""

import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from dotenv import load_dotenv

# FLUKE GPT-5 utilities
from fluke_gpt5_utils import GPT5Client, GPT5_CONFIGS, GPT5_MODELS

load_dotenv()


DEFAULT_HEADER = (
    "Follow the instruction below EXACTLY as written. "
    "Satisfy every constraint and formatting requirement. "
    "Do not add extra commentary unless explicitly asked.\n\n"
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def generate_pair_outputs(
    client: GPT5Client,
    dataset_path: Path,
    header: str,
    reasoning_effort: str,
    prediction_cache: Optional[Dict[str, Dict[str, str]]] = None,
    max_tokens: int = 4096,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    items = load_jsonl(dataset_path)
    total = len(items)
    for i, obj in enumerate(items):
        print(f"Processing {i+1}/{total}", end="\r")
        key = obj.get("key")
        orig = obj.get("text") or ""
        mod = obj.get("modified") or ""
        p0 = (header or "") + orig
        p1 = (header or "") + mod
        if prediction_cache is not None and p0 in prediction_cache and str(prediction_cache[p0].get("content", "")).strip().lower() not in ("", "nan", "none", "null"):
            r0 = prediction_cache[p0]
        else:
            r0 = client.generate(p0, max_tokens=max_tokens, reasoning_effort=reasoning_effort)
            if prediction_cache is not None:
                prediction_cache[p0] = r0
            time.sleep(0.1)
        if prediction_cache is not None and p1 in prediction_cache and str(prediction_cache[p1].get("content", "")).strip().lower() not in ("", "nan", "none", "null"):
            r1 = prediction_cache[p1]
        else:
            r1 = client.generate(p1, max_tokens=max_tokens, reasoning_effort=reasoning_effort)
            if prediction_cache is not None:
                prediction_cache[p1] = r1
            time.sleep(0.1)
        rows.append(
            {
                "key": key,
                "original_text": orig,
                "text": mod,  # keep parity with OpenRouter script
                "original_raw_output": r0.get("content", ""),
                "original_reasoning": r0.get("reasoning", ""),
                "raw_output": r1.get("content", ""),
                "reasoning": r1.get("reasoning", ""),
            }
        )
    print()
    return rows


def maybe_analyze(dataset_path: Path, outputs_csv: Path, scores_root: Path, aggregates_root: Path, model_name: str, mod: str):
    orig_scores = scores_root / mod / f"{model_name}_original.csv"
    mod_scores = scores_root / mod / f"{model_name}_modified.csv"
    orig_scores.parent.mkdir(parents=True, exist_ok=True)

    os.system(
        f"python ifeval_evaluate.py --dataset {dataset_path} --outputs {outputs_csv} --side original --out_csv {orig_scores}"
    )
    os.system(
        f"python ifeval_evaluate.py --dataset {dataset_path} --outputs {outputs_csv} --side modified --out_csv {mod_scores}"
    )

    out_csv = aggregates_root / mod / f"{model_name}_comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    os.system(
        f"python ifeval_analysis.py --orig_csv {orig_scores} --mod_csv {mod_scores} --model {model_name} --mod {mod} --out_csv {out_csv} --dataset {dataset_path}"
    )


def run_single(dataset_path: Path, out_dir: Path, model_name: str, client: GPT5Client, reasoning_effort: str, prompt_mode: str, prompt_file: Path, analyze: bool, scores_root: Path, aggregates_root: Path, mod_for_agg: str, force_regenerate: bool = False, max_tokens: int = 4096):
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    # Resolve header
    if prompt_mode == "canonical":
        header = ""
    else:
        if prompt_file and prompt_file.exists():
            header = prompt_file.read_text(encoding="utf-8")
        else:
            header = DEFAULT_HEADER
    print(f"Prompt mode: {prompt_mode}{' (custom file)' if prompt_file else ''}")

    mod_stem = dataset_path.name.replace(".jsonl", "").replace(".json", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{model_name}-0shot-{mod_stem}.csv"
    print(f"Writing outputs to: {out_csv}")

    # Small preview
    try:
        items = load_jsonl(dataset_path)
        if items:
            demo = items[0]
            demo_orig = (demo.get("text") or "").strip()
            demo_mod = (demo.get("modified") or "").strip()
            print("\nDemo sample (original):")
            print(demo_orig[:400] + ("..." if len(demo_orig) > 400 else ""))
            r0 = client.generate((header or "") + demo_orig, max_tokens=max_tokens, reasoning_effort=reasoning_effort)
            print("Original output (truncated):")
            print((r0.get("content", "")[:400] + ("..." if len(r0.get("content", "")) > 400 else "")))
            print(f"Reasoning available: {bool(r0.get('reasoning'))}")

            print("\nDemo sample (modified):")
            print(demo_mod[:400] + ("..." if len(demo_mod) > 400 else ""))
            r1 = client.generate((header or "") + demo_mod, max_tokens=max_tokens, reasoning_effort=reasoning_effort)
            print("Modified output (truncated):")
            print((r1.get("content", "")[:400] + ("..." if len(r1.get("content", "")) > 400 else "")))
            print(f"Reasoning available: {bool(r1.get('reasoning'))}")
    except Exception as e:
        print(f"Demo preview failed: {e}")

    # Seed cache from existing outputs if available
    prediction_cache: Dict[str, Dict[str, str]] = {}
    if out_csv.exists() and not force_regenerate:
        try:
            df_prev = pd.read_csv(out_csv)
            if {"original_text", "original_raw_output"}.issubset(df_prev.columns):
                for _, row in df_prev.iterrows():
                    p = (header or "") + str(row.get("original_text", ""))
                    c = row.get("original_raw_output", "")
                    # Reasoning may be NaN for IFEval; normalize to empty string
                    r = ""
                    if "original_reasoning" in df_prev.columns:
                        v = row.get("original_reasoning", "")
                        r = "" if (str(v).strip().lower() in ("", "nan", "none", "null")) else str(v)
                    if p.strip() and str(c).strip().lower() not in ("", "nan", "none", "null"):
                        prediction_cache[p] = {"content": str(c), "reasoning": r}
            if {"text", "raw_output"}.issubset(df_prev.columns):
                for _, row in df_prev.iterrows():
                    p = (header or "") + str(row.get("text", ""))
                    c = row.get("raw_output", "")
                    r = ""
                    if "reasoning" in df_prev.columns:
                        v = row.get("reasoning", "")
                        r = "" if (str(v).strip().lower() in ("", "nan", "none", "null")) else str(v)
                    if p.strip() and str(c).strip().lower() not in ("", "nan", "none", "null"):
                        prediction_cache[p] = {"content": str(c), "reasoning": r}
            print(f"Seeded cache with {len(prediction_cache)} items from existing outputs")
        except Exception as e:
            print(f"Warning: failed to load existing outputs for cache seeding: {e}")

    rows = generate_pair_outputs(client, dataset_path, header, reasoning_effort, prediction_cache=prediction_cache, max_tokens=max_tokens)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved to: {out_csv}")

    if analyze:
        maybe_analyze(dataset_path, out_csv, scores_root, aggregates_root, model_name, mod_for_agg)


def main():
    ap = argparse.ArgumentParser(description="Run IFEval with GPT-5 models (OpenAI API)")
    ap.add_argument(
        "--mod",
        default="all",
        help="Modification name (e.g., length_bias). Use 'all' to run every file under data/modified_data/ifeval",
    )
    ap.add_argument(
        "--dataset",
        type=Path,
        help="Path to a single dataset file. If omitted and --mod is given, defaults to ../../../data/modified_data/ifeval/<mod>_100.jsonl. If --mod=all, iterates over all *.jsonl in that directory.",
    )
    ap.add_argument("--outputs_root", type=Path, default=Path("../results/ifeval"))
    ap.add_argument("--scores_root", type=Path, default=Path("../results/ifeval_scores"))
    ap.add_argument("--aggregates_root", type=Path, default=Path("../results/ifeval_aggregates"))
    ap.add_argument("--analyze", action="store_true", help="After generation, run scoring + analysis")
    ap.add_argument("--prompt_mode", choices=["canonical", "header"], default="canonical")
    ap.add_argument("--prompt_file", type=Path)
    ap.add_argument(
        "--config",
        default="standard",
        choices=list(GPT5_CONFIGS.keys()),
        help="GPT-5 config variant controlling model + reasoning",
    )
    ap.add_argument("--enable_reasoning", action="store_true", help="Enable GPT-5 reasoning effort from config (default: disabled for IFEval)")
    ap.add_argument("--force_regenerate", action="store_true", help="Ignore existing outputs CSV and do not seed cache from it")
    ap.add_argument("--max_tokens", type=int, default=4096, help="Max output tokens for model responses (default: 4096)")

    args = ap.parse_args()

    # API and model
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set. Put it in your environment or .env file.")

    config = GPT5_CONFIGS[args.config]
    model_name = config["model"]
    model_id = GPT5_MODELS[model_name]
    # IFEval: disable reasoning by default (minimal). Allow enabling via flag.
    reasoning_effort = "minimal" if not args.enable_reasoning else config.get("reasoning_effort", "medium")
    client = GPT5Client(api_key, model_id)

    print(f"Model: {model_name} ({model_id}) | Reasoning effort: {reasoning_effort}")

    if args.mod and args.mod.lower() == "all" and not args.dataset:
        root = Path("../../../data/modified_data/ifeval")
        if not root.exists():
            raise SystemExit(f"Modified data root not found: {root}")
        files = sorted([p for p in root.glob("*.jsonl")])
        if not files:
            raise SystemExit(f"No .jsonl files found in {root}")
        processed_mods = []
        for p in files:
            mod_name = p.stem
            mod_for_agg = mod_name[:-4] if mod_name.endswith("_100") else mod_name
            run_single(
                dataset_path=p,
                out_dir=args.outputs_root,
                model_name=model_name,
                client=client,
                reasoning_effort=reasoning_effort,
                prompt_mode=args.prompt_mode,
                prompt_file=args.prompt_file,
                analyze=args.analyze,
                scores_root=args.scores_root,
                aggregates_root=args.aggregates_root,
                mod_for_agg=mod_for_agg,
                force_regenerate=args.force_regenerate,
                max_tokens=args.max_tokens,
            )
            processed_mods.append(mod_for_agg)
        try:
            uniq = sorted(set(processed_mods))
            print(f"\nProcessed modifications ({len(uniq)}): {', '.join(uniq)}")
            print(f"Model: {model_name}")
        except Exception:
            pass
        # Auto sanity report after analysis
        if args.analyze:
            try:
                print("\nRunning sanity report...")
                os.system("python sanity_report.py")
            except Exception as e:
                print(f"Warning: sanity report failed: {e}")
        print("Done.")
        return

    dataset = args.dataset or Path(f"../../../data/modified_data/ifeval/{args.mod}_100.jsonl")
    mod_for_agg = args.mod
    run_single(
        dataset_path=dataset,
        out_dir=args.outputs_root,
        model_name=model_name,
        client=client,
        reasoning_effort=reasoning_effort,
        prompt_mode=args.prompt_mode,
        prompt_file=args.prompt_file,
        analyze=args.analyze,
        scores_root=args.scores_root,
        aggregates_root=args.aggregates_root,
        mod_for_agg=mod_for_agg,
        force_regenerate=args.force_regenerate,
        max_tokens=args.max_tokens,
    )
    if args.analyze:
        try:
            print("\nRunning sanity report...")
            os.system("python sanity_report.py")
        except Exception as e:
            print(f"Warning: sanity report failed: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
