#!/usr/bin/env python3
"""
Run IFEval (instruction following) with OpenAI GPT-5 (context-aware).

Adds a lightweight modification context note for the modified prompt while
keeping the original prompt canonical. Supports caching (seeded from prior
outputs), --mod all, --analyze, and --max_tokens like the non-context script.
"""

import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from dotenv import load_dotenv

from fluke_gpt5_utils import GPT5Client, GPT5_CONFIGS, GPT5_MODELS

load_dotenv()


MODIFICATION_DESCRIPTIONS: Dict[str, str] = {
    'typo_bias': 'modified to contain typos',
    'capitalization': 'modified capitalization patterns',
    'punctuation': 'modified punctuation patterns',
    'negation': 'modified with added or removed negation',
    'sentiment': 'modified sentiment expressions',
    'active_to_passive': 'modified voice from active to passive',
    'casual': 'modified to be more casual/informal',
    'dialectal': 'modified to use dialectal variations',
    'compound_word': 'modified with compound word variations',
    'derivation': 'modified using derivational morphology',
    'concept_replacement': 'modified by replacing concepts with equivalents',
    'length_bias': 'modified to change text length or verbosity',
    'geographical_bias': 'modified to use different geographic contexts',
    'temporal_bias': 'modified to use words preferred in different time periods',
    'coordinating_conjunction': 'modified to include coordinating conjunctions',
    'grammatical_role': 'modified by changing grammatical roles',
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    dec = json.JSONDecoder()
    with path.open('r', encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                items.append(json.loads(s))
            except json.JSONDecodeError:
                try:
                    obj, end = dec.raw_decode(s)
                    items.append(obj)
                except Exception:
                    print(f"Warning: failed to parse {path.name} line {ln}")
    return items


def mod_from_filename(p: Path) -> str:
    stem = p.stem
    return stem[:-4] if stem.endswith('_100') else stem


def build_prompts(header: str, orig: str, mod: str, mod_key: str) -> (str, str):
    # Original: canonical/header only
    p0 = (header or '') + orig
    # Modified: include a short context header, then the modified instruction
    desc = MODIFICATION_DESCRIPTIONS.get(mod_key, None)
    if desc:
        ctx = f"Note: This instruction was {desc}. You must still follow every constraint exactly.\n\n"
    else:
        ctx = ""
    p1 = (header or '') + ctx + mod
    return p0, p1


def generate_pair_outputs(
    client: GPT5Client,
    dataset_path: Path,
    header: str,
    mod_key: str,
    reasoning_effort: str,
    prediction_cache: Optional[Dict[str, Dict[str, str]]] = None,
    max_tokens: int = 4096,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    items = load_jsonl(dataset_path)
    total = len(items)
    for i, obj in enumerate(items):
        print(f"Processing {i+1}/{total}", end='\r')
        key = obj.get('key')
        orig = obj.get('text') or ''
        mod = obj.get('modified') or ''
        p0, p1 = build_prompts(header, orig, mod, mod_key)

        # Original
        if prediction_cache is not None and p0 in prediction_cache and str(prediction_cache[p0].get('content', '')).strip().lower() not in ('', 'nan', 'none', 'null'):
            r0 = prediction_cache[p0]
        else:
            r0 = client.generate(p0, max_tokens=max_tokens, reasoning_effort=reasoning_effort)
            if prediction_cache is not None:
                prediction_cache[p0] = r0
            time.sleep(0.1)

        # Modified with context
        if prediction_cache is not None and p1 in prediction_cache and str(prediction_cache[p1].get('content', '')).strip().lower() not in ('', 'nan', 'none', 'null'):
            r1 = prediction_cache[p1]
        else:
            r1 = client.generate(p1, max_tokens=max_tokens, reasoning_effort=reasoning_effort)
            if prediction_cache is not None:
                prediction_cache[p1] = r1
            time.sleep(0.1)

        rows.append({
            'key': key,
            'original_text': orig,
            'text': mod,
            'original_raw_output': r0.get('content', ''),
            'original_reasoning': r0.get('reasoning', ''),
            'raw_output': r1.get('content', ''),
            'reasoning': r1.get('reasoning', ''),
        })
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


def main():
    ap = argparse.ArgumentParser(description='Run IFEval (GPT-5) with context-aware modified prompts')
    ap.add_argument('--mod', default='all', help="Modification name (e.g., length_bias). Use 'all' to process all *.jsonl in modified_data/ifeval")
    ap.add_argument('--dataset', type=Path, help='Explicit dataset path; overrides --mod')
    ap.add_argument('--outputs_root', type=Path, default=Path('../results/ifeval'))
    ap.add_argument('--scores_root', type=Path, default=Path('../results/ifeval_scores'))
    ap.add_argument('--aggregates_root', type=Path, default=Path('../results/ifeval_aggregates'))
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--max_tokens', type=int, default=4096)
    ap.add_argument('--force_regenerate', action='store_true')
    ap.add_argument('--config', default='standard', choices=list(GPT5_CONFIGS.keys()))
    ap.add_argument('--enable_reasoning', action='store_true', help='Enable GPT-5 reasoning effort from config (default: disabled/minimal)')
    args = ap.parse_args()

    # Model + effort
    cfg = GPT5_CONFIGS[args.config]
    model_name = cfg['model']
    model_id = GPT5_MODELS[model_name]
    reasoning_effort = cfg.get('reasoning_effort', 'medium') if args.enable_reasoning else 'minimal'

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise SystemExit('OPENAI_API_KEY not set')
    client = GPT5Client(api_key, model_id)

    def run_single(dataset_path: Path):
        if not dataset_path.exists():
            raise SystemExit(f'Dataset not found: {dataset_path}')
        mod_key = mod_from_filename(dataset_path)

        # Header is empty by default; context is injected only on modified side
        header = ''

        # Outputs path
        out_dir = args.outputs_root
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{model_name}-context-aware-0shot-{dataset_path.stem}.csv"
        print(f"Writing outputs to: {out_csv}")

        # Seed cache from existing outputs if allowed
        prediction_cache: Dict[str, Dict[str, str]] = {}
        if out_csv.exists() and not args.force_regenerate:
            try:
                df_prev = pd.read_csv(out_csv)
                # original
                if {'original_text', 'original_raw_output'}.issubset(df_prev.columns):
                    for _, row in df_prev.iterrows():
                        p = (header or '') + str(row.get('original_text', ''))
                        c = row.get('original_raw_output', '')
                        r = row.get('original_reasoning', '')
                        if p.strip() and str(c).strip().lower() not in ('', 'nan', 'none', 'null'):
                            prediction_cache[p] = {'content': str(c), 'reasoning': str(r) if isinstance(r, str) else ''}
                # modified (with context)
                if {'text', 'raw_output'}.issubset(df_prev.columns):
                    for _, row in df_prev.iterrows():
                        # Build context header for keying
                        desc = MODIFICATION_DESCRIPTIONS.get(mod_key, None)
                        ctx = f"Note: This instruction was {desc}. You must still follow every constraint exactly.\n\n" if desc else ''
                        p = (header or '') + ctx + str(row.get('text', ''))
                        c = row.get('raw_output', '')
                        r = row.get('reasoning', '')
                        if p.strip() and str(c).strip().lower() not in ('', 'nan', 'none', 'null'):
                            prediction_cache[p] = {'content': str(c), 'reasoning': str(r) if isinstance(r, str) else ''}
                print(f"Seeded cache with {len(prediction_cache)} items from existing outputs")
            except Exception as e:
                print(f"Warning: failed to seed cache: {e}")

        # Generate
        rows = generate_pair_outputs(
            client,
            dataset_path,
            header,
            mod_key,
            reasoning_effort,
            prediction_cache=prediction_cache,
            max_tokens=args.max_tokens,
        )
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved to: {out_csv}")

        if args.analyze:
            maybe_analyze(dataset_path, out_csv, args.scores_root, args.aggregates_root, model_name, mod_key)

    # Dispatch
    if args.mod.lower() == 'all' and not args.dataset:
        root = Path('../../../data/modified_data/ifeval')
        files = sorted([p for p in root.glob('*.jsonl')])
        if not files:
            raise SystemExit(f'No datasets in {root}')
        processed = []
        for p in files:
            print(f"\n=== Context-aware IFEval: {p.name} ===")
            run_single(p)
            processed.append(mod_from_filename(p))
        print(f"\nProcessed modifications ({len(set(processed))}): {', '.join(sorted(set(processed)))}")
        if args.analyze:
            try:
                print("\nRunning sanity report...")
                os.system('python sanity_report.py')
            except Exception as e:
                print(f"Warning: sanity report failed: {e}")
    else:
        dataset = args.dataset or Path(f"../../../data/modified_data/ifeval/{args.mod}_100.jsonl")
        run_single(dataset)
        if args.analyze:
            try:
                print("\nRunning sanity report...")
                os.system('python sanity_report.py')
            except Exception as e:
                print(f"Warning: sanity report failed: {e}")


if __name__ == '__main__':
    main()

