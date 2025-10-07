#!/usr/bin/env python3
"""
Run IFEval (instruction following) on OpenRouter models, handling reasoning models differently.

Generates original and modified outputs for a given modification file (length_bias) and
optionally runs the scorer and analysis to produce comparison CSVs.
"""

import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

import requests
from dotenv import load_dotenv

# Load environment variables from .env if present (same as GSM script)
load_dotenv()


# Common presets for convenience (override with --model_id as needed)
MODEL_PRESETS = {
    # Reasoning-capable
    "deepseek-r1": {"id": "deepseek/deepseek-r1", "reasoning": True},
    "o1-preview": {"id": "openai/o1-preview", "reasoning": True},
    "o3-2025-04-16": {"id": "openai/o3-2025-04-16", "reasoning": True},
    # Non-reasoning (standard chat)
    "gpt4o": {"id": "openai/gpt-4o", "reasoning": False},
    "gpt-5": {"id": "openai/gpt-5", "reasoning": False},
    "claude-3-5-sonnet": {"id": "anthropic/claude-3.5-sonnet", "reasoning": False},
    "llama": {"id": "meta-llama/llama-3.1-70b-instruct", "reasoning": False},
}


class OpenRouterClient:
    def __init__(self, api_key: str, model_id: str, use_reasoning: bool = False, reasoning_effort: str = "medium"):
        self.api_key = api_key
        self.model_id = model_id
        self.use_reasoning = use_reasoning
        self.reasoning_effort = reasoning_effort
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.2) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body: Dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Provider-specific reasoning controls
        if self.use_reasoning:
            low = self.model_id.lower()
            if "deepseek" in low:
                body["include_reasoning"] = True
            elif "o1" in low:
                body["reasoning"] = {"effort": "high", "max_tokens": 2000, "exclude": False}
            else:
                # Generic hint that some providers use
                body["reasoning"] = {"effort": self.reasoning_effort}

        # Retry on transient network/provider errors (e.g., "Response ended prematurely")
        for attempt in range(3):
            try:
                resp = requests.post(self.base_url, headers=headers, json=body, timeout=120)
                resp.raise_for_status()
                try:
                    data = resp.json()
                except json.JSONDecodeError as je:
                    # Treat invalid JSON as transient; retry
                    msg = f"JSON decode error: {je} | body preview: {resp.text[:200]}"
                    if attempt < 2:
                        time.sleep(1 + attempt)
                        continue
                    print(f"OpenRouter API error: {msg}")
                    return {"content": "", "reasoning": ""}
            except Exception as e:
                msg = str(e)
                if attempt < 2 and ("premature" in msg.lower() or "chunked" in msg.lower() or "timeout" in msg.lower() or isinstance(e, (requests.exceptions.ChunkedEncodingError, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError))):
                    time.sleep(1 + attempt)
                    continue
                print(f"OpenRouter API error: {e}")
                return {"content": "", "reasoning": ""}
            content = ""
            reasoning = ""
            if data.get("choices"):
                choice = data["choices"][0]
                message = choice.get("message", {})
                content = message.get("content", "")
                if self.use_reasoning:
                    if "reasoning" in message:
                        reasoning = message.get("reasoning", "")
                    elif "reasoning" in choice:
                        r = choice.get("reasoning", {})
                        if isinstance(r, dict):
                            reasoning = r.get("content", "")
                        else:
                            reasoning = str(r)
                    elif "thinking" in choice:
                        reasoning = choice.get("thinking", "")
            return {"content": content or "", "reasoning": reasoning or ""}
        # Should not reach here
        return {"content": "", "reasoning": ""}


DEFAULT_HEADER = (
    "Follow the instruction below EXACTLY as written. "
    "Satisfy every constraint and formatting requirement. "
    "Do not add extra commentary unless explicitly asked.\n\n"
)

def load_jsonl_objs(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    dec = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                items.append(json.loads(s))
            except json.JSONDecodeError as e:
                # Attempt to salvage the first valid JSON object on the line
                try:
                    obj, end = dec.raw_decode(s)
                    items.append(obj)
                    # Optionally warn if trailing junk exists
                    if any(ch.strip() for ch in s[end:]):
                        print(f"Warning: trailing data ignored in {path.name} at line {ln}")
                except Exception:
                    print(f"Warning: failed to parse {path.name} line {ln}: {e}")
                    continue
    return items

def _is_valid_content(val: Any) -> bool:
    try:
        import pandas as pd  # type: ignore
        if pd.isna(val):
            return False
    except Exception:
        pass
    s = str(val).strip()
    if not s:
        return False
    if s.lower() in {"nan", "none", "null"}:
        return False
    # Require minimal length to avoid junk
    return len(s) >= 2

def _norm_reason(val: Any) -> str:
    try:
        import pandas as pd  # type: ignore
        if pd.isna(val):
            return ""
    except Exception:
        pass
    s = str(val).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def generate_pair_outputs(
    client: OpenRouterClient,
    dataset_path: Path,
    prompt_header: str,
    rate_limit_s: float = 0.2,
    prediction_cache: Optional[Dict[str, Dict[str, str]]] = None,
    max_tokens: int = 4096,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    items = load_jsonl_objs(dataset_path)
    total = len(items)
    for i, obj in enumerate(items):
        print(f"Processing {i+1}/{total}", end='\r')
        key = obj.get("key")
        orig = obj.get("text") or ""
        mod = obj.get("modified") or ""
        # Compose prompts
        p0 = (prompt_header or "") + orig
        p1 = (prompt_header or "") + mod
        # Original with cache
        if prediction_cache is not None and p0 in prediction_cache and _is_valid_content(prediction_cache[p0].get("content")):
            r0 = prediction_cache[p0]
        else:
            r0 = client.generate(p0, max_tokens=max_tokens)
            if prediction_cache is not None:
                prediction_cache[p0] = r0
            time.sleep(rate_limit_s)
        # Modified with cache
        if prediction_cache is not None and p1 in prediction_cache and _is_valid_content(prediction_cache[p1].get("content")):
            r1 = prediction_cache[p1]
        else:
            r1 = client.generate(p1, max_tokens=max_tokens)
            if prediction_cache is not None:
                prediction_cache[p1] = r1
            time.sleep(rate_limit_s)
        rows.append({
            "key": key,
            "original_text": orig,
            "text": mod,  # follow other tasks: 'text' refers to modified
            "original_raw_output": r0.get("content", ""),
            "original_reasoning": r0.get("reasoning", ""),
            "raw_output": r1.get("content", ""),
            "reasoning": r1.get("reasoning", ""),
        })
        time.sleep(rate_limit_s)
    print()
    return rows


def maybe_analyze(dataset_path: Path, outputs_csv: Path, scores_root: Path, aggregates_root: Path, model_name: str, mod: str):
    # Score both sides from the combined CSV
    orig_scores = scores_root / mod / f"{model_name}_original.csv"
    mod_scores = scores_root / mod / f"{model_name}_modified.csv"
    orig_scores.parent.mkdir(parents=True, exist_ok=True)

    os.system(
        f"python ifeval_evaluate.py --dataset {dataset_path} --outputs {outputs_csv} --side original --out_csv {orig_scores}"
    )
    os.system(
        f"python ifeval_evaluate.py --dataset {dataset_path} --outputs {outputs_csv} --side modified --out_csv {mod_scores}"
    )

    # Aggregate
    out_csv = aggregates_root / mod / f"{model_name}_comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    os.system(
        f"python ifeval_analysis.py --orig_csv {orig_scores} --mod_csv {mod_scores} --model {model_name} --mod {mod} --out_csv {out_csv} --dataset {dataset_path}"
    )


def main():
    ap = argparse.ArgumentParser(description="Run IFEval on OpenRouter models (handles reasoning models)")
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
    ap.add_argument("--model", help=f"Model preset name: one of {list(MODEL_PRESETS.keys())}")
    ap.add_argument("--model_id", help="Override with a raw OpenRouter model id (e.g., openai/gpt-4o)")
    ap.add_argument("--reasoning", action="store_true", help="Force reasoning mode for the client")
    ap.add_argument("--no-reasoning", dest="no_reasoning", action="store_true", help="Force disable reasoning mode")
    ap.add_argument("--outputs_root", type=Path, default=Path("../results/ifeval"))
    ap.add_argument("--scores_root", type=Path, default=Path("../results/ifeval_scores"))
    ap.add_argument("--aggregates_root", type=Path, default=Path("../results/ifeval_aggregates"))
    ap.add_argument("--analyze", action="store_true", help="After generation, run scoring + analysis")
    ap.add_argument("--force_regenerate", action="store_true", help="Ignore existing outputs CSV and do not seed cache from it")
    ap.add_argument("--prompt_mode", choices=["canonical", "header"], default="canonical", help="canonical: pass instruction verbatim; header: prepend a header")
    ap.add_argument("--prompt_file", type=Path, help="Optional path to a custom header file (used when --prompt_mode header)")
    ap.add_argument("--max_tokens", type=int, default=4096, help="Max output tokens for model responses (default: 4096)")
    args = ap.parse_args()

    def run_single(dataset_path: Path, mod_name: str):
        nonlocal use_reasoning, model_name, client

        if not dataset_path.exists():
            raise SystemExit(f"Dataset not found: {dataset_path}")

        print(f"\n=== Running IFEval for mod '{mod_name}' on {model_name} ===")

        # Determine prompt header
        if args.prompt_mode == "canonical":
            prompt_header = ""
        else:
            if args.prompt_file and args.prompt_file.exists():
                prompt_header = args.prompt_file.read_text(encoding="utf-8")
            else:
                prompt_header = DEFAULT_HEADER
        print(f"Prompt mode: {args.prompt_mode}{' (custom file)' if args.prompt_file else ''}")

        # Determine output file
        mod_stem_local = dataset_path.name.replace(".jsonl", "").replace(".json", "")
        out_dir = args.outputs_root
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{model_name}-0shot-{mod_stem_local}.csv"
        print(f"Writing outputs to: {out_csv}")

        # Seed cache from existing outputs if available
        prediction_cache: Dict[str, Dict[str, str]] = {}
        if out_csv.exists() and not args.force_regenerate:
            try:
                import pandas as pd
                df_prev = pd.read_csv(out_csv)
                # Seed original side
                if {"original_text", "original_raw_output"}.issubset(df_prev.columns):
                    for _, row in df_prev.iterrows():
                        p = (prompt_header or "") + str(row.get("original_text", ""))
                        c = row.get("original_raw_output", "")
                        r = _norm_reason(row.get("original_reasoning", "")) if "original_reasoning" in df_prev.columns else ""
                        if p.strip() and _is_valid_content(c):
                            prediction_cache[p] = {"content": str(c), "reasoning": r}
                # Seed modified side
                if {"text", "raw_output"}.issubset(df_prev.columns):
                    for _, row in df_prev.iterrows():
                        p = (prompt_header or "") + str(row.get("text", ""))
                        c = row.get("raw_output", "")
                        r = _norm_reason(row.get("reasoning", "")) if "reasoning" in df_prev.columns else ""
                        if p.strip() and _is_valid_content(c):
                            prediction_cache[p] = {"content": str(c), "reasoning": r}
                print(f"Seeded cache with {len(prediction_cache)} items from existing outputs")
            except Exception as e:
                print(f"Warning: failed to load existing outputs for cache seeding: {e}")

        # Preview first sample
        try:
            items_local = load_jsonl_objs(dataset_path)
            if items_local:
                demo = items_local[0]
                demo_orig = (demo.get("text") or "").strip()
                demo_mod = (demo.get("modified") or "").strip()
                print("\nDemo sample (original):")
                print(demo_orig[:400] + ("..." if len(demo_orig) > 400 else ""))
                r0 = client.generate(demo_orig if args.prompt_mode == "canonical" else (prompt_header + demo_orig), max_tokens=args.max_tokens)
                print("Original output (truncated):")
                o_content = r0.get("content", "")
                print((o_content[:400] + ("..." if len(o_content) > 400 else "")))
                print(f"Reasoning available: {bool(r0.get('reasoning'))}")

                print("\nDemo sample (modified):")
                print(demo_mod[:400] + ("..." if len(demo_mod) > 400 else ""))
                r1 = client.generate(demo_mod if args.prompt_mode == "canonical" else (prompt_header + demo_mod), max_tokens=args.max_tokens)
                print("Modified output (truncated):")
                m_content = r1.get("content", "")
                print((m_content[:400] + ("..." if len(m_content) > 400 else "")))
                print(f"Reasoning available: {bool(r1.get('reasoning'))}")
        except Exception as e:
            print(f"Demo preview failed: {e}")

        # Generate and save combined CSV with both sides
        rows = generate_pair_outputs(
            client,
            dataset_path,
            prompt_header if args.prompt_mode == "header" else "",
            prediction_cache=prediction_cache,
            max_tokens=args.max_tokens,
        )
        import pandas as pd
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved to: {out_csv}")

        if args.analyze:
            maybe_analyze(dataset_path, out_csv, args.scores_root, args.aggregates_root, model_name, mod_name)

    # Resolve model id and reasoning flag
    model_name = None
    use_reasoning: Optional[bool] = None
    if args.model_id:
        model_id = args.model_id
        model_name = model_id.split("/")[-1]
    else:
        if not args.model:
            raise SystemExit("Please provide --model preset or --model_id")
        if args.model not in MODEL_PRESETS:
            raise SystemExit(f"Unknown model preset: {args.model}")
        preset = MODEL_PRESETS[args.model]
        model_id = preset["id"]
        model_name = args.model
        use_reasoning = preset.get("reasoning", False)

    # Override reasoning flags if requested
    if args.reasoning:
        use_reasoning = True
    if args.no_reasoning:
        use_reasoning = False
    if use_reasoning is None:
        use_reasoning = False
    # IFEval: force reasoning off by default unless explicitly enabled
    if not args.reasoning:
        use_reasoning = False

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set. Put it in your environment or .env file.")

    client = OpenRouterClient(api_key, model_id, use_reasoning=use_reasoning, reasoning_effort="medium")

    print(f"Model: {model_name} ({model_id}) | Reasoning: {use_reasoning}")
    # Determine run mode: single mod or all
    if args.mod and args.mod.lower() == "all" and not args.dataset:
        root = Path("../../../data/modified_data/ifeval")
        if not root.exists():
            raise SystemExit(f"Modified data root not found: {root}")
        files = sorted([p for p in root.glob("*.jsonl")])
        if not files:
            raise SystemExit(f"No .jsonl files found in {root}")
        processed_mods = []
        for p in files:
            mod_name = p.stem  # e.g., length_bias_100
            # Strip trailing _100 for aggregation grouping
            if mod_name.endswith("_100"):
                mod_for_agg = mod_name[:-4]
            else:
                mod_for_agg = mod_name
            run_single(p, mod_for_agg)
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

    # Single dataset path (either explicit or derived from --mod)
    dataset = args.dataset or Path(f"../../../data/modified_data/ifeval/{args.mod}_100.jsonl")
    run_single(dataset, args.mod)
    if args.analyze:
        try:
            print("\nRunning sanity report...")
            os.system("python sanity_report.py")
        except Exception as e:
            print(f"Warning: sanity report failed: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
