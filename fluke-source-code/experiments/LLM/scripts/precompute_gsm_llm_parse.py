#!/usr/bin/env python3
"""
Precompute parsed GSM predictions using an LLM for non-DeepSeek, non-LLaMA models.
- Adds two columns to each CSV under ../results/gsm:
  - parsed_original_pred
  - parsed_modified_pred
- Requires OPENAI_API_KEY and USE_LLM_GSM_PARSE=1 to actually call the API.
  If not set, the script will skip files without existing parsed columns.
"""

import os
import re
import glob
from pathlib import Path
import time
import pandas as pd

try:
    from .llm_utils import llm_extract_number_from_texts, load_dotenv_if_present  # type: ignore
except Exception:
    try:
        from llm_utils import llm_extract_number_from_texts, load_dotenv_if_present  # type: ignore
    except Exception:
        llm_extract_number_from_texts = None
        def load_dotenv_if_present():
            pass

SCRIPT_DIR = Path(__file__).resolve().parent

def _map_model_from_filename(base: str) -> str:
    # rough mapping similar to gsm_analysis
    if base.startswith('gpt-5-standard-context-aware-'):
        return 'gpt-5-standard-context-aware'
    if base.startswith('gpt-5-standard-') or base.startswith('gpt-5-'):
        return 'gpt-5-standard'
    if base.startswith('gpt4o-') or base.startswith('gpt-4o-'):
        return 'gpt4o'
    if base.startswith('claude-'):
        return 'claude'
    if base.startswith('deepseek-') or base.startswith('deepseek-r1-'):
        return 'deepseek-r1'
    if base.startswith('llama-'):
        return 'llama'
    return base.split('-')[0]

def _looks_ok(v: str) -> bool:
    return bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\d+\s*/\s*\d+)", str(v)))

def parse_cell_with_llm(row: pd.Series, side: str) -> str | None:
    if llm_extract_number_from_texts is None:
        return None
    if not os.environ.get('USE_LLM_GSM_PARSE'):
        return None
    if side == 'original':
        texts = [
            str(row.get('original_step_by_step_reasoning', '')),
            str(row.get('original_raw_output', '')),
            str(row.get('original_reasoning', '')),
        ]
    else:
        texts = [
            str(row.get('modified_step_by_step_reasoning', row.get('step_by_step_reasoning', ''))),
            str(row.get('raw_output', '')),
            str(row.get('modified_reasoning', row.get('reasoning', ''))),
        ]
    val = llm_extract_number_from_texts(texts)
    return val

def parse_cell_with_simple_patterns(row: pd.Series, side: str) -> str | None:
    # Safe fallback: only trust explicit finals — #### or "final answer:"
    if side == 'original':
        candidates = [
            str(row.get('original_step_by_step_reasoning', '')),
            str(row.get('original_raw_output', '')),
            str(row.get('original_reasoning', '')),
        ]
    else:
        candidates = [
            str(row.get('modified_step_by_step_reasoning', row.get('step_by_step_reasoning', ''))),
            str(row.get('raw_output', '')),
            str(row.get('modified_reasoning', row.get('reasoning', ''))),
        ]
    for s in candidates:
        if not isinstance(s, str) or not s.strip():
            continue
        m = re.search(r"####\s*[$€£¥₹₽]?\s*([+-]?(?:\d+|\d{1,3}(?:,\d{3})*)(?:\.\d+)?(?:[eE][+-]?\d+)?|[+-]?\d+\s*/\s*\d+)", s)
        if m:
            return m.group(1).replace(',', '')
        m = re.search(r"(?i)(?:final\s+answer|answer)\s*[:\-]?\s*([+-]?(?:\d+|\d{1,3}(?:,\d{3})*)(?:\.\d+)?(?:[eE][+-]?\d+)?|[+-]?\d+\s*/\s*\d+)", s)
        if m:
            return m.group(1).replace(',', '')
        # As a last resort, scan the entire output and take the last numeric token
        nums = re.findall(r'[+-]?(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\d+\s*/\s*\d+)', s.replace(',', ''))
        if nums:
            return nums[-1]
    return None

def main():
    # Load .env and default USE_LLM_GSM_PARSE
    load_dotenv_if_present()
    results_dir = (SCRIPT_DIR / '../results/gsm').resolve()
    files = sorted(glob.glob(str(results_dir / '*.csv')))
    if not files:
        print(f"No files under {results_dir}")
        return

    # Identify targets (non-DeepSeek/LLaMA, not already parsed)
    targets: list[tuple[str, str]] = []  # (filepath, model)
    for fp in files:
        base = os.path.basename(fp)
        # Skip comparison and helper files
        if any(skip in base for skip in ['_comparison', '_backup', 'negation_change', 'DP']):
            continue
        model = _map_model_from_filename(base)
        # Only process non-DeepSeek/LLaMA
        if model not in ('deepseek-r1', 'llama'):
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                print(f"[WARN] Cannot read {base}: {e}")
                continue
            if 'parsed_original_pred' in df.columns and 'parsed_modified_pred' in df.columns:
                # Already parsed, skip
                continue
            targets.append((fp, model))

    total_files = len(targets)
    print(f"Found {len(files)} GSM files; parsing {total_files} with LLM (others skipped).")
    if total_files == 0:
        if llm_extract_number_from_texts is None or not os.environ.get('OPENAI_API_KEY'):
            print("Note: LLM parsing disabled — set OPENAI_API_KEY and ensure USE_LLM_GSM_PARSE=1 in .env")
        return
    if llm_extract_number_from_texts is None or not os.environ.get('OPENAI_API_KEY'):
        print("[INFO] LLM parsing appears disabled — will proceed with pattern-only fallback.")

    for idx, (fp, model) in enumerate(targets, 1):
        base = os.path.basename(fp)
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[{idx}/{total_files}] [SKIP] {base}: read error: {e}")
            continue
        n = len(df)
        print(f"[{idx}/{total_files}] Processing {base} (model={model}, rows={n}) …")
        t0 = time.time()
        parsed_op = [None] * n
        parsed_mp = [None] * n
        # Progress strategy: print every 10% or at least every 50 rows
        step = max(1, min(50, n // 10 or 1))
        filled_op = 0
        filled_mp = 0
        llm_hits_op = 0
        llm_hits_mp = 0
        for i in range(n):
            row = df.iloc[i]
            op = parse_cell_with_llm(row, 'original')
            mp = parse_cell_with_llm(row, 'modified')
            if op:
                llm_hits_op += 1
            if not op:
                op = parse_cell_with_simple_patterns(row, 'original')
            else:
                # If both present, prefer LLM; keep op
                pass
            if mp:
                llm_hits_mp += 1
            if not mp:
                mp = parse_cell_with_simple_patterns(row, 'modified')
            else:
                pass
            parsed_op[i] = op
            parsed_mp[i] = mp
            if op:
                filled_op += 1
            if mp:
                filled_mp += 1
            if (i + 1) % step == 0 or (i + 1) == n:
                pct = int(((i + 1) / n) * 100)
                print(f"  … {i+1}/{n} ({pct}%) parsed (orig {filled_op}, mod {filled_mp})", end='\r', flush=True)
        print()  # newline after progress
        df['parsed_original_pred'] = parsed_op
        df['parsed_modified_pred'] = parsed_mp
        # Write back with backup
        backup = fp + '.backup'
        try:
            if not os.path.exists(backup):
                os.rename(fp, backup)
        except Exception:
            pass
        df.to_csv(fp, index=False)
        dt = time.time() - t0
        print(f"[OK] Wrote parsed columns -> {base} in {dt:.1f}s (orig {filled_op}/{n}, mod {filled_mp}/{n}); LLM hits: orig {llm_hits_op}, mod {llm_hits_mp}")

if __name__ == '__main__':
    main()
