#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from ifeval_checkers import check_constraint
import re


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
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
                # Attempt to salvage the first JSON object and ignore trailing garbage
                try:
                    obj, end = dec.raw_decode(s)
                    items.append(obj)
                    if any(ch.strip() for ch in s[end:]):
                        print(f"Warning: trailing data ignored in {path.name} at line {ln}")
                except Exception:
                    print(f"Warning: failed to parse {path.name} line {ln}: {e}")
                    continue
    return items


def load_outputs_map(path: Path, side: str) -> Dict[Any, str]:
    """Load model outputs JSONL into a dict key->output_text.

    Expected fields per line: {"key": <id>, "output_text": <str>}.
    Also supports {"id": <id>, "response": <str>} as a fallback.
    """
    outputs: Dict[Any, str] = {}
    if not path.exists():
        return outputs
    # Support either JSONL (key/output_text) or CSV with raw_output columns
    if path.suffix.lower() == ".jsonl":
        for obj in load_jsonl(path):
            k = obj.get("key", obj.get("id"))
            text = obj.get("output_text", obj.get("response", obj.get("prediction")))
            if k is None:
                continue
            if text is None:
                text = ""
            outputs.setdefault(k, str(text))
    else:
        # Assume CSV produced by run_ifeval_openrouter.py
        df = pd.read_csv(path)
        text_col = "original_raw_output" if side == "original" else "raw_output"
        key_col = "key" if "key" in df.columns else None
        if key_col is None:
            # Derive key as row index if absent
            for i, row in df.iterrows():
                outputs.setdefault(i, str(row.get(text_col, "")))
        else:
            for _, row in df.iterrows():
                k = row.get(key_col)
                outputs.setdefault(k, str(row.get(text_col, "")))
    return outputs


def extract_eval_text(text: str) -> str:
    """Extract the main prediction segment from model output.
    Prefer fenced code blocks (```...```). If multiple, choose the longest.
    Fallback to full text if none.
    """
    if not text:
        return ""
    # Find fenced code blocks
    blocks = re.findall(r"```(?:\w+)?\n([\s\S]*?)```", text)
    if blocks:
        return max((b.strip() for b in blocks), key=len, default=text)
    return text


def score_sample(output_text: str, instruction_ids: List[str], kwargs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    eval_text = extract_eval_text(output_text or "")
    num_constraints = len(instruction_ids)
    results = []
    satisfied = 0
    for cid, params in zip(instruction_ids, kwargs_list):
        ok = check_constraint(cid, params or {}, eval_text)
        results.append({"constraint_id": cid, "passed": bool(ok)})
        satisfied += 1 if ok else 0
    compliance_rate = 100.0 * satisfied / num_constraints if num_constraints > 0 else 0.0
    strict_success = 1 if num_constraints > 0 and satisfied == num_constraints else 0
    return {
        "num_constraints": num_constraints,
        "num_satisfied": satisfied,
        "compliance_rate": compliance_rate,
        "strict_success": strict_success,
        "constraints": results,
        "eval_text": eval_text,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate IFEval-style constraints for model outputs")
    ap.add_argument("--dataset", required=True, type=Path, help="Path to modified_data/ifeval/<mod>_100.jsonl")
    ap.add_argument("--outputs", required=True, type=Path, help="Path to model outputs JSONL for this side")
    ap.add_argument("--side", required=True, choices=["original", "modified"], help="Which prompt side the outputs correspond to")
    ap.add_argument("--out_csv", required=True, type=Path, help="Path to write per-sample scores CSV")
    args = ap.parse_args()

    dataset = load_jsonl(args.dataset)
    outputs_map = load_outputs_map(args.outputs, args.side)

    rows = []
    missing = 0
    for obj in dataset:
        key = obj.get("key")
        instruction_ids = obj.get("instruction_id_list") or []
        kwargs_list = obj.get("kwargs") or []
        if not isinstance(instruction_ids, list) or not isinstance(kwargs_list, list):
            continue
        if len(instruction_ids) != len(kwargs_list):
            # Skip malformed row
            continue
        output_text = outputs_map.get(key, "")
        if key not in outputs_map:
            missing += 1
        score = score_sample(output_text, instruction_ids, kwargs_list)
        rows.append({
            "key": key,
            "side": args.side,
            "num_constraints": score["num_constraints"],
            "num_satisfied": score["num_satisfied"],
            "compliance_rate": score["compliance_rate"],
            "strict_success": score["strict_success"],
            "eval_text": score.get("eval_text", output_text or ""),
            # Store minimal diagnostics for potential viewer extensions
            "constraints_json": json.dumps(score["constraints"], ensure_ascii=False),
        })

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print(f"Scored {len(df)} samples. Missing outputs: {missing}. Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
