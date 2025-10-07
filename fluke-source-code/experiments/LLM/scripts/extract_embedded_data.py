#!/usr/bin/env python3
"""
Extract embedded dataset from fluke_comparison_viewer.html into a standalone JSON file.

It finds the `const embeddedData = {...};` block, parses it as JSON, flattens all
`dataset.data` arrays into a single list, and writes it to `data/full.json`.

Usage:
  python3 extract_embedded_data.py
"""

import os
import re
import sys
import json
from pathlib import Path

HERE = Path(__file__).parent
HTML_PATH = HERE / 'fluke_comparison_viewer.html'
OUT_DIR = HERE / 'data'
OUT_JSON = OUT_DIR / 'full.json'


def main() -> int:
    if not HTML_PATH.exists():
        print(f"Error: HTML file not found at {HTML_PATH}")
        return 1

    # Read the large HTML file
    with HTML_PATH.open('r', encoding='utf-8') as f:
        s = f.read()

    # Locate the embeddedData assignment. We rely on the terminating `};` after the object.
    m = re.search(r"const\s+embeddedData\s*=\s*(\{[\s\S]*?\});", s)
    if not m:
        print("Error: Could not find `const embeddedData = {...};` block")
        return 1

    obj_text = m.group(1)
    # Remove trailing semicolon if present (regex captures it, but we'll be cautious)
    if obj_text.endswith(';'):
        obj_text = obj_text[:-1]

    # Convert JS object literal into strict JSON:
    # - Escape raw newlines inside double-quoted strings
    # - Remove trailing commas before } or ]
    def to_strict_json(js_text: str) -> str:
        out_chars = []
        in_str = False
        esc = False
        for ch in js_text:
            if in_str:
                if esc:
                    out_chars.append(ch)
                    esc = False
                else:
                    if ch == '\\':
                        out_chars.append(ch)
                        esc = True
                    elif ch == '"':
                        out_chars.append(ch)
                        in_str = False
                    elif ch == '\n':
                        out_chars.append('\\n')
                    elif ch == '\r':
                        out_chars.append('\\r')
                    else:
                        out_chars.append(ch)
            else:
                if ch == '"':
                    out_chars.append(ch)
                    in_str = True
                    esc = False
                else:
                    out_chars.append(ch)
        tmp = ''.join(out_chars)
        # Remove trailing commas before closing brackets outside strings
        out = []
        i = 0
        in_str = False
        esc = False
        L = len(tmp)
        while i < L:
            ch = tmp[i]
            if in_str:
                out.append(ch)
                if esc:
                    esc = False
                else:
                    if ch == '\\':
                        esc = True
                    elif ch == '"':
                        in_str = False
                i += 1
                continue
            # outside string
            if ch == '"':
                out.append(ch)
                in_str = True
                esc = False
                i += 1
                continue
            if ch == ',':
                # lookahead for only whitespace then ] or }
                j = i + 1
                while j < L and tmp[j] in (' ', '\t', '\n', '\r'):
                    j += 1
                if j < L and tmp[j] in (']', '}'):
                    # skip this comma
                    i += 1
                    continue
            out.append(ch)
            i += 1
        return ''.join(out)

    strict = to_strict_json(obj_text)
    try:
        data = json.loads(strict)
    except json.JSONDecodeError as e:
        # Provide a brief context for debugging
        start = max(0, e.pos - 120)
        end = min(len(strict), e.pos + 120)
        snippet = strict[start:end]
        print("JSON decode error after fixes at pos", e.pos, e)
        print("Context snippet:\n", snippet)
        return 1

    # Flatten all `.data` arrays from each dataset entry
    all_rows = []
    total_expected = 0
    for key, ds in data.items():
        try:
            total_expected += int(ds.get('samples') or 0)
        except Exception:
            pass
        rows = ds.get('data') or []
        if not isinstance(rows, list):
            continue
        all_rows.extend(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open('w', encoding='utf-8') as out:
        json.dump(all_rows, out, ensure_ascii=False)

    print(f"Wrote {len(all_rows)} rows to {OUT_JSON}")
    if total_expected:
        print(f"(Sum of per-dataset 'samples' counts: {total_expected})")
    try:
        size_mb = OUT_JSON.stat().st_size / (1024 * 1024)
        print(f"Output size: {size_mb:.2f} MB")
    except Exception:
        pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
