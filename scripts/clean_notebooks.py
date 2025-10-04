#!/usr/bin/env python3
import json
import re
from pathlib import Path

NB_DIR = Path('fluke-source-code/data_generation')
PATTERNS = [
    re.compile(r"/Users/[^\s'\"`]+"),
    re.compile(r"/home/[^\s'\"`]+"),
    re.compile(r"[A-Za-z]:\\\\Users\\\\[^\s'\"`]+"),  # Windows paths
]

def scrub_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    out = s
    for pat in PATTERNS:
        out = pat.sub('<LOCAL_PATH>', out)
    return out

def clean_notebook(path: Path) -> bool:
    try:
        with path.open('r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception:
        return False

    changed = False
    for cell in nb.get('cells', []):
        # Clear outputs and execution counts
        if cell.get('outputs'):
            cell['outputs'] = []
            changed = True
        if 'execution_count' in cell and cell['execution_count'] is not None:
            cell['execution_count'] = None
            changed = True
        # Scrub local paths in source
        src = cell.get('source')
        if isinstance(src, list):
            new_src = [scrub_text(x) for x in src]
            if new_src != src:
                cell['source'] = new_src
                changed = True
        elif isinstance(src, str):
            new_src = scrub_text(src)
            if new_src != src:
                cell['source'] = new_src
                changed = True

    # Scrub top-level metadata if present
    md = nb.get('metadata', {})
    for k in list(md.keys()):
        v = md[k]
        if isinstance(v, str):
            new_v = scrub_text(v)
            if new_v != v:
                md[k] = new_v
                changed = True
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, str):
                    new_vv = scrub_text(vv)
                    if new_vv != vv:
                        v[kk] = new_vv
                        changed = True
    nb['metadata'] = md

    if changed:
        with path.open('w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
    return changed

def main():
    count = 0
    for nb in NB_DIR.glob('*.ipynb'):
        if clean_notebook(nb):
            count += 1
            print(f"Cleaned: {nb}")
        else:
            print(f"Checked: {nb}")
    print(f"Done. Updated {count} notebooks.")

if __name__ == '__main__':
    main()

