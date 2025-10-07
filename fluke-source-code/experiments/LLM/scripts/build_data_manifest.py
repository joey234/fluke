#!/usr/bin/env python3
"""
Scan the current directory for *_data.json files and write data_manifest.json
listing their relative paths for the viewer to load dynamically.
"""
from pathlib import Path
import json

HERE = Path(__file__).parent

def main() -> int:
    files = sorted([p.name for p in HERE.glob('*_data.json')])
    manifest = {
        "chunks": files
    }
    out = HERE / 'data_manifest.json'
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Wrote {out} with {len(files)} entries")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

