#!/usr/bin/env python3
"""
Standalone script to re-extract predictions from existing GSM result files
using the improved answer parsing logic.

This script reads all CSV files in results/gsm/ and re-extracts predictions
from the raw_output and original_raw_output columns using the fixed
extract_answer_prediction function.
"""

import os
import glob
import pandas as pd
import re
from typing import List
import argparse
from pathlib import Path

def extract_answer_prediction_fixed(text: str) -> str:
    """Unified, strict GSM extractor prioritizing #### and robust numerics."""
    if not text:
        return "0"

    # 0) Strict #### rule (allow currency, commas, fractions)
    m = re.findall(r"####\s*[$€£¥₹₽]?\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?|[+-]?\d+\s*/\s*\d+)", text)
    if m:
        return m[-1].replace(',', '')

    # 1) Answer cues
    for pattern, _ in [
        (r'[Ff]inal\s+[Aa]nswer[:\s]*', 'final_answer'),
        (r'[Aa]nswer:\s*', 'answer_colon'),
        (r'[Aa]nswer\s*=\s*', 'answer_equals'),
        (r'[Aa]nswer\s+is\s+', 'answer_is'),
        (r'[Tt]herefore,?\s*', 'therefore'),
        (r'[Ss]o\s+the\s+answer\s+is\s+', 'so_answer_is')
    ]:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if not matches:
            continue
        last = matches[-1]
        remaining = text[last.end():]
        # Emphasis
        for emp in [
            r'\*\*[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\s*\w*\*\*', r'\*\*[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\*\*',
            r'\*[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\s*\w*\*', r'\*[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\*',
            r'__[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\s*\w*__', r'__[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)__',
            r'_[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\s*\w*_', r'_[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)_',
            r'\*\*([+-]?\d+(?:\.\d+)?)\s*\w*\*\*', r'\*\*([+-]?\d+(?:\.\d+)?)\*\*',
            r'\*([+-]?\d+(?:\.\d+)?)\s*\w*\*', r'\*([+-]?\d+(?:\.\d+)?)\*',
            r'__([+-]?\d+(?:\.\d+)?)\s*\w*__', r'__([+-]?\d+(?:\.\d+)?)__',
            r'_([+-]?\d+(?:\.\d+)?)\s*\w*_', r'_([+-]?\d+(?:\.\d+)?)_'
        ]:
            em = re.findall(emp, remaining)
            if em:
                return em[-1]
        # Currency
        cur = re.findall(r'[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)', remaining)
        if cur:
            try:
                vals = [(float(n), n) for n in cur]
                return max(vals, key=lambda x: x[0])[1]
            except:
                return cur[-1]
        # Numbers (incl. fractions)
        nums = re.findall(r'([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?|[+-]?\d+\s*/\s*\d+)', remaining)
        if nums:
            perc = re.search(r'(\d+(?:\.\d+)?)%', remaining)
            if perc:
                return perc.group(1)
            first_sentence = remaining.split('.')[0] if '.' in remaining else remaining.split('\n')[0]
            fs_nums = re.findall(r'([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?|[+-]?\d+\s*/\s*\d+)', first_sentence)
            if fs_nums:
                if len(fs_nums) == 1:
                    return fs_nums[0].replace(',', '')
                try:
                    def to_float(x):
                        x2 = x.replace(',', '')
                        if '/' in x2:
                            a,b = x2.split('/')
                            return float(a)/float(b) if float(b) != 0 else float('nan')
                        return float(x2)
                    vals = [(to_float(n), n) for n in fs_nums]
                    return max(vals, key=lambda x: x[0])[1].replace(',', '')
                except:
                    return fs_nums[0].replace(',', '')
            if len(nums) > 1:
                try:
                    def to_float(x):
                        x2 = x.replace(',', '')
                        if '/' in x2:
                            a,b = x2.split('/')
                            return float(a)/float(b) if float(b) != 0 else float('nan')
                        return float(x2)
                    vals = [(to_float(n), n.replace(',', '')) for n in nums]
                    return max(vals, key=lambda x: x[0])[1]
                except:
                    return nums[-1].replace(',', '')
            return nums[0].replace(',', '')

    # 2) Emphasis anywhere
    for emp in [
        r'\*\*[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\s*\w*\*\*', r'\*\*[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\*\*',
        r'\*[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\s*\w*\*', r'\*[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\*',
        r'__[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\s*\w*__', r'__[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)__',
        r'_[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)\s*\w*_', r'_[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)_',
        r'\*\*([+-]?\d+(?:\.\d+)?)\s*\w*\*\*', r'\*\*([+-]?\d+(?:\.\d+)?)\*\*',
        r'\*([+-]?\d+(?:\.\d+)?)\s*\w*\*', r'\*([+-]?\d+(?:\.\d+)?)\*',
        r'__([+-]?\d+(?:\.\d+)?)\s*\w*__', r'__([+-]?\d+(?:\.\d+)?)__',
        r'_([+-]?\d+(?:\.\d+)?)\s*\w*_', r'_([+-]?\d+(?:\.\d+)?)_'
    ]:
        em = re.findall(emp, text)
        if em:
            return em[-1]

    # 3) Currency anywhere
    cur = re.findall(r'[$€£¥₹₽]([+-]?\d+(?:\.\d+)?)', text)
    if cur:
        return cur[-1]

    # 4) Line-based heuristics
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        m2 = re.match(r'^([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?|[+-]?\d+\s*/\s*\d+)', line)
        if m2:
            return m2.group(1).replace(',', '')
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        nums = re.findall(r'([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?|[+-]?\d+\s*/\s*\d+)', line)
        if nums:
            try:
                def to_float(x):
                    x2 = x.replace(',', '')
                    if '/' in x2:
                        a,b = x2.split('/')
                        return float(a)/float(b) if float(b) != 0 else float('nan')
                    return float(x2)
                vals = [(to_float(n), n.replace(',', '')) for n in nums]
                return max(vals, key=lambda x: x[0])[1]
            except:
                return nums[-1].replace(',', '')

    # 5) Fallback
    nums = re.findall(r'([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?|[+-]?\d+\s*/\s*\d+)', text)
    if nums:
        return nums[-1].replace(',', '')
    return "0"

def process_gsm_file(file_path: str, backup_original: bool = True) -> dict:
    """Process a single GSM results file and re-extract predictions"""
    
    print(f"Processing: {os.path.basename(file_path)}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check required columns
        required_cols = ['raw_output', 'original_raw_output']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return {
                'status': 'skipped',
                'reason': f'Missing columns: {missing_cols}',
                'file': file_path
            }
        
        # Create backup if requested
        if backup_original:
            backup_path = file_path.replace('.csv', '_backup.csv')
            if not os.path.exists(backup_path):
                df.to_csv(backup_path, index=False)
                print(f"  Created backup: {os.path.basename(backup_path)}")
        
        # Track changes
        original_modified_changes = 0
        original_original_changes = 0
        
        # Re-extract modified predictions
        if 'modified_pred' in df.columns:
            new_modified_preds = []
            for idx, row in df.iterrows():
                if pd.notna(row['raw_output']):
                    new_pred = extract_answer_prediction_fixed(str(row['raw_output']))
                    new_modified_preds.append(new_pred)
                    if str(new_pred) != str(row.get('modified_pred', '')):
                        original_modified_changes += 1
                else:
                    new_modified_preds.append(row.get('modified_pred', '0'))
            
            df['modified_pred'] = new_modified_preds
        
        # Re-extract original predictions
        if 'original_pred' in df.columns:
            new_original_preds = []
            for idx, row in df.iterrows():
                if pd.notna(row['original_raw_output']):
                    new_pred = extract_answer_prediction_fixed(str(row['original_raw_output']))
                    new_original_preds.append(new_pred)
                    if str(new_pred) != str(row.get('original_pred', '')):
                        original_original_changes += 1
                else:
                    new_original_preds.append(row.get('original_pred', '0'))
            
            df['original_pred'] = new_original_preds
        
        # Save the updated file
        df.to_csv(file_path, index=False)
        
        return {
            'status': 'success',
            'file': file_path,
            'total_rows': len(df),
            'modified_pred_changes': original_modified_changes,
            'original_pred_changes': original_original_changes,
            'backup_created': backup_original and not os.path.exists(file_path.replace('.csv', '_backup.csv'))
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'reason': str(e),
            'file': file_path
        }

def main():
    parser = argparse.ArgumentParser(description='Re-extract predictions from GSM result files')
    parser.add_argument('--results-dir', '-d', default='../results/gsm',
                        help='Directory containing GSM result files (default: ../results/gsm)')
    parser.add_argument('--pattern', '-p', default='*.csv',
                        help='File pattern to match (default: *.csv)')
    parser.add_argument('--no-backup', action='store_true',
                        help='Do not create backup files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be processed without making changes')
    args = parser.parse_args()
    
    # Get the results directory path
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' does not exist")
        return
    
    # Find all CSV files
    pattern = os.path.join(results_dir, args.pattern)
    csv_files = glob.glob(pattern)
    
    # Filter out backup files
    csv_files = [f for f in csv_files if '_backup.csv' not in f and 'negation_change' not in f]
    
    if not csv_files:
        print(f"No CSV files found in {results_dir} matching pattern '{args.pattern}'")
        return
    
    print(f"Found {len(csv_files)} GSM result files to process")
    print(f"Results directory: {results_dir.absolute()}")
    print(f"Backup original files: {not args.no_backup}")
    print("=" * 60)
    
    if args.dry_run:
        print("DRY RUN - No files will be modified")
        for file_path in sorted(csv_files):
            print(f"Would process: {os.path.basename(file_path)}")
        return
    
    # Process each file
    results = []
    total_modified_changes = 0
    total_original_changes = 0
    
    for file_path in sorted(csv_files):
        result = process_gsm_file(file_path, backup_original=not args.no_backup)
        results.append(result)
        
        if result['status'] == 'success':
            total_modified_changes += result['modified_pred_changes']
            total_original_changes += result['original_pred_changes']
            print(f"  ✓ Modified predictions changed: {result['modified_pred_changes']}")
            print(f"  ✓ Original predictions changed: {result['original_pred_changes']}")
        elif result['status'] == 'skipped':
            print(f"  ⚠️  Skipped: {result['reason']}")
        elif result['status'] == 'error':
            print(f"  ✗ Error: {result['reason']}")
        
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')
    
    print(f"Files processed successfully: {successful}")
    print(f"Files skipped: {skipped}")
    print(f"Files with errors: {errors}")
    print(f"Total modified predictions changed: {total_modified_changes}")
    print(f"Total original predictions changed: {total_original_changes}")
    
    if not args.no_backup:
        backup_count = sum(1 for r in results if r.get('backup_created', False))
        print(f"Backup files created: {backup_count}")
    
    # Show examples of changes if any
    if total_modified_changes > 0 or total_original_changes > 0:
        print(f"\n🎉 Successfully re-extracted predictions using improved parsing logic!")
        print(f"The fixed logic now prioritizes the first number after 'Final answer:' patterns.")
    else:
        print(f"\nℹ️  No prediction changes were needed - all files already had correct extractions.")

if __name__ == "__main__":
    main()
