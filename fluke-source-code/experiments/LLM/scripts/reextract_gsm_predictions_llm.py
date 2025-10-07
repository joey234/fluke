#!/usr/bin/env python3
"""
LLM-based re-extraction script for GSM predictions using moonshotai/kimi-k2
This is a more robust version that uses an LLM instead of regex parsing
"""

import os
import glob
import pandas as pd
import argparse
from pathlib import Path
from llm_answer_extractor import LLMAnswerExtractor
import time
import logging
from dotenv import load_dotenv
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_batch_extraction(extractor: LLMAnswerExtractor, batch_data: list, max_workers: int = 10) -> list:
    """Process multiple extractions in parallel using ThreadPoolExecutor"""
    results = [None] * len(batch_data)
    
    def extract_single(idx, question, raw_output):
        try:
            result = extractor.extract_answer(question, raw_output)
            return idx, result, None
        except Exception as e:
            return idx, None, str(e)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_single, idx, item['question'], item['raw_output']): idx 
            for idx, item in enumerate(batch_data)
        }
        
        for future in as_completed(futures):
            idx, result, error = future.result()
            if error:
                logger.error(f"Batch extraction error at index {idx}: {error}")
                results[idx] = batch_data[idx].get('fallback', '0')
            else:
                results[idx] = result
    
    return results

def process_gsm_file_llm(file_path: str, extractor: LLMAnswerExtractor, backup_original: bool = True, batch_size: int = 50, max_workers: int = 10) -> dict:
    """Process a single GSM results file using LLM-based answer extraction"""
    
    print(f"Processing: {os.path.basename(file_path)}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check required columns
        required_cols = ['text', 'raw_output', 'original_raw_output']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return {
                'status': 'skipped',
                'reason': f'Missing columns: {missing_cols}',
                'file': file_path
            }
        
        # Create backup if requested
        if backup_original:
            backup_path = file_path.replace('.csv', '_llm_backup.csv')
            if not os.path.exists(backup_path):
                df.to_csv(backup_path, index=False)
                print(f"  Created backup: {os.path.basename(backup_path)}")
        
        # Track changes
        modified_changes = 0
        original_changes = 0
        
        # Re-extract modified predictions using LLM (batch processing)
        if 'modified_pred' in df.columns:
            print(f"  Processing modified predictions with LLM (batch_size={batch_size}, workers={max_workers})...")
            
            # Prepare batch data for modified predictions
            batch_data = []
            for idx, row in df.iterrows():
                if pd.notna(row['raw_output']):
                    batch_data.append({
                        'question': str(row['text']),
                        'raw_output': str(row['raw_output']),
                        'fallback': row.get('modified_pred', '0'),
                        'original_pred': row.get('modified_pred', '0')
                    })
                else:
                    batch_data.append({
                        'question': '',
                        'raw_output': '',
                        'fallback': row.get('modified_pred', '0'),
                        'original_pred': row.get('modified_pred', '0')
                    })
            
            # Process in batches for better performance
            new_modified_preds = []
            total_batches = (len(batch_data) + batch_size - 1) // batch_size
            
            for i in range(0, len(batch_data), batch_size):
                batch_num = i // batch_size + 1
                print(f"    Processing batch {batch_num}/{total_batches}...")
                
                batch = batch_data[i:i + batch_size]
                # Only process non-empty batches
                batch_to_process = [item for item in batch if item['raw_output']]
                
                if batch_to_process:
                    batch_results = process_batch_extraction(extractor, batch_to_process, max_workers)
                    
                    # Map results back to original batch
                    result_idx = 0
                    for item in batch:
                        if item['raw_output']:
                            new_pred = batch_results[result_idx]
                            new_modified_preds.append(new_pred)
                            if str(new_pred) != str(item['original_pred']):
                                modified_changes += 1
                            result_idx += 1
                        else:
                            new_modified_preds.append(item['fallback'])
                else:
                    # All items in batch are empty
                    new_modified_preds.extend([item['fallback'] for item in batch])
            
            df['modified_pred'] = new_modified_preds
        
        # Re-extract original predictions using LLM (batch processing)
        if 'original_pred' in df.columns:
            print(f"  Processing original predictions with LLM (batch_size={batch_size}, workers={max_workers})...")
            
            # Prepare batch data for original predictions
            batch_data = []
            for idx, row in df.iterrows():
                if pd.notna(row['original_raw_output']):
                    batch_data.append({
                        'question': str(row['text']),
                        'raw_output': str(row['original_raw_output']),
                        'fallback': row.get('original_pred', '0'),
                        'original_pred': row.get('original_pred', '0')
                    })
                else:
                    batch_data.append({
                        'question': '',
                        'raw_output': '',
                        'fallback': row.get('original_pred', '0'),
                        'original_pred': row.get('original_pred', '0')
                    })
            
            # Process in batches for better performance
            new_original_preds = []
            total_batches = (len(batch_data) + batch_size - 1) // batch_size
            
            for i in range(0, len(batch_data), batch_size):
                batch_num = i // batch_size + 1
                print(f"    Processing batch {batch_num}/{total_batches}...")
                
                batch = batch_data[i:i + batch_size]
                # Only process non-empty batches
                batch_to_process = [item for item in batch if item['raw_output']]
                
                if batch_to_process:
                    batch_results = process_batch_extraction(extractor, batch_to_process, max_workers)
                    
                    # Map results back to original batch
                    result_idx = 0
                    for item in batch:
                        if item['raw_output']:
                            new_pred = batch_results[result_idx]
                            new_original_preds.append(new_pred)
                            if str(new_pred) != str(item['original_pred']):
                                original_changes += 1
                            result_idx += 1
                        else:
                            new_original_preds.append(item['fallback'])
                else:
                    # All items in batch are empty
                    new_original_preds.extend([item['fallback'] for item in batch])
            
            df['original_pred'] = new_original_preds
        
        # Save the updated file
        df.to_csv(file_path, index=False)
        
        return {
            'status': 'success',
            'file': file_path,
            'total_rows': len(df),
            'modified_pred_changes': modified_changes,
            'original_pred_changes': original_changes
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'reason': str(e),
            'file': file_path
        }

def main():
    parser = argparse.ArgumentParser(description='Re-extract GSM predictions using LLM-based approach with parallel processing')
    parser.add_argument('--results-dir', '-d', default='../results/gsm',
                        help='Directory containing GSM result files (default: ../results/gsm)')
    parser.add_argument('--pattern', '-p', default='*.csv',
                        help='File pattern to match (default: *.csv)')
    parser.add_argument('--no-backup', action='store_true',
                        help='Do not create backup files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be processed without making changes')
    parser.add_argument('--api-key', 
                        help='OpenRouter API key for LLM extraction (default: from .env file)')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Number of predictions to process in each batch (default: 50)')
    parser.add_argument('--max-workers', type=int, default=10,
                        help='Maximum number of parallel workers for API calls (default: 10)')
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
    csv_files = [f for f in csv_files if '_backup.csv' not in f and '_llm_backup.csv' not in f]
    
    if not csv_files:
        print(f"No CSV files found in {results_dir} matching pattern '{args.pattern}'")
        return
    
    print(f"Found {len(csv_files)} GSM result files to process")
    print(f"Results directory: {results_dir.absolute()}")
    print(f"Using LLM-based extraction with moonshotai/kimi-k2 via OpenRouter")
    print(f"Backup original files: {not args.no_backup}")
    print(f"Batch size: {args.batch_size}, Max workers: {args.max_workers}")
    print("=" * 80)
    
    if args.dry_run:
        print("DRY RUN - No files will be modified")
        for file_path in sorted(csv_files):
            print(f"Would process: {os.path.basename(file_path)}")
        return
    
    # Initialize LLM extractor
    try:
        # Use API key from argument or environment variable
        api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
        
        if not api_key:
            print("❌ No API key provided!")
            print("Either use --api-key argument or set OPENROUTER_API_KEY in .env file")
            print("Example .env file content:")
            print("OPENROUTER_API_KEY=your_actual_openrouter_api_key_here")
            print("Get your API key from: https://openrouter.ai/")
            return
            
        extractor = LLMAnswerExtractor(api_key)
        print("✅ LLM extractor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize LLM extractor: {str(e)}")
        return
    
    # Process each file
    results = []
    total_modified_changes = 0
    total_original_changes = 0
    
    for file_path in sorted(csv_files):
        result = process_gsm_file_llm(
            file_path, 
            extractor, 
            backup_original=not args.no_backup,
            batch_size=args.batch_size,
            max_workers=args.max_workers
        )
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
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')
    
    print(f"Files processed successfully: {successful}")
    print(f"Files skipped: {skipped}")
    print(f"Files with errors: {errors}")
    print(f"Total modified predictions changed: {total_modified_changes}")
    print(f"Total original predictions changed: {total_original_changes}")
    
    if not args.no_backup:
        print(f"LLM backup files created with '_llm_backup.csv' suffix")
    
    if total_modified_changes > 0 or total_original_changes > 0:
        print(f"\n🎉 Successfully re-extracted predictions using LLM-based parsing!")
        print(f"The LLM approach is much more robust than regex-based extraction.")
    else:
        print(f"\nℹ️  No prediction changes were needed - all files already had correct extractions.")

if __name__ == "__main__":
    main()