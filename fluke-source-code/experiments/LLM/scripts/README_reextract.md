# GSM Prediction Re-extraction Script

This script re-extracts predictions from existing GSM result files using the improved answer parsing logic.

## Problem it Solves

The original `extract_answer_prediction` function had issues with extracting the correct numerical answer from math problem responses. Specifically:

- It would sometimes extract the wrong number when multiple numbers appeared after "Final answer:"
- Example: `"Final Answer: The fog will take **140 minutes**, or **2 hours and 20 minutes**, to cover the entire city."` 
- Old logic extracted: `20` (wrong)
- New logic extracts: `140` (correct)

## Usage

### Basic Usage
```bash
python reextract_gsm_predictions.py
```

### With Options
```bash
# Dry run (see what would be processed)
python reextract_gsm_predictions.py --dry-run

# Specify different results directory
python reextract_gsm_predictions.py --results-dir /path/to/results/gsm

# Don't create backup files
python reextract_gsm_predictions.py --no-backup

# Process only specific files
python reextract_gsm_predictions.py --pattern "gpt4o*.csv"
```

## What the Script Does

1. **Finds all CSV files** in `../results/gsm/` (or specified directory)
2. **Creates backups** of original files (unless `--no-backup` specified)
3. **Re-extracts predictions** from `raw_output` and `original_raw_output` columns
4. **Updates** `modified_pred` and `original_pred` columns with improved extractions
5. **Reports changes** made to each file

## Improved Parsing Logic

The new parsing logic uses these strategies in order:

1. **Answer Patterns (Priority)**: Look for "Final answer:", "Answer:", "Therefore", etc. and extract the **first** number after the pattern
2. **Emphasized Numbers**: Look for numbers in `**bold**` or `*italic*` formatting
3. **Line-start Numbers**: Numbers that appear at the beginning of lines
4. **Largest Numbers**: When multiple numbers exist in the last line, prioritize the largest
5. **Fallback**: Last number in the entire text

## Safety Features

- **Automatic backups** created before modifications
- **Dry-run mode** to preview changes
- **Error handling** for corrupted files
- **Column validation** to ensure required columns exist
- **Change tracking** to report exactly what was modified

## Output

The script provides detailed output including:
- Files processed successfully/skipped/failed
- Number of prediction changes per file
- Summary statistics
- Backup file creation status

Example output:
```
Found 8 GSM result files to process
============================================================
Processing: gpt4o-gpt4o-0shot-active_to_passive_100.csv
  Created backup: gpt4o-gpt4o-0shot-active_to_passive_100_backup.csv
  ✓ Modified predictions changed: 15
  ✓ Original predictions changed: 12
...
SUMMARY
============================================================
Files processed successfully: 8
Total modified predictions changed: 89
Total original predictions changed: 76
🎉 Successfully re-extracted predictions using improved parsing logic!
```