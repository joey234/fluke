#!/usr/bin/env python3
"""
Create a single centralized HTML viewer that loads CSV data dynamically with client-side filtering (Fixed version)
"""

import pandas as pd
import html
import json
import re
import os
import glob
import warnings

# Silence benign SyntaxWarning from backslash sequences inside the large HTML f-string
warnings.filterwarnings("ignore", category=SyntaxWarning)

def create_centralized_html_viewer():
    """Create a single HTML viewer that loads and filters all comparison data"""
    
    # Get valid modifications from all available model files
    all_result_files = [
        f for f in glob.glob("../results/*/*.csv")
        if ('_backup.csv' not in f and 'negation_change' not in f)
    ]
    valid_modifications = set()
    
    # Extract modifications from all model files using the same patterns as create_comparison_all_models.py
    patterns = [
        (r'gpt-5-standard-context-aware-0shot-(.+)\.csv', 'gpt-5-standard-context-aware'),
        (r'gpt-5-context-aware-0shot-(.+)\.csv', 'gpt-5-standard-context-aware'),
        (r'gpt-5-standard-0shot-(.+)\.csv', 'gpt-5-standard'), 
        (r'gpt-5-0shot-(.+)\.csv', 'gpt-5-standard'),
        (r'claude-3-5-sonnet-0shot-(.+)\.csv', 'claude-3-5-sonnet'),
        (r'claude-0shot-(.+)\.csv', 'claude-3-5-sonnet'),
        (r'deepseek-r1-deepseek-0shot-(.+)\.csv', 'deepseek-r1'),
        (r'deepseek-r1-0shot-(.+)\.csv', 'deepseek-r1'),
        (r'gpt4o-0shot-(.+)\.csv', 'gpt4o'),
        (r'llama-0shot-(.+)\.csv', 'llama'),
        (r'mixtral-0shot-(.+)\.csv', 'mixtral'),
        (r'gpt4o-gpt4o-0shot-(.+)\.csv', 'gpt4o'),
        (r'claude-claude-0shot-(.+)\.csv', 'claude'),
        (r'llama-llama-0shot-(.+)\.csv', 'llama')
    ]
    
    def _normalize_mod(mod: str) -> str:
        if mod.endswith('_100_new'):
            mod = mod[:-4]
        if mod.startswith('singlish'):
            mod = mod.replace('singlish', 'dialectal', 1)
        return mod

    for file in all_result_files:
        filename = os.path.basename(file)
        for pattern, model_name in patterns:
            match = re.search(pattern, filename)
            if match:
                modification = _normalize_mod(match.group(1))
                valid_modifications.add(modification)
                break
    
    print(f"Valid modifications from all model files: {sorted(valid_modifications)}")
    
    # Find all comparison CSV files
    # NOTE: Exclude the combined all_models_comparison_*.csv files to avoid double-counting,
    # since those rows will also appear in each per-model CSV and are filtered by row.model in the UI.
    all_csv_files = [
        f for f in glob.glob("*_comparison_*.csv")
        if (
            '_backup.csv' not in f
            and 'negation_change' not in f
            and not os.path.basename(f).startswith('all_models_comparison_')
        )
    ]
    
    if not all_csv_files:
        print("No comparison CSV files found. Run create_comparison_all_models.py first.")
        return
    
    # Filter to only include files with valid modifications
    csv_files = []
    for csv_file in all_csv_files:
        try:
            # Check if this file contains samples with valid modifications
            df = pd.read_csv(csv_file, nrows=5)  # Just check first few rows
            if 'modification' in df.columns:
                file_modifications = set(df['modification'].dropna().unique())
                if file_modifications.intersection(valid_modifications):
                    csv_files.append(csv_file)
                else:
                    print(f"  Skipping {csv_file}: no valid modifications found")
            else:
                print(f"  Skipping {csv_file}: no modification column")
        except Exception as e:
            print(f"  Error checking {csv_file}: {e}")
    
    print(f"Found {len(csv_files)} comparison files with valid modifications (out of {len(all_csv_files)} total)")
    
    # Load and convert CSV files to JSON for JavaScript
    json_data = {}
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Extract model and task from filename
            base_name = csv_file.replace('.csv', '')
            if '_comparison_' in base_name:
                model_name = base_name.split('_comparison_')[0]
                task_name = base_name.split('_comparison_')[1]
            else:
                # Handle old format
                if csv_file.startswith('gpt5_context_aware_'):
                    model_name = 'gpt-5-standard-context-aware'
                    task_name = csv_file.replace('gpt5_context_aware_comparison_', '').replace('.csv', '')
                else:
                    continue
            
            # Exclude Mixtral from viewer per request
            if model_name == 'mixtral':
                continue

            # Convert DataFrame to list of dictionaries and clean data
            data_list = []
            for _, row in df.iterrows():
                # Skip rows with invalid modifications
                mod_val = _normalize_mod(str(row.get('modification', '')))
                if mod_val not in valid_modifications:
                    continue
                    
                # Clean and prepare the row data - handle NaN values properly
                def clean_value(val):
                    if pd.isna(val):
                        return ''
                    return str(val)
                
                clean_row = {
                    'task': clean_value(row.get('task', task_name)),
                    'model': clean_value(row.get('model', model_name)),
                    'modification': clean_value(mod_val),
                    'original_text': clean_value(row.get('original_text', '')).replace('"', ''),
                    'modified_text': clean_value(row.get('modified_text', '')).replace('"', ''),
                    'original_pred': clean_value(row.get('original_pred', '')),
                    'modified_pred': clean_value(row.get('modified_pred', '')),
                    'original_correct': clean_value(row.get('original_correct', 'False')),
                    'modified_correct': clean_value(row.get('modified_correct', 'False')),
                    'performance': clean_value(row.get('performance', '')),
                    'original_reasoning': clean_value(row.get('original_reasoning', '')),
                    'modified_reasoning': clean_value(row.get('modified_reasoning', '')),
                    'original_step_by_step_reasoning': clean_value(row.get('original_step_by_step_reasoning', '')),
                    'modified_step_by_step_reasoning': clean_value(row.get('modified_step_by_step_reasoning', '')),
                    'row_index': clean_value(row.get('row_index', ''))
                }
                
                # Add task-specific fields
                if task_name == 'gsm':
                    clean_row['original_answer'] = clean_value(row.get('original_answer', ''))
                    clean_row['modified_answer'] = clean_value(row.get('modified_answer', ''))
                    # pass through negation subtype if any
                    clean_row['negation_subtype'] = clean_value(row.get('negation_subtype', row.get('type', '')))
                elif task_name == 'ner':
                    clean_row['original_entities'] = clean_value(row.get('original_entities', ''))
                    clean_row['modified_entities'] = clean_value(row.get('modified_entities', ''))
                elif task_name == 'dialogue':
                    clean_row['original_dialog_label'] = clean_value(row.get('original_dialog_label', ''))
                    clean_row['modified_dialog_label'] = clean_value(row.get('modified_dialog_label', ''))
                elif task_name == 'sa':
                    clean_row['original_label'] = clean_value(row.get('original_label', ''))
                    clean_row['modified_label'] = clean_value(row.get('modified_label', ''))
                elif task_name == 'coref':
                    clean_row['original_clusters'] = clean_value(row.get('original_clusters', ''))
                    clean_row['modified_clusters'] = clean_value(row.get('modified_clusters', ''))
                elif task_name == 'ifeval':
                    clean_row['num_constraints'] = clean_value(row.get('num_constraints', ''))
                    clean_row['original_num_satisfied'] = clean_value(row.get('original_num_satisfied', ''))
                    clean_row['modified_num_satisfied'] = clean_value(row.get('modified_num_satisfied', ''))
                    clean_row['original_compliance_rate'] = clean_value(row.get('original_compliance_rate', ''))
                    clean_row['modified_compliance_rate'] = clean_value(row.get('modified_compliance_rate', ''))
                    clean_row['constraints_detail'] = clean_value(row.get('constraints_detail', ''))
                    # Pass through negation subtype if available
                    clean_row['negation_subtype'] = clean_value(row.get('negation_subtype', row.get('type', '')))
                
                data_list.append(clean_row)
            
            # Embed exactly the dataset rows without additional sampling to avoid duplicates
            embedded_samples = data_list
            
            key = f"{model_name}_{task_name}"
            json_data[key] = {
                'model': model_name,
                'task': task_name,
                'samples': len(data_list),
                'data': embedded_samples
            }
            
            # Also save full data as separate JSON file for on-demand loading
            with open(f"{key}_data.json", 'w', encoding='utf-8') as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)
            
            print(f"  {key}: {len(data_list)} samples")
            
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    # Canonical display names for models
    model_display_map = {
        'bert': 'BERT',
        'gpt2': 'GPT-2',
        't5': 'T5',
        'gpt4o': 'GPT-4o',
        'gpt-4o': 'GPT-4o',
        'claude': 'Claude-3.5-Sonnet',
        'claude-3-5-sonnet': 'Claude-3.5-Sonnet',
        'llama': 'Llama 3.1 405B',
        'llama-3': 'Llama 3.1 405B',
        'deepseek-r1': 'DeepSeek R1',
        'deepseek-r1-deepseek': 'DeepSeek R1',
        'gpt-5-standard': 'GPT-5',
        'gpt-5-standard-context-aware': 'GPT-5 (w. context)'
    }

    # Generate the HTML
    html_content = fr"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FLUKE Model Comparison Viewer</title>
    <style>
        :root {{
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --success-color: #28a745;
            --danger-color: #dc3545;
            --warning-color: #ffc107;
            --info-color: #17a2b8;
            --light-color: #f8f9fa;
            --dark-color: #343a40;
        }}
        
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        
        .header {{
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .controls {{
            background: white;
            padding: 20px;
            margin: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .control-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .control-group {{
            display: flex;
            flex-direction: column;
        }}
        
        .control-group label {{
            font-weight: bold;
            margin-bottom: 5px;
            color: var(--dark-color);
        }}
        
        .control-group select, .control-group input {{
            padding: 8px 12px;
            border: 2px solid #e9ecef;
            border-radius: 6px;
            font-size: 14px;
            transition: border-color 0.3s;
        }}
        
        .control-group select:focus, .control-group input:focus {{
            outline: none;
            border-color: var(--primary-color);
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.2s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-2px);
        }}
        
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: var(--dark-color);
            font-size: 14px;
        }}
        
        .stat-card .stat-number {{
            font-size: 24px;
            font-weight: bold;
            color: var(--primary-color);
        }}
        
        .results {{
            margin: 20px;
        }}
        
        .comparison {{
            background: white;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .comparison-header {{
            background-color: var(--light-color);
            padding: 15px 20px;
            border-bottom: 2px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }}
        
        .badge {{
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        
        .badge-model {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
        }}
        
        .badge-modification {{
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            color: white;
        }}
        
        .badge-performance {{
            font-size: 0.8em;
        }}
        
        .badge-original_better {{ background-color: #d4edda; color: #155724; }}
        .badge-modified_better {{ background-color: #cce7ff; color: #004085; }}
        .badge-both_correct {{ background-color: #e6f3ff; color: #0066cc; }}
        .badge-both_wrong {{ background-color: #f8d7da; color: #721c24; }}
        
        .content-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0;
        }}
        
        .text-section {{
            padding: 20px;
            border-right: 1px solid #e9ecef;
        }}
        
        .text-section:last-child {{
            border-right: none;
        }}
        
        .text-section h3 {{
            margin-top: 0;
            margin-bottom: 15px;
            color: var(--dark-color);
            border-bottom: 2px solid #eee;
            padding-bottom: 8px;
        }}
        
        .text-content {{
            background-color: var(--light-color);
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid var(--primary-color);
            margin-bottom: 15px;
            font-family: 'SF Mono', Monaco, monospace;
            white-space: pre-wrap;
            line-height: 1.5;
            max-height: 200px;
            overflow-y: auto;
        }}
        
        .prediction {{
            padding: 10px 15px;
            border-radius: 6px;
            margin-bottom: 10px;
            font-weight: bold;
        }}
        
        .prediction.correct {{
            background-color: #d4edda;
            color: #155724;
            border-left: 4px solid var(--success-color);
        }}
        
        .prediction.incorrect {{
            background-color: #f8d7da;
            color: #721c24;
            border-left: 4px solid var(--danger-color);
        }}
        .violation-mark {{ background-color: #ffebee; color: #c62828; text-decoration: underline; }}
        
        .ground-truth {{
            margin-top: 10px;
            color: #666;
            font-size: 0.9em;
            padding: 8px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }}

        .constraints-section {{
            margin: 15px 20px;
            padding: 10px;
            background: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 6px;
        }}
        .constraints-title {{ font-weight: bold; color: #333; margin-bottom: 8px; }}
        .constraint-item {{ display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px dashed #eee; }}
        .constraint-item:last-child {{ border-bottom: none; }}
        .constraint-id {{ color: #555; }}
        .constraint-status {{ display: flex; gap: 12px; }}
        .constraint-ok {{ color: #2e7d32; font-weight: bold; }}
        .constraint-fail {{ color: #c62828; font-weight: bold; }}
        
        .loading {{
            text-align: center;
            padding: 40px;
            color: var(--dark-color);
        }}
        
        .loading-spinner {{
            display: inline-block;
            width: 40px;
            height: 40px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid var(--primary-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 20px;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .no-results {{
            text-align: center;
            padding: 40px;
            color: #666;
            font-style: italic;
        }}
        
        .debug-info {{
            background: #f8f9fa;
            padding: 10px;
            margin: 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 12px;
            border-left: 3px solid var(--info-color);
        }}
        
        @media (max-width: 768px) {{
            .control-grid {{
                grid-template-columns: 1fr;
            }}
            
            .content-grid {{
                grid-template-columns: 1fr;
            }}
            
            .text-section {{
                border-right: none;
                border-bottom: 1px solid #e9ecef;
            }}
            
            .text-section:last-child {{
                border-bottom: none;
            }}
            
            .comparison-header {{
                flex-direction: column;
                align-items: stretch;
                text-align: center;
            }}
        }}
        
        .reasoning-section {{
            margin-top: 15px;
            padding: 10px;
            background-color: #f8f9fa;
            border-left: 4px solid var(--info-color);
            border-radius: 4px;
        }}
        
        .reasoning-toggle {{
            cursor: pointer;
            font-weight: bold;
            color: var(--info-color);
            user-select: none;
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .reasoning-toggle:hover {{
            color: var(--primary-color);
        }}
        
        .reasoning-content {{
            display: none;
            margin-top: 10px;
            white-space: pre-wrap;
            font-size: 0.9em;
            line-height: 1.5;
            max-height: 300px;
            overflow-y: auto;
            padding: 10px;
            background-color: white;
            border-radius: 4px;
            border: 1px solid #dee2e6;
        }}
        
        .reasoning-content.expanded {{
            display: block;
        }}
        
        .reasoning-type {{
            font-weight: bold;
            color: var(--dark-color);
            margin-bottom: 8px;
        }}
        
        .text-diff {{
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
        }}
        
        .text-diff-added {{
            background-color: #d4edda;
            color: #155724;
        }}
        
        .text-diff-removed {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        
        .text-diff-changed {{
            background-color: #fff3cd;
            color: #856404;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 FLUKE Model Comparison Viewer</h1>
        <p>Interactive analysis of model performance across text modifications</p>
    </div>

    <div class="controls">
        <div class="control-grid">
            <div class="control-group">
                <label for="modelSelect">Model:</label>
                <select id="modelSelect">
                    <option value="">All Models</option>
                </select>
            </div>
            
            <div class="control-group">
                <label for="taskSelect">Task:</label>
                <select id="taskSelect">
                    <option value="">All Tasks</option>
                    <option value="ner">NER (Named Entity Recognition)</option>
                    <option value="dialogue">Dialogue Classification</option>
                    <option value="sa">Sentiment Analysis</option>
                    <option value="coref">Coreference Resolution</option>
                    <option value="gsm">GSM (Grade School Math)</option>
                    <option value="ifeval">Instruction Following (IFEval)</option>
                </select>
            </div>
            
            <div class="control-group">
                <label for="performanceSelect">Performance:</label>
                <select id="performanceSelect">
                    <option value="">All Performance Types</option>
                    <option value="original_better">Original Better</option>
                    <option value="modified_better">Modified Better</option>
                    <option value="both_correct">Both Correct</option>
                    <option value="both_wrong">Both Wrong</option>
                </select>
            </div>
            
            <div class="control-group">
                <label for="modificationSelect">Modification:</label>
                <select id="modificationSelect">
                    <option value="">All Modifications</option>
                    <option value="active_to_passive_100">Active to Passive</option>
                    <option value="capitalization_100">Capitalization</option>
                    <option value="casual_100">Casual Language</option>
                    <option value="compound_word_100">Compound Words</option>
                    <option value="concept_replacement_100">Concept Replacement</option>
                    <option value="coordinating_conjunction_100">Coordinating Conjunction</option>
                    <option value="derivation_100">Derivation</option>
                    <option value="dialectal_100">Dialectal</option>
                    <option value="discourse_100">Discourse</option>
                    <option value="geographical_bias_100">Geographical Bias</option>
                    <option value="grammatical_role_100">Grammatical Role</option>
                    <option value="length_bias_100">Length Bias</option>
                    <option value="negation_100">Negation</option>
                    <option value="negation_change_100">Negation Change</option>
                    <option value="punctuation_100">Punctuation</option>
                    <option value="sentiment_100">Sentiment</option>
                    <option value="temporal_bias_100">Temporal Bias</option>
                    <option value="typo_bias_100">Typo Bias</option>
                </select>
            </div>
            
        </div>
    </div>

    <div class="stats" id="stats">
        <div class="stat-card">
            <h3>Available Samples</h3>
            <div class="stat-number" id="totalSamples">0</div>
        </div>
        <div class="stat-card">
            <h3>Filtered Results</h3>
            <div class="stat-number" id="filteredCount">0</div>
        </div>
        <div class="stat-card">
            <h3>Original Better</h3>
            <div class="stat-number" id="originalBetter">0</div>
        </div>
        <div class="stat-card">
            <h3>Modified Better</h3>
            <div class="stat-number" id="modifiedBetter">0</div>
        </div>
        <div class="stat-card">
            <h3>Both Correct</h3>
            <div class="stat-number" id="bothCorrect">0</div>
        </div>
        <div class="stat-card">
            <h3>Both Wrong</h3>
            <div class="stat-number" id="bothWrong">0</div>
        </div>
    </div>
    
    <div class="debug-info" id="debugInfo">
        Loading data...
    </div>

    <div class="results" id="results">
        <div class="loading">
            <div class="loading-spinner"></div>
            <div>Loading comparison data...</div>
        </div>
    </div>

    <script>
        // Embedded data (first 50 samples per dataset for immediate loading)
        const embeddedData = {json.dumps(json_data, indent=8)};
        
        // Global variables
        let allData = [];
        let filteredData = [];
        
        // Initialize the viewer
        async function initialize() {{
            try {{
                loadEmbeddedData();
                populateModelSelect();
                populateModificationSelect('');
                setupEventListeners();
                updateDebugInfo('Initialized with embedded data');
                applyFilters();
            }} catch (error) {{
                console.error('Initialization error:', error);
                updateDebugInfo('Error: ' + error.message);
                document.getElementById('results').innerHTML = `
                    <div class="no-results">
                        <h3>Error loading data</h3>
                        <p>${{error.message}}</p>
                    </div>
                `;
            }}
        }}
        
        // Simple HTML escaper
        function escapeHtml(s) {{
            return String(s)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
        }}

        // Simple text difference highlighting
        function highlightTextDifferences(originalText, modifiedText) {{
            if (!originalText || !modifiedText) return {{ original: originalText, modified: modifiedText }};
            
            const originalWords = originalText.split(' ');
            const modifiedWords = modifiedText.split(' ');
            
            // Simple word-by-word comparison
            let originalHighlighted = '';
            let modifiedHighlighted = '';
            
            const maxLen = Math.max(originalWords.length, modifiedWords.length);
            
            for (let i = 0; i < maxLen; i++) {{
                const origWord = originalWords[i] || '';
                const modWord = modifiedWords[i] || '';
                
                if (origWord === modWord) {{
                    // Words are the same
                    if (origWord) {{
                        originalHighlighted += origWord + ' ';
                        modifiedHighlighted += modWord + ' ';
                    }}
                }} else {{
                    // Words are different
                    if (origWord) {{
                        originalHighlighted += `<span class="text-diff text-diff-removed">${{origWord}}</span> `;
                    }}
                    if (modWord) {{
                        modifiedHighlighted += `<span class="text-diff text-diff-added">${{modWord}}</span> `;
                    }}
                }}
            }}
            
            return {{
                original: originalHighlighted.trim(),
                modified: modifiedHighlighted.trim()
            }};
        }}
        
        // IFEval violation highlighter
        function highlightIfevalViolations(text, constraintsDetail, side) {{
            if (!text) return '';
            let html = text;
            try {{
                const escapeRegExp = (s) => s.replace(new RegExp('[-/\\^$*+?.()|[\\]{{}}]', 'g'), '\\$&');
                const cons = JSON.parse(constraintsDetail || '[]');
                // Apply per-constraint highlights only for failed ones on the chosen side
                cons.forEach(c => {{
                    const failed = side === 'orig' ? !c.orig : !c.mod;
                    if (!failed) return;
                    const id = c.id || '';
                    const p = c.params || {{}};
                    if (id === 'punctuation:no_comma') {{
                        // highlight commas
                        html = html.replace(/,/g, '<span class="violation-mark">,</span>');
                    }} else if (id === 'keywords:letter_frequency') {{
                        const letter = p.letter || '';
                        if (letter) {{
                            const re = new RegExp(escapeRegExp(letter), 'g');
                            html = html.replace(re, m => `<span class="violation-mark">${{m}}</span>`);
                        }}
                    }} else if (id === 'keywords:frequency' || id === 'keywords:forbidden_words' || id === 'keywords:forbidden') {{
                        const kw = p.keyword || '';
                        const list = Array.isArray(p.forbidden_words) ? p.forbidden_words : [];
                        if (kw) {{
                            const re = new RegExp(escapeRegExp(kw), 'gi');
                            html = html.replace(re, m => `<span class=\"violation-mark\">${{m}}</span>`);
                        }}
                        if (list.length) {{
                            list.forEach(w => {{
                                if (!w) return;
                                const re = new RegExp(escapeRegExp(w), 'gi');
                                html = html.replace(re, m => `<span class=\"violation-mark\">${{m}}</span>`);
                            }});
                        }}
                    }} else if (id === 'startend:end_checker') {{
                        const endPhrase = (p.end_phrase || '').toString();
                        if (endPhrase && !html.endsWith(endPhrase)) {{
                            const n = endPhrase.length;
                            if (n > 0 && n <= html.length) {{
                                const tail = html.slice(-n);
                                html = html.slice(0, -n) + `<span class=\"violation-mark\">${{tail}}</span>`;
                            }}
                        }}
                    }} else if (id === 'combination:repeat_prompt') {{
                        const prompt = (p.prompt_to_repeat || '').toString();
                        if (prompt && !html.startsWith(prompt)) {{
                            const n = prompt.length;
                            if (n > 0 && n <= html.length) {{
                                const head = html.slice(0, n);
                                html = `<span class=\"violation-mark\">${{head}}</span>` + html.slice(n);
                            }}
                        }}
                    }} else if (id === 'length_constraints:number_words') {{
                        const relation = (p.relation || 'at most').toLowerCase();
                        const num = parseInt(p.num_words || 0, 10) || 0;
                        const words = html.split(/\s+/);
                        if (relation.includes('at most') && words.length > num && num > 0) {{
                            const okPart = words.slice(0, num).join(' ');
                            const extra = words.slice(num).join(' ');
                            html = okPart + ' ' + `<span class=\"violation-mark\">${{extra}}</span>`;
                        }} else if (relation.includes('at least') && words.length < num) {{
                            // Avoid over-highlighting entire text; keep output readable
                        }}
                    }}
                }});
            }} catch (e) {{
                // ignore parse errors
            }}
            return html;
        }}
        
        // Toggle reasoning section
        function toggleReasoning(elementId) {{
            const content = document.getElementById(elementId);
            const toggle = content.previousElementSibling;
            
            if (content.classList.contains('expanded')) {{
                content.classList.remove('expanded');
                toggle.innerHTML = '▶️ Show Reasoning';
            }} else {{
                content.classList.add('expanded');
                toggle.innerHTML = '🔽 Hide Reasoning';
            }}
        }}
        
        // Format NER entities for display
        function formatNEREntities(entitiesStr) {{
            if (!entitiesStr || entitiesStr === 'nan' || entitiesStr === '') {{
                return 'No entities';
            }}
            
            try {{
                // Try to parse as JSON array
                const entities = JSON.parse(entitiesStr.replace(/'/g, '"'));
                if (Array.isArray(entities) && entities.length === 0) {{
                    return 'No entities';
                }}
                if (Array.isArray(entities)) {{
                    return entities.map(e => {{
                        if (typeof e === 'object') {{
                            // Handle {{text: "name", value: "type"}} format
                            if (e.text && e.value) {{
                                return `${{e.text}} (${{e.value}})`;
                            }}
                            // Handle {{"name": "type"}} format
                            const keys = Object.keys(e);
                            if (keys.length > 0) {{
                                return keys.map(k => `${{k}} (${{e[k]}})`).join(', ');
                            }}
                        }}
                        return String(e);
                    }}).join(', ');
                }}
            }} catch (e) {{
                // If JSON parsing fails, try to extract entity info from string
                const regex = /["']([^"']+)["']\\s*:\\s*["']([^"']+)["']/g;
                const matches = entitiesStr.match(regex);
                if (matches) {{
                    return matches.map(match => {{
                        const parts = match.match(/["']([^"']+)["']\\s*:\\s*["']([^"']+)["']/);
                        return parts ? `${{parts[1]}} (${{parts[2]}})` : match;
                    }}).join(', ');
                }}
            }}
            
            return entitiesStr;
        }}
        
        // Canonical display mapping for model names
        const modelMap = {json.dumps(model_display_map)};

        // Load embedded data into allData array
        function loadEmbeddedData() {{
            allData = [];
            Object.entries(embeddedData).forEach(([key, dataset]) => {{
                if (dataset.data && Array.isArray(dataset.data)) {{
                    allData.push(...dataset.data);
                }}
            }});
            
            const totalSamples = Object.values(embeddedData).reduce((sum, dataset) => sum + (dataset.samples || 0), 0);
            document.getElementById('totalSamples').textContent = totalSamples.toLocaleString();
            
            updateDebugInfo(`Loaded ${{allData.length}} samples from embedded data (out of ${{totalSamples}} total)`);
        }}
        
        // Populate model select dropdown
        function populateModelSelect() {{
            const models = [...new Set(allData.map(row => row.model))].sort();
            const modelSelect = document.getElementById('modelSelect');
            
            models.forEach(model => {{
                const option = document.createElement('option');
                option.value = model;
                option.textContent = modelMap[model] || model;
                modelSelect.appendChild(option);
            }});
            
            updateDebugInfo(`Found models: ${{models.join(', ')}}`);
        }}

        // Populate modification select based on current task
        function populateModificationSelect(currentTask) {{
            const modSelect = document.getElementById('modificationSelect');
            if (!modSelect) return;
            const prev = modSelect.value;
            const mods = [...new Set(allData
                .filter(row => !currentTask || row.task === currentTask)
                .map(row => row.modification))].filter(Boolean).sort();
            modSelect.innerHTML = '';
            const optAll = document.createElement('option');
            optAll.value = '';
            optAll.textContent = 'All Modifications';
            modSelect.appendChild(optAll);
            mods.forEach(m => {{
                const o = document.createElement('option');
                o.value = m;
                o.textContent = m;
                modSelect.appendChild(o);
            }});
            if ([...modSelect.options].some(o => o.value === prev)) {{
                modSelect.value = prev;
            }} else {{
                modSelect.value = '';
            }}
        }}
        
        // Setup event listeners for all controls
        function setupEventListeners() {{
            const controls = ['modelSelect', 'taskSelect', 'performanceSelect', 'modificationSelect'];
            controls.forEach(controlId => {{
                document.getElementById(controlId).addEventListener('change', applyFilters);
            }});
        }}
        
        // Apply current filters
        function applyFilters() {{
            const filters = {{
                model: document.getElementById('modelSelect').value,
                task: document.getElementById('taskSelect').value,
                performance: document.getElementById('performanceSelect').value,
                modification: document.getElementById('modificationSelect').value
            }};

            // Keep modification options in sync with selected task
            try {{
                populateModificationSelect(filters.task);
            }} catch (e) {{
                // ignore if not available
            }}
            
            // Enhanced debugging
            let debugMsg = `Filters: model="${{filters.model}}" task="${{filters.task}}" perf="${{filters.performance}}" mod="${{filters.modification}}"`;
            
            // Count what each filter eliminates
            let afterModel = filters.model ? allData.filter(r => r.model === filters.model) : allData;
            let afterTask = filters.task ? afterModel.filter(r => r.task === filters.task) : afterModel;
            let afterPerf = filters.performance ? afterTask.filter(r => r.performance === filters.performance) : afterTask;
            let afterMod = filters.modification ? afterPerf.filter(r => r.modification === filters.modification) : afterPerf;
            
            debugMsg += ` | After model: ${{afterModel.length}}, task: ${{afterTask.length}}, perf: ${{afterPerf.length}}, mod: ${{afterMod.length}}`;
            
            // Filter the data
            filteredData = allData.filter(row => {{
                if (filters.model && row.model !== filters.model) return false;
                if (filters.task && row.task !== filters.task) return false;
                if (filters.performance && row.performance !== filters.performance) return false;
                if (filters.modification && row.modification !== filters.modification) return false;
                return true;
            }});
            
            // Apply 2000 sample limit for performance
            if (filteredData.length > 2000) {{
                filteredData = filteredData.slice(0, 2000);
            }}
            
            updateDebugInfo(debugMsg);
            
            updateStats();
            renderResults();
        }}
        
        // Update debug information
        function updateDebugInfo(message) {{
            document.getElementById('debugInfo').textContent = `Debug: ${{message}} (Total data: ${{allData.length}} samples)`;
        }}
        
        // Update statistics display
        function updateStats() {{
            const stats = {{
                original_better: 0,
                modified_better: 0,
                both_correct: 0,
                both_wrong: 0
            }};
            
            filteredData.forEach(row => {{
                if (row.performance in stats) {{
                    stats[row.performance]++;
                }}
            }});
            
            document.getElementById('filteredCount').textContent = filteredData.length;
            document.getElementById('originalBetter').textContent = stats.original_better || 0;
            document.getElementById('modifiedBetter').textContent = stats.modified_better || 0;
            document.getElementById('bothCorrect').textContent = stats.both_correct || 0;
            document.getElementById('bothWrong').textContent = stats.both_wrong || 0;
        }}
        
        // Render filtered results
        function renderResults() {{
            const resultsContainer = document.getElementById('results');
            
            if (filteredData.length === 0) {{
                resultsContainer.innerHTML = `
                    <div class="no-results">
                        <h3>No samples found</h3>
                        <p>Try adjusting your filters to see more results.</p>
                        <p>Available data: ${{allData.length}} total samples</p>
                    </div>
                `;
                return;
            }}
            
            const html = filteredData.map((row, idx) => {{
                let originalCorrect = row.original_correct === 'True';
                let modifiedCorrect = row.modified_correct === 'True';
                let performance = row.performance || '';
                const modification = (row.modification || '').replace(/_/g, ' ').replace(/100/g, '').trim();
                
                const originalText = (row.original_text || '').substring(0, 500);
                const modifiedText = (row.modified_text || '').substring(0, 500);
                const taskName = row.task;
                
                // For IFEval: show inputs as raw (no diff), and highlight violations in prediction text only
                let textLeftHtml = '';
                let textRightHtml = '';
                let predLeftHtml = row.original_pred || '';
                let predRightHtml = row.modified_pred || '';
                // GSM display: prefer the LAST ####-parsed answer from reasoning/raw for all except GPT-5 standard
                if (taskName === 'gsm') {{
                    try {{
                        const isGpt5Std = (row.model || '') === 'gpt-5-standard';
                        const extractLastHash = (text) => {{
                            if (!text) return null;
                            const re = /####\s*[$€£¥₹₽]?\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?|[+-]?\d+\s*\/\s*\d+)/g;
                            let m, last = null;
                            while ((m = re.exec(text)) !== null) {{ last = m[1]; }}
                            return last ? last.replace(/,/g, '') : null;
                        }};
                        if (!isGpt5Std) {{
                            const oCat = ((row.original_step_by_step_reasoning || '') + ' ' + (row.original_reasoning || '') + ' ' + (row.original_raw_output || ''));
                            const mCat = ((row.modified_step_by_step_reasoning || row.step_by_step_reasoning || '') + ' ' + (row.modified_reasoning || row.reasoning || '') + ' ' + (row.raw_output || ''));
                            const op = extractLastHash(oCat);
                            const mp = extractLastHash(mCat);
                            if (op) predLeftHtml = op;
                            if (mp) predRightHtml = mp;
                        }}
                    }} catch (e) {{}}
                    // Recompute correctness on the client for robustness
                    try {{
                        const parseNum = (s) => {{
                            if (!s) return null;
                            const t = String(s).replace(/,/g, '').trim();
                            // Prefer the LAST #### match if present
                            let tok = t;
                            const reAll = /####\s*[$€£¥₹₽]?\s*([+-]?(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\d+\s*\/\s*\d+))/g;
                            let mLast = null, m;
                            while ((m = reAll.exec(t)) !== null) {{ mLast = m[1]; }}
                            if (mLast) tok = mLast;
                            const mFrac = tok.match(/^\s*([+-]?\d+)\s*\/\s*(\d+)\s*$/);
                            if (mFrac) {{
                                const a = parseFloat(mFrac[1]);
                                const b = parseFloat(mFrac[2]);
                                return b !== 0 ? a / b : null;
                            }}
                            const mNum = tok.match(/[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$/);
                            if (mNum) return parseFloat(mNum[0]);
                            return null;
                        }};
                        const oa = parseNum(row.original_answer || '');
                        const ma = parseNum(row.modified_answer || '');
                        const opn = parseNum(predLeftHtml || row.original_pred || '');
                        const mpn = parseNum(predRightHtml || row.modified_pred || '');
                        if (oa !== null && opn !== null) originalCorrect = Math.abs(opn - oa) < 1e-9;
                        const isNeg = (row.modification || '').toLowerCase().startsWith('negation');
                        const subtype = (row.negation_subtype || row.type || '').toLowerCase();
                        if (isNeg) {{
                            const eq = (mpn !== null && oa !== null) ? (Math.abs(mpn - oa) < 1e-9) : (String(row.modified_pred||'').trim() === String(row.original_answer||'').trim());
                            if (subtype.includes('approximate') || subtype.includes('double')) {{
                                modifiedCorrect = eq;
                            }} else {{
                                modifiedCorrect = !eq;
                            }}
                        }} else {{
                            if (ma !== null && mpn !== null) modifiedCorrect = Math.abs(mpn - ma) < 1e-9;
                        }}
                        // Update performance label based on recomputed correctness
                        performance = (originalCorrect && modifiedCorrect) ? 'both_correct' : (originalCorrect ? 'original_better' : (modifiedCorrect ? 'modified_better' : 'both_wrong'));
                    }} catch (e) {{}}
                }}
                if (taskName === 'ifeval') {{
                    // inputs
                    textLeftHtml = escapeHtml(originalText);
                    textRightHtml = escapeHtml(modifiedText);
                    // predictions with violation highlights
                    predLeftHtml = highlightIfevalViolations(row.original_pred || '', row.constraints_detail || '[]', 'orig');
                    predRightHtml = highlightIfevalViolations(row.modified_pred || '', row.constraints_detail || '[]', 'mod');
                    // Adjust correctness and performance for negation subtypes (verbal/lexical/absolute)
                    try {{
                        const n = parseInt(row.num_constraints || '0') || 0;
                        const oc = parseInt(row.original_num_satisfied || '0') || 0;
                        const mc = parseInt(row.modified_num_satisfied || '0') || 0;
                        const strictO = (n > 0 && oc === n);
                        const strictM = (n > 0 && mc === n);
                        const modStr = (row.modification || '').toString();
                        const subtype = (row.negation_subtype || row.type || '').toString().toLowerCase();
                        if (modStr.startsWith('negation') && ['verbal','lexical','absolute'].includes(subtype)) {{
                            originalCorrect = strictO;
                            modifiedCorrect = !strictM;
                            if (originalCorrect && modifiedCorrect) performance = 'both_correct';
                            else if (originalCorrect && !modifiedCorrect) performance = 'original_better';
                            else if (!originalCorrect && modifiedCorrect) performance = 'modified_better';
                            else performance = 'both_wrong';
                        }}
                    }} catch (e) {{}}
                }} else {{
                    const textDiff = highlightTextDifferences(originalText, modifiedText);
                    textLeftHtml = textDiff.original;
                    textRightHtml = textDiff.modified;
                }}
                
                // Format labels based on task
                let originalLabel = 'N/A';
                let modifiedLabel = 'N/A';
                
                if (taskName === 'gsm') {{
                    originalLabel = `Answer: ${{row.original_answer || ''}}`;
                    modifiedLabel = `Answer: ${{row.modified_answer || ''}}`;
                }} else if (taskName === 'ner') {{
                    originalLabel = `Entities: ${{formatNEREntities(row.original_entities)}}`;
                    modifiedLabel = `Entities: ${{formatNEREntities(row.modified_entities)}}`;
                }} else if (taskName === 'dialogue') {{
                    originalLabel = `${{row.original_dialog_label || ''}}`;
                    modifiedLabel = `${{row.modified_dialog_label || ''}}`;
                }} else if (taskName === 'sa') {{
                    originalLabel = `${{row.original_label || ''}}`;
                    modifiedLabel = `${{row.modified_label || ''}}`;
                }} else if (taskName === 'coref') {{
                    originalLabel = `${{row.original_clusters || 'None'}}`;
                    modifiedLabel = `${{row.modified_clusters || 'None'}}`;
                }} else if (taskName === 'ifeval') {{
                    const ncons = row.num_constraints || '0';
                    const oSat = row.original_num_satisfied || '0';
                    const mSat = row.modified_num_satisfied || '0';
                    const oRate = row.original_compliance_rate || '0';
                    const mRate = row.modified_compliance_rate || '0';
                    originalLabel = `Constraints satisfied: ${{oSat}}/${{ncons}} (${{oRate}}%)`;
                    modifiedLabel = `Constraints satisfied: ${{mSat}}/${{ncons}} (${{mRate}}%)`;
                }}
                
                // Prepare reasoning content for each side separately
                const hasOriginalReasoning = (row.original_reasoning && row.original_reasoning !== '' && row.original_reasoning !== 'nan') ||
                                           (row.original_step_by_step_reasoning && row.original_step_by_step_reasoning !== '' && row.original_step_by_step_reasoning !== 'nan');
                
                const hasModifiedReasoning = (row.modified_reasoning && row.modified_reasoning !== '' && row.modified_reasoning !== 'nan') ||
                                           (row.modified_step_by_step_reasoning && row.modified_step_by_step_reasoning !== '' && row.modified_step_by_step_reasoning !== 'nan');
                
                let originalReasoningSection = '';
                if (hasOriginalReasoning) {{
                    const originalReasoningId = `original_reasoning_${{idx}}`;
                    originalReasoningSection = `
                        <div class="reasoning-section">
                            <div class="reasoning-toggle" onclick="toggleReasoning('${{originalReasoningId}}')">
                                ▶️ Show Reasoning
                            </div>
                            <div id="${{originalReasoningId}}" class="reasoning-content">`;
                    
                    if (row.original_reasoning && row.original_reasoning !== '' && row.original_reasoning !== 'nan') {{
                        originalReasoningSection += `<div class="reasoning-type">Reasoning:</div>${{row.original_reasoning}}<br><br>`;
                    }}
                    if (row.original_step_by_step_reasoning && row.original_step_by_step_reasoning !== '' && row.original_step_by_step_reasoning !== 'nan') {{
                        originalReasoningSection += `<div class="reasoning-type">Step-by-Step Reasoning:</div>${{row.original_step_by_step_reasoning}}`;
                    }}
                    
                    originalReasoningSection += `</div></div>`;
                }}
                
                let modifiedReasoningSection = '';
                if (hasModifiedReasoning) {{
                    const modifiedReasoningId = `modified_reasoning_${{idx}}`;
                    modifiedReasoningSection = `
                        <div class="reasoning-section">
                            <div class="reasoning-toggle" onclick="toggleReasoning('${{modifiedReasoningId}}')">
                                ▶️ Show Reasoning
                            </div>
                            <div id="${{modifiedReasoningId}}" class="reasoning-content">`;
                    
                    if (row.modified_reasoning && row.modified_reasoning !== '' && row.modified_reasoning !== 'nan') {{
                        modifiedReasoningSection += `<div class="reasoning-type">Reasoning:</div>${{row.modified_reasoning}}<br><br>`;
                    }}
                    if (row.modified_step_by_step_reasoning && row.modified_step_by_step_reasoning !== '' && row.modified_step_by_step_reasoning !== 'nan') {{
                        modifiedReasoningSection += `<div class="reasoning-type">Step-by-Step Reasoning:</div>${{row.modified_step_by_step_reasoning}}`;
                    }}
                    
                    modifiedReasoningSection += `</div></div>`;
                }}
                // Constraints detail for IFEval
                let constraintsSection = '';
                if (taskName === 'ifeval' && row.constraints_detail && row.constraints_detail !== '' && row.constraints_detail !== 'nan') {{
                    try {{
                        const cons = JSON.parse(row.constraints_detail);
                        if (Array.isArray(cons) && cons.length > 0) {{
                            const items = cons.map(c => {{
                                const o = c.orig ? '<span class="constraint-ok">✓</span>' : '<span class="constraint-fail">✗</span>';
                                const m = c.mod ? '<span class="constraint-ok">✓</span>' : '<span class="constraint-fail">✗</span>';
                                return `<div class=\"constraint-item\"><div class=\"constraint-id\">${{c.id}}</div><div class=\"constraint-status\">Orig: ${{o}} &nbsp; Mod: ${{m}}</div></div>`;
                            }}).join('');
                            constraintsSection = `<div class=\"constraints-section\"><div class=\"constraints-title\">Constraint Satisfaction</div>${{items}}</div>`;
                        }}
                    }} catch (e) {{
                        // ignore JSON parse errors
                    }}
                }}

                return `
                    <div class="comparison">
                        <div class="comparison-header">
                            <div>
                                <span class="badge badge-model">${{modelMap[row.model] || row.model}}</span>
                                <span class="badge badge-modification">${{modification}}</span> ${{ ((taskName === 'gsm' || taskName === 'ifeval') && modification.toLowerCase().startsWith('negation') && row.negation_subtype && row.negation_subtype !== 'nan') ? `<span class=\"badge badge-negation\">Negation: ${{row.negation_subtype}}</span>` : "" }}
                            </div>
                            <div>
                                <span class="badge badge-performance badge-${{performance}}">${{performance.replace(/_/g, ' ').toUpperCase()}}</span>
                            </div>
                        </div>
                        <div class="content-grid">
                            <div class="text-section">
                                <h3>Original Text (${{taskName.toUpperCase()}})</h3>
                                <div class="text-content">${{textLeftHtml}}</div>
                                <div class="prediction ${{originalCorrect ? 'correct' : 'incorrect'}}">
                                    Prediction: ${{predLeftHtml}}
                                </div>
                                <div class="ground-truth">
                                    <strong>Ground Truth:</strong> ${{originalLabel}}
                                </div>
                                ${{originalReasoningSection}}
                            </div>
                            <div class="text-section">
                                <h3>Modified Text</h3>
                                <div class="text-content">${{textRightHtml}}</div>
                                <div class="prediction ${{modifiedCorrect ? 'correct' : 'incorrect'}}">
                                    Prediction: ${{predRightHtml}}
                                </div>
                                <div class="ground-truth">
                                    <strong>Ground Truth:</strong> ${{modifiedLabel}}
                                </div>
                                ${{modifiedReasoningSection}}
                            </div>
                        </div>
                        ${{constraintsSection}}
                    </div>
                `;
            }}).join('');
            
            resultsContainer.innerHTML = html;
        }}
        
        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', initialize);
    </script>
</body>
</html>
"""
    
    # Write the HTML file
    with open('fluke_comparison_viewer.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ Created centralized HTML viewer: fluke_comparison_viewer.html")
    print(f"📊 Includes {len(json_data)} datasets with embedded sample data")
    print("🌐 Open the file in a web browser to use the interactive viewer")

def main():
    print("Creating centralized HTML viewer with embedded data...")
    print("=" * 60)
    create_centralized_html_viewer()

if __name__ == "__main__":
    main()
