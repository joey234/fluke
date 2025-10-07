#!/usr/bin/env python3
"""
Create a single centralized HTML viewer that loads CSV data dynamically with client-side filtering
"""

import pandas as pd
import html
import json
import re
import os
import glob

def format_dialogue_text_js_safe(text):
    """Format dialogue text for JavaScript embedding"""
    if pd.isna(text) or str(text).strip() == '':
        return ""
    
    text_str = str(text).strip()
    
    # Replace agent patterns with HTML line breaks
    formatted = re.sub(r'(agent\s+\d+:)', r'<br><strong>\\1</strong>', text_str, flags=re.IGNORECASE)
    
    # Remove leading <br> if it exists
    if formatted.startswith('<br>'):
        formatted = formatted[4:]
    
    # Escape for JavaScript
    formatted = formatted.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
    
    return formatted

def create_centralized_html_viewer():
    """Create a single HTML viewer that loads and filters all comparison data"""
    
    # Find all comparison CSV files (ignore backups and negation_change helpers)
    csv_files = [
        f for f in glob.glob("*_comparison_*.csv")
        if ('_backup.csv' not in f and 'negation_change' not in f)
    ]
    
    if not csv_files:
        print("No comparison CSV files found. Run create_comparison_all_models.py first.")
        return
    
    print(f"Found {len(csv_files)} comparison files to include in viewer")
    
    # Prepare data structure for JavaScript
    csv_data = {}
    
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
                    
            key = f"{model_name}_{task_name}"
            csv_data[key] = {
                'file': csv_file,
                'model': model_name,
                'task': task_name,
                'samples': len(df)
            }
            
            print(f"  {key}: {len(df)} samples")
            
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
        'claude': 'Claude-3.5',
        'claude-3-5-sonnet': 'Claude-3.5',
        'llama': 'Llama 3.1',
        'llama-3': 'Llama 3.1',
        'deepseek-r1': 'DS R1',
        'deepseek-r1-deepseek': 'DS R1',
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
        
        .badge-original-better {{ background-color: #d4edda; color: #155724; }}
        .badge-modified-better {{ background-color: #cce7ff; color: #004085; }}
        .badge-both-correct {{ background-color: #e6f3ff; color: #0066cc; }}
        .badge-both-wrong {{ background-color: #f8d7da; color: #721c24; }}
        
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
        
        .ground-truth {{
            margin-top: 10px;
            color: #666;
            font-size: 0.9em;
            padding: 8px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }}
        
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
                    <option value="punctuation_100">Punctuation</option>
                    <option value="sentiment_100">Sentiment</option>
                    <option value="temporal_bias_100">Temporal Bias</option>
                    <option value="typo_bias_100">Typo Bias</option>
                </select>
            </div>
            
            <div class="control-group">
                <label for="maxResults">Max Results:</label>
                <select id="maxResults">
                    <option value="50">50</option>
                    <option value="100" selected>100</option>
                    <option value="200">200</option>
                    <option value="500">500</option>
                    <option value="-1">All</option>
                </select>
            </div>
        </div>
    </div>

    <div class="stats" id="stats">
        <div class="stat-card">
            <h3>Total Samples</h3>
            <div class="stat-number" id="totalSamples">0</div>
        </div>
        <div class="stat-card">
            <h3>Models</h3>
            <div class="stat-number" id="totalModels">0</div>
        </div>
        <div class="stat-card">
            <h3>Tasks</h3>
            <div class="stat-number" id="totalTasks">0</div>
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

    <div class="results" id="results">
        <div class="loading">
            <div class="loading-spinner"></div>
            <div>Loading comparison data...</div>
        </div>
    </div>

    <script>
        // Available CSV files and their metadata
        const csvData = {json.dumps(csv_data, indent=8)};
        
        // Global variables
        let allData = [];
        let filteredData = [];
        let currentData = null;
        
        // Canonical display mapping for model names
        const modelMap = {json.dumps(model_display_map)};

        // Initialize the viewer
        async function initialize() {{
            try {{
                await loadInitialData();
                populateModelSelect();
                setupEventListeners();
                applyFilters();
            }} catch (error) {{
                console.error('Initialization error:', error);
                document.getElementById('results').innerHTML = `
                    <div class="no-results">
                        <h3>Error loading data</h3>
                        <p>${{error.message}}</p>
                    </div>
                `;
            }}
        }}
        
        // Load initial metadata and sample data
        async function loadInitialData() {{
            const totalSamples = Object.values(csvData).reduce((sum, data) => sum + data.samples, 0);
            const totalModels = new Set(Object.values(csvData).map(data => data.model)).size;
            const totalTasks = new Set(Object.values(csvData).map(data => data.task)).size;
            
            document.getElementById('totalSamples').textContent = totalSamples.toLocaleString();
            document.getElementById('totalModels').textContent = totalModels;
            document.getElementById('totalTasks').textContent = totalTasks;
        }}
        
        // Populate model select dropdown
        function populateModelSelect() {{
            const models = [...new Set(Object.values(csvData).map(data => data.model))].sort();
            const modelSelect = document.getElementById('modelSelect');
            
            models.forEach(model => {{
                const option = document.createElement('option');
                option.value = model;
                option.textContent = modelMap[model] || model;
                modelSelect.appendChild(option);
            }});
        }}
        
        // Setup event listeners for all controls
        function setupEventListeners() {{
            const controls = ['modelSelect', 'taskSelect', 'performanceSelect', 'modificationSelect', 'maxResults'];
            controls.forEach(controlId => {{
                document.getElementById(controlId).addEventListener('change', applyFilters);
            }});
        }}
        
        // Apply current filters and load data
        async function applyFilters() {{
            const filters = {{
                model: document.getElementById('modelSelect').value,
                task: document.getElementById('taskSelect').value,
                performance: document.getElementById('performanceSelect').value,
                modification: document.getElementById('modificationSelect').value,
                maxResults: parseInt(document.getElementById('maxResults').value)
            }};
            
            // Show loading
            document.getElementById('results').innerHTML = `
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <div>Loading filtered data...</div>
                </div>
            `;
            
            try {{
                await loadFilteredData(filters);
                updateStats();
                renderResults();
            }} catch (error) {{
                console.error('Filter error:', error);
                document.getElementById('results').innerHTML = `
                    <div class="no-results">
                        <h3>Error applying filters</h3>
                        <p>${{error.message}}</p>
                    </div>
                `;
            }}
        }}
        
        // Load data based on current filters
        async function loadFilteredData(filters) {{
            filteredData = [];
            
            // Determine which CSV files to load
            const filesToLoad = Object.entries(csvData).filter(([key, data]) => {{
                if (filters.model && data.model !== filters.model) return false;
                if (filters.task && data.task !== filters.task) return false;
                return true;
            }});
            
            if (filesToLoad.length === 0) {{
                return;
            }}
            
            // Load data from matching CSV files
            for (const [key, data] of filesToLoad) {{
                try {{
                    const csvText = await fetch(data.file).then(response => {{
                        if (!response.ok) throw new Error(`Failed to load ${{data.file}}`);
                        return response.text();
                    }});
                    
                    const rows = parseCSV(csvText);
                    
                    // Apply additional filters
                    const filtered = rows.filter(row => {{
                        if (filters.performance && row.performance !== filters.performance) return false;
                        if (filters.modification && row.modification !== filters.modification) return false;
                        return true;
                    }});
                    
                    filteredData.push(...filtered);
                }} catch (error) {{
                    console.warn(`Failed to load ${{data.file}}:`, error);
                }}
            }}
            
            // Apply max results limit
            if (filters.maxResults > 0 && filteredData.length > filters.maxResults) {{
                // Shuffle and limit results for variety
                filteredData = shuffleArray(filteredData).slice(0, filters.maxResults);
            }}
        }}
        
        // Simple CSV parser
        function parseCSV(csvText) {{
            const lines = csvText.split('\\n');
            if (lines.length < 2) return [];
            
            const headers = lines[0].split(',').map(h => h.replace(/"/g, '').trim());
            const rows = [];
            
            for (let i = 1; i < lines.length; i++) {{
                const line = lines[i].trim();
                if (!line) continue;
                
                const values = parseCSVLine(line);
                if (values.length !== headers.length) continue;
                
                const row = {{}};
                headers.forEach((header, idx) => {{
                    row[header] = values[idx];
                }});
                rows.push(row);
            }}
            
            return rows;
        }}
        
        // Parse a single CSV line (handles quoted values)
        function parseCSVLine(line) {{
            const result = [];
            let current = '';
            let inQuotes = false;
            
            for (let i = 0; i < line.length; i++) {{
                const char = line[i];
                
                if (char === '"') {{
                    inQuotes = !inQuotes;
                }} else if (char === ',' && !inQuotes) {{
                    result.push(current.trim());
                    current = '';
                }} else {{
                    current += char;
                }}
            }}
            
            result.push(current.trim());
            return result;
        }}
        
        // Shuffle array for random sampling
        function shuffleArray(array) {{
            const shuffled = [...array];
            for (let i = shuffled.length - 1; i > 0; i--) {{
                const j = Math.floor(Math.random() * (i + 1));
                [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
            }}
            return shuffled;
        }}
        
        // Update statistics display
        function updateStats() {{
            const stats = {{
                originalBetter: 0,
                modifiedBetter: 0,
                bothCorrect: 0,
                bothWrong: 0
            }};
            
            filteredData.forEach(row => {{
                if (row.performance in stats) {{
                    stats[row.performance]++;
                }}
            }});
            
            document.getElementById('originalBetter').textContent = stats.originalBetter || 0;
            document.getElementById('modifiedBetter').textContent = stats.modifiedBetter || 0;
            document.getElementById('bothCorrect').textContent = stats.bothCorrect || 0;
            document.getElementById('bothWrong').textContent = stats.bothWrong || 0;
        }}
        
        // Render filtered results
        function renderResults() {{
            const resultsContainer = document.getElementById('results');
            
            if (filteredData.length === 0) {{
                resultsContainer.innerHTML = `
                    <div class="no-results">
                        <h3>No samples found</h3>
                        <p>Try adjusting your filters to see more results.</p>
                    </div>
                `;
                return;
            }}
            
            const html = filteredData.map((row, idx) => {{
                const modelNameRaw = row.model || 'Unknown';
                const modelName = modelMap[modelNameRaw] || modelNameRaw;
                const taskName = row.task || 'Unknown';
                const originalCorrect = row.original_correct === 'True';
                const modifiedCorrect = row.modified_correct === 'True';
                const performance = row.performance || '';
                const modification = (row.modification || '').replace(/_/g, ' ').replace(/100/g, '').trim();
                
                const originalText = (row.original_text || '').replace(/"/g, '');
                const modifiedText = (row.modified_text || '').replace(/"/g, '');
                let originalPred = row.original_pred || '';
                let modifiedPred = row.modified_pred || '';
                // GSM: if CoT includes a #### answer, prefer it for display
                if (taskName === 'gsm') {{
                    const re = /####\s*[$€£¥₹₽]?\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?|[+-]?\d+\s*\/\s*\d+)/;
                    const oCot = (row.original_step_by_step_reasoning || '') + ' ' + (row.original_reasoning || '');
                    const mCot = (row.modified_step_by_step_reasoning || '') + ' ' + (row.modified_reasoning || '');
                    const om = oCot.match(re);
                    const mm = mCot.match(re);
                    if (om && om[1]) originalPred = om[1].replace(/,/g, '');
                    if (mm && mm[1]) modifiedPred = mm[1].replace(/,/g, '');
                }}
                
                // Format labels based on task
                let originalLabel = 'N/A';
                let modifiedLabel = 'N/A';
                
                if (taskName === 'gsm') {{
                    originalLabel = `Answer: ${{row.original_answer || ''}}`;
                    modifiedLabel = `Answer: ${{row.modified_answer || ''}}`;
                }} else if (taskName === 'ner') {{
                    originalLabel = `Entities: ${{row.original_entities || 'None'}}`;
                    modifiedLabel = `Entities: ${{row.modified_entities || 'None'}}`;
                }} else if (taskName === 'dialogue') {{
                    originalLabel = `Speaker: ${{row.original_dialog_label || ''}}`;
                    modifiedLabel = `Speaker: ${{row.modified_dialog_label || ''}}`;
                }} else if (taskName === 'sa') {{
                    originalLabel = `Sentiment: ${{row.original_label || ''}}`;
                    modifiedLabel = `Sentiment: ${{row.modified_label || ''}}`;
                }} else if (taskName === 'coref') {{
                    originalLabel = `Clusters: ${{row.original_clusters || 'None'}}`;
                    modifiedLabel = `Clusters: ${{row.modified_clusters || 'None'}}`;
                }} else if (taskName === 'ifeval') {{
                    const n = parseInt(row.num_constraints || '0') || 0;
                    const oc = parseInt(row.original_num_satisfied || '0') || 0;
                    const mc = parseInt(row.modified_num_satisfied || '0') || 0;
                    const orate = parseFloat(row.original_compliance_rate || '0') || 0;
                    const mrate = parseFloat(row.modified_compliance_rate || '0') || 0;
                    originalLabel = `Constraints: ${{oc}}/${{n}}; Compliance: ${{orate.toFixed(1)}}%`;
                    modifiedLabel = `Constraints: ${{mc}}/${{n}}; Compliance: ${{mrate.toFixed(1)}}%`;
                }}
                
                // Optional negation subtype badge for GSM/IFEVAL
                const subtype = (row.negation_subtype || row.type || '').toString();
                const subtypeBadge = (modification.toLowerCase().startsWith('negation') && (taskName === 'gsm' || taskName === 'ifeval') && subtype && subtype.toLowerCase() !== 'nan')
                    ? `<span class="badge" style="background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%); color: white;">Subtype: ${subtype}</span>`
                    : '';

                const perfClass = performance.replace(/_/g, '-');
                return `
                    <div class="comparison">
                        <div class="comparison-header">
                            <div>
                                <span class="badge badge-model">${{modelName}}</span>
                                <span class="badge badge-modification">${{modification}}</span>
                                ${{subtypeBadge}}
                            </div>
                            <div>
                                <span class="badge badge-performance badge-${{perfClass}}">${{performance.replace(/_/g, ' ').toUpperCase()}}</span>
                            </div>
                        </div>
                        <div class="content-grid">
                            <div class="text-section">
                                <h3>Original Text (${{taskName.toUpperCase()}})</h3>
                                <div class="text-content">${{originalText}}</div>
                                <div class="prediction ${{originalCorrect ? 'correct' : 'incorrect'}}">
                                    Prediction: ${{originalPred}}
                                </div>
                                <div class="ground-truth">
                                    <strong>Ground Truth:</strong> ${{originalLabel}}
                                </div>
                            </div>
                            <div class="text-section">
                                <h3>Modified Text</h3>
                                <div class="text-content">${{modifiedText}}</div>
                                <div class="prediction ${{modifiedCorrect ? 'correct' : 'incorrect'}}">
                                    Prediction: ${{modifiedPred}}
                                </div>
                                <div class="ground-truth">
                                    <strong>Ground Truth:</strong> ${{modifiedLabel}}
                                </div>
                            </div>
                        </div>
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
    print(f"📊 Includes {len(csv_data)} datasets")
    print("🌐 Open the file in a web browser to use the interactive viewer")

def main():
    print("Creating centralized HTML viewer with client-side filtering...")
    print("=" * 60)
    create_centralized_html_viewer()

if __name__ == "__main__":
    main()
