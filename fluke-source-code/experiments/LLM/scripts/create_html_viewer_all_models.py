#!/usr/bin/env python3
"""
Create HTML viewers for comparison CSV files with highlighted differences (all models)
"""

import pandas as pd
import difflib
import html
import json
import re
import os
import glob

def format_labels(label_data, task_name):
    """Format label data for display based on task type"""
    if pd.isna(label_data) or str(label_data).strip() == '':
        return "No labels"
    
    label_str = str(label_data).strip()
    
    if task_name == 'gsm':
        # GSM labels are numerical answers
        return f"Answer: {label_str}"
    elif task_name == 'ner':
        # Try to parse as JSON first
        try:
            if label_str.startswith('[') and label_str.endswith(']'):
                entities = json.loads(label_str)
                if isinstance(entities, list) and entities:
                    # Handle both formats: [{"text": "Name", "value": "TYPE"}] and [{"Name": "TYPE"}]
                    formatted = []
                    for entity in entities:
                        if isinstance(entity, dict):
                            if 'text' in entity and 'value' in entity:
                                # Format: {"text": "Name", "value": "TYPE"}
                                formatted.append(f"{entity['text']} ({entity['value']})")
                            else:
                                # Format: {"Name": "TYPE"}
                                for text, entity_type in entity.items():
                                    formatted.append(f"{text} ({entity_type})")
                    return ", ".join(formatted) if formatted else "No entities"
                else:
                    return "No entities"
            else:
                return label_str
        except (json.JSONDecodeError, KeyError):
            return label_str
    elif task_name == 'dialogue':
        # Dialogue labels are speaker/agent classifications
        return f"Speaker: {label_str}"
    elif task_name == 'sa':
        # Sentiment analysis labels
        return f"Sentiment: {label_str.title()}"
    elif task_name == 'coref':
        # Coreference resolution clusters
        try:
            if label_str.startswith('[') and label_str.endswith(']'):
                clusters = json.loads(label_str)
                if isinstance(clusters, list) and clusters:
                    cluster_strs = []
                    for i, cluster in enumerate(clusters):
                        if isinstance(cluster, list):
                            cluster_text = ", ".join([str(mention) for mention in cluster])
                            cluster_strs.append(f"Cluster {i+1}: [{cluster_text}]")
                    return "<br>".join(cluster_strs) if cluster_strs else "No clusters"
                else:
                    return "No clusters"
            else:
                return label_str
        except (json.JSONDecodeError, KeyError):
            return label_str
    else:
        return label_str

def format_dialogue_text(text):
    """Format dialogue text with proper line breaks after each agent"""
    if pd.isna(text) or str(text).strip() == '':
        return ""
    
    text_str = str(text).strip()
    
    # Replace agent patterns with HTML line breaks
    # Look for patterns like "agent 0:", "Agent 1:", etc.
    formatted = re.sub(r'(agent\s+\d+:)', r'<br><strong>\1</strong>', text_str, flags=re.IGNORECASE)
    
    # Remove leading <br> if it exists (use replace, not lstrip)
    if formatted.startswith('<br>'):
        formatted = formatted[4:]  # Remove exactly '<br>'
    
    # Escape HTML but preserve our formatting tags
    formatted = html.escape(formatted)
    formatted = formatted.replace('&lt;br&gt;', '<br>')
    formatted = formatted.replace('&lt;strong&gt;', '<strong>')
    formatted = formatted.replace('&lt;/strong&gt;', '</strong>')
    
    return formatted

def get_diff_html(original_text, modified_text, task_name):
    """Generate HTML showing differences between original and modified text"""
    if task_name == 'dialogue':
        original_formatted = format_dialogue_text(original_text)
        modified_formatted = format_dialogue_text(modified_text)
    else:
        original_formatted = html.escape(str(original_text) if not pd.isna(original_text) else "")
        modified_formatted = html.escape(str(modified_text) if not pd.isna(modified_text) else "")
    
    # Create diff
    diff = difflib.unified_diff(
        original_formatted.splitlines(keepends=True),
        modified_formatted.splitlines(keepends=True),
        fromfile='Original',
        tofile='Modified',
        lineterm=''
    )
    
    diff_html = []
    for line in diff:
        if line.startswith('+++') or line.startswith('---'):
            continue
        elif line.startswith('@@'):
            diff_html.append(f'<div class="diff-header">{html.escape(line)}</div>')
        elif line.startswith('+'):
            diff_html.append(f'<div class="diff-added">{line[1:]}</div>')
        elif line.startswith('-'):
            diff_html.append(f'<div class="diff-removed">{line[1:]}</div>')
        else:
            diff_html.append(f'<div class="diff-context">{line}</div>')
    
    return ''.join(diff_html) if diff_html else '<div class="diff-context">No differences detected</div>'

def create_html_viewer(csv_file, output_file, filter_performance=None, filter_modification=None):
    """Create HTML viewer for comparison CSV"""
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return
    
    # Apply filters
    filtered_df = df.copy()
    
    if filter_performance:
        filtered_df = filtered_df[filtered_df['performance'] == filter_performance]
        if len(filtered_df) == 0:
            print(f"No samples found for performance filter: {filter_performance}")
            return
    
    if filter_modification:
        filtered_df = filtered_df[filtered_df['modification'] == filter_modification]
        if len(filtered_df) == 0:
            print(f"No samples found for modification filter: {filter_modification}")
            return
    
    task_name = filtered_df['task'].iloc[0] if len(filtered_df) > 0 else 'unknown'
    
    # Handle backward compatibility for files without 'model' column
    if 'model' in filtered_df.columns:
        model_name = filtered_df['model'].iloc[0] if len(filtered_df) > 0 else 'unknown'
    else:
        # Extract model name from CSV filename
        import os
        csv_filename = os.path.basename(csv_file)
        if csv_filename.startswith('gpt5_context_aware_'):
            model_name = 'gpt-5-standard-context-aware'
        else:
            model_name = csv_filename.split('_comparison_')[0] if '_comparison_' in csv_filename else 'unknown'
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{model_name.upper()} Comparison Viewer - {task_name.upper()}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .model-badge {{
            background: rgba(255, 255, 255, 0.2);
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px 0;
            display: inline-block;
        }}
        .stats {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            flex: 1;
            min-width: 200px;
        }}
        .comparison {{
            background: white;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .comparison-header {{
            background-color: #f8f9fa;
            padding: 15px 20px;
            border-bottom: 2px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .modification-badge {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        .performance-badge {{
            padding: 5px 12px;
            border-radius: 15px;
            font-weight: bold;
            font-size: 0.8em;
        }}
        .performance-original_better {{ background-color: #d4edda; color: #155724; }}
        .performance-modified_better {{ background-color: #cce7ff; color: #004085; }}
        .performance-both_correct {{ background-color: #e6f3ff; color: #0066cc; }}
        .performance-both_wrong {{ background-color: #f8d7da; color: #721c24; }}
        .content-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0;
        }}
        .text-section {{
            padding: 20px;
        }}
        .text-section h3 {{
            margin-top: 0;
            margin-bottom: 15px;
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 8px;
        }}
        .text-content {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #007bff;
            margin-bottom: 15px;
            font-family: 'SF Mono', Monaco, monospace;
            white-space: pre-wrap;
            line-height: 1.5;
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
            border-left: 4px solid #28a745;
        }}
        .prediction.incorrect {{
            background-color: #f8d7da;
            color: #721c24;
            border-left: 4px solid #dc3545;
        }}
        .diff-section {{
            grid-column: 1 / -1;
            padding: 20px;
            background-color: #f8f9fa;
            border-top: 1px solid #e9ecef;
        }}
        .diff-content {{
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.9em;
            line-height: 1.4;
        }}
        .diff-added {{
            background-color: #d4edda;
            color: #155724;
            padding: 2px 4px;
            margin: 1px 0;
        }}
        .diff-removed {{
            background-color: #f8d7da;
            color: #721c24;
            padding: 2px 4px;
            margin: 1px 0;
        }}
        .diff-context {{
            background-color: #f8f9fa;
            padding: 2px 4px;
            margin: 1px 0;
        }}
        .diff-header {{
            background-color: #e9ecef;
            padding: 2px 4px;
            font-weight: bold;
            margin: 1px 0;
        }}
        .reasoning-section {{
            grid-column: 1 / -1;
            padding: 20px;
            background-color: #fff;
            border-top: 1px solid #e9ecef;
        }}
        .reasoning-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .reasoning-content {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #6c757d;
            font-family: 'SF Mono', Monaco, monospace;
            white-space: pre-wrap;
            line-height: 1.4;
            max-height: 300px;
            overflow-y: auto;
            font-size: 0.9em;
        }}
        @media (max-width: 768px) {{
            .content-grid, .reasoning-grid {{
                grid-template-columns: 1fr;
            }}
            .comparison-header {{
                flex-direction: column;
                align-items: stretch;
                text-align: center;
            }}
            .stats {{
                flex-direction: column;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{model_name.upper()} Comparison Viewer</h1>
        <div class="model-badge">{model_name}</div>
        <h2>{task_name.upper()} Task</h2>
        <p>Analyzing original vs modified predictions with highlighted differences</p>
    </div>
"""
    
    # Add statistics
    total_samples = len(filtered_df)
    perf_counts = filtered_df['performance'].value_counts()
    
    html_content += f"""
    <div class="stats">
        <div class="stat-card">
            <h3>Total Samples</h3>
            <h2>{total_samples}</h2>
        </div>
        <div class="stat-card">
            <h3>Original Better</h3>
            <h2>{perf_counts.get('original_better', 0)}</h2>
        </div>
        <div class="stat-card">
            <h3>Modified Better</h3>
            <h2>{perf_counts.get('modified_better', 0)}</h2>
        </div>
        <div class="stat-card">
            <h3>Both Correct</h3>
            <h2>{perf_counts.get('both_correct', 0)}</h2>
        </div>
        <div class="stat-card">
            <h3>Both Wrong</h3>
            <h2>{perf_counts.get('both_wrong', 0)}</h2>
        </div>
    </div>
"""
    
    # Add each comparison
    for idx, row in filtered_df.iterrows():
        original_correct = row.get('original_correct', False)
        modified_correct = row.get('modified_correct', False)
        performance = row.get('performance', '')
        modification = row.get('modification', '')
        
        # Get labels based on task
        if task_name == 'gsm':
            original_label = format_labels(row.get('original_answer', ''), task_name)
            modified_label = format_labels(row.get('modified_answer', ''), task_name)
        elif task_name == 'dialogue':
            original_label = format_labels(row.get('original_dialog_label', ''), task_name)
            modified_label = format_labels(row.get('modified_dialog_label', ''), task_name)
        elif task_name == 'ner':
            original_label = format_labels(row.get('original_entities', ''), task_name)
            modified_label = format_labels(row.get('modified_entities', ''), task_name)
        elif task_name == 'sa':
            original_label = format_labels(row.get('original_label', ''), task_name)
            modified_label = format_labels(row.get('modified_label', ''), task_name)
        elif task_name == 'coref':
            original_label = format_labels(row.get('original_clusters', ''), task_name)
            modified_label = format_labels(row.get('modified_clusters', ''), task_name)
        else:
            original_label = "N/A"
            modified_label = "N/A"
        
        diff_html = get_diff_html(row.get('original_text', ''), row.get('modified_text', ''), task_name)
        
        html_content += f"""
    <div class="comparison">
        <div class="comparison-header">
            <div class="modification-badge">{modification.replace('_', ' ').title()}</div>
            <div class="performance-badge performance-{performance}">{performance.replace('_', ' ').title()}</div>
        </div>
        <div class="content-grid">
            <div class="text-section">
                <h3>Original Text</h3>
                <div class="text-content">{''.join(format_dialogue_text(row.get('original_text', '')) if task_name == 'dialogue' else html.escape(str(row.get('original_text', ''))))}</div>
                <div class="prediction {'correct' if original_correct else 'incorrect'}">
                    Prediction: {html.escape(str(row.get('original_pred', '')))}
                </div>
                <div style="margin-top: 10px; color: #666; font-size: 0.9em;">
                    <strong>Ground Truth:</strong> {original_label}
                </div>
            </div>
            <div class="text-section">
                <h3>Modified Text</h3>
                <div class="text-content">{''.join(format_dialogue_text(row.get('modified_text', '')) if task_name == 'dialogue' else html.escape(str(row.get('modified_text', ''))))}</div>
                <div class="prediction {'correct' if modified_correct else 'incorrect'}">
                    Prediction: {html.escape(str(row.get('modified_pred', '')))}
                </div>
                <div style="margin-top: 10px; color: #666; font-size: 0.9em;">
                    <strong>Ground Truth:</strong> {modified_label}
                </div>
            </div>
            <div class="diff-section">
                <h3>Text Differences</h3>
                <div class="diff-content">
                    {diff_html}
                </div>
            </div>
            <div class="reasoning-section">
                <h3>Model Reasoning</h3>
                <div class="reasoning-grid">
                    <div>
                        <h4>Original Reasoning</h4>
                        <div class="reasoning-content">
{html.escape(str(row.get('original_reasoning', 'No reasoning available')) if not pd.isna(row.get('original_reasoning')) else 'No reasoning available')}
                        </div>
                        <h4 style="margin-top: 15px;">Original Step-by-Step</h4>
                        <div class="reasoning-content">
{html.escape(str(row.get('original_step_by_step_reasoning', 'No step-by-step reasoning available')) if not pd.isna(row.get('original_step_by_step_reasoning')) else 'No step-by-step reasoning available')}
                        </div>
                    </div>
                    <div>
                        <h4>Modified Reasoning</h4>
                        <div class="reasoning-content">
{html.escape(str(row.get('modified_reasoning', 'No reasoning available')) if not pd.isna(row.get('modified_reasoning')) else 'No reasoning available')}
                        </div>
                        <h4 style="margin-top: 15px;">Modified Step-by-Step</h4>
                        <div class="reasoning-content">
{html.escape(str(row.get('modified_step_by_step_reasoning', 'No step-by-step reasoning available')) if not pd.isna(row.get('modified_step_by_step_reasoning')) else 'No step-by-step reasoning available')}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    # Write HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Created HTML viewer: {output_file} ({len(filtered_df)} samples)")

def main():
    print("Creating HTML viewers for all model comparison files...")
    print("=" * 50)
    
    # Find all comparison CSV files
    csv_files = glob.glob("*_comparison_*.csv")
    
    if not csv_files:
        print("No comparison CSV files found. Run create_comparison_all_models.py first.")
        return
    
    # Performance filters
    performance_filters = ['original_better', 'modified_better', 'both_wrong']
    
    # Modification types  
    modification_types = [
        'active_to_passive_100', 'capitalization_100', 'casual_100', 'compound_word_100',
        'concept_replacement_100', 'coordinating_conjunction_100', 'derivation_100',
        'dialectal_100', 'discourse_100', 'geographical_bias_100', 'grammatical_role_100',
        'length_bias_100', 'negation_100', 'punctuation_100', 'sentiment_100',
        'temporal_bias_100', 'typo_bias_100'
    ]
    
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file}...")
        
        # Extract model and task from filename
        base_name = csv_file.replace('.csv', '')
        
        # Create base viewer
        base_html = f"{base_name}_viewer.html"
        create_html_viewer(csv_file, base_html)
        
        # Create performance-specific viewers
        for perf_filter in performance_filters:
            perf_html = f"{base_name}_viewer_{perf_filter}.html"
            create_html_viewer(csv_file, perf_html, filter_performance=perf_filter)
        
        # Create modification-specific viewers
        for mod_type in modification_types:
            mod_html = f"{base_name}_viewer_mod_{mod_type}.html"
            create_html_viewer(csv_file, mod_html, filter_modification=mod_type)
            
            # Create combined filters (modification + performance)
            for perf_filter in performance_filters:
                combined_html_path = f"{base_name}_viewer_mod_{mod_type}_{perf_filter}.html"
                create_html_viewer(csv_file, combined_html_path, 
                                 filter_performance=perf_filter, 
                                 filter_modification=mod_type)
    
    print("\n" + "=" * 50)
    print("HTML viewers created successfully!")

if __name__ == "__main__":
    main()