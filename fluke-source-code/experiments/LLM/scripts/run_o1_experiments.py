#!/usr/bin/env python3
"""
FLUKE Experiments Runner for OpenAI o1 Reasoning Models

This script automates running FLUKE robustness evaluations across different o1 models
and reasoning configurations.

Usage:
    python run_o1_experiments.py --task sentiment --model o1-preview --config standard
    python run_o1_experiments.py --task all --model o1-mini --samples 25
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path
import time

# Configuration for different o3 and o1 reasoning models
REASONING_CONFIGS = {
    'o3-2025-04-16': {
        'model_id': 'openai/o3-2025-04-16',
        'description': 'Latest o3 reasoning model',
        'cost_tier': 'high',
        'recommended_samples': 50
    },
    'o1-preview': {
        'model_id': 'openai/o1-preview',
        'description': 'Full o1 reasoning model',
        'cost_tier': 'high',
        'recommended_samples': 50
    },
    'o1-mini': {
        'model_id': 'openai/o1-mini', 
        'description': 'Efficient o1 reasoning model',
        'cost_tier': 'medium',
        'recommended_samples': 100
    },
    'o1': {
        'model_id': 'openai/o1',
        'description': 'Latest o1 model (when available)',
        'cost_tier': 'high',
        'recommended_samples': 50
    }
}

REASONING_MODES = {
    'standard': 'Standard reasoning approach',
    'detailed': 'Detailed step-by-step reasoning', 
    'efficient': 'Concise reasoning for faster processing'
}

AVAILABLE_TASKS = {
    'sentiment': {
        'notebook': 'llm_sentiment_o1.ipynb',
        'description': 'Sentiment analysis robustness testing'
    },
    'coref': {
        'notebook': 'llm_coref_o3.ipynb', 
        'description': 'Coreference resolution robustness testing'
    },
    'ner': {
        'notebook': 'llm_ner_o3.ipynb',
        'description': 'Named entity recognition robustness testing'
    },
    'dialogue': {
        'notebook': 'llm_dialogue_o3.ipynb',
        'description': 'Dialogue understanding robustness testing'
    }
}

def check_requirements():
    """Check if required packages and API keys are available."""
    required_packages = ['dspy', 'openai', 'pandas', 'datasets', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("OPENAI_API_KEY environment variable not set")
        return False
    
    return True

def estimate_cost(model, num_samples, task):
    """Provide rough cost estimates for o3 and o1 model usage."""
    # Rough estimates based on OpenAI pricing (update as needed)
    cost_per_sample = {
        'o3-2025-04-16': 0.08,  # Estimated - o3 may be more expensive
        'o1-preview': 0.05,     # Estimated
        'o1-mini': 0.01,        # Estimated  
        'o1': 0.05              # Estimated
    }
    
    base_cost = cost_per_sample.get(model, 0.02) * num_samples
    
    # Task complexity multipliers
    task_multipliers = {
        'sentiment': 1.0,
        'coref': 1.5,
        'ner': 1.2,
        'dialogue': 1.3
    }
    
    estimated_cost = base_cost * task_multipliers.get(task, 1.0)
    
    return estimated_cost

def run_notebook(notebook_path, parameters=None):
    """Execute a Jupyter notebook with optional parameters."""
    try:
        # Convert notebook to Python script and execute
        cmd = [
            'jupyter', 'nbconvert', 
            '--to', 'python',
            '--execute',
            '--stdout',
            str(notebook_path)
        ]
        
        print(f"Executing notebook: {notebook_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print("Notebook executed successfully")
            return True
        else:
            print(f"Notebook execution failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("Notebook execution timed out (1 hour limit)")
        return False
    except Exception as e:
        print(f"Error executing notebook: {e}")
        return False

def create_experiment_config(model, task, reasoning_mode, num_samples):
    """Create configuration for the experiment."""
    return {
        'model': model,
        'model_id': REASONING_CONFIGS[model]['model_id'],
        'task': task,
        'reasoning_mode': reasoning_mode,
        'num_samples': num_samples,
        'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S'),
        'estimated_cost': estimate_cost(model, num_samples, task)
    }

def main():
    parser = argparse.ArgumentParser(description='Run FLUKE experiments with OpenAI o3 and o1 reasoning models')
    
    parser.add_argument('--task', 
                       choices=list(AVAILABLE_TASKS.keys()) + ['all'],
                       default='sentiment',
                       help='Task to run (default: sentiment)')
    
    parser.add_argument('--model',
                       choices=list(REASONING_CONFIGS.keys()),
                       default='o3-2025-04-16', 
                       help='Reasoning model to use (default: o3-2025-04-16)')
    
    parser.add_argument('--config',
                       choices=list(REASONING_MODES.keys()),
                       default='standard',
                       help='Reasoning configuration (default: standard)')
    
    parser.add_argument('--samples',
                       type=int,
                       help='Number of samples to test (default: model-specific)')
    
    parser.add_argument('--modifications',
                       nargs='*',
                       help='Specific modifications to test (default: all available)')
    
    parser.add_argument('--dry-run',
                       action='store_true',
                       help='Show experiment plan without executing')
    
    parser.add_argument('--force',
                       action='store_true', 
                       help='Skip cost confirmation')
    
    args = parser.parse_args()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Determine number of samples
    if args.samples is None:
        args.samples = REASONING_CONFIGS[args.model]['recommended_samples']
    
    # Create experiment configuration
    if args.task == 'all':
        tasks_to_run = list(AVAILABLE_TASKS.keys())
    else:
        tasks_to_run = [args.task]
    
    # Show experiment plan
    print(\"=\" * 60)
    print(\"FLUKE o3/o1 Reasoning Model Experiments Plan\")
    print(\"=\" * 60)
    print(f\"Model: {args.model} ({REASONING_CONFIGS[args.model]['description']})\")\n    print(f\"Reasoning Mode: {args.config} ({REASONING_MODES[args.config]})\")\n    print(f\"Tasks: {', '.join(tasks_to_run)}\")\n    print(f\"Samples per task: {args.samples}\")\n    \n    total_cost = 0\n    for task in tasks_to_run:\n        cost = estimate_cost(args.model, args.samples, task)\n        total_cost += cost\n        print(f\"  - {task}: ~${cost:.2f}\")\n    \n    print(f\"\\nEstimated total cost: ~${total_cost:.2f}\")\n    print(f\"Note: Costs are rough estimates and may vary\")\n    \n    if args.dry_run:\n        print(\"\\nDry run - no experiments executed\")\n        return\n    \n    # Cost confirmation\n    if not args.force and total_cost > 10:\n        response = input(f\"\\nEstimated cost is ${total_cost:.2f}. Continue? (y/N): \")\n        if response.lower() != 'y':\n            print(\"Experiment cancelled\")\n            return\n    \n    # Execute experiments\n    print(\"\\n\" + \"=\"*60)\n    print(\"Starting Experiments\")\n    print(\"=\"*60)\n    \n    results_summary = []\n    \n    for task in tasks_to_run:\n        print(f\"\\nRunning {task} task...\")\n        \n        # Check if notebook exists\n        notebook_file = AVAILABLE_TASKS[task]['notebook']\n        notebook_path = Path(notebook_file)\n        \n        if not notebook_path.exists():\n            print(f\"Notebook not found: {notebook_path}\")\n            print(\"Make sure you're running from the correct directory\")\n            continue\n        \n        # Create experiment config\n        config = create_experiment_config(args.model, task, args.config, args.samples)\n        \n        # Save config for the notebook to read\n        config_file = f'experiment_config_{task}.json'\n        with open(config_file, 'w') as f:\n            json.dump(config, f, indent=2)\n        \n        print(f\"Configuration saved to: {config_file}\")\n        \n        # Run the experiment\n        success = run_notebook(notebook_path)\n        \n        results_summary.append({\n            'task': task,\n            'model': args.model,\n            'config': args.config,\n            'samples': args.samples,\n            'success': success,\n            'timestamp': config['timestamp']\n        })\n        \n        # Clean up config file\n        try:\n            os.remove(config_file)\n        except:\n            pass\n        \n        if success:\n            print(f\"✓ {task} completed successfully\")\n        else:\n            print(f\"✗ {task} failed\")\n        \n        # Add delay between tasks\n        if len(tasks_to_run) > 1:\n            print(\"Waiting 30 seconds before next task...\")\n            time.sleep(30)\n    \n    # Summary\n    print(\"\\n\" + \"=\"*60)\n    print(\"Experiment Summary\")\n    print(\"=\"*60)\n    \n    successful_tasks = [r for r in results_summary if r['success']]\n    failed_tasks = [r for r in results_summary if not r['success']]\n    \n    print(f\"Successful: {len(successful_tasks)}/{len(results_summary)}\")\n    \n    if successful_tasks:\n        print(\"\\nSuccessful tasks:\")\n        for result in successful_tasks:\n            print(f\"  ✓ {result['task']} ({result['model']}-{result['config']})\")\n    \n    if failed_tasks:\n        print(\"\\nFailed tasks:\")\n        for result in failed_tasks:\n            print(f\"  ✗ {result['task']} ({result['model']}-{result['config']})\")\n    \n    print(f\"\\nResults saved in results/ directory\")\n    print(f\"Look for files with prefix: {args.model}-{args.config}-\")\n\nif __name__ == '__main__':\n    main()