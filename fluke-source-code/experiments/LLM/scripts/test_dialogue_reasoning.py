#!/usr/bin/env python3
"""
Test script to examine dialogue reasoning output
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import FLUKE utilities
from fluke_gpt5_utils import (
    remove_space, extract_classification_prediction,
    GPT5_MODELS, GPT5_CONFIGS, GPT5Client
)

# Load environment variables
load_dotenv()

class DialogueContradictionDetector:
    def __init__(self, client: GPT5Client, reasoning_effort: str = "medium"):
        self.client = client
        self.reasoning_effort = reasoning_effort
        self.prompt_template = """Given a dialogue with agent labels (agent 0 and agent 1 alternating), determine if the last utterance contradicts the dialogue context. Answer with 1 if it contradicts, 0 if it does not contradict.

Dialogue:
{dialogue}

Answer:"""
    
    def predict(self, dialogue: str) -> Dict[str, str]:
        """Predict dialogue contradiction for given dialogue"""
        prompt = self.prompt_template.format(dialogue=dialogue)
        response = self.client.generate(prompt, reasoning_effort=self.reasoning_effort)
        return response

def add_agent_labels(dialogue_list: List[str]) -> str:
    """Add agent 0/1 labels to each turn in the dialogue."""
    labeled_dialogue = []
    for i, turn in enumerate(dialogue_list):
        agent_label = f"agent {i % 2}: {turn}"
        labeled_dialogue.append(agent_label)
    return '\n'.join(labeled_dialogue)

def main():
    # Configuration
    CONFIG_NAME = 'standard'
    config = GPT5_CONFIGS[CONFIG_NAME]
    MODEL_NAME = config['model']
    MODEL_ID = GPT5_MODELS[MODEL_NAME]
    
    print(f"Configuration: {CONFIG_NAME}")
    print(f"Model: {MODEL_NAME} ({MODEL_ID})")
    print(f"Reasoning Effort: {config.get('reasoning_effort', 'medium')}")
    
    # Get API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
    
    # Initialize client and detector
    client = GPT5Client(openai_api_key, MODEL_ID)
    detector = DialogueContradictionDetector(client, reasoning_effort=config.get('reasoning_effort', 'medium'))
    
    # Load a single example
    data_path = '../../../data/train_dev_test_data/dialog/test.json'
    ds = pd.read_json(data_path)
    example = ds.iloc[59].to_dict()
    
    # Create dialogue with agent labels
    dialogue = add_agent_labels(example["dialogue"])
    dialogue_clean = remove_space(dialogue)
    
    print(f"\nExample dialogue with agent labels:")
    print(f"{dialogue}")
    print(f"\nLabel: {example['label']} ({'Contradiction' if example['label'] == 1 else 'No contradiction'})")
    
    # Get prediction
    test_pred = detector.predict(dialogue_clean)
    test_pred_label = extract_classification_prediction(test_pred['content'])
    
    print(f"\nPrediction: {test_pred_label}")
    print(f"\n" + "="*80)
    print("FULL RAW OUTPUT:")
    print("="*80)
    print(f"{test_pred['content']}")
    
    print(f"\n" + "="*80)
    print("REASONING AVAILABLE:")
    print("="*80)
    print(f"Reasoning available: {bool(test_pred.get('reasoning'))}")
    print(f"Reasoning content length: {len(test_pred.get('reasoning', ''))}")
    
    if test_pred.get('reasoning'):
        print(f"\n" + "="*80)
        print("FULL REASONING:")
        print("="*80)
        print(f"{test_pred['reasoning']}")
    else:
        print("No reasoning content found")
    
    print(f"\n" + "="*80)
    print("DEBUG INFO:")
    print("="*80)
    debug_info = test_pred.get('debug_reasoning', {})
    for key, value in debug_info.items():
        print(f"{key}: {value}")
    
    print(f"\nDebug fields: {test_pred.get('debug_fields', [])}")
    print(f"Reasoning effort used: {test_pred.get('reasoning_effort', 'unknown')}")

if __name__ == "__main__":
    main()