#!/usr/bin/env python3
"""
Test script to verify reasoning is saved in CSV
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
    
    print(f"Testing reasoning CSV output...")
    
    # Get API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
    
    # Initialize client and detector
    client = GPT5Client(openai_api_key, MODEL_ID)
    detector = DialogueContradictionDetector(client, reasoning_effort=config.get('reasoning_effort', 'medium'))
    
    # Create a simple test result
    dialogue = "agent 0: Hello there! agent 1: Hi back!"
    response = detector.predict(dialogue)
    
    # Create result dictionary as done in the main script
    result = {
        'original_dialog': dialogue,
        'modified_dialog': dialogue,
        'modified_label': 0,
        'original_label': 0,
        'modified_pred': extract_classification_prediction(response['content']),
        'original_pred': extract_classification_prediction(response['content']),
        'raw_output': response['content'],
        'reasoning': response['reasoning'],
        'original_raw_output': response['content'],
        'original_reasoning': response['reasoning'],
        'type': 'test',
        'id': 0
    }
    
    print(f"Result keys: {list(result.keys())}")
    print(f"Reasoning present: {bool(result['reasoning'])}")
    print(f"Reasoning length: {len(result['reasoning'])}")
    
    # Create DataFrame and check columns
    df = pd.DataFrame([result])
    print(f"DataFrame columns: {list(df.columns)}")
    print(f"Reasoning column present: {'reasoning' in df.columns}")
    
    if 'reasoning' in df.columns:
        print(f"Sample reasoning in DataFrame: {df['reasoning'].iloc[0][:200]}...")
    
    # Save to CSV and verify
    test_file = '../results/dialogue/test_reasoning.csv'
    os.makedirs('../results/dialogue', exist_ok=True)
    df.to_csv(test_file, index=False)
    
    # Read back and verify
    df_read = pd.read_csv(test_file)
    print(f"Read back columns: {list(df_read.columns)}")
    print(f"Reasoning column after CSV read: {'reasoning' in df_read.columns}")
    
    if 'reasoning' in df_read.columns:
        print(f"Reasoning preserved in CSV: {len(str(df_read['reasoning'].iloc[0]))}")
        print(f"Reasoning content: {str(df_read['reasoning'].iloc[0])[:200]}...")
    else:
        print("ERROR: Reasoning column missing from CSV!")

if __name__ == "__main__":
    main()