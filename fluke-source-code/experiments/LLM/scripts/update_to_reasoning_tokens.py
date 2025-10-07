#!/usr/bin/env python3
"""
Quick script to update all OpenRouter scripts to use reasoning tokens
"""

import os

scripts = [
    'run_coref_openrouter.py',
    'run_dialogue_openrouter.py', 
    'run_ner_openrouter.py'
]

for script in scripts:
    print(f"Updating {script}...")
    
    with open(script, 'r') as f:
        content = f.read()
    
    # Update generate method signature and implementation
    content = content.replace(
        'def generate(self, prompt: str, max_tokens: int = 20000, temperature: float = 1.0) -> str:',
        'def generate(self, prompt: str, max_tokens: int = 20000, temperature: float = 1.0, use_reasoning: bool = True) -> Dict[str, str]:'
    )
    
    content = content.replace(
        '"""Generate response from OpenRouter API"""',
        '"""Generate response from OpenRouter API with reasoning support"""'
    )
    
    # Replace the payload and response handling
    old_payload = '''        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"API Error: {e}")
            return ""'''
    
    new_payload = '''        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Add reasoning support for compatible models
        if use_reasoning and ("deepseek" in self.model.lower() or "o1" in self.model.lower()):
            payload["reasoning"] = {
                "effort": "high",
                "max_tokens": 2000,
                "exclude": False
            }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            message = result["choices"][0]["message"]
            return {
                "content": message.get("content", ""),
                "reasoning": message.get("reasoning", "")
            }
        except Exception as e:
            print(f"API Error: {e}")
            return {"content": "", "reasoning": ""}'''
    
    content = content.replace(old_payload, new_payload)
    
    # Update class constructors to remove use_cot parameter
    content = content.replace(
        'def __init__(self, client: OpenRouterClient, use_cot: bool = False):',
        'def __init__(self, client: OpenRouterClient):'
    )
    
    # Update predict method return types
    content = content.replace(
        'def predict(self, text: str) -> str:',
        'def predict(self, text: str) -> Dict[str, str]:'
    )
    content = content.replace(
        'def predict(self, text: str, pronoun: str, candidates: str) -> str:',
        'def predict(self, text: str, pronoun: str, candidates: str) -> Dict[str, str]:'
    )
    content = content.replace(
        'def predict(self, dialogue: str) -> str:',
        'def predict(self, dialogue: str) -> Dict[str, str]:'
    )
    
    # Update prompt handling
    content = content.replace(
        'response = self.client.generate(prompt)\n        return response',
        'response = self.client.generate(prompt)\n        return response'
    )
    
    print(f"Updated {script}")

print("All scripts updated! Please manually verify and adjust as needed.")