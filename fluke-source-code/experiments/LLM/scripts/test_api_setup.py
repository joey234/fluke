#!/usr/bin/env python3
"""
Simple test to check OpenRouter API setup
"""
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if API key exists
api_key = os.getenv('OPENROUTER_API_KEY')
if not api_key:
    print("❌ OPENROUTER_API_KEY not found in environment variables")
    print("Please create a .env file with: OPENROUTER_API_KEY=your_key_here")
    exit(1)

print(f"✅ API key found: {api_key[:8]}...")

# Test simple API call
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Test different DeepSeek model IDs
model_ids = [
    "deepseek/deepseek-r1",
    "deepseek/deepseek-v3",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-coder",
    "deepseek-chat",
    "deepseek-coder"
]

for model_id in model_ids:
    print(f"\nTesting model: {model_id}")
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Hello, can you say hi?"}],
        "max_tokens": 100
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        print("✅ API call successful!")
        print(f"Response: {result['choices'][0]['message']['content']}")
        print(f"✅ Working model found: {model_id}")
        break
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                print(f"Error: {error_details.get('error', {}).get('message', 'Unknown error')}")
            except:
                print(f"Response text: {e.response.text}")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")