#!/usr/bin/env python3
"""
Fix for DeepSeek R1 signature to prevent None outputs in sentiment analysis.
This script provides an improved signature definition and better error handling.
"""

import dspy
from fluke_reasoning_utils import extract_classification_prediction

def create_improved_deepseek_signature():
    """Create an improved DeepSeek signature that prevents None outputs."""
    
    class DeepSeekSentiment(dspy.Signature):
        """Classify sentiment of the given text. Analyze the emotional tone, word choice, and overall sentiment. 
        You must respond with ONLY a single digit: 1 for positive sentiment, 0 for negative sentiment. 
        Do not include any other text or explanation."""
        text = dspy.InputField()
        label = dspy.OutputField(desc="The sentiment label: 1 for positive, 0 for negative")

    class DeepSeekSentimentModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.prog = dspy.Predict(DeepSeekSentiment)

        def forward(self, text):
            return self.prog(text=text)

    return DeepSeekSentimentModule()

def create_robust_eval_metric():
    """Create a robust evaluation metric that handles None outputs."""
    
    def eval_metric(true, prediction, trace=None):
        pred = prediction.label
        print(f"Raw prediction: {pred}")
        
        # Handle None or invalid predictions
        if pred is None:
            print("WARNING: Model returned None prediction")
            return False
        
        # Convert to string and clean
        pred_str = str(pred).strip()
        if not pred_str:
            print("WARNING: Model returned empty prediction")
            return False
        
        # Try to extract the classification
        parsed_answer = extract_classification_prediction(pred_str)
        if parsed_answer is None:
            print(f"WARNING: Could not parse prediction '{pred_str}'")
            return False
        
        result = parsed_answer == str(true.label)
        print(f"Parsed: {parsed_answer}, True: {true.label}, Correct: {result}")
        return result
    
    return eval_metric

def create_chain_of_thought_version():
    """Create a Chain-of-Thought version for better reasoning."""
    
    class CoTDeepSeekSentiment(dspy.Signature):
        """Classify sentiment of the given text. First think through the emotional tone, word choice, and overall sentiment.
        Then provide your final answer as ONLY a single digit: 1 for positive sentiment, 0 for negative sentiment."""
        text = dspy.InputField()
        reasoning = dspy.OutputField(desc="Your step-by-step reasoning about the sentiment")
        label = dspy.OutputField(desc="The final sentiment label: 1 for positive, 0 for negative")

    class CoTDeepSeekSentimentModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.prog = dspy.ChainOfThought(CoTDeepSeekSentiment)

        def forward(self, text):
            return self.prog(text=text)

    return CoTDeepSeekSentimentModule()

def create_fallback_signature():
    """Create a fallback signature with explicit formatting."""
    
    class FallbackDeepSeekSentiment(dspy.Signature):
        """Classify sentiment of the given text. 
        
        Instructions:
        1. Analyze the emotional tone, word choice, and overall sentiment
        2. Respond with ONLY a single digit: 1 for positive, 0 for negative
        3. Do not include any other text, punctuation, or explanation
        
        Example output: 1
        """
        text = dspy.InputField()
        label = dspy.OutputField(desc="Single digit: 1=positive, 0=negative")

    class FallbackDeepSeekSentimentModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.prog = dspy.Predict(FallbackDeepSeekSentiment)

        def forward(self, text):
            return self.prog(text=text)

    return FallbackDeepSeekSentimentModule()

if __name__ == "__main__":
    print("DeepSeek R1 Signature Fixes Available:")
    print("1. Improved signature with better description")
    print("2. Robust evaluation metric with None handling")
    print("3. Chain-of-Thought version for better reasoning")
    print("4. Fallback signature with explicit formatting")
    print("\nImport and use these functions in your notebook to fix the None output issue.") 