#!/usr/bin/env python3
"""
LLM-based answer extraction using moonshotai/kimi-k2 model
This provides a much more robust solution than regex-based parsing
"""

import openai
import re
import time
import logging
from typing import Optional
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAnswerExtractor:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        """Initialize the LLM answer extractor"""
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = "moonshotai/kimi-k2"  # Using moonshotai/kimi-k2 via OpenRouter
        
    def extract_answer(self, question_text: str, raw_output: str, max_retries: int = 3) -> str:
        """
        Extract the final numerical answer from the raw output using LLM
        
        Args:
            question_text: The original math problem/question
            raw_output: The model's response containing the solution
            max_retries: Number of retry attempts
            
        Returns:
            String containing just the numerical answer (e.g., "975", "24500", "60")
        """
        
        prompt = f"""You are an expert at extracting final numerical answers from mathematical solutions.

QUESTION:
{question_text}

SOLUTION PROVIDED:
{raw_output}

TASK: Extract ONLY the final numerical answer to the question. Return just the number, without any units, currency symbols, or additional text.

EXAMPLES:
- If the answer is "$975", return: 975
- If the answer is "60%", return: 60  
- If the answer is "24,500 fils", return: 24500
- If the answer is "3.5 hours", return: 3.5

RULES:
1. Return ONLY the numerical value
2. Remove currency symbols ($, €, £, etc.)
3. Remove units (km, hours, %, etc.)  
4. Remove commas from large numbers
5. If there are multiple numbers in the conclusion, choose the one that directly answers the question
6. For percentage questions, return the percentage value (60% → 60)

FINAL ANSWER (number only):"""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,  # Deterministic output
                    max_tokens=50,    # Short answer expected
                    timeout=30
                )
                
                answer = response.choices[0].message.content.strip()
                
                # Clean and validate the answer
                cleaned_answer = self._clean_and_validate_answer(answer)
                if cleaned_answer:
                    return cleaned_answer
                
                logger.warning(f"Attempt {attempt + 1}: Invalid answer format: {answer}")
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(0.5 + (0.5 * attempt))  # Lighter backoff for parallel processing
                
        # Fallback to regex if LLM fails
        logger.warning("LLM extraction failed, falling back to regex")
        return self._fallback_regex_extraction(raw_output)
    
    def _clean_and_validate_answer(self, answer: str) -> Optional[str]:
        """Clean and validate the LLM's answer"""
        if not answer:
            return None
            
        # Remove any remaining non-numeric characters except decimal points and negative signs
        cleaned = re.sub(r'[^\d.-]', '', answer)
        
        # Validate it's a proper number
        try:
            float(cleaned)
            return cleaned
        except ValueError:
            return None
    
    def _fallback_regex_extraction(self, text: str) -> str:
        """Fallback regex extraction (simplified version of our previous logic)"""
        if not text:
            return "0"
            
        # Look for explicit answer patterns
        answer_patterns = [
            r'[Tt]herefore,?\s*.*?(\d+(?:\.\d+)?)',
            r'[Aa]nswer.*?(\d+(?:\.\d+)?)',
            r'[Ff]inal.*?(\d+(?:\.\d+)?)',
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1]
        
        # Last resort: find any number
        numbers = re.findall(r'(\d+(?:\.\d+)?)', text)
        return numbers[-1] if numbers else "0"


def create_llm_extractor() -> LLMAnswerExtractor:
    """Create an LLM extractor instance with API key from .env file"""
    # Load API key from environment variable (set in .env file)
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    if not api_key:
        raise ValueError(
            "Please set OPENROUTER_API_KEY in your .env file or environment variables.\n"
            "Example .env file content:\n"
            "OPENROUTER_API_KEY=your_actual_openrouter_api_key_here\n"
            "Get your API key from: https://openrouter.ai/"
        )
    
    return LLMAnswerExtractor(api_key)


# Test function
def test_llm_extractor():
    """Test the LLM extractor with our problematic samples"""
    
    test_cases = [
        {
            "question": "Leilani invested 300 vatu into a scheme that accumulates simple interest at the rate of three-quarters of the original amount per year. How much in total will she have after 3 years?",
            "raw_output": """Let me solve this step by step.

1) First, let's identify the key information:
   * Principal (P) = 300 vatu
   * Rate (r) = 3/4 = 0.75 (or 75%) per year
   * Time (t) = 3 years

2) For simple interest, we use the formula:
   * Total Amount = Principal + Interest
   * Where Interest = Principal × Rate × Time

3) Calculate the Interest:
   * Interest = 300 × 0.75 × 3
   * Interest = 300 × 2.25
   * Interest = 675 vatu

4) Calculate Total Amount:
   * Total Amount = Principal + Interest
   * Total Amount = 300 + 675
   * Total Amount = 975 vatu

Therefore, after 3 years, Leilani will have 975 vatu in total.""",
            "expected": "975",
            "description": "Leilani geographical bias case"
        },
        {
            "question": "Ali is distributing vouchers for his pizza restaurant via post. He plans to distribute 700 small vouchers and twice as many large vouchers. If each small voucher costs 5 fils to send and each large voucher costs 15 fils, how much does he spend on postage total?",
            "raw_output": """Let me solve this step by step:

1. First, let's determine the number of large vouchers:
* Small vouchers = 700
* Large vouchers = 2 × 700 = 1,400 vouchers

2. Calculate the cost for small vouchers:
* Cost per small voucher = 5 fils
* Total cost for small vouchers = 700 × 5 = 3,500 fils

3. Calculate the cost for large vouchers:
* Cost per large voucher = 15 fils
* Total cost for large vouchers = 1,400 × 15 = 21,000 fils

4. Add both costs to get total postage:
* Total postage = Cost of small vouchers + Cost of large vouchers
* Total postage = 3,500 + 21,000 = 24,500 fils

Therefore, Ali will spend 24,500 fils on postage in total.""",
            "expected": "24500",
            "description": "Ali comma-separated numbers case"
        },
        {
            "question": "Amadou is traveling on the national road at an average speed of 50km/hr (taking into account all the necessary stops for fuel, rest, etc.). If he continues like this for 3 days, and the distance between Niamey and Zinder is 6000 km, what percentage of this distance has he covered?",
            "raw_output": """Let me solve this step by step.

1. First, let's find out how many hours Amadou travels in 3 days:
   * 1 day = 24 hours
   * 3 days = 24 × 3 = 72 hours

2. Next, let's calculate the distance Amadou covers in 72 hours:
   * Distance = Speed × Time
   * Distance = 50 km/hr × 72 hours
   * Distance = 3,600 km

3. Now we can calculate what percentage 3,600 km is of 6,000 km:
   * Percentage = (Distance covered ÷ Total distance) × 100
   * Percentage = (3,600 ÷ 6,000) × 100
   * Percentage = 0.6 × 100
   * Percentage = 60%

Therefore, Amadou has covered 60% of the total distance between Niamey and Zinder.

Note: The actual distance between Niamey and Zinder is much less than 6,000 km (it's around 900 km), but I solved the problem using the given values in the question.""",
            "expected": "60",
            "description": "Amadou percentage case"
        }
    ]
    
    try:
        extractor = create_llm_extractor()
        
        print("Testing LLM Answer Extractor")
        print("=" * 60)
        
        all_passed = True
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test_case['description']}")
            print("-" * 40)
            
            result = extractor.extract_answer(
                test_case["question"], 
                test_case["raw_output"]
            )
            
            print(f"Expected: {test_case['expected']}")
            print(f"LLM Result: {result}")
            
            is_correct = (result == test_case["expected"])
            if is_correct:
                print("✅ PASSED")
            else:
                print("❌ FAILED")
                all_passed = False
        
        print("\n" + "=" * 60)
        if all_passed:
            print("🎉 All LLM extraction tests PASSED!")
        else:
            print("❌ Some LLM extraction tests FAILED.")
            
    except ValueError as e:
        print(f"Error: {e}")
        print("\nTo set up your API key, create a .env file in this directory with:")
        print("OPENROUTER_API_KEY=your_actual_openrouter_api_key_here")


if __name__ == "__main__":
    test_llm_extractor()