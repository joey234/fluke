# LLM Experiments

This repository contains the code for the paper *FLUKE: A Task-Agnostic Framework for Linguistic Capability Testing*.

## Structure 


experiments/
├── .env # Environment configuration
└── LLM/
├── dialogue_prompt_cot.txt # Chain of thought dialogue prompts
├── coref_prompt_direct.txt # Direct coreference prompts
└── coref_prompt_cot.txt # Chain of thought coreference prompts

data/
├── dialogue_data.json # Dialogue data
├── coref_data.json # Coreference data
└── ... # Other data
data_generation/
├── dialogue_generation.py # Dialogue generation code
├── coref_generation.py # Coreference generation code
└── ... # Other generation code
README.md
