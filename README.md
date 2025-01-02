# LLM Experiments

This repository contains the code for the paper *FLUKE: A Task-Agnostic Framework for Linguistic Capability Testing*.

## Structure 


### Data
This folder contains the data generated based on the FLUKE framework.

```
data/
├── modified_data.json # Modified data
    ├── coref
    ├── dialogue
    ├── sa
    ├── ner
├── scripts/ # Scripts for data analysis
```

### Data Generation
This folder contains the script and prompt for generating data following the FLUKE framework.

```
data_generation/
├── dialogue_prompt.ipynb # Dialogue generation code
├── coref_prompt.ipynb # Coreference generation code
├── ner_prompt.ipynb # NER generation code
├── sentiment_prompt.ipynb # Sentiment generation code

experiments/
├── .env # Environment configuration
├── LLM/ # LLM experiments
├── PLM/ # PLM experiments
├── analysis/ # Results analysis scripts
```

### Usage

To generate all the results reported in the paper, run the following notebook:
```
experiments/analysis/parse_coref_dialog.ipynb
experiments/analysis/parse_ner.ipynb
experiments/analysis/parse_sa.ipynb
```

To run the LLM experiments, run the following notebook:
```
experiments/LLM/llm_{task}_{model}.ipynb
```

To run the PLM experiments, run the following notebook:
```
experiments/PLM/{task}/eval_{model}.py
```

