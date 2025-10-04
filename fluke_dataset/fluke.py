"""FLUKE: A Task-Agnostic Framework for Linguistic Capability Testing dataset."""

import json
import os
from typing import List, Dict, Any

import datasets


_CITATION = """\
@article{fluke2024,
  title={FLUKE: A Task-Agnostic Framework for Linguistic Capability Testing},
  author={Your Name and Co-author},
  journal={Conference/Journal Name},
  year={2024}
}
"""

_DESCRIPTION = """\
FLUKE (Framework for Linguistic Capability Testing) is a comprehensive evaluation dataset 
designed to test the robustness of language models across multiple linguistic dimensions. 
This dataset contains systematically modified test data across four core NLP tasks: 
Coreference Resolution, Named Entity Recognition (NER), Sentiment Analysis, and Dialogue Understanding.

The dataset applies 17 different types of linguistic modifications to evaluate model performance, 
revealing vulnerabilities that standard benchmarks might miss.
"""

_HOMEPAGE = "https://huggingface.co/datasets/your-username/fluke"

_LICENSE = "MIT"

_URLS = {
    "coref": "data/coref/",
    "ner": "data/ner/", 
    "sa": "data/sa/",
    "dialogue": "data/dialogue/"
}

_MODIFICATION_TYPES = [
    "active_to_passive", "capitalization", "casual", "compound_word", 
    "concept_replacement", "coordinating_conjunction", "derivation", 
    "dialectal", "discourse", "geographical_bias", "grammatical_role",
    "length_bias", "negation", "punctuation", "sentiment", 
    "temporal_bias", "typo_bias"
]

class FlukeDataset(datasets.GeneratorBasedBuilder):
    """FLUKE: A Task-Agnostic Framework for Linguistic Capability Testing dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="coref", 
            version=VERSION, 
            description="Coreference Resolution task with linguistic modifications"
        ),
        datasets.BuilderConfig(
            name="ner", 
            version=VERSION, 
            description="Named Entity Recognition task with linguistic modifications"
        ),
        datasets.BuilderConfig(
            name="sa", 
            version=VERSION, 
            description="Sentiment Analysis task with linguistic modifications"
        ),
        datasets.BuilderConfig(
            name="dialogue", 
            version=VERSION, 
            description="Dialogue Understanding task with linguistic modifications"
        ),
        datasets.BuilderConfig(
            name="all", 
            version=VERSION, 
            description="All tasks combined"
        ),
    ]

    DEFAULT_CONFIG_NAME = "all"

    def _info(self):
        features = datasets.Features({
            "original": datasets.Value("string"),
            "modified": datasets.Value("string"),
            "label": datasets.Value("string"),
            "modification_type": datasets.Value("string"),
            "task": datasets.Value("string"),
        })

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        
        if self.config.name == "all":
            # Return all tasks
            return [
                datasets.SplitGenerator(
                    name=task,
                    gen_kwargs={
                        "task": task,
                        "data_dir": _URLS[task]
                    },
                )
                for task in ["coref", "ner", "sa", "dialogue"]
            ]
        else:
            # Return specific task
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "task": self.config.name,
                        "data_dir": _URLS[self.config.name]
                    },
                ),
            ]

    def _generate_examples(self, task: str, data_dir: str):
        """Yields examples as (key, example) tuples."""
        
        key = 0
        
        for modification_type in _MODIFICATION_TYPES:
            file_path = os.path.join(data_dir, f"{modification_type}_100.json")
            
            # Check if file exists (some modifications might not exist for all tasks)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                for example in data:
                    yield key, {
                        "original": example.get("original_text", example.get("original", "")),
                        "modified": example.get("modified_text", example.get("modified", "")),
                        "label": str(example.get("modified_label", example.get("label", ""))),
                        "modification_type": example.get("test", modification_type),
                        "task": task,
                    }
                    key += 1 