# FLUKE LLM Benchmark Analysis Report
Generated: 2025-08-18 15:42:39

---
## Main Benchmark Results
Performance on full benchmark datasets (not modification subsets):

### Model Performance Ranking
| Rank | Model | Average | Tasks Evaluated |
|------|-------|---------|----------------|
| 1 | llama3_405B | 85.89% | 1 |
| 2 | deepseek-r1 | 81.48% | 1 |
| 3 | mixtral | 71.66% | 7 |
| 4 | gpt-5 | 69.84% | 4 |
| 5 | llama | 48.50% | 4 |
| 6 | claude | 39.08% | 7 |
| 7 | gpt4o | 31.38% | 21 |
| 8 | llama3.1_70b | 0.00% | 1 |
| 9 | llama3.1_8B | 0.00% | 1 |

### Detailed Benchmark Results
Accuracy (%) on main benchmark tasks:

| Model | 3-5-sonnet-cot-sst2 | 8x22b-cot-sst2 | 8x22b-sst2 | coref | cot-coref | cot-coref-active_to_passive | cot-coref-casual | cot-coref_capitalization | cot-coref_punctuation | cot-coref_typo_bias | cot-dialogue | cot-dialogue_capitalization | cot-dialogue_punctuation | cot-dialogue_typo_bias | cot-sst2 | cot-sst2_capitalization | cot-sst2_punctuation | cot-sst2_typo_bias | dialogue | ner | sst2 | sst2_capitalization | sst2_punctuation | sst2_typo_bias | Average |
|-------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| llama3_405B | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 85.89 | - | - | - | 85.89 |
| deepseek-r1 | - | - | - | 81.48 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 81.48 |
| mixtral | - | 88.53 | 83.37 | 61.02 | 0.00 | - | - | - | - | - | 88.41 | - | - | - | - | - | - | - | 85.89 | - | 94.38 | - | - | - | 71.66 |
| gpt-5 | - | - | - | 82.93 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 93.31 | 8.03 | 95.07 | - | - | - | 69.84 |
| llama | - | - | - | 0.00 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 91.72 | 6.77 | 95.53 | - | - | - | 48.50 |
| claude | 0.00 | - | - | 78.84 | 0.00 | - | - | - | - | - | 0.00 | - | - | - | - | - | - | - | 94.83 | 8.37 | 91.51 | - | - | - | 39.08 |
| gpt4o | - | - | - | 73.47 | 78.19 | 64.46 | 70.20 | 0.00 | 0.00 | 0.00 | 90.73 | 0.00 | 0.00 | 0.00 | 87.16 | 0.00 | 0.00 | 0.00 | 93.12 | 7.97 | 93.69 | 0.00 | 0.00 | 0.00 | 31.38 |
| llama3.1_70b | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 0.00 | - | - | - | 0.00 |
| llama3.1_8B | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 0.00 | - | - | - | 0.00 |

## Task-Specific Analysis

### Coreference Resolution
Top Performers:
1. **gpt-5**: 82.93%
2. **deepseek-r1**: 81.48%
3. **claude**: 78.84%
4. **gpt4o**: 78.19%
5. **mixtral**: 61.10%

Statistics:
- Average accuracy: 51.69%
- Best accuracy: 82.93%
- Models evaluated: 6

### Dialogue Understanding
Top Performers:
1. **claude**: 94.83%
2. **gpt-5**: 93.31%
3. **gpt4o**: 93.12%
4. **llama**: 91.72%
5. **mixtral**: 88.41%

Statistics:
- Average accuracy: 80.43%
- Best accuracy: 94.83%
- Models evaluated: 5

### Named Entity Recognition
Top Performers:
1. **claude**: 8.37%
2. **gpt-5**: 8.03%
3. **gpt4o**: 7.97%
4. **llama**: 6.77%

Statistics:
- Average accuracy: 7.78%
- Best accuracy: 8.37%
- Models evaluated: 4

### Sentiment Analysis
Top Performers:
1. **llama**: 95.53%
2. **gpt-5**: 95.07%
3. **mixtral**: 94.38%
4. **gpt4o**: 93.69%
5. **claude**: 91.51%

Statistics:
- Average accuracy: 71.47%
- Best accuracy: 95.53%
- Models evaluated: 8

## Frontier Models Analysis

### GPT-5 Performance
- **sst2**: 95.07%
- **dialogue**: 93.31%
- **coref**: 82.93%
- **ner**: 8.03%

**GPT-5 Overall Average**: 69.83%

### DeepSeek R1 Performance
- **coref**: 81.48%

**DeepSeek R1 Overall Average**: 81.48%

### GPT-5 vs GPT-4o Direct Comparison
| Benchmark | GPT-5 | GPT-4o | Difference |
|-----------|-------|--------|------------|
| coref | 82.93% | 73.47% | +9.46% |
| dialogue | 93.31% | 93.12% | +0.19% |
| ner | 8.03% | 7.97% | +0.06% |
| sst2 | 95.07% | 93.69% | +1.38% |

## Modification Robustness Analysis
Performance drop on linguistic modifications (100-sample tests):

### Model Robustness Ranking
Average accuracy across all modifications:

1. **gpt4o**: 1.08%
2. **claude**: 0.00%
3. **deepseek-r1**: 0.00%
4. **gpt-5**: 0.00%
5. **llama**: 0.00%
6. **mixtral**: 0.00%

## Key Insights
- **Best Overall Model**: llama3_405B (85.89% average)
- **Most Consistent Model**: mixtral (σ=29.55)
- **Most Challenging Task**: Named Entity Recognition (7.78% avg)
- **Easiest Task**: Dialogue Understanding (60.32% avg)
- **Chain-of-Thought Impact**: -28.65% average improvement
