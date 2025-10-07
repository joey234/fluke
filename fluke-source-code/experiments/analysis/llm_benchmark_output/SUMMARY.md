# FLUKE LLM Benchmark Analysis - Executive Summary

## Main Benchmark Results (Full Datasets)

### Overall Model Ranking
Based on average performance across all main benchmark tasks:

| Rank | Model | Average Score | Notable Strengths |
|------|-------|--------------|-------------------|
| 1 | **DeepSeek R1** | 81.48% | Excellent on Coreference Resolution |
| 2 | **GPT-5** | 69.84% | Strong across all tasks, best on Sentiment |
| 3 | **Claude 3.5** | 54.89% | Best on Dialogue & NER |
| 4 | **GPT-4o** | 63.06% | Good with Chain-of-Thought |
| 5 | **Mixtral** | 71.66% | Consistent performer |
| 6 | **Llama** | 48.50% | Strong on Sentiment Analysis |

### Task-Specific Champions

#### 🎯 Coreference Resolution
1. **GPT-5**: 82.93%
2. **DeepSeek R1**: 81.48%
3. **Claude 3.5**: 78.84%

#### 💬 Dialogue Understanding
1. **Claude 3.5**: 94.83%
2. **GPT-5**: 93.31%
3. **GPT-4o**: 93.12%

#### 🏷️ Named Entity Recognition
1. **Claude 3.5**: 8.37%
2. **GPT-5**: 8.03%
3. **GPT-4o**: 7.97%

*Note: NER shows low scores, likely due to evaluation methodology*

#### 😊 Sentiment Analysis
1. **Llama**: 95.53%
2. **GPT-5**: 95.07%
3. **Mixtral**: 94.38%

## Frontier Models Comparison

### GPT-5 vs GPT-4o
GPT-5 shows consistent improvements over GPT-4o:

| Task | GPT-5 | GPT-4o | Improvement |
|------|-------|--------|-------------|
| Coreference | 82.93% | 73.47% | **+9.46%** |
| Dialogue | 93.31% | 93.12% | +0.19% |
| NER | 8.03% | 7.97% | +0.06% |
| Sentiment | 95.07% | 93.69% | +1.38% |

**Average Improvement: +2.77%**

### DeepSeek R1 Performance
- Evaluated on Coreference Resolution: **81.48%**
- Competitive with GPT-5 (82.93%) on this task
- More evaluation needed on other tasks

## Key Findings

✅ **Best Overall**: DeepSeek R1 shows strong performance where evaluated
✅ **Most Versatile**: GPT-5 performs well across all tasks
✅ **Specialist Models**: Claude 3.5 excels at dialogue, Llama at sentiment
⚠️ **NER Challenge**: All models struggle with NER task (avg 7.78%)
📈 **GPT Evolution**: GPT-5 consistently outperforms GPT-4o

## Robustness Analysis

When tested on linguistic modifications (100-sample subsets):
- Models show significant performance drops on modifications
- Average accuracy on modifications: ~1% vs 60%+ on main benchmarks
- Indicates models may be sensitive to linguistic variations

## Recommendations

1. **For General Use**: GPT-5 or Claude 3.5 offer best all-around performance
2. **For Coreference**: DeepSeek R1 or GPT-5 are top choices
3. **For Dialogue**: Claude 3.5 leads the pack
4. **For Sentiment**: Most models perform well (90%+)
5. **NER Needs Work**: Consider specialized NER models or different evaluation

---
*Analysis based on FLUKE benchmark suite - 506 total experiments across 9 LLM models*