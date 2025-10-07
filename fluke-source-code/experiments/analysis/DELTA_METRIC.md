# FLUKE Weighted Delta Metric

## Formula

The FLUKE weighted delta metric is calculated as:

```
Weighted Delta = (B - A) × log₁₀(A) / log₁₀(100)
```

Where:
- **A** = Original/baseline accuracy (in percentage, 0-100)
- **B** = Modified accuracy (in percentage, 0-100)

## Interpretation

- **Negative values**: Performance drop (model performs worse on modified data)
- **Positive values**: Performance improvement (model performs better on modified data)
- **Magnitude**: Larger absolute values indicate more significant changes

## Key Properties

1. **Baseline-aware**: The metric considers the original accuracy level
   - Drops from high accuracy (e.g., 95% → 90%) are weighted more heavily
   - Drops from low accuracy (e.g., 30% → 25%) are weighted less

2. **Log-scaled**: Uses logarithmic scaling to account for the non-linear nature of accuracy improvements
   - It's harder to improve from 90% to 95% than from 50% to 55%

3. **Normalized**: Divided by log₁₀(100) to normalize the scale

## Examples

| Original | Modified | Absolute Drop | Relative Drop | Weighted Delta |
|----------|----------|--------------|---------------|----------------|
| 95%      | 90%      | 5%           | 5.3%          | -4.94          |
| 95%      | 70%      | 25%          | 26.3%         | -24.72         |
| 60%      | 45%      | 15%          | 25.0%         | -13.34         |
| 30%      | 20%      | 10%          | 33.3%         | -7.39          |
| 92%      | 89%      | 3%           | 3.3%          | -2.95          |

## Implementation in Analysis Scripts

The metric is implemented in:

1. **consolidated_analysis.py**: `analyze_performance_drop()` method
2. **utils.py**: `MetricsCalculator.performance_drop()` with `method='weighted_delta'`

### Usage Example

```python
from consolidated_analysis import FLUKEAnalyzer

analyzer = FLUKEAnalyzer()
drops_df = analyzer.analyze_performance_drop('coref', use_weighted_delta=True)

# Results include:
# - baseline_accuracy: Original accuracy
# - modified_accuracy: Accuracy on modified data
# - absolute_drop: Simple difference
# - relative_drop_%: Percentage change
# - weighted_delta: FLUKE weighted delta metric
```

## Why Use Weighted Delta?

1. **Fair Comparison**: Accounts for the difficulty of maintaining high accuracy
2. **Robustness Measure**: Better captures model robustness to linguistic modifications
3. **Research Standard**: Used in the FLUKE paper for all reported results

## Statistical Significance

The analysis also includes statistical tests (Wilcoxon, Mann-Whitney U) to determine if performance changes are statistically significant (p < 0.05).