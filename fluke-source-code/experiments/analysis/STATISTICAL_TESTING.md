# Statistical Testing in FLUKE Analysis

## Overview

Proper statistical significance testing requires per-sample binary outcomes (correct/incorrect) rather than aggregated accuracy values. This document explains the correct implementation.

## Key Concepts

### 1. Data Structure for Testing

**Correct Approach:**
```python
# Binary arrays where 1 = correct prediction, 0 = incorrect
orig_correct = [1 if pred == label else 0 for pred, label in zip(orig_pred, orig_label)]
mod_correct = [1 if pred == label else 0 for pred, label in zip(mod_pred, mod_label)]

# Statistical test on paired samples
_, p_value = stats.wilcoxon(orig_correct, mod_correct)
```

**Incorrect Approach:**
```python
# Using aggregated accuracies
orig_acc = 85.5
mod_acc = 82.3
# Cannot perform proper significance test on single values!
```

### 2. Weighted Delta Calculation

The weighted delta metric is calculated from aggregated accuracies:

```python
# A = original accuracy (0-100 scale)
# B = modified accuracy (0-100 scale)
weighted_delta = (B - A) × log₁₀(A) / log₁₀(100)
```

### 3. Complete Analysis Flow

```python
# Step 1: Load individual predictions
orig_pred = [...]  # List of predictions
orig_label = [...]  # List of true labels
mod_pred = [...]   # Modified predictions
mod_label = [...]  # Modified labels

# Step 2: Create binary correctness arrays
orig_correct = np.array([1 if p == l else 0 for p, l in zip(orig_pred, orig_label)])
mod_correct = np.array([1 if p == l else 0 for p, l in zip(mod_pred, mod_label)])

# Step 3: Calculate accuracies for metrics
orig_acc = orig_correct.mean() * 100
mod_acc = mod_correct.mean() * 100

# Step 4: Calculate weighted delta
weighted_delta = (mod_acc - orig_acc) * np.log10(orig_acc) / np.log10(100)

# Step 5: Statistical significance tests
_, wilcoxon_p = stats.wilcoxon(orig_correct, mod_correct)
_, mannwhitney_p = stats.mannwhitneyu(orig_correct, mod_correct)
```

## Statistical Tests Used

### Wilcoxon Signed-Rank Test
- **Use case**: Paired samples (same examples, different conditions)
- **Null hypothesis**: No difference between original and modified
- **Best for**: FLUKE modifications where we test the same examples

### Mann-Whitney U Test
- Not used in the cleaned scripts for paired comparisons.
- Kept here for completeness; only appropriate for independent samples.

### McNemar's Test
- **Use case**: Specifically designed for paired binary data
- **Focus**: Changes in individual predictions (discordant pairs)
- **Implementation**: Exact binomial test on discordant counts (`n10` vs `n01`)
- **Usage in scripts**: Combined with Wilcoxon (take min p-value) for robustness

### Robustness (Unrobustness) Metric
- We report an unrobustness magnitude alongside weighted delta:
  - Binary tasks (coref, dialogue, sa, gsm): discordant rate = percentage of examples where correctness flips between conditions: U = 100 × mean(|m_i − o_i|).
  - NER: mean absolute per-sample F1 change: U = mean(|F1_mod_i − F1_orig_i|).
- No separate p-value or CI is provided for U; it is a descriptive measure of sensitivity to modification.

## Significance Levels

| Symbol | P-value Range | Interpretation |
|--------|--------------|----------------|
| *** | p < 0.001 | Very highly significant |
| ** | p < 0.01 | Highly significant |
| * | p < 0.05 | Significant |
| . | p < 0.1 | Marginally significant |
| ns | p ≥ 0.1 | Not significant |

## Implementation Files

1. **statistical_analysis.py**: Complete implementation with per-sample testing
2. **consolidated_analysis.py**: Updated with `statistical_significance_test()` method
3. **utils.py**: Contains `performance_drop()` with weighted_delta option

## Example Usage

```python
from statistical_analysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# Analyze a specific modification
result = analyzer.analyze_modification_impact(
    original_file='results/gpt5-original.csv',
    modified_file='results/gpt5-negation.csv',
    task='coref'
)

print(f"Weighted Delta: {result['weighted_delta']:.3f}")
print(f"Significance: {result['significance']} (p={result['min_pvalue']:.4f})")
```

## Important Notes

1. **Data Requirements**: Need individual predictions, not just accuracies
2. **Sample Size**: Tests require sufficient samples (typically n ≥ 30)
3. **Paired Nature**: FLUKE modifications test the same examples, making Wilcoxon most appropriate
4. **Multiple Comparisons**: Consider Bonferroni correction when testing many modifications

## Verification

The implementation has been verified to:
- ✓ Calculate weighted delta correctly: (B-A) × log₁₀(A) / log₁₀(100)
- ✓ Perform tests on binary arrays (not aggregated values)
- ✓ Handle edge cases (zero accuracy, perfect accuracy, no change)
- ✓ Assign correct significance levels based on p-values
