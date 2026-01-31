# Cross-Validation Framework Documentation

**Version:** 1.0  
**Last Updated:** 2026-01-23  
**Author:** Agent_Validation

---

## Table of Contents

1. [Overview](#overview)
2. [Key Concepts](#key-concepts)
3. [Framework Architecture](#framework-architecture)
4. [Cross-Validation Methods](#cross-validation-methods)
5. [Robustness Metrics](#robustness-metrics)
6. [Usage Guide](#usage-guide)
7. [Best Practices](#best-practices)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The Cross-Validation Framework provides comprehensive time-series cross-validation procedures to ensure pattern robustness across different time periods and market conditions. It implements walk-forward analysis with expanding or rolling windows to test pattern performance on out-of-sample data.

### Key Features

- **Time-Series Cross-Validation:** Walk-forward analysis with configurable fold parameters
- **Out-of-Sample Testing:** Evaluates pattern performance on unseen data
- **Stability Analysis:** Measures pattern consistency across time periods
- **Market Condition Analysis:** Evaluates performance across volatility and trend regimes
- **Robustness Scoring:** Comprehensive scoring based on multiple metrics
- **Visualization:** Generates detailed visualizations of cross-validation results

---

## Key Concepts

### Time-Series Cross-Validation

Unlike traditional k-fold cross-validation, time-series cross-validation respects the temporal order of data. The framework uses walk-forward analysis where:

1. Training data is always before test data
2. Each fold uses a different time period for testing
3. Patterns are evaluated on genuinely out-of-sample data

### Walk-Forward Analysis

Walk-forward analysis simulates real-world trading by:

- Training on historical data
- Testing on subsequent (future) data
- Rolling or expanding the training window
- Repeating for multiple time periods

### Pattern Robustness

A pattern is considered robust if it:

1. Maintains performance on out-of-sample data
2. Shows stability across different time periods
3. Performs consistently across market conditions
4. Doesn't overfit to specific market regimes

---

## Framework Architecture

### Core Classes

#### `CrossValidationResult`

Data class storing comprehensive cross-validation results for a single pattern.

**Attributes:**
- `pattern_id`: Unique identifier for the pattern
- `pattern_name`: Human-readable pattern name
- In-sample metrics (success rate, occurrences, false positive rate)
- Out-of-sample metrics (fold-wise and aggregated)
- Stability metrics (std, CV, stability score)
- Performance degradation (in-sample vs out-of-sample)
- Consistency metrics (consistent folds, consistency rate)
- Market condition performance
- Robustness score and validation status

#### `FoldResult`

Data class storing results for individual cross-validation folds.

**Attributes:**
- `fold_id`: Fold identifier
- Time period boundaries (train/test start/end)
- Performance metrics (success rate, occurrences, false positive rate)
- Market conditions (volatility regime, trend regime)

#### `TimeSeriesCrossValidator`

Main class implementing cross-validation procedures.

**Key Methods:**
- `create_folds()`: Create train/test folds for cross-validation
- `evaluate_pattern_on_fold()`: Evaluate a pattern on a single fold
- `cross_validate_pattern()`: Perform cross-validation on a single pattern
- `cross_validate_patterns()`: Perform cross-validation on multiple patterns
- `analyze_robustness()`: Analyze overall robustness across all patterns
- `generate_robustness_report()`: Generate comprehensive report
- `generate_visualizations()`: Generate visualizations
- `run_cross_validation()`: Run complete cross-validation workflow

---

## Cross-Validation Methods

### Expanding Window Cross-Validation

The default method uses an expanding window where:

- Initial training set size is 40% of data
- Test set size is 20% of data
- Training window grows with each fold
- All historical data is used for training

**Advantages:**
- Uses maximum available data
- Suitable for long-term pattern discovery
- Reduces variance in estimates

**Disadvantages:**
- May not adapt to changing market conditions
- Older data may be less relevant

### Rolling Window Cross-Validation

Alternative method using a fixed-size rolling window:

- Training window size is fixed
- Window slides forward with each fold
- Oldest data is dropped as new data is added

**Advantages:**
- Adapts to changing market conditions
- More recent data has more weight
- Better for short-term patterns

**Disadvantages:**
- Uses less data for training
- Higher variance in estimates

### Configuration

```python
config = {
    # Cross-validation parameters
    'n_folds': 5,              # Number of cross-validation folds
    'min_train_size': 0.4,     # Minimum training size (40% of data)
    'test_size': 0.2,          # Test size (20% of data)
    'expanding_window': True,  # Use expanding window vs rolling
    
    # Validation thresholds
    'min_success_rate': 70.0,      # Minimum out-of-sample success rate
    'max_performance_degradation': 15.0,  # Max % drop from in-sample
    'min_stability_score': 0.6,    # Minimum stability score
    'min_consistency_rate': 0.7,   # Minimum % of folds meeting criteria
    
    # Robustness scoring weights
    'stability_weight': 0.3,       # Weight for stability in robustness score
    'consistency_weight': 0.3,     # Weight for consistency in robustness score
    'performance_weight': 0.2,     # Weight for performance in robustness score
    'degradation_weight': 0.2      # Weight for degradation in robustness score
}
```

---

## Robustness Metrics

### 1. Success Rate Metrics

- **In-Sample Success Rate:** Success rate on full training data
- **Out-of-Sample Success Rate:** Average success rate across all test folds
- **Performance Degradation:** Difference between in-sample and out-of-sample success rates

### 2. Stability Metrics

- **Success Rate Std:** Standard deviation of success rates across folds
- **Success Rate CV:** Coefficient of variation (std / mean)
- **Stability Score:** 1 - CV (higher is more stable)

### 3. Consistency Metrics

- **Consistent Folds:** Number of folds meeting minimum success rate threshold
- **Consistency Rate:** Percentage of folds meeting threshold

### 4. Robustness Score

Composite score combining multiple metrics:

```
Robustness Score = 
    0.3 × Stability Score +
    0.3 × Consistency Rate +
    0.2 × (Out-of-Sample Success Rate / 100) +
    0.2 × Normalized Performance Degradation
```

Where normalized degradation = max(0, 1 - degradation / 100)

### 5. Robustness Criteria

A pattern is considered robust if it meets ALL of:

1. **Out-of-Sample Success Rate ≥ 70%**
2. **Stability Score ≥ 0.6**
3. **Consistency Rate ≥ 70%**
4. **Performance Degradation ≤ 15%**

---

## Usage Guide

### Basic Usage

```python
from src.cross_validation_framework import TimeSeriesCrossValidator
import pandas as pd
import json

# Load data
features_df = pd.read_csv('data/features_matrix.csv', index_col='Date', parse_dates=True)

# Load patterns
with open('data/patterns.json', 'r') as f:
    patterns = json.load(f)

# Initialize cross-validator
validator = TimeSeriesCrossValidator(features_df)

# Run cross-validation
cv_results = validator.run_cross_validation(patterns)

# Access results
results = cv_results['results']
analysis = cv_results['analysis']
```

### Custom Configuration

```python
# Custom configuration
config = {
    'n_folds': 10,
    'min_train_size': 0.5,
    'test_size': 0.15,
    'expanding_window': False,  # Use rolling window
    'min_success_rate': 75.0,
    'max_performance_degradation': 10.0
}

validator = TimeSeriesCrossValidator(features_df, config=config)
```

### Analyzing Individual Patterns

```python
# Cross-validate a single pattern
pattern = patterns[0]
result = validator.cross_validate_pattern(pattern, pattern_id='my_pattern')

# Access metrics
print(f"Robustness Score: {result.robustness_score:.3f}")
print(f"Out-of-Sample SR: {result.avg_out_sample_success_rate:.1f}%")
print(f"Stability Score: {result.stability_score:.3f}")
print(f"Consistency Rate: {result.consistency_rate:.1%}")
print(f"Is Robust: {result.is_robust}")

# View validation notes
for note in result.validation_notes:
    print(f"- {note}")
```

### Accessing Fold Results

```python
# Get fold results for a pattern
fold_results = validator.fold_results['pattern_0']

for fold in fold_results:
    print(f"Fold {fold.fold_id}:")
    print(f"  Test Period: {fold.test_start} to {fold.test_end}")
    print(f"  Success Rate: {fold.success_rate:.1f}%")
    print(f"  Occurrences: {fold.occurrences}")
    print(f"  Market: {fold.volatility_regime}_{fold.trend_regime}")
```

### Generating Reports and Visualizations

```python
# Generate robustness report
report_path = validator.generate_robustness_report(
    results, 
    analysis,
    output_path='data/my_cv_report.md'
)

# Generate visualizations
viz_path = validator.generate_visualizations(
    results,
    output_path='data/my_cv_viz.png'
)
```

---

## Best Practices

### 1. Data Preparation

- Ensure data is sorted by date
- Use sufficient historical data (minimum 5 years recommended)
- Include market regime labels (volatility and trend)
- Handle missing values before cross-validation

### 2. Fold Configuration

- Use at least 3-5 folds for reliable estimates
- Ensure minimum training size provides enough pattern occurrences
- Balance between fold count and fold size
- Consider market cycles when setting fold boundaries

### 3. Pattern Selection

- Only cross-validate patterns with sufficient in-sample occurrences
- Filter patterns by minimum success rate before cross-validation
- Consider pattern diversity when selecting patterns for validation

### 4. Interpretation

- Focus on out-of-sample performance, not in-sample
- High stability score indicates consistent performance
- Low performance degradation indicates minimal overfitting
- Consider market condition performance for regime-specific patterns

### 5. Threshold Selection

- Adjust thresholds based on pattern discovery goals
- Higher thresholds for production patterns
- Lower thresholds for exploratory analysis
- Consider trade-offs between robustness and pattern count

### 6. Common Pitfalls

- **Look-ahead Bias:** Ensure test data is always after training data
- **Data Leakage:** Don't use future information in pattern discovery
- **Overfitting:** High in-sample but low out-of-sample performance
- **Insufficient Data:** Too few occurrences per fold for reliable estimates

---

## Examples

### Example 1: Basic Cross-Validation

```python
from src.cross_validation_framework import TimeSeriesCrossValidator
import pandas as pd
import json

# Load data
features_df = pd.read_csv('data/features_matrix.csv', index_col='Date', parse_dates=True)

# Load patterns
with open('data/enhanced_patterns.json', 'r') as f:
    patterns = json.load(f)

# Initialize and run
validator = TimeSeriesCrossValidator(features_df)
cv_results = validator.run_cross_validation(patterns)

# Print summary
analysis = cv_results['analysis']
print(f"Robustness Rate: {analysis['robustness_rate']:.1%}")
print(f"Avg Robustness Score: {analysis['avg_robustness_score']:.3f}")
```

### Example 2: Custom Configuration

```python
config = {
    'n_folds': 8,
    'min_train_size': 0.5,
    'test_size': 0.15,
    'expanding_window': False,
    'min_success_rate': 75.0,
    'max_performance_degradation': 10.0,
    'min_stability_score': 0.7,
    'min_consistency_rate': 0.8
}

validator = TimeSeriesCrossValidator(features_df, config=config)
cv_results = validator.run_cross_validation(patterns)
```

### Example 3: Analyzing Robust Patterns

```python
# Get robust patterns
robust_patterns = {
    pid: result for pid, result in cv_results['results'].items()
    if result.is_robust
}

print(f"Found {len(robust_patterns)} robust patterns")

# Sort by robustness score
sorted_robust = sorted(
    robust_patterns.items(),
    key=lambda x: x[1].robustness_score,
    reverse=True
)

for pid, result in sorted_robust[:5]:
    print(f"{pid}: Score={result.robustness_score:.3f}, "
          f"SR={result.avg_out_sample_success_rate:.1f}%")
```

### Example 4: Market Condition Analysis

```python
# Analyze performance by market condition
for pid, result in cv_results['results'].items():
    print(f"\n{pid}:")
    for condition, sr in result.market_condition_performance.items():
        print(f"  {condition}: {sr:.1f}%")
```

### Example 5: Comparing Pattern Sets

```python
# Cross-validate ML patterns
ml_validator = TimeSeriesCrossValidator(features_df)
ml_results = ml_validator.run_cross_validation(ml_patterns)

# Cross-validate rule-based patterns
rule_validator = TimeSeriesCrossValidator(features_df)
rule_results = rule_validator.run_cross_validation(rule_patterns)

# Compare
ml_robustness = ml_results['analysis']['robustness_rate']
rule_robustness = rule_results['analysis']['robustness_rate']

print(f"ML Robustness: {ml_robustness:.1%}")
print(f"Rule-Based Robustness: {rule_robustness:.1%}")
```

---

## Troubleshooting

### Issue: No robust patterns found

**Possible Causes:**
- Thresholds too high
- Patterns overfit to training data
- Insufficient training data
- Market conditions changed significantly

**Solutions:**
- Lower validation thresholds
- Increase training window size
- Use more fundamental features
- Filter patterns by in-sample performance first

### Issue: High performance degradation

**Possible Causes:**
- Patterns overfit to training data
- Market regime shift
- Data leakage in pattern discovery

**Solutions:**
- Add regularization to pattern discovery
- Use more stable features
- Implement regime-specific patterns
- Check for look-ahead bias

### Issue: Low consistency rate

**Possible Causes:**
- Patterns sensitive to market conditions
- Insufficient occurrences per fold
- High variance in pattern performance

**Solutions:**
- Increase minimum occurrences threshold
- Use longer test periods
- Filter patterns by stability
- Consider regime-specific validation

### Issue: Insufficient occurrences in folds

**Possible Causes:**
- Pattern too rare
- Test window too small
- Training window too small

**Solutions:**
- Increase test window size
- Use more frequent patterns
- Adjust fold configuration
- Consider pattern frequency in selection

### Issue: Visualization errors

**Possible Causes:**
- Empty results
- Invalid data types
- Missing matplotlib backend

**Solutions:**
- Verify results are not empty
- Check data types in results
- Ensure matplotlib is properly installed
- Use non-interactive backend if needed

---

## Related Documentation

- [Pattern Validation Framework](validation_framework.md)
- [ML Pattern Discovery](ml_pattern_discovery.md)
- [Rule-Based Pattern Enhancement](rule_based_patterns.md)
- [Feature Engineering](feature_engineering.md)

---

## References

- Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213.

- Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: principles and practice* (2nd ed.). OTexts.

- Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. *International Journal of Forecasting*, 16(4), 437-450.

---

*Documentation maintained by Agent_Validation*