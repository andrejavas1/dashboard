# Pattern Validation Framework Documentation

## Overview

The Pattern Validation Framework is a comprehensive system for evaluating pattern quality in trading pattern discovery. It provides tools to measure success rates, false positive rates, pattern diversity, and automated testing procedures.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Quick Start](#quick-start)
4. [Core Components](#core-components)
5. [Validation Metrics](#validation-metrics)
6. [Usage Examples](#usage-examples)
7. [Automated Testing](#automated-testing)
8. [Best Practices](#best-practices)
9. [API Reference](#api-reference)

## Introduction

Pattern validation is critical for ensuring that discovered trading patterns are robust, statistically significant, and likely to perform well in live trading. This framework provides a standardized approach to pattern evaluation that can be applied across different pattern discovery methods.

### Why Validation Matters

- **Prevents Overfitting**: Patterns that look good on historical data may fail in live trading
- **Ensures Statistical Significance**: Validates that patterns are not due to random chance
- **Measures Real-World Performance**: Evaluates patterns on multiple dimensions of quality
- **Enables Comparison**: Provides standardized metrics for comparing different discovery methods

## Features

### Core Validation Features

- **Success Rate Measurement**: Calculate the percentage of profitable pattern occurrences
- **False Positive Tracking**: Monitor incorrect pattern signals
- **Statistical Significance Testing**: Use binomial tests to validate pattern significance
- **Stability Analysis**: Measure pattern performance across different time periods
- **Diversity Metrics**: Evaluate pattern diversity across feature categories

### Advanced Features

- **Composite Scoring**: Combine multiple metrics into a single quality score
- **Pattern Comparison**: Compare different pattern sets to evaluate improvements
- **Automated Testing**: Run predefined test cases against pattern discovery methods
- **Report Generation**: Create detailed validation reports with recommendations

## Quick Start

```python
import pandas as pd
from src.pattern_validation_framework import PatternValidationFramework, PatternType

# Load your features data
features_df = pd.read_csv('data/features_matrix.csv', index_col='Date', parse_dates=True)

# Initialize the framework
framework = PatternValidationFramework(features_df)

# Load patterns to validate
import json
with open('data/enhanced_patterns.json', 'r') as f:
    patterns = json.load(f)

# Run validation
report = framework.validate_patterns(patterns, PatternType.RULE_BASED)

# Save the report
framework.save_report(report, 'data/validation_report.json')
```

## Core Components

### PatternMetrics

A dataclass containing comprehensive metrics for a single pattern:

```python
@dataclass
class PatternMetrics:
    pattern_id: str
    pattern_type: PatternType
    direction: str
    label_col: str
    
    # Success metrics
    occurrences: int
    success_count: int
    success_rate: float
    
    # False positive metrics
    false_positive_count: int
    false_positive_rate: float
    
    # Statistical metrics
    p_value: float
    statistical_significance: bool
    
    # Stability metrics
    stability_score: float
    yearly_success_rates: Dict[int, float]
    
    # Performance metrics
    avg_move: float
    avg_time_to_target: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Diversity metrics
    feature_categories: List[str]
    num_conditions: int
    
    # Quality scores
    composite_score: float
    validation_status: ValidationStatus
```

### ValidationReport

A comprehensive report containing validation results for multiple patterns:

```python
@dataclass
class ValidationReport:
    report_id: str
    timestamp: str
    pattern_type: PatternType
    
    # Summary statistics
    total_patterns: int
    passed_patterns: int
    failed_patterns: int
    warning_patterns: int
    
    # Aggregate metrics
    avg_success_rate: float
    avg_false_positive_rate: float
    avg_stability_score: float
    avg_composite_score: float
    
    # Diversity metrics
    unique_feature_categories: int
    pattern_diversity_index: float
    
    # Performance metrics
    avg_occurrences: float
    total_occurrences: int
    
    # Detailed results
    patterns: List[PatternMetrics]
    recommendations: List[str]
```

### PatternValidationFramework

The main class for running validations:

```python
class PatternValidationFramework:
    def __init__(self, features_df: pd.DataFrame = None, config: Dict = None)
    
    def evaluate_pattern(self, pattern: Dict, pattern_id: str = None, 
                        pattern_type: PatternType = PatternType.RULE_BASED) -> PatternMetrics
    
    def validate_patterns(self, patterns: List[Dict], pattern_type: PatternType = PatternType.RULE_BASED,
                         pattern_prefix: str = "") -> ValidationReport
    
    def compare_pattern_sets(self, set1: List[PatternMetrics], set2: List[PatternMetrics],
                            set1_name: str = "Set 1", set2_name: str = "Set 2") -> Dict
    
    def run_automated_tests(self, test_cases: List[Dict]) -> Dict
    
    def save_report(self, report: ValidationReport, output_path: str = None)
    
    def generate_summary_report(self, reports: List[ValidationReport] = None) -> Dict
```

## Validation Metrics

### Success Rate

The percentage of pattern occurrences that result in profitable trades.

```
Success Rate = (Successful Occurrences / Total Occurrences) × 100
```

**Threshold**: Minimum 70% for pattern acceptance

### False Positive Rate

The percentage of pattern occurrences that result in losses.

```
False Positive Rate = (Failed Occurrences / Total Occurrences) × 100
```

**Threshold**: Maximum 15% for pattern acceptance

### Statistical Significance (p-value)

The probability that the pattern's success rate is due to random chance.

```
p-value = BinomialTest(success_count, total_count, p=0.5, alternative='greater')
```

**Threshold**: p-value < 0.05 for statistical significance

### Stability Score

Measures how consistently the pattern performs across different time periods.

```
Stability Score = 1 - Coefficient of Variation
```

**Threshold**: Minimum 0.5 for pattern acceptance

### Composite Score

A weighted combination of all quality metrics:

```
Composite Score = 
    0.35 × Success Rate Score +
    0.15 × Frequency Score +
    0.20 × False Positive Score +
    0.15 × Stability Score +
    0.15 × Diversity Score
```

**Threshold**: Minimum 0.6 for pattern acceptance

### Pattern Diversity Index

An entropy-based measure of pattern diversity across feature categories.

```
Diversity Index = -Σ(p_i × log(p_i)) / log(n_categories)
```

Higher values indicate better diversity across feature categories.

## Usage Examples

### Example 1: Basic Pattern Validation

```python
from src.pattern_validation_framework import PatternValidationFramework, PatternType
import json

# Initialize framework
framework = PatternValidationFramework()

# Load patterns
with open('data/enhanced_patterns.json', 'r') as f:
    patterns = json.load(f)

# Validate patterns
report = framework.validate_patterns(patterns, PatternType.RULE_BASED)

# Print summary
print(f"Total Patterns: {report.total_patterns}")
print(f"Passed: {report.passed_patterns}")
print(f"Failed: {report.failed_patterns}")
print(f"Average Success Rate: {report.avg_success_rate:.1f}%")
```

### Example 2: Custom Thresholds

```python
# Initialize with custom thresholds
config = {
    'thresholds': {
        'min_success_rate': 75.0,
        'max_false_positive_rate': 10.0,
        'min_stability_score': 0.6,
        'min_occurrences': 30
    }
}

framework = PatternValidationFramework(config=config)
```

### Example 3: Comparing Pattern Sets

```python
# Load two pattern sets
with open('data/baseline_patterns.json', 'r') as f:
    baseline_patterns = json.load(f)

with open('data/improved_patterns.json', 'r') as f:
    improved_patterns = json.load(f)

# Validate both sets
baseline_report = framework.validate_patterns(baseline_patterns, PatternType.RULE_BASED)
improved_report = framework.validate_patterns(improved_patterns, PatternType.RULE_BASED)

# Compare
comparison = framework.compare_pattern_sets(
    baseline_report.patterns,
    improved_report.patterns,
    "Baseline",
    "Improved"
)

print(comparison['overall_assessment'])
```

### Example 4: Single Pattern Evaluation

```python
# Define a pattern
pattern = {
    'conditions': {
        'RSI_14': {'operator': '<=', 'value': 30},
        'MACD_Histogram': {'operator': '>=', 'value': 0}
    },
    'direction': 'long',
    'label_col': 'Label_3pct_5d',
    'occurrences': 45,
    'success_rate': 0.78,
    'avg_move': 4.2,
    'false_positive_rate': 12.0,
    'stability_score': 0.65
}

# Evaluate
metrics = framework.evaluate_pattern(pattern, "test_pattern_1", PatternType.RULE_BASED)
print(f"Composite Score: {metrics.composite_score:.3f}")
print(f"Validation Status: {metrics.validation_status.value}")
```

## Automated Testing

The framework supports automated testing of pattern discovery methods:

```python
test_cases = [
    {
        'name': 'High Success Rate Test',
        'type': PatternType.RULE_BASED,
        'patterns': high_success_patterns,
        'expected': {
            'min_success_rate': 80.0,
            'max_false_positive_rate': 15.0
        }
    },
    {
        'name': 'Low False Positive Test',
        'type': PatternType.ML_BASED,
        'patterns': low_fpr_patterns,
        'expected': {
            'max_false_positive_rate': 10.0,
            'min_pass_rate': 70.0
        }
    }
]

results = framework.run_automated_tests(test_cases)
```

### Test Case Structure

```python
test_case = {
    'name': 'Test Name',
    'type': PatternType.RULE_BASED,  # or PatternType.ML_BASED
    'patterns': [...],  # List of patterns to test
    'expected': {
        'min_success_rate': 70.0,
        'max_false_positive_rate': 15.0,
        'min_pass_rate': 80.0
    }
}
```

## Best Practices

### 1. Use Adequate Sample Sizes

Always ensure patterns have sufficient occurrences for statistical significance.

```python
# Bad: Too few occurrences
pattern = {'occurrences': 5, 'success_rate': 1.0}

# Good: Adequate occurrences
pattern = {'occurrences': 50, 'success_rate': 0.80}
```

### 2. Check Stability Across Time Periods

Patterns should perform consistently across different market conditions.

```python
# Check yearly performance
for pattern in patterns:
    if pattern.stability_score < 0.5:
        print(f"Warning: {pattern.pattern_id} has low stability")
```

### 3. Maintain Pattern Diversity

Use patterns from multiple feature categories to reduce risk.

```python
# Ensure diversity
report = framework.validate_patterns(patterns)
if report.unique_feature_categories < 3:
    print("Warning: Low pattern diversity")
```

### 4. Set Appropriate Thresholds

Adjust thresholds based on your risk tolerance and market conditions.

```python
# Conservative thresholds
config = {
    'thresholds': {
        'min_success_rate': 80.0,
        'max_false_positive_rate': 10.0,
        'min_stability_score': 0.7
    }
}
```

### 5. Regular Validation

Run validation regularly to catch degrading pattern performance.

```python
# Run validation weekly
schedule.every().week.do(framework.validate_patterns, patterns)
```

## API Reference

### PatternType Enum

```python
class PatternType(Enum):
    ML_BASED = "ml_based"
    RULE_BASED = "rule_based"
    HYBRID = "hybrid"
    GUARANTEED = "guaranteed"
    CONTEXT7_ENHANCED = "context7_enhanced"
```

### ValidationStatus Enum

```python
class ValidationStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    INCONCLUSIVE = "inconclusive"
```

### Default Thresholds

```python
DEFAULT_THRESHOLDS = {
    'min_success_rate': 70.0,
    'max_false_positive_rate': 15.0,
    'min_p_value': 0.05,
    'min_stability_score': 0.5,
    'min_occurrences': 20,
    'min_composite_score': 0.6,
    'max_patterns_per_category': 10,
    'min_feature_categories': 2
}
```

## Integration with Other Modules

### Integration with ML Pattern Discovery

```python
from src.ml_pattern_discovery import MLPatternDiscovery
from src.pattern_validation_framework import PatternValidationFramework, PatternType

# Discover patterns using ML
ml_discovery = MLPatternDiscovery(features_df)
ml_patterns = ml_discovery.discover_patterns_ml()

# Validate using the framework
framework = PatternValidationFramework(features_df)
report = framework.validate_patterns(ml_patterns, PatternType.ML_BASED)
```

### Integration with Rule-Based Patterns

```python
from src.enhanced_rule_based_patterns import EnhancedRuleBasedPatternDiscovery
from src.pattern_validation_framework import PatternValidationFramework, PatternType

# Discover patterns using rule-based method
rb_discovery = EnhancedRuleBasedPatternDiscovery(features_df)
rb_patterns = rb_discovery.discover_patterns()

# Validate using the framework
framework = PatternValidationFramework(features_df)
report = framework.validate_patterns(rb_patterns, PatternType.RULE_BASED)
```

## Troubleshooting

### Common Issues

1. **Low Success Rates**
   - Check if patterns are overfitted
   - Consider increasing the minimum occurrence threshold
   - Review feature selection

2. **High False Positive Rates**
   - Add additional filtering conditions
   - Use stricter thresholds
   - Consider regime filtering

3. **Low Stability Scores**
   - Patterns may be market regime dependent
   - Consider regime-aware pattern discovery
   - Reduce pattern complexity

4. **Low Diversity Index**
   - Use more feature categories
   - Add patterns from different market conditions
   - Balance long and short patterns

## Conclusion

The Pattern Validation Framework provides a comprehensive approach to evaluating pattern quality. By following the guidelines and examples in this documentation, you can effectively validate patterns from any discovery method and ensure only high-quality patterns are used in live trading.

For questions or support, please refer to the project documentation or contact the development team.