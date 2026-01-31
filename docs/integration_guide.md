# Integrated Pattern Discovery System - Integration Guide

**Version:** 1.0  
**Date:** 2026-01-23  
**Author:** Agent_CodebaseRefactor

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Component Integration](#component-integration)
4. [API Reference](#api-reference)
5. [Usage Guide](#usage-guide)
6. [Configuration](#configuration)
7. [Data Flow](#data-flow)
8. [Testing](#testing)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The Integrated Pattern Discovery System is a cohesive framework that combines multiple pattern discovery methods, validation frameworks, and testing procedures into a unified system. It provides:

- **Multiple Discovery Methods**: ML-based, rule-based, hybrid, and ensemble approaches
- **Integrated Validation**: Pattern validation framework with comprehensive metrics
- **Cross-Validation**: Time-series cross-validation for robustness testing
- **Unified API**: Simple interfaces for component communication
- **Result Persistence**: Save and load discovered patterns
- **Pattern Ranking**: Select top patterns based on various criteria

### Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| Integrated System | Main system orchestrator | [`src/integrated_system.py`](../src/integrated_system.py) |
| ML Pattern Discovery | Machine learning-based patterns | [`src/ml_pattern_discovery.py`](../src/ml_pattern_discovery.py) |
| Rule-Based Patterns | Enhanced rule-based discovery | [`src/enhanced_rule_based_patterns.py`](../src/enhanced_rule_based_patterns.py) |
| Validation Framework | Pattern quality validation | [`src/pattern_validation_framework.py`](../src/pattern_validation_framework.py) |
| Cross-Validation | Time-series robustness testing | [`src/cross_validation_framework.py`](../src/cross_validation_framework.py) |
| FPR Validator | False positive reduction | [`src/false_positive_reduction_validator.py`](../src/false_positive_reduction_validator.py) |

---

## System Architecture

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Integrated Pattern System                      │
│                     (IntegratedPatternSystem)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │   ML Pattern │    │  Rule-Based  │    │   Hybrid     │        │
│  │  Discovery   │    │  Discovery   │    │  Discovery   │        │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘        │
│         │                   │                   │                 │
│         └───────────────────┼───────────────────┘                 │
│                             │                                     │
│                    ┌────────▼────────┐                           │
│                    │ Pattern Results │                           │
│                    └────────┬────────┘                           │
│                             │                                     │
│                    ┌────────▼────────┐                           │
│                    │  Validation     │                           │
│                    │  Framework      │                           │
│                    └────────┬────────┘                           │
│                             │                                     │
│                    ┌────────▼────────┐                           │
│                    │ Cross-Validation│                           │
│                    │  Framework      │                           │
│                    └────────┬────────┘                           │
│                             │                                     │
│                    ┌────────▼────────┐                           │
│                    │ Pattern Ranking │                           │
│                    │ & Selection      │                           │
│                    └────────┬────────┘                           │
│                             │                                     │
│                    ┌────────▼────────┐                           │
│                    │  Results       │                           │
│                    │  Persistence    │                           │
│                    └─────────────────┘                           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Data Loading**: Features matrix loaded from CSV
2. **Pattern Discovery**: Multiple methods discover patterns
3. **Pattern Validation**: Patterns validated using validation framework
4. **Cross-Validation**: Robustness testing across time periods
5. **Pattern Ranking**: Patterns ranked by composite score
6. **Result Export**: Top patterns exported to files

---

## Component Integration

### ML Pattern Discovery Integration

The ML pattern discovery component integrates through the [`MLPatternDiscovery`](../src/ml_pattern_discovery.py:97) class:

```python
from src.integrated_system import IntegratedPatternSystem, SystemConfig

config = SystemConfig(
    discovery_methods=["ml_based"],
    ml_models=["random_forest", "gradient_boosting"],
    n_features=50,
    feature_selection_method="mutual_info"
)

system = IntegratedPatternSystem(config)
results = system.discover_patterns()
```

**Key Integration Points:**
- Feature loading and preprocessing
- Target variable creation
- Model training and evaluation
- Pattern extraction from models

### Rule-Based Pattern Discovery Integration

The rule-based component integrates through [`EnhancedRuleBasedPatternDiscovery`](../src/enhanced_rule_based_patterns.py:23):

```python
config = SystemConfig(
    discovery_methods=["rule_based"],
    max_conditions=5,
    min_conditions=2,
    max_false_positive_rate=15.0
)

system = IntegratedPatternSystem(config)
results = system.discover_patterns()
```

**Key Integration Points:**
- Feature categorization
- Smart threshold generation
- Pattern evaluation with false positive reduction
- Diversity enhancement

### Validation Framework Integration

The validation framework integrates through [`PatternValidationFramework`](../src/pattern_validation_framework.py:179):

```python
config = SystemConfig(
    enable_validation=True,
    min_success_rate=70.0,
    max_false_positive_rate=15.0
)

system = IntegratedPatternSystem(config)
system.discover_patterns()
validated = system.validate_patterns()
```

**Key Integration Points:**
- Pattern metrics calculation
- Statistical significance testing
- Composite scoring
- Validation status assignment

### Cross-Validation Integration

The cross-validation framework integrates through [`TimeSeriesCrossValidator`](../src/cross_validation_framework.py:84):

```python
config = SystemConfig(
    enable_cross_validation=True,
    cv_folds=5,
    min_train_size=0.4,
    test_size=0.2
)

system = IntegratedPatternSystem(config)
system.discover_patterns()
system.validate_patterns()
cv_results = system.cross_validate_patterns()
```

**Key Integration Points:**
- Time-series fold creation
- Out-of-sample evaluation
- Stability scoring
- Robustness determination

---

## API Reference

### IntegratedPatternSystem

Main system class that orchestrates all components.

#### Constructor

```python
IntegratedPatternSystem(config: SystemConfig = None)
```

**Parameters:**
- `config`: System configuration (optional, uses defaults if not provided)

#### Methods

##### `load_data(features_path: str = None) -> pd.DataFrame`

Load features data for pattern discovery.

**Parameters:**
- `features_path`: Path to features CSV file (optional)

**Returns:**
- DataFrame with features

**Example:**
```python
system = IntegratedPatternSystem()
df = system.load_data("data/features_matrix.csv")
```

##### `discover_patterns(methods: List[str] = None) -> Dict[str, List[PatternResult]]`

Discover patterns using specified methods.

**Parameters:**
- `methods`: List of discovery methods (optional)
  - `"ml_based"`: Machine learning patterns
  - `"rule_based"`: Rule-based patterns
  - `"hybrid"`: Combined ML and rule-based
  - `"ensemble"`: Ensemble of all methods

**Returns:**
- Dictionary mapping method names to pattern results

**Example:**
```python
results = system.discover_patterns(["ml_based", "rule_based"])
```

##### `validate_patterns(patterns: List[PatternResult] = None) -> List[PatternResult]`

Validate discovered patterns.

**Parameters:**
- `patterns`: Patterns to validate (optional, uses discovered patterns)

**Returns:**
- Validated patterns with validation status

**Example:**
```python
validated = system.validate_patterns()
```

##### `cross_validate_patterns(patterns: List[PatternResult] = None) -> Dict`

Perform cross-validation on patterns.

**Parameters:**
- `patterns`: Patterns to cross-validate (optional)

**Returns:**
- Cross-validation results

**Example:**
```python
cv_results = system.cross_validate_patterns()
```

##### `rank_patterns(patterns: List[PatternResult] = None, method: str = None) -> List[PatternResult]`

Rank patterns using specified method.

**Parameters:**
- `patterns`: Patterns to rank (optional)
- `method`: Ranking method (optional)
  - `"composite"`: By composite score (default)
  - `"success_rate"`: By success rate
  - `"robustness"`: By robustness score

**Returns:**
- Ranked patterns

**Example:**
```python
ranked = system.rank_patterns(method="composite")
```

##### `select_top_patterns(n: int = 20) -> List[PatternResult]`

Select top N patterns from all discovered patterns.

**Parameters:**
- `n`: Number of top patterns to select

**Returns:**
- Top N patterns

**Example:**
```python
top_10 = system.select_top_patterns(n=10)
```

##### `run_full_pipeline() -> Dict`

Run the complete pattern discovery pipeline.

**Returns:**
- Complete pipeline results

**Example:**
```python
results = system.run_full_pipeline()
```

##### `save_results() -> None`

Save all results to files.

**Example:**
```python
system.save_results()
```

##### `load_results() -> None`

Load previously saved results.

**Example:**
```python
system.load_results()
```

##### `get_system_status() -> Dict`

Get current system status.

**Returns:**
- Dictionary with system status

**Example:**
```python
status = system.get_system_status()
```

##### `get_pattern_summary() -> pd.DataFrame`

Get a summary DataFrame of all patterns.

**Returns:**
- DataFrame with pattern summary

**Example:**
```python
summary = system.get_pattern_summary()
print(summary)
```

##### `export_patterns(output_path: str = None, format: str = 'json') -> None`

Export patterns to file.

**Parameters:**
- `output_path`: Output file path (optional)
- `format`: Export format (`'json'` or `'csv'`)

**Example:**
```python
system.export_patterns("patterns.json", format="json")
system.export_patterns("patterns.csv", format="csv")
```

### SystemConfig

Configuration dataclass for the integrated system.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `features_path` | str | `"data/features_matrix.csv"` | Path to features file |
| `output_dir` | str | `"data"` | Output directory |
| `discovery_methods` | List[str] | `["ml_based", "rule_based"]` | Discovery methods |
| `max_patterns_per_method` | int | `50` | Max patterns per method |
| `min_success_rate` | float | `70.0` | Minimum success rate % |
| `min_occurrences` | int | `20` | Minimum occurrences |
| `enable_validation` | bool | `True` | Enable validation |
| `enable_cross_validation` | bool | `True` | Enable cross-validation |
| `cv_folds` | int | `5` | CV folds |
| `min_train_size` | float | `0.4` | Min training size |
| `test_size` | float | `0.2` | Test size |
| `ml_models` | List[str] | `["random_forest", "gradient_boosting"]` | ML models |
| `n_features` | int | `50` | Number of features |
| `feature_selection_method` | str | `"mutual_info"` | Feature selection method |
| `max_conditions` | int | `5` | Max conditions |
| `min_conditions` | int | `2` | Min conditions |
| `max_false_positive_rate` | float | `15.0` | Max FPR % |
| `ensemble_method` | str | `"weighted"` | Ensemble method |
| `pattern_ranking_method` | str | `"composite"` | Ranking method |

### PatternResult

Dataclass representing a discovered pattern.

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `pattern_id` | str | Unique pattern identifier |
| `pattern_name` | str | Pattern name |
| `method` | str | Discovery method |
| `conditions` | Dict | Pattern conditions |
| `direction` | str | Trade direction (`"long"` or `"short"`) |
| `label_col` | str | Target label column |
| `success_rate` | float | Success rate (0-100) |
| `occurrences` | int | Number of occurrences |
| `false_positive_rate` | float | False positive rate (0-100) |
| `p_value` | float | Statistical p-value |
| `stability_score` | float | Stability score (0-1) |
| `regime_coverage` | float | Regime coverage (0-1) |
| `composite_score` | float | Composite quality score |
| `cv_success_rate` | float | CV success rate |
| `cv_stability` | float | CV stability |
| `cv_robustness` | float | CV robustness |
| `is_robust` | bool | Whether pattern is robust |
| `validation_status` | str | Validation status |
| `creation_date` | str | Creation timestamp |
| `source_file` | str | Source file |

---

## Usage Guide

### Quick Start

```python
from src.integrated_system import IntegratedPatternSystem, SystemConfig

# Initialize system with default configuration
system = IntegratedPatternSystem()

# Run full pipeline
results = system.run_full_pipeline()

# Get summary
summary = system.get_pattern_summary()
print(summary)

# Export top patterns
system.export_patterns("top_patterns.json")
```

### Custom Configuration

```python
from src.integrated_system import IntegratedPatternSystem, SystemConfig

# Create custom configuration
config = SystemConfig(
    discovery_methods=["ml_based", "rule_based"],
    max_patterns_per_method=30,
    min_success_rate=75.0,
    enable_validation=True,
    enable_cross_validation=True,
    cv_folds=5
)

# Initialize system
system = IntegratedPatternSystem(config)

# Run pipeline
results = system.run_full_pipeline()
```

### Step-by-Step Usage

```python
from src.integrated_system import IntegratedPatternSystem

# Initialize system
system = IntegratedPatternSystem()

# Load data
system.load_data("data/features_matrix.csv")

# Discover patterns
discovery_results = system.discover_patterns(["ml_based", "rule_based"])

# Validate patterns
validated_patterns = system.validate_patterns()

# Cross-validate patterns
cv_results = system.cross_validate_patterns()

# Rank patterns
ranked_patterns = system.rank_patterns(method="composite")

# Select top patterns
top_patterns = system.select_top_patterns(n=20)

# Save results
system.save_results()

# Export patterns
system.export_patterns("patterns.json", format="json")
```

### Using Specific Discovery Methods

#### ML-Based Only

```python
config = SystemConfig(
    discovery_methods=["ml_based"],
    ml_models=["random_forest"],
    n_features=50
)
system = IntegratedPatternSystem(config)
results = system.discover_patterns()
```

#### Rule-Based Only

```python
config = SystemConfig(
    discovery_methods=["rule_based"],
    max_conditions=5,
    max_false_positive_rate=15.0
)
system = IntegratedPatternSystem(config)
results = system.discover_patterns()
```

#### Hybrid Approach

```python
config = SystemConfig(
    discovery_methods=["hybrid"],
    max_patterns_per_method=30
)
system = IntegratedPatternSystem(config)
results = system.discover_patterns()
```

#### Ensemble Approach

```python
config = SystemConfig(
    discovery_methods=["ensemble"],
    ensemble_method="weighted"
)
system = IntegratedPatternSystem(config)
results = system.discover_patterns()
```

### Loading Saved Results

```python
from src.integrated_system import IntegratedPatternSystem, SystemConfig

# Create system with same configuration
config = SystemConfig(output_dir="data")
system = IntegratedPatternSystem(config)

# Load saved results
system.load_results()

# Access patterns
print(f"Loaded {len(system.patterns)} patterns")
print(f"Loaded {len(system.robust_patterns)} robust patterns")

# Get summary
summary = system.get_pattern_summary()
print(summary)
```

---

## Configuration

### Default Configuration

```python
SystemConfig(
    features_path="data/features_matrix.csv",
    output_dir="data",
    discovery_methods=["ml_based", "rule_based"],
    max_patterns_per_method=50,
    min_success_rate=70.0,
    min_occurrences=20,
    enable_validation=True,
    enable_cross_validation=True,
    cv_folds=5,
    min_train_size=0.4,
    test_size=0.2,
    ml_models=["random_forest", "gradient_boosting"],
    n_features=50,
    feature_selection_method="mutual_info",
    max_conditions=5,
    min_conditions=2,
    max_false_positive_rate=15.0,
    ensemble_method="weighted",
    pattern_ranking_method="composite"
)
```

### Configuration File

You can also load configuration from a YAML file:

```yaml
# config.yaml
system:
  features_path: "data/features_matrix.csv"
  output_dir: "data"
  discovery_methods: ["ml_based", "rule_based"]
  max_patterns_per_method: 50
  min_success_rate: 70.0
  min_occurrences: 20
  enable_validation: true
  enable_cross_validation: true

cross_validation:
  cv_folds: 5
  min_train_size: 0.4
  test_size: 0.2

ml:
  models: ["random_forest", "gradient_boosting"]
  n_features: 50
  feature_selection_method: "mutual_info"

rule_based:
  max_conditions: 5
  min_conditions: 2
  max_false_positive_rate: 15.0
```

```python
import yaml

with open("config.yaml", "r") as f:
    config_dict = yaml.safe_load(f)

config = SystemConfig(**config_dict['system'])
system = IntegratedPatternSystem(config)
```

---

## Data Flow

### Input Data

The system expects a features matrix CSV file with the following structure:

| Column | Type | Description |
|--------|------|-------------|
| Date | datetime | Date index |
| Open | float | Opening price |
| High | float | High price |
| Low | float | Low price |
| Close | float | Closing price |
| Volume | float | Trading volume |
| * | * | Technical features |
| Label_* | str | Movement labels |

### Output Data

The system generates the following output files:

| File | Description |
|------|-------------|
| `integrated_patterns.json` | All discovered patterns |
| `validated_patterns.json` | Validated patterns |
| `robust_patterns.json` | Robust patterns |
| `validation_results.json` | Validation results |
| `cv_results.json` | Cross-validation results |
| `system_state.json` | System state |

### Data Flow Diagram

```
┌─────────────────┐
│  Features CSV   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Load Data      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Discover       │
│  Patterns       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Validate       │
│  Patterns       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Cross-Validate │
│  Patterns       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Rank & Select  │
│  Top Patterns   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Export Results │
└─────────────────┘
```

---

## Testing

### Running Tests

The integrated system includes comprehensive tests:

```bash
# Run all tests
python -m pytest tests/test_integrated_system.py -v

# Run specific test class
python -m pytest tests/test_integrated_system.py::TestSystemInitialization -v

# Run specific test
python -m pytest tests/test_integrated_system.py::TestSystemInitialization::test_system_initialization -v
```

### Test Coverage

The test suite covers:

- **System Initialization**: Configuration and setup
- **Data Loading**: Feature matrix loading
- **Pattern Discovery**: All discovery methods
- **Validation Integration**: Validation framework
- **Cross-Validation Integration**: CV framework
- **Pattern Ranking**: Ranking methods
- **Results Persistence**: Save/load functionality
- **System Status**: State management
- **Pattern Summary**: Summary generation
- **Full Pipeline**: Complete workflow
- **Component Communication**: Data flow
- **Error Handling**: Edge cases

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.13.3
collected 38 items

tests/test_integrated_system.py::TestSystemInitialization::test_system_initialization PASSED
tests/test_integrated_system.py::TestSystemInitialization::test_default_configuration PASSED
tests/test_integrated_system.py::TestSystemInitialization::test_output_directory_creation PASSED
tests/test_integrated_system.py::TestSystemInitialization::test_custom_configuration PASSED
tests/test_integrated_system.py::TestDataLoading::test_load_features PASSED
tests/test_integrated_system.py::TestDataLoading::test_features_have_required_columns PASSED
...

=========================== short test summary info ===========================
28 passed, 10 skipped
```

---

## Best Practices

### 1. Use Appropriate Discovery Methods

- **ML-Based**: Best for complex, non-linear patterns
- **Rule-Based**: Best for interpretable, transparent patterns
- **Hybrid**: Combines strengths of both approaches
- **Ensemble**: Best for robust, production-ready patterns

### 2. Configure Thresholds Appropriately

```python
# Conservative settings for production
config = SystemConfig(
    min_success_rate=80.0,      # Higher success rate
    min_occurrences=30,          # More occurrences
    max_false_positive_rate=10.0 # Lower FPR
)

# Aggressive settings for research
config = SystemConfig(
    min_success_rate=65.0,      # Lower success rate
    min_occurrences=15,          # Fewer occurrences
    max_false_positive_rate=20.0 # Higher FPR
)
```

### 3. Enable Validation and Cross-Validation

```python
config = SystemConfig(
    enable_validation=True,      # Always enable validation
    enable_cross_validation=True # Always enable CV for robustness
)
```

### 4. Save Results Regularly

```python
# After each major operation
system.discover_patterns()
system.save_results()

system.validate_patterns()
system.save_results()

system.cross_validate_patterns()
system.save_results()
```

### 5. Use Pattern Ranking

```python
# Rank by composite score for overall quality
ranked = system.rank_patterns(method="composite")

# Rank by success rate for highest returns
ranked = system.rank_patterns(method="success_rate")

# Rank by robustness for stability
ranked = system.rank_patterns(method="robustness")
```

### 6. Export Patterns in Multiple Formats

```python
# Export to JSON for programmatic access
system.export_patterns("patterns.json", format="json")

# Export to CSV for spreadsheet analysis
system.export_patterns("patterns.csv", format="csv")
```

### 7. Monitor System Status

```python
# Check status before and after operations
status_before = system.get_system_status()
system.run_full_pipeline()
status_after = system.get_system_status()

print(f"Patterns discovered: {status_after['patterns_discovered']}")
print(f"Patterns validated: {status_after['patterns_validated']}")
print(f"Patterns robust: {status_after['patterns_robust']}")
```

---

## Troubleshooting

### Common Issues

#### Issue: No Patterns Discovered

**Symptom:** `discovered_patterns` list is empty

**Possible Causes:**
1. Thresholds too strict
2. Insufficient data
3. Feature quality issues

**Solutions:**
```python
# Lower thresholds
config = SystemConfig(
    min_success_rate=60.0,      # Lower from 70%
    min_occurrences=10,          # Lower from 20
    max_false_positive_rate=25.0 # Higher from 15%
)
```

#### Issue: Validation Fails

**Symptom:** All patterns fail validation

**Possible Causes:**
1. Overfitting in discovery
2. Data quality issues
3. Incorrect label columns

**Solutions:**
```python
# Check validation results
system.validate_patterns()
print(system.validation_results)

# Adjust validation thresholds
config = SystemConfig(
    min_success_rate=65.0,      # Lower validation threshold
    max_false_positive_rate=20.0 # Higher FPR threshold
)
```

#### Issue: Cross-Validation Takes Too Long

**Symptom:** CV execution is slow

**Possible Causes:**
1. Too many patterns
2. Too many CV folds
3. Large dataset

**Solutions:**
```python
# Reduce CV complexity
config = SystemConfig(
    cv_folds=3,              # Reduce from 5
    max_patterns_per_method=20 # Reduce from 50
)
```

#### Issue: Memory Errors

**Symptom:** Out of memory errors

**Possible Causes:**
1. Too many patterns in memory
2. Large feature matrix
3. Multiple ML models

**Solutions:**
```python
# Reduce memory usage
config = SystemConfig(
    max_patterns_per_method=20,  # Reduce patterns
    n_features=30,               # Reduce features
    ml_models=["random_forest"]  # Use fewer models
)
```

### Debug Mode

Enable debug logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
system = IntegratedPatternSystem(config)
```

### Getting Help

1. Check the logs for error messages
2. Review the validation results
3. Verify data quality
4. Check configuration settings
5. Run tests to verify installation

---

## Appendix

### File Structure

```
project/
├── src/
│   ├── integrated_system.py          # Main system
│   ├── ml_pattern_discovery.py       # ML patterns
│   ├── enhanced_rule_based_patterns.py # Rule patterns
│   ├── pattern_validation_framework.py # Validation
│   ├── cross_validation_framework.py  # Cross-validation
│   └── false_positive_reduction_validator.py # FPR validator
├── tests/
│   └── test_integrated_system.py     # Integration tests
├── docs/
│   └── integration_guide.md          # This guide
├── data/
│   ├── features_matrix.csv           # Input data
│   ├── integrated_patterns.json      # Output patterns
│   ├── validated_patterns.json       # Validated patterns
│   ├── robust_patterns.json          # Robust patterns
│   ├── validation_results.json       # Validation results
│   ├── cv_results.json               # CV results
│   └── system_state.json             # System state
└── config.yaml                       # Configuration file
```

### Glossary

| Term | Definition |
|------|------------|
| **Pattern** | A set of conditions that predict market movements |
| **Success Rate** | Percentage of correct predictions |
| **False Positive Rate** | Percentage of incorrect predictions |
| **Stability** | Consistency of pattern performance over time |
| **Robustness** | Pattern's ability to maintain performance across time periods |
| **Cross-Validation** | Testing patterns on out-of-sample data |
| **Composite Score** | Combined quality metric |
| **Regime Coverage** | Pattern's performance across market conditions |

### References

- [`src/integrated_system.py`](../src/integrated_system.py) - Main system implementation
- [`src/ml_pattern_discovery.py`](../src/ml_pattern_discovery.py) - ML pattern discovery
- [`src/enhanced_rule_based_patterns.py`](../src/enhanced_rule_based_patterns.py) - Rule-based patterns
- [`src/pattern_validation_framework.py`](../src/pattern_validation_framework.py) - Validation framework
- [`src/cross_validation_framework.py`](../src/cross_validation_framework.py) - Cross-validation
- [`tests/test_integrated_system.py`](../tests/test_integrated_system.py) - Integration tests

---

**End of Integration Guide**