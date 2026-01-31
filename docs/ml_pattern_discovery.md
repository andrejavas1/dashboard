# Machine Learning Pattern Discovery Documentation

## Overview

This document provides comprehensive documentation for the Machine Learning (ML) Pattern Discovery module. The module implements data-driven pattern discovery algorithms using scikit-learn to identify high success rate trading patterns that meet specific performance criteria (>80% success rate with 12+ occurrences per year).

## Architecture

### Core Components

1. **MLPatternDiscovery**: Main class for ML-based pattern discovery
2. **PatternMetrics**: Dataclass for storing pattern evaluation metrics
3. **MLModelConfig**: Configuration class for ML model parameters

### Supported ML Algorithms

| Algorithm | Type | Use Case | Strengths |
|-----------|------|----------|-----------|
| Random Forest | Ensemble | General pattern discovery | Handles non-linear relationships, provides feature importance |
| Gradient Boosting | Ensemble | Complex patterns | High accuracy, handles imbalanced data |
| Logistic Regression | Linear | Baseline models | Interpretable, fast training |
| SVM | Kernel-based | Complex boundaries | Effective in high-dimensional spaces |
| Decision Tree | Tree-based | Simple rules | Highly interpretable |
| KNN | Instance-based | Local patterns | Simple, no training phase |
| MLP | Neural Network | Complex patterns | Learns non-linear relationships |
| AdaBoost | Ensemble | Boosting | Improves weak learners |

## Data Pipeline

### 1. Feature Loading

```python
from src.ml_pattern_discovery import MLPatternDiscovery

# Initialize discovery system
ml_discovery = MLPatternDiscovery()

# Load pre-calculated features
ml_discovery.load_features("data/features_matrix.csv")
```

### 2. Target Variable Creation

The target variable is created based on future returns:

```python
# Create target: 1 = success (>=2% profit), 0 = failure (<=-1% loss)
ml_discovery.create_target_variable(
    profit_threshold=2.0,
    loss_threshold=-1.0,
    holding_period=5
)
```

**Parameters:**
- `profit_threshold`: Minimum profit percentage for success (default: 2.0%)
- `loss_threshold`: Maximum loss percentage for failure (default: -1.0%)
- `holding_period`: Number of days to hold position (default: 5)

### 3. Feature Preparation

```python
# Prepare features for ML (handles categorical variables)
X = ml_discovery.prepare_features_for_ml(
    exclude_cols=['Open', 'High', 'Low', 'Close', 'Volume', 'TR'],
    handle_categorical=True
)
```

### 4. Feature Selection

Multiple feature selection methods are supported:

```python
# Method 1: Mutual Information (recommended)
selected_features, selector = ml_discovery.feature_selection(
    X, ml_discovery.target,
    method='mutual_info',
    k=50
)

# Method 2: ANOVA F-test
selected_features, selector = ml_discovery.feature_selection(
    X, ml_discovery.target,
    method='kbest',
    k=50
)

# Method 3: Recursive Feature Elimination
selected_features, selector = ml_discovery.feature_selection(
    X, ml_discovery.target,
    method='rfe',
    k=50
)

# Method 4: Model-based selection
selected_features, selector = ml_discovery.feature_selection(
    X, ml_discovery.target,
    method='model_based',
    k=50
)
```

**Feature Selection Methods:**

| Method | Description | Best For |
|--------|-------------|----------|
| `variance` | Removes low-variance features | Initial filtering |
| `kbest` | Selects top k features using ANOVA F-test | Quick selection |
| `mutual_info` | Selects features based on mutual information | Non-linear relationships |
| `rfe` | Recursive Feature Elimination | Optimal subset |
| `model_based` | Uses model feature importance | Model-specific selection |

### 5. Data Splitting

```python
# Time series split (maintains chronological order)
X_train, X_val, X_test, y_train, y_val, y_test = ml_discovery.split_data(
    X_selected, ml_discovery.target,
    test_size=0.2,
    validation_size=0.1,
    time_series=True
)
```

**Important:** Time series split is used by default to maintain chronological order and prevent data leakage.

### 6. Feature Scaling

```python
# Standard scaling (z-score normalization)
X_train_scaled, X_val_scaled, X_test_scaled = ml_discovery.scale_features(
    X_train, X_val, X_test,
    method='standard'
)

# Alternative scaling methods:
# - 'minmax': Min-max scaling to [0, 1]
# - 'robust': Robust scaling using median and IQR
```

## Model Training

### Single Model Training

```python
# Train a single model with hyperparameter tuning
model, training_history = ml_discovery.train_model(
    model_type='random_forest',
    X_train=X_train_scaled,
    y_train=y_train,
    hyperparameter_tuning=True,
    cv_folds=5
)
```

### Hyperparameter Tuning

The system uses RandomizedSearchCV for efficient hyperparameter optimization:

```python
# Hyperparameter grids are pre-configured for each model type
# Example: Random Forest hyperparameters
{
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', None]
}
```

### Custom Hyperparameter Grid

```python
# Override default hyperparameter grid
ml_discovery.hyperparameter_grids['random_forest'] = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15],
    'min_samples_split': [5, 10]
}
```

## Model Evaluation

### Evaluation Metrics

```python
# Evaluate model on test set
metrics = ml_discovery.evaluate_model(model, X_test_scaled, y_test)

# Metrics include:
# - accuracy: Overall accuracy
# - precision: True positives / (true positives + false positives)
# - recall: True positives / (true positives + false negatives)
# - f1: Harmonic mean of precision and recall
# - auc: Area under ROC curve
# - confusion_matrix: TP, FP, TN, FN counts
```

### Cross-Validation

```python
# Time series cross-validation
cv_results = ml_discovery.cross_validate_model(
    model_type='random_forest',
    n_splits=5
)

# Returns:
# - cv_scores: List of CV scores
# - mean_score: Mean CV score
# - std_score: Standard deviation of CV scores
# - min_score: Minimum CV score
# - max_score: Maximum CV score
```

## Pattern Discovery

### Single Model Pattern Discovery

```python
# Discover patterns using a single model
patterns = ml_discovery.discover_patterns_ml(
    model_type='random_forest',
    feature_selection_method='mutual_info',
    n_features=50,
    min_occurrences=20,
    probability_threshold=0.7
)
```

**Parameters:**
- `model_type`: Type of ML model to use
- `feature_selection_method`: Feature selection method
- `n_features`: Number of features to select
- `min_occurrences`: Minimum pattern occurrences
- `probability_threshold`: Threshold for pattern confidence (0-1)

### Pattern Discovery Process

1. **Model Training**: Train ML model on training data
2. **Prediction**: Get probability predictions on full dataset
3. **High-Confidence Filtering**: Filter predictions above threshold
4. **Clustering**: Use DBSCAN to cluster similar patterns
5. **Pattern Extraction**: Extract patterns from each cluster
6. **Metrics Calculation**: Calculate comprehensive metrics for each pattern

### Full Pipeline Execution

```python
# Run complete ML pattern discovery pipeline
results = ml_discovery.run_full_pipeline(
    model_types=['random_forest', 'gradient_boosting'],
    feature_selection_method='mutual_info',
    n_features=50
)

# Results include:
# - model_types: List of models trained
# - num_patterns: Total patterns discovered
# - num_meeting_criteria: Patterns meeting success criteria
# - patterns: List of pattern dictionaries
# - metrics: List of pattern metrics
# - report: Pattern summary report
# - evaluation_results: Model evaluation results
```

## Ensemble Methods

### Voting Ensemble

```python
# Create ensemble of multiple models
ensemble, ensemble_metrics = ml_discovery.ensemble_models(
    model_types=['random_forest', 'gradient_boosting', 'logistic_regression'],
    voting='soft'  # or 'hard' for majority voting
)
```

**Voting Methods:**
- `soft`: Average probability predictions (recommended)
- `hard`: Majority vote on class predictions

## Pattern Metrics

### PatternMetrics Dataclass

```python
@dataclass
class PatternMetrics:
    pattern_id: int
    pattern_name: str
    success_rate: float
    total_occurrences: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    avg_profit_loss: float
    avg_holding_period: float
    sharpe_ratio: float
    max_drawdown: float
    annual_occurrences: float
    meets_criteria: bool
```

### Success Criteria

A pattern meets the success criteria if:
- **Success Rate** ≥ 80%
- **Annual Occurrences** ≥ 12

### Calculating Pattern Metrics

```python
# Calculate metrics for discovered patterns
pattern_metrics = ml_discovery.calculate_pattern_metrics(
    patterns=patterns,
    X=X_selected,
    y=ml_discovery.target
)
```

### Pattern Report Generation

```python
# Generate summary report
report = ml_discovery.generate_pattern_report()

# Report includes:
# - Pattern ID and Name
# - Success Rate
# - Total and Annual Occurrences
# - Precision, Recall, F1 Score
# - Average Profit/Loss
# - Sharpe Ratio
# - Max Drawdown
# - Meets Criteria flag
```

## Saving and Loading

### Saving Models and Results

```python
# Save trained models
ml_discovery.save_models(output_dir="data")

# Saves:
# - ml_model_{model_name}.pkl: Trained models
# - ml_scaler.pkl: Feature scaler
# - ml_feature_selector.pkl: Feature selector
# - ml_selected_features.json: Selected feature names

# Save patterns and metrics
ml_discovery.save_patterns(output_dir="data")

# Saves:
# - ml_discovered_patterns.json: Pattern definitions
# - ml_pattern_metrics.json: Pattern metrics
# - ml_evaluation_results.json: Evaluation results
```

### Loading Saved Models

```python
import pickle

# Load model
with open("data/ml_model_random_forest.pkl", 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open("data/ml_scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

# Load feature selector
with open("data/ml_feature_selector.pkl", 'rb') as f:
    selector = pickle.load(f)

# Load selected features
import json
with open("data/ml_selected_features.json", 'r') as f:
    selected_features = json.load(f)
```

## Hyperparameter Tuning Guidelines

### Random Forest

**Recommended Settings:**
- `n_estimators`: 100-200 (more trees = better performance, slower training)
- `max_depth`: 10-20 (prevents overfitting)
- `min_samples_split`: 5-10 (controls tree complexity)
- `min_samples_leaf`: 2-4 (prevents overfitting)
- `max_features`: 'sqrt' (default, works well)
- `class_weight`: 'balanced' (for imbalanced data)

**Tuning Strategy:**
1. Start with default parameters
2. Increase `n_estimators` if performance plateaus
3. Reduce `max_depth` if overfitting
4. Increase `min_samples_split` if overfitting

### Gradient Boosting

**Recommended Settings:**
- `n_estimators`: 100-200
- `learning_rate`: 0.05-0.1 (lower = more trees needed)
- `max_depth`: 3-7 (shallower trees prevent overfitting)
- `min_samples_split`: 5-10
- `subsample`: 0.8-1.0 (stochastic gradient boosting)

**Tuning Strategy:**
1. Start with `learning_rate=0.1`, `n_estimators=100`
2. If overfitting, reduce `learning_rate` and increase `n_estimators`
3. If underfitting, increase `learning_rate`
4. Use `subsample < 1.0` for regularization

### Logistic Regression

**Recommended Settings:**
- `C`: 0.1-10 (inverse regularization strength)
- `penalty`: 'l2' (default) or 'l1' (sparse solutions)
- `solver`: 'saga' (supports all penalties)
- `class_weight`: 'balanced' (for imbalanced data)
- `max_iter`: 1000-2000

**Tuning Strategy:**
1. Start with `C=1.0`, `penalty='l2'`
2. If overfitting, decrease `C`
3. If underfitting, increase `C`
4. Try `penalty='l1'` for feature selection

### SVM

**Recommended Settings:**
- `C`: 1-100 (regularization parameter)
- `kernel`: 'rbf' (default) or 'linear' (for high-dimensional data)
- `gamma`: 'scale' (default) or 0.01-0.1
- `class_weight`: 'balanced' (for imbalanced data)

**Tuning Strategy:**
1. Start with `C=10`, `kernel='rbf'`, `gamma='scale'`
2. If overfitting, decrease `C`
3. If underfitting, increase `C`
4. Try `kernel='linear'` if data has many features

## Performance Optimization

### Computational Efficiency

1. **Use RandomizedSearchCV** instead of GridSearchCV for large hyperparameter spaces
2. **Limit n_iter_search** to 50 for faster tuning
3. **Use n_jobs=-1** to utilize all CPU cores
4. **Enable caching** for expensive operations

### Memory Optimization

1. **Use sparse matrices** for high-dimensional data
2. **Batch processing** for large datasets
3. **Feature selection** to reduce dimensionality
4. **Delete intermediate variables** when not needed

### Model Selection Guidelines

| Scenario | Recommended Model |
|----------|-------------------|
| Quick prototyping | Logistic Regression, Decision Tree |
| Best accuracy | Gradient Boosting, Random Forest |
| Interpretability | Decision Tree, Logistic Regression |
| Imbalanced data | Random Forest (balanced), Gradient Boosting |
| Large datasets | Random Forest, Logistic Regression |
| Small datasets | SVM, KNN |

## Best Practices

### Data Preparation

1. **Always use time series split** for financial data
2. **Scale features** before training most models
3. **Handle missing values** before feature selection
4. **Remove constant features** to improve performance

### Model Training

1. **Use cross-validation** for robust evaluation
2. **Monitor for overfitting** (train vs. validation performance)
3. **Use class weights** for imbalanced data
4. **Save models** after training for reproducibility

### Pattern Discovery

1. **Start with probability_threshold=0.7** for balanced precision/recall
2. **Use min_occurrences=20** to ensure statistical significance
3. **Validate patterns** on out-of-sample data
4. **Check annual occurrences** to ensure trading frequency

### Evaluation

1. **Use multiple metrics** (not just accuracy)
2. **Focus on F1 score** for imbalanced data
3. **Check confusion matrix** for error analysis
4. **Validate on test set** after all tuning

## Troubleshooting

### Common Issues

**Issue: Low success rate patterns**
- Solution: Increase `probability_threshold` to 0.8 or 0.85
- Solution: Try different model types (e.g., Gradient Boosting)
- Solution: Increase `n_features` to capture more information

**Issue: Too few pattern occurrences**
- Solution: Decrease `min_occurrences` to 15 or 10
- Solution: Decrease `probability_threshold` to 0.6
- Solution: Check if data spans enough time period

**Issue: Overfitting**
- Solution: Reduce model complexity (max_depth, n_estimators)
- Solution: Increase regularization (decrease C for LR/SVM)
- Solution: Use more cross-validation folds
- Solution: Add more training data

**Issue: Long training time**
- Solution: Reduce `n_iter_search` for hyperparameter tuning
- Solution: Use simpler models (Logistic Regression)
- Solution: Reduce `n_features` via feature selection
- Solution: Use `n_jobs=-1` for parallel processing

## Configuration

### Config File (config.yaml)

```yaml
ml_patterns:
  min_success_rate: 0.80
  min_annual_occurrences: 12
  max_patterns: 50
  feature_selection_method: 'mutual_info'
  n_features_to_select: 50
  models:
    - 'random_forest'
    - 'gradient_boosting'
    - 'logistic_regression'
```

### Programmatic Configuration

```python
# Modify configuration
ml_discovery.model_config = MLModelConfig(
    model_type='random_forest',
    test_size=0.2,
    validation_size=0.1,
    random_state=42,
    cv_folds=5,
    scoring_metric='f1',
    n_iter_search=50,
    min_success_rate=0.80,
    min_annual_occurrences=12
)
```

## Example Usage

### Complete Example

```python
from src.ml_pattern_discovery import MLPatternDiscovery

# Initialize
ml_discovery = MLPatternDiscovery()

# Load features
ml_discovery.load_features("data/features_matrix.csv")

# Create target
ml_discovery.create_target_variable(
    profit_threshold=2.0,
    loss_threshold=-1.0,
    holding_period=5
)

# Run full pipeline
results = ml_discovery.run_full_pipeline(
    model_types=['random_forest', 'gradient_boosting'],
    feature_selection_method='mutual_info',
    n_features=50
)

# View results
print(f"Total Patterns: {results['num_patterns']}")
print(f"Meeting Criteria: {results['num_meeting_criteria']}")

# Generate report
report = ml_discovery.generate_pattern_report()
print(report.to_string(index=False))

# Save results
ml_discovery.save_models()
ml_discovery.save_patterns()
```

### Quick Pattern Discovery

```python
from src.ml_pattern_discovery import MLPatternDiscovery

# Initialize and discover
ml_discovery = MLPatternDiscovery()
ml_discovery.load_features()
ml_discovery.create_target_variable()

# Discover patterns
patterns = ml_discovery.discover_patterns_ml(
    model_type='random_forest',
    probability_threshold=0.75
)

# View patterns
for pattern in patterns:
    if pattern['meets_criteria']:
        print(f"{pattern['pattern_name']}: "
              f"Success Rate={pattern['success_rate']:.2%}, "
              f"Annual Occurrences={pattern['annual_occurrences']:.1f}")
```

## References

### Scikit-learn Documentation
- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
- [Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting)
- [Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

### Additional Resources
- [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Hyperparameter Tuning](https://scikit-learn.org/stable/modules/grid_search.html)
- [Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-01-22 | Initial implementation |

---

**Note:** This module is designed for financial pattern discovery and should be used with appropriate risk management practices. Past performance does not guarantee future results.