# Pattern Filtering Solution

## Problem

The pipeline was generating duplicate and redundant patterns, wasting resources in validation:
- **Pattern 166** and **Pattern 4532** had identical conditions but different validation rates
- **Pattern 4531** was a subset (missing one condition)
- Total: 6,396 patterns with ~35% duplicates

## Solution

Added a **Pattern Filter** module ([`src/pattern_filter.py`](src/pattern_filter.py)) integrated into **Phase 5** (Pattern Optimization).

### Where to Add Filtering: Phase 5

**Phase 5** is the optimal place because:
1. **Prevents wasted resources** - duplicates won't be validated in Phase 6
2. **Cleaner pipeline** - downstream phases work with unique patterns
3. **Logical place** - Phase 5 is responsible for pattern refinement

### Filtering Methods

The filter implements three levels of deduplication:

#### 1. Exact Duplicate Removal
Removes patterns with identical:
- Conditions
- `label_col` (target window/threshold)
- Direction (long/short)

**Example**: Pattern 4532 removed (duplicate of 166)

#### 2. Condition Duplicate Removal
For patterns with identical conditions but different `label_col`, keeps only the one with highest:
- `validation_success_rate` (preferred)
- `training_success_rate` (fallback)

**Example**: If two patterns have same conditions but different windows (5d vs 10d), keeps the one with higher validation rate.

#### 3. Subset Pattern Removal
Removes patterns that are subsets of other patterns with the same direction.
Keeps the more specific pattern (more conditions).

**Example**: Pattern 4531 removed (subset of 166 - missing Dist_100d_Low condition)

### Integration

Modified [`src/phase5_pattern_optimization.py`](src/phase5_pattern_optimization.py:17) to import and use the filter:

```python
from pattern_filter import PatternFilter

# In optimize_all_patterns():
pattern_filter = PatternFilter()
optimized = pattern_filter.filter_patterns(
    optimized,
    remove_exact_duplicates=True,
    remove_condition_duplicates=True,
    remove_subsets=True
)
```

### Results

Test on existing `data/patterns.json`:

| Metric | Value |
|--------|-------|
| Input patterns | 6,396 |
| Exact duplicates removed | 1,876 |
| Condition duplicates removed | 0 |
| Subset patterns removed | 340 |
| Output patterns | 4,180 |
| **Reduction** | **34.6%** |

### Example: Patterns 166, 4531, 4532

| Pattern | Conditions | Training Rate | Validation Rate | Result |
|---------|------------|---------------|-----------------|--------|
| 166 | OBV_ROC_20d ≥ -2.13<br>Dist_100d_Low ≥ 19.60<br>Dist_50d_Low ≥ 19.94 | 87.67% | **85.71%** | **KEPT** |
| 4531 | OBV_ROC_20d ≥ -2.13<br>Dist_50d_Low ≥ 19.94 | 87.67% | 61.64% | **REMOVED** (subset) |
| 4532 | OBV_ROC_20d ≥ -2.13<br>Dist_100d_Low ≥ 19.60<br>Dist_50d_Low ≥ 19.94 | 87.67% | 61.64% | **REMOVED** (duplicate) |

### Benefits

1. **Reduced validation time** - 35% fewer patterns to validate
2. **Better pattern quality** - keeps only the best variants
3. **Prevents overfitting** - removes redundant patterns
4. **Resource savings** - less computation in Phase 6

### Testing

Run the test script:
```bash
python test_pattern_filter.py
```

This will show filtering statistics and what happened to example duplicates.

## Files Modified

1. **New**: [`src/pattern_filter.py`](src/pattern_filter.py) - Pattern filtering module
2. **Modified**: [`src/phase5_pattern_optimization.py`](src/phase5_pattern_optimization.py:17) - Integrated filter
3. **New**: [`test_pattern_filter.py`](test_pattern_filter.py) - Test script
4. **New**: [`debug_filter.py`](debug_filter.py) - Debug script

## Usage

The filter is automatically applied when running Phase 5:

```bash
python main.py --phase 5
```

Or run the full pipeline:
```bash
python main.py