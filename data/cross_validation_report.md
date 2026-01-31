# Cross-Validation Robustness Report

**Generated:** 2026-01-23 06:15:11

## Executive Summary

- **Total Patterns Tested:** 20
- **Robust Patterns:** 0
- **Robustness Rate:** 0.0%

### Robustness Criteria
- Minimum Out-of-Sample Success Rate: 70.0%
- Minimum Stability Score: 0.6
- Minimum Consistency Rate: 70%
- Maximum Performance Degradation: 15.0%

## Cross-Validation Configuration

- **Number of Folds:** 5
- **Minimum Training Size:** 40%
- **Test Size:** 20%
- **Window Type:** Expanding

## Overall Robustness Analysis

| Metric | Value |
|--------|-------|
| Total Patterns | 20 |
| Robust Patterns | 0 |
| Robustness Rate | 0.0% |
| Average Robustness Score | 0.548 |
| Average Stability Score | 0.851 |
| Average Consistency Rate | 0.0% |
| Average Performance Degradation | -2.14% |
| Out-of-Sample SR Mean | 44.23% |
| Out-of-Sample SR Std | 8.05% |
| Out-of-Sample SR Range | 26.9% - 52.7% |
| Out-of-Sample Occurrences Mean | 653.2 |
| Out-of-Sample Occurrences Std | 138.2 |

## Pattern-by-Pattern Results

| Pattern ID | Pattern Name | Robust | Robustness Score | Out-Sample SR | Stability | Consistency |
|------------|-------------|--------|-----------------|---------------|-----------|-------------|
| pattern_15 | Label_1pct_10d | ✗ | 0.597 | 52.6% | 0.973 | 0.0% |
| pattern_3 | Label_1pct_10d | ✗ | 0.593 | 51.4% | 0.964 | 0.0% |
| pattern_13 | Label_1pct_5d | ✗ | 0.588 | 50.7% | 0.949 | 0.0% |
| pattern_18 | Label_1pct_5d | ✗ | 0.584 | 52.7% | 0.924 | 0.0% |
| pattern_0 | Label_1pct_5d | ✗ | 0.583 | 51.1% | 0.930 | 0.0% |
| pattern_1 | Label_1pct_5d | ✗ | 0.583 | 51.1% | 0.930 | 0.0% |
| pattern_7 | Label_1pct_10d | ✗ | 0.580 | 48.4% | 0.948 | 0.0% |
| pattern_5 | Label_1pct_5d | ✗ | 0.571 | 44.6% | 0.945 | 0.0% |
| pattern_14 | Label_1pct_3d | ✗ | 0.569 | 50.2% | 0.873 | 0.0% |
| pattern_11 | Label_2pct_10d | ✗ | 0.567 | 43.3% | 0.927 | 0.0% |
| pattern_19 | Label_1pct_3d | ✗ | 0.564 | 47.2% | 0.869 | 0.0% |
| pattern_6 | Label_1pct_3d | ✗ | 0.559 | 48.7% | 0.850 | 0.0% |
| pattern_17 | Label_1pct_3d | ✗ | 0.558 | 49.6% | 0.832 | 0.0% |
| pattern_16 | Label_1pct_3d | ✗ | 0.549 | 40.2% | 0.901 | 0.0% |
| pattern_9 | Label_3pct_10d | ✗ | 0.513 | 37.9% | 0.765 | 0.0% |
| pattern_2 | Label_3pct_10d | ✗ | 0.512 | 40.0% | 0.749 | 0.0% |
| pattern_4 | Label_3pct_10d | ✗ | 0.512 | 40.0% | 0.748 | 0.0% |
| pattern_12 | Label_2pct_3d | ✗ | 0.481 | 26.9% | 0.753 | 0.0% |
| pattern_8 | Label_2pct_3d | ✗ | 0.451 | 29.5% | 0.602 | 0.0% |
| pattern_10 | Label_2pct_3d | ✗ | 0.444 | 28.5% | 0.583 | 0.0% |

## Detailed Results: Top 10 Patterns

### pattern_15: Label_1pct_10d

**Robustness Score:** 0.597
**Is Robust:** No

#### In-Sample Performance
- Success Rate: 52.8%
- Occurrences: 2814
- False Positive Rate: 47.0%

#### Out-of-Sample Performance
- Average Success Rate: 52.6%
- Average Occurrences: 524.0
- Average False Positive Rate: 47.1%
- Performance Degradation: 0.20%

#### Stability Metrics
- Success Rate Std: 1.40%
- Success Rate CV: 0.027
- Stability Score: 0.973

#### Consistency Metrics
- Consistent Folds: 0/3
- Consistency Rate: 0.0%

#### Market Condition Performance
- Low_Strong Bear: 53.9%
- High_Strong Bear: 50.7%
- Medium_Strong Bear: 53.1%

#### Validation Notes
- Out-of-sample success rate (52.6%) below threshold (70.0%)
- Consistency rate (0.0%) below threshold (70%)

### pattern_3: Label_1pct_10d

**Robustness Score:** 0.593
**Is Robust:** No

#### In-Sample Performance
- Success Rate: 50.8%
- Occurrences: 4000
- False Positive Rate: 48.9%

#### Out-of-Sample Performance
- Average Success Rate: 51.4%
- Average Occurrences: 801.7
- Average False Positive Rate: 48.3%
- Performance Degradation: -0.54%

#### Stability Metrics
- Success Rate Std: 1.83%
- Success Rate CV: 0.036
- Stability Score: 0.964

#### Consistency Metrics
- Consistent Folds: 0/3
- Consistency Rate: 0.0%

#### Market Condition Performance
- Low_Strong Bull: 53.2%
- High_Strong Bear: 48.9%
- Medium_Strong Bull: 52.1%

#### Validation Notes
- Out-of-sample success rate (51.4%) below threshold (70.0%)
- Consistency rate (0.0%) below threshold (70%)

### pattern_13: Label_1pct_5d

**Robustness Score:** 0.588
**Is Robust:** No

#### In-Sample Performance
- Success Rate: 49.9%
- Occurrences: 3008
- False Positive Rate: 45.9%

#### Out-of-Sample Performance
- Average Success Rate: 50.7%
- Average Occurrences: 561.0
- Average False Positive Rate: 45.7%
- Performance Degradation: -0.82%

#### Stability Metrics
- Success Rate Std: 2.59%
- Success Rate CV: 0.051
- Stability Score: 0.949

#### Consistency Metrics
- Consistent Folds: 0/3
- Consistency Rate: 0.0%

#### Market Condition Performance
- Low_Strong Bull: 47.5%
- High_Strong Bull: 50.7%
- Medium_Strong Bull: 53.8%

#### Validation Notes
- Out-of-sample success rate (50.7%) below threshold (70.0%)
- Consistency rate (0.0%) below threshold (70%)

### pattern_18: Label_1pct_5d

**Robustness Score:** 0.584
**Is Robust:** No

#### In-Sample Performance
- Success Rate: 51.6%
- Occurrences: 2545
- False Positive Rate: 44.8%

#### Out-of-Sample Performance
- Average Success Rate: 52.7%
- Average Occurrences: 493.0
- Average False Positive Rate: 43.6%
- Performance Degradation: -1.02%

#### Stability Metrics
- Success Rate Std: 4.03%
- Success Rate CV: 0.076
- Stability Score: 0.924

#### Consistency Metrics
- Consistent Folds: 0/3
- Consistency Rate: 0.0%

#### Market Condition Performance
- Low_Strong Bull: 48.8%
- High_Strong Bear: 50.9%
- Medium_Strong Bull: 58.2%

#### Validation Notes
- Out-of-sample success rate (52.7%) below threshold (70.0%)
- Consistency rate (0.0%) below threshold (70%)

### pattern_0: Label_1pct_5d

**Robustness Score:** 0.583
**Is Robust:** No

#### In-Sample Performance
- Success Rate: 50.3%
- Occurrences: 4031
- False Positive Rate: 46.2%

#### Out-of-Sample Performance
- Average Success Rate: 51.1%
- Average Occurrences: 806.0
- Average False Positive Rate: 45.6%
- Performance Degradation: -0.81%

#### Stability Metrics
- Success Rate Std: 3.59%
- Success Rate CV: 0.070
- Stability Score: 0.930

#### Consistency Metrics
- Consistent Folds: 0/3
- Consistency Rate: 0.0%

#### Market Condition Performance
- Low_Strong Bull: 47.1%
- High_Strong Bear: 50.2%
- Medium_Strong Bull: 55.8%

#### Validation Notes
- Out-of-sample success rate (51.1%) below threshold (70.0%)
- Consistency rate (0.0%) below threshold (70%)

### pattern_1: Label_1pct_5d

**Robustness Score:** 0.583
**Is Robust:** No

#### In-Sample Performance
- Success Rate: 50.3%
- Occurrences: 4031
- False Positive Rate: 46.2%

#### Out-of-Sample Performance
- Average Success Rate: 51.1%
- Average Occurrences: 806.0
- Average False Positive Rate: 45.6%
- Performance Degradation: -0.81%

#### Stability Metrics
- Success Rate Std: 3.59%
- Success Rate CV: 0.070
- Stability Score: 0.930

#### Consistency Metrics
- Consistent Folds: 0/3
- Consistency Rate: 0.0%

#### Market Condition Performance
- Low_Strong Bull: 47.1%
- High_Strong Bear: 50.2%
- Medium_Strong Bull: 55.8%

#### Validation Notes
- Out-of-sample success rate (51.1%) below threshold (70.0%)
- Consistency rate (0.0%) below threshold (70%)

### pattern_7: Label_1pct_10d

**Robustness Score:** 0.580
**Is Robust:** No

#### In-Sample Performance
- Success Rate: 49.1%
- Occurrences: 3718
- False Positive Rate: 50.8%

#### Out-of-Sample Performance
- Average Success Rate: 48.4%
- Average Occurrences: 743.0
- Average False Positive Rate: 51.4%
- Performance Degradation: 0.69%

#### Stability Metrics
- Success Rate Std: 2.52%
- Success Rate CV: 0.052
- Stability Score: 0.948

#### Consistency Metrics
- Consistent Folds: 0/3
- Consistency Rate: 0.0%

#### Market Condition Performance
- Low_Strong Bull: 45.8%
- High_Strong Bull: 51.8%
- Medium_Strong Bull: 47.5%

#### Validation Notes
- Out-of-sample success rate (48.4%) below threshold (70.0%)
- Consistency rate (0.0%) below threshold (70%)

### pattern_5: Label_1pct_5d

**Robustness Score:** 0.571
**Is Robust:** No

#### In-Sample Performance
- Success Rate: 45.3%
- Occurrences: 3803
- False Positive Rate: 51.0%

#### Out-of-Sample Performance
- Average Success Rate: 44.6%
- Average Occurrences: 738.7
- Average False Positive Rate: 52.0%
- Performance Degradation: 0.70%

#### Stability Metrics
- Success Rate Std: 2.46%
- Success Rate CV: 0.055
- Stability Score: 0.945

#### Consistency Metrics
- Consistent Folds: 0/3
- Consistency Rate: 0.0%

#### Market Condition Performance
- Low_Strong Bull: 43.1%
- High_Strong Bull: 48.1%
- Medium_Strong Bull: 42.7%

#### Validation Notes
- Out-of-sample success rate (44.6%) below threshold (70.0%)
- Consistency rate (0.0%) below threshold (70%)

### pattern_14: Label_1pct_3d

**Robustness Score:** 0.569
**Is Robust:** No

#### In-Sample Performance
- Success Rate: 47.0%
- Occurrences: 2880
- False Positive Rate: 41.2%

#### Out-of-Sample Performance
- Average Success Rate: 50.2%
- Average Occurrences: 564.7
- Average False Positive Rate: 40.9%
- Performance Degradation: -3.18%

#### Stability Metrics
- Success Rate Std: 6.39%
- Success Rate CV: 0.127
- Stability Score: 0.873

#### Consistency Metrics
- Consistent Folds: 0/3
- Consistency Rate: 0.0%

#### Market Condition Performance
- Low_Strong Bear: 41.6%
- High_Strong Bear: 52.3%
- Medium_Strong Bear: 56.8%

#### Validation Notes
- Out-of-sample success rate (50.2%) below threshold (70.0%)
- Consistency rate (0.0%) below threshold (70%)

### pattern_11: Label_2pct_10d

**Robustness Score:** 0.567
**Is Robust:** No

#### In-Sample Performance
- Success Rate: 41.9%
- Occurrences: 3422
- False Positive Rate: 46.9%

#### Out-of-Sample Performance
- Average Success Rate: 43.3%
- Average Occurrences: 678.0
- Average False Positive Rate: 48.4%
- Performance Degradation: -1.37%

#### Stability Metrics
- Success Rate Std: 3.18%
- Success Rate CV: 0.073
- Stability Score: 0.927

#### Consistency Metrics
- Consistent Folds: 0/3
- Consistency Rate: 0.0%

#### Market Condition Performance
- Low_Strong Bull: 39.0%
- High_Strong Bull: 44.5%
- Medium_Strong Bull: 46.5%

#### Validation Notes
- Out-of-sample success rate (43.3%) below threshold (70.0%)
- Consistency rate (0.0%) below threshold (70%)

## Recommendations

1. **Low Robustness** - Only {robustness_rate:.0%} of patterns are robust. Pattern discovery needs significant improvement to ensure time-period stability.

2. **Low Performance Degradation** - Patterns maintain performance well on out-of-sample data (avg {avg_degradation:.1f}% degradation).

3. **Good Stability** - Patterns show good stability across time periods (avg stability score: {avg_stability:.3f}).

## Conclusion

The cross-validation analysis tested pattern robustness across multiple time periods using 5-fold time-series cross-validation. The results indicate that the pattern discovery approach needs improvement to produce more robust patterns. Key metrics include robustness score, stability score, consistency rate, and performance degradation from in-sample to out-of-sample data.

---

*Report generated by TimeSeriesCrossValidator*
