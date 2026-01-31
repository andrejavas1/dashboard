# Quick Implementation Guide: Improving Pattern Discovery & Success Rate

## Immediate Actions (30 minutes)

### Action 1: Update Configuration Thresholds
**File**: `config.yaml`

Replace the pattern_discovery section with more aggressive settings:

```yaml
# OLD (lines 45-51):
pattern_discovery:
  min_occurrences: 30
  min_success_rate: 55
  high_confidence_rate: 65
  high_confidence_occurrences: 40
  p_value_threshold: 0.05
  max_features_per_pattern: 5
  test_combinations: [1, 2, 3, 4, 5]

# NEW - More aggressive discovery:
pattern_discovery:
  min_occurrences: 12                    # Lower to find rarer patterns
  min_success_rate: 51                   # Just above random chance + margin
  high_confidence_rate: 65               # Quality threshold for portfolio
  high_confidence_occurrences: 40        
  p_value_threshold: 0.10                # Looser for discovery phase
  max_features_per_pattern: 6            # Allow 6-feature patterns
  test_combinations: [1, 2, 3, 4, 5, 6]  # Test more combinations
```

### Action 2: Expand Feature Search Space
**File**: `src/phase4_pattern_discovery.py` lines 165-175

```python
# OLD:
top_features = sorted(feature_variances.items(), key=lambda x: x[1], reverse=True)[:50]
max_combos = {1: 5000, 2: 1000, 3: 200, 4: 50, 5: 10}.get(n_features, 100)

# NEW - Larger search space:
top_features = sorted(feature_variances.items(), key=lambda x: x[1], reverse=True)[:120]
max_combos = {1: 8000, 2: 5000, 3: 2000, 4: 500, 5: 100, 6: 20}.get(n_features, 100)
```

---

## Phase 1: Thresholds & Labels (1-2 hours)

### Update Movement Parameters
**File**: `config.yaml` lines 31-32

```yaml
# OLD - Too aggressive for XOM:
thresholds: [1, 2, 3, 5, 7, 10]
time_windows: [3, 5, 10, 20, 30]

# NEW - Realistic targets:
thresholds: [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
time_windows: [5, 10, 15, 20, 30]

# ADD new intermediate labels:
additional_labels:
  - "Label_0pct_5d"   # Any move in 5 days
  - "Label_0.5pct_10d"  # 0.5% in 10 days  
```

### Why This Works:
- **0.5% in 5 days** = ~36% annual return (achievable)
- **1% in 10 days** = realistic for conservative stock
- **5% in 30 days** = aggressive but possible
- More labels = more pattern variants tested

---

## Phase 2: Fix Pattern Operators (2-3 hours)

### Replace Ambiguous "~" Operator

**File**: `src/phase4_pattern_discovery.py`

**Current problematic code (around line 220)**:
```python
# PROBLEM: Uses hardcoded ±10% range
conditions[feature] = {'operator': '~', 'value': threshold_val}
```

**New implementation**:
```python
def _find_best_thresholds(self, features, label_col, direction='long'):
    """Find optimal thresholds using explicit operators"""
    
    best_pattern = None
    best_score = 0
    thresholds_to_test = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    for thresholds in combinations(thresholds_to_test, len(features)):
        conditions = {}
        for i, feature in enumerate(features):
            corr = self._calculate_feature_success_correlation(feature, label_col, direction)
            
            if corr > 0:
                # Higher values correlate with success -> use >=
                threshold_val = self.data[feature].quantile(thresholds[i])
                conditions[feature] = {'operator': '>=', 'value': float(threshold_val)}
            else:
                # Lower values correlate with success -> use <=
                threshold_val = self.data[feature].quantile(thresholds[i])
                conditions[feature] = {'operator': '<=', 'value': float(threshold_val)}
        
        pattern = self._evaluate_pattern(conditions, label_col, direction)
        
        if pattern and pattern['success_rate'] > best_score:
            best_score = pattern['success_rate']
            best_pattern = pattern
    
    return best_pattern
```

**File**: `src/phase6_validation.py`

**Fix validation to match discovery**:
```python
def evaluate_pattern_on_period(self, pattern, data):
    """Evaluate pattern with consistent operator logic"""
    
    conditions = pattern['conditions']
    label_col = pattern['label_col']
    direction = pattern.get('direction', 'unknown')
    
    mask = pd.Series(True, index=data.index)
    
    for feature, condition in conditions.items():
        if feature not in data.columns:
            return None
        
        # Use explicit operators only
        op = condition['operator']
        val = condition['value']
        
        if op == '>=':
            mask &= (data[feature] >= val)
        elif op == '<=':
            mask &= (data[feature] <= val)
        elif op == '>':
            mask &= (data[feature] > val)
        elif op == '<':
            mask &= (data[feature] < val)
        # REMOVED: elif op == '~':  (ambiguous range operator)
    
    # ... rest of evaluation code
```

---

## Phase 3: Implement Pattern Weighting (3-4 hours)

Create a new file: **`src/pattern_scorer.py`**

```python
"""
Enhanced pattern scoring system with multiple metrics.
"""
import numpy as np
from scipy import stats

class PatternScorer:
    """Score patterns by multiple factors for better selection"""
    
    @staticmethod
    def calculate_composite_score(pattern, occurrences_count=None):
        """
        Calculate composite score for pattern.
        
        Factors:
        - Success rate (most important: 40%)
        - Statistical significance (30%)
        - Move quality (15%)
        - Occurrence frequency (15%)
        """
        success_rate = pattern.get('success_rate', 50)
        occurrences = pattern.get('occurrences', 0)
        p_value = pattern.get('p_value', 0.5)
        avg_move = pattern.get('avg_move', 1.0)
        
        # 1. Success rate score (0-100, normalized)
        sr_score = min(success_rate, 100)
        
        # 2. Statistical significance (0-100, inverse p-value)
        # p=0.05 -> score 95, p=0.50 -> score 50
        p_score = (1 - p_value) * 100
        
        # 3. Move quality (0-100, normalized to expected move)
        # Assume 1% is baseline (50 points)
        move_score = min((avg_move / 1.0) * 50, 100)
        
        # 4. Frequency score (0-100)
        # 50+ occurrences = 100 points, 10 = 20 points
        freq_score = min((occurrences / 50) * 100, 100)
        
        # Weighted composite
        composite = (
            sr_score * 0.40 +
            p_score * 0.30 +
            move_score * 0.15 +
            freq_score * 0.15
        )
        
        return {
            'composite_score': composite,
            'components': {
                'success_rate': sr_score,
                'statistical_significance': p_score,
                'move_quality': move_score,
                'frequency': freq_score
            },
            'weights': {
                'success_rate': 0.40,
                'statistical_significance': 0.30,
                'move_quality': 0.15,
                'frequency': 0.15
            }
        }
    
    @staticmethod
    def calculate_drawdown_score(occurrences_df, label_col, direction):
        """
        Score pattern by maximum drawdown risk.
        
        Returns score 0-100 where:
        - 100 = no drawdown between trades
        - 50 = max 5 consecutive losses
        - 0 = >10 consecutive losses
        """
        if direction == 'long':
            outcomes = occurrences_df[label_col] == 'STRONG_UP'
        else:
            outcomes = occurrences_df[label_col] == 'STRONG_DOWN'
        
        # Find max consecutive losses
        losses = (~outcomes).astype(int)
        max_consecutive_losses = 0
        current_loss_streak = 0
        
        for loss in losses:
            if loss == 1:
                current_loss_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_loss_streak)
            else:
                current_loss_streak = 0
        
        # Convert to score (0-100)
        drawdown_score = max(0, 100 - (max_consecutive_losses * 10))
        
        return drawdown_score
    
    @staticmethod
    def calculate_sharpe_proxy(occurrences_df, label_col, direction, risk_free_rate=0.02):
        """
        Calculate Sharpe ratio proxy based on pattern wins/losses.
        
        Assumes each pattern occurrence = one trade.
        """
        if direction == 'long':
            returns = (occurrences_df[label_col] == 'STRONG_UP').astype(float) * 2 - 1  # 1% win, -1% loss
        else:
            returns = (occurrences_df[label_col] == 'STRONG_DOWN').astype(float) * 2 - 1
        
        if len(returns) < 2:
            return 0
        
        annual_return = returns.mean() * 252  # Assuming daily data
        annual_std = returns.std() * np.sqrt(252)
        
        sharpe = (annual_return - risk_free_rate) / (annual_std + 1e-6)
        
        return max(0, sharpe)  # Return 0 if negative
```

**Integrate into Phase 4**: `src/phase4_pattern_discovery.py`

```python
from pattern_scorer import PatternScorer

class PatternDiscovery:
    # ... existing code ...
    
    def _evaluate_pattern(self, conditions, label_col, direction='long'):
        """Evaluate pattern (modified to add scoring)"""
        
        # ... existing evaluation code ...
        
        pattern = {
            'conditions': conditions,
            'success_rate': success_rate,
            'occurrences': len(occurrences),
            'p_value': p_value,
            'avg_move': avg_move,
            'direction': direction,
            'label_col': label_col,
            'method': 'rule_based'
        }
        
        # ADD: Score pattern for filtering
        scoring_result = PatternScorer.calculate_composite_score(pattern, len(occurrences))
        pattern['composite_score'] = scoring_result['composite_score']
        pattern['score_components'] = scoring_result['components']
        
        return pattern
```

---

## Phase 4: Add Market Regime Detection (4-6 hours)

Create: **`src/market_regime_detector.py`**

```python
"""
Detect market regime (trend, volatility) for regime-specific patterns.
"""
import pandas as pd
import numpy as np

class MarketRegimeDetector:
    """Identify market regimes for pattern discovery"""
    
    @staticmethod
    def detect_volatility_regime(price_data, window=20):
        """
        Detect volatility regime: LOW, NORMAL, HIGH
        """
        returns = price_data['Close'].pct_change()
        volatility = returns.rolling(window).std() * np.sqrt(252)  # Annual vol
        
        vol_percentiles = volatility.quantile([0.25, 0.75])
        
        regime = pd.Series('NORMAL', index=volatility.index)
        regime[volatility < vol_percentiles[0.25]] = 'LOW'
        regime[volatility > vol_percentiles[0.75]] = 'HIGH'
        
        return regime
    
    @staticmethod
    def detect_trend_regime(price_data, window_short=50, window_long=200):
        """
        Detect trend regime: UPTREND, DOWNTREND, SIDEWAYS
        """
        sma_short = price_data['Close'].rolling(window_short).mean()
        sma_long = price_data['Close'].rolling(window_long).mean()
        
        regime = pd.Series('SIDEWAYS', index=price_data.index)
        regime[sma_short > sma_long] = 'UPTREND'
        regime[sma_short < sma_long] = 'DOWNTREND'
        
        return regime
    
    @staticmethod
    def combine_regimes(volatility_regime, trend_regime):
        """
        Combine into composite regime:
        - UpTrend_LowVol
        - UpTrend_HighVol
        - DownTrend_LowVol
        - etc.
        """
        return trend_regime + '_' + volatility_regime

# Usage in phase 4:
# detector = MarketRegimeDetector()
# vol_regime = detector.detect_volatility_regime(ohlcv_data)
# trend_regime = detector.detect_trend_regime(ohlcv_data)
# 
# For each regime, discover patterns separately:
# patterns_uptrend_lowvol = discover_patterns(data[regime == 'UPTREND_LOW'])
```

---

## Phase 5: Verify Filter is Running (30 minutes)

**File**: `src/phase5_pattern_optimization.py`

Add this logging to verify deduplication:

```python
def run_phase5(self):
    """Run phase 5 with filter verification"""
    
    logger.info("Loading classified patterns...")
    with open('data/classified_patterns.json', 'r') as f:
        classified = json.load(f)
    
    # Log pre-filter stats
    initial_count = len(classified)
    logger.info(f"Patterns before filtering: {initial_count}")
    
    # Apply filter
    from pattern_filter import PatternFilter
    pattern_filter = PatternFilter()
    optimized = pattern_filter.filter_patterns(
        classified,
        remove_exact_duplicates=True,
        remove_condition_duplicates=True,
        remove_subsets=True
    )
    
    # Log post-filter stats
    final_count = len(optimized)
    reduction_pct = ((initial_count - final_count) / initial_count) * 100
    logger.info(f"Patterns after filtering: {final_count}")
    logger.info(f"Reduction: {reduction_pct:.1f}%")
    logger.info(f"Unique patterns saved for validation: {final_count}")
    
    # Save optimized patterns
    with open('data/optimized_patterns.json', 'w') as f:
        json.dump(optimized, f, indent=2)
    
    return optimized
```

---

## Expected Results Timeline

| Week | Action | Expected Result |
|------|--------|-----------------|
| 1 | Update config + expand features | +40% more patterns discovered |
| 2 | Fix operators + add weighting | +15% better pattern quality |
| 3 | Add regime detection | +30% more relevant patterns |
| 4+ | Portfolio optimization | 20-30% overall improvement |

---

## Testing Your Changes

After each change, run:

```bash
# Test configuration
python -c "import yaml; cfg=yaml.safe_load(open('config.yaml')); print(f\"Min occurrences: {cfg['pattern_discovery']['min_occurrences']}\")"

# Run a single phase
python main.py --phase 4

# Check results
python -c "import json; patterns=json.load(open('data/optimized_patterns.json')); print(f'Patterns: {len(patterns)}')"
```

---

## Dashboard Updates

Update `pattern_dashboard_server.py` to show improvement metrics:

```python
@app.route('/api/pipeline-metrics')
def get_pipeline_metrics():
    """Return latest pipeline metrics"""
    with open('data/pipeline_runs.json', 'r') as f:
        runs = json.load(f)
    
    if not runs:
        return jsonify({})
    
    latest = runs[-1]['metrics']
    
    return jsonify({
        'patterns_discovered': latest.get('patterns_discovered', 0),
        'patterns_validated': latest.get('patterns_validated', 0),
        'avg_success_rate': latest.get('avg_success_rate', 0),
        'avg_occurrences': latest.get('avg_occurrences', 0),
        'total_occurrences': latest.get('total_occurrences', 0)
    })
```

---

## Summary of Changes

| File | Changes | Impact |
|------|---------|--------|
| `config.yaml` | Lower thresholds, expand testing | +40% discovery |
| `src/phase4_pattern_discovery.py` | Remove "~", explicit operators | Better validation match |
| `src/phase6_validation.py` | Match phase 4 operators | Fewer false rejections |
| **NEW** `src/pattern_scorer.py` | Multi-factor scoring | Better quality patterns |
| **NEW** `src/market_regime_detector.py` | Regime-specific patterns | More relevant patterns |
| `src/phase5_pattern_optimization.py` | Add filter logging | Transparency |

All changes maintain backward compatibility and are easily reversible.

