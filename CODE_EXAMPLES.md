# Code Implementation Examples

## Example 1: Update config.yaml

**Location**: `config.yaml` lines 45-80

Replace this:
```yaml
# Pattern Discovery Parameters - Balanced for Realistic Data
pattern_discovery:
  min_occurrences: 30
  min_success_rate: 55
  high_confidence_rate: 65
  high_confidence_occurrences: 40
  p_value_threshold: 0.05
  max_features_per_pattern: 5
  test_combinations: [1, 2, 3, 4, 5]

# Movement Parameters
movement:
  time_windows: [3, 5, 10, 20, 30]
  thresholds: [1, 2, 3, 5, 7, 10]
```

With this:
```yaml
# Pattern Discovery Parameters - Improved Discovery
pattern_discovery:
  min_occurrences: 12              # More patterns discovered
  min_success_rate: 51             # Just above statistical random (50%)
  high_confidence_rate: 65         # Quality filter for portfolio
  high_confidence_occurrences: 40  
  p_value_threshold: 0.10          # Looser for discovery (tighter in validation)
  max_features_per_pattern: 6      # Allow 6-feature combinations
  test_combinations: [1, 2, 3, 4, 5, 6]  # Test all combo lengths

# Movement Parameters - Realistic for Conservative Stock
movement:
  time_windows: [5, 10, 15, 20, 30]     # Focus on 5-30 day moves
  thresholds: [0.5, 1.0, 1.5, 2.0, 3.0] # Realistic percentage moves
  
  # Optional: Add intermediate targets
  extended_thresholds: [0.25, 0.75, 2.5, 4.0, 5.0]
  extended_time_windows: [3, 7, 12, 25]
```

---

## Example 2: Expand Feature Search in Phase 4

**File**: `src/phase4_pattern_discovery.py`

**Section to replace** (around line 165-180):
```python
        # Limit features for computational efficiency
        # Use top features by variance
        feature_variances = {f: self.data[f].var() for f in self.numeric_features}
        top_features = sorted(feature_variances.items(), key=lambda x: x[1], reverse=True)[:50]
        top_feature_names = [f[0] for f in top_features]
        
        # Test combinations based on config
        for n_features in self.test_combinations:
            logger.info(f"  Testing {n_features}-feature combinations...")
            
            feature_combos = list(combinations(top_feature_names, n_features))
            
            # Limit combinations for efficiency (more restrictive for higher n_features)
            max_combos = {1: 5000, 2: 1000, 3: 200, 4: 50, 5: 10}.get(n_features, 100)
```

**Replace with**:
```python
        # Expand feature search space
        # Use top features by variance but test more features
        feature_variances = {f: self.data[f].var() for f in self.numeric_features}
        top_features = sorted(feature_variances.items(), key=lambda x: x[1], reverse=True)[:120]
        top_feature_names = [f[0] for f in top_features]
        
        logger.info(f"  Using {len(top_feature_names)} features (top by variance) for pattern testing")
        
        # Test combinations based on config with expanded limits
        for n_features in self.test_combinations:
            logger.info(f"  Testing {n_features}-feature combinations...")
            
            feature_combos = list(combinations(top_feature_names, n_features))
            
            # Expanded combination testing limits
            # Allows testing more combinations to increase pattern discovery
            max_combos = {
                1: 8000,    # was 5000
                2: 5000,    # was 1000
                3: 2000,    # was 200
                4: 500,     # was 50
                5: 100,     # was 10
                6: 20       # new
            }.get(n_features, 100)
            
            if len(feature_combos) > max_combos:
                # Log sampling rate
                sampling_pct = (max_combos / len(feature_combos)) * 100
                feature_combos = feature_combos[:max_combos]
                logger.info(f"    Testing {max_combos} combinations ({sampling_pct:.1f}% of {len(list(combinations(top_feature_names, n_features)))} possible)")
            else:
                logger.info(f"    Testing all {len(feature_combos)} combinations")
```

---

## Example 3: Remove Ambiguous "~" Operator

### Part A: Fix Phase 4 (Pattern Discovery)

**File**: `src/phase4_pattern_discovery.py`

**Find this method** (around line 220):
```python
    def _find_best_thresholds(self, features: Tuple[str, ...], label_col: str, direction: str = 'long') -> Optional[Dict]:
        """
        Find optimal thresholds for a given feature combination.
        """
        # ... existing code ...
        
        for thresholds in combinations(thresholds_to_test, len(features)):
            conditions = {}
            for i, feature in enumerate(features):
                # Determine direction based on correlation with success
                corr = self._calculate_feature_success_correlation(feature, label_col, direction)
                
                if corr > 0:
                    # Higher values correlate with success
                    threshold_val = self.data[feature].quantile(thresholds[i])
                    conditions[feature] = {'operator': '~', 'value': threshold_val}  # PROBLEM!
                else:
                    # Lower values correlate with success
                    threshold_val = self.data[feature].quantile(thresholds[i])
                    conditions[feature] = {'operator': '~', 'value': threshold_val}  # PROBLEM!
```

**Replace with**:
```python
    def _find_best_thresholds(self, features: Tuple[str, ...], label_col: str, direction: str = 'long') -> Optional[Dict]:
        """
        Find optimal thresholds for a given feature combination.
        
        Uses explicit >= and <= operators instead of ambiguous ~
        """
        thresholds_to_test = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        best_pattern = None
        best_score = 0
        
        for thresholds in combinations(thresholds_to_test, len(features)):
            conditions = {}
            for i, feature in enumerate(features):
                # Determine direction based on correlation with success
                corr = self._calculate_feature_success_correlation(feature, label_col, direction)
                
                if corr > 0:
                    # Higher values correlate with success -> use >= operator
                    threshold_val = self.data[feature].quantile(thresholds[i])
                    conditions[feature] = {
                        'operator': '>=',
                        'value': float(threshold_val)  # Explicit >=
                    }
                else:
                    # Lower values correlate with success -> use <= operator
                    threshold_val = self.data[feature].quantile(thresholds[i])
                    conditions[feature] = {
                        'operator': '<=',
                        'value': float(threshold_val)  # Explicit <=
                    }
            
            # Evaluate pattern with new operators
            pattern = self._evaluate_pattern(conditions, label_col, direction)
            
            if pattern and pattern['success_rate'] > best_score:
                best_score = pattern['success_rate']
                best_pattern = pattern
        
        return best_pattern
```

### Part B: Fix Phase 6 (Validation)

**File**: `src/phase6_validation.py`

**Find this section** (around line 140):
```python
        for feature, condition in conditions.items():
            if feature not in data.columns:
                return None
            
            if condition['operator'] == '>=':
                mask &= (data[feature] >= condition['value'])
            elif condition['operator'] == '<=':
                mask &= (data[feature] <= condition['value'])
            elif condition['operator'] == '>':
                mask &= (data[feature] > condition['value'])
            elif condition['operator'] == '<':
                mask &= (data[feature] < condition['value'])
            elif condition['operator'] == '~':  # PROBLEM: Ambiguous range operator
                mask &= (abs(data[feature] - condition['value']) < condition['value'] * 0.1)
```

**Replace with**:
```python
        for feature, condition in conditions.items():
            if feature not in data.columns:
                logger.warning(f"Feature {feature} not found in data, skipping pattern validation")
                return None
            
            op = condition['operator']
            val = condition['value']
            
            # Use only explicit operators
            if op == '>=':
                mask &= (data[feature] >= val)
            elif op == '<=':
                mask &= (data[feature] <= val)
            elif op == '>':
                mask &= (data[feature] > val)
            elif op == '<':
                mask &= (data[feature] < val)
            elif op == '~':
                # DEPRECATED: Ambiguous range operator from old patterns
                # Convert to equivalent >= operator for compatibility
                logger.warning(f"Pattern uses deprecated ~ operator for {feature}. Converting to >= for validation.")
                mask &= (data[feature] >= val)
            else:
                logger.error(f"Unknown operator: {op}")
                return None
```

---

## Example 4: Create Pattern Scoring Module

**New File**: `src/pattern_scorer.py`

```python
"""
Enhanced pattern scoring system with multiple metrics.
Scores patterns by success rate, statistical significance, move quality, and frequency.
"""
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class PatternScorer:
    """Score patterns by multiple factors for better selection"""
    
    @staticmethod
    def calculate_composite_score(pattern, occurrences_count=None):
        """
        Calculate composite score for pattern selection.
        
        Factors:
        - Success rate (40%): Core metric
        - Statistical significance (30%): Based on p-value
        - Move quality (15%): Average move size achieved
        - Occurrence frequency (15%): How often pattern appears
        
        Args:
            pattern: Pattern dict with metrics
            occurrences_count: Number of occurrences (optional override)
        
        Returns:
            Dict with composite_score and component scores
        """
        success_rate = pattern.get('success_rate', 50)
        occurrences = occurrences_count if occurrences_count else pattern.get('occurrences', 0)
        p_value = pattern.get('p_value', 0.5)
        avg_move = pattern.get('avg_move', 1.0)
        
        # 1. Success rate score (0-100, normalized)
        # Raw success rate, capped at 100
        sr_score = min(success_rate, 100)
        
        # 2. Statistical significance score (0-100, inverse p-value)
        # p=0.05 -> score 95, p=0.50 -> score 50
        # Better patterns have smaller p-values
        p_value_safe = max(p_value, 0.0001)  # Avoid division by zero
        p_score = (1 - p_value_safe) * 100
        
        # 3. Move quality score (0-100, normalized to expected move)
        # Assume 1% move = baseline (50 points)
        # 2% move = 100 points, 0.5% move = 25 points
        move_score = min((avg_move / 1.0) * 50, 100)
        
        # 4. Frequency score (0-100)
        # 50+ occurrences = 100 points
        # 10 occurrences = 20 points
        # 100 occurrences = 100 points (capped)
        freq_score = min((occurrences / 50) * 100, 100)
        
        # Weighted composite score
        composite = (
            sr_score * 0.40 +              # Success rate: 40% weight
            p_score * 0.30 +               # Statistical significance: 30% weight
            move_score * 0.15 +            # Move quality: 15% weight
            freq_score * 0.15              # Frequency: 15% weight
        )
        
        return {
            'composite_score': round(composite, 2),
            'components': {
                'success_rate': round(sr_score, 2),
                'statistical_significance': round(p_score, 2),
                'move_quality': round(move_score, 2),
                'frequency': round(freq_score, 2)
            },
            'weights': {
                'success_rate': 0.40,
                'statistical_significance': 0.30,
                'move_quality': 0.15,
                'frequency': 0.15
            },
            'explanation': f"Pattern with {success_rate:.1f}% success, {occurrences} occurrences, " \
                          f"{avg_move:.2f}% avg move, p-value={p_value:.4f}"
        }
    
    @staticmethod
    def calculate_win_loss_ratio(success_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate win/loss ratio from pattern metrics.
        
        Args:
            success_rate: Success percentage (0-100)
            avg_win: Average move when pattern succeeds (percentage)
            avg_loss: Average loss when pattern fails (percentage)
        
        Returns:
            Win/loss ratio (higher is better, 1.0 is breakeven)
        """
        sr = success_rate / 100
        
        if avg_loss == 0:
            return float('inf') if avg_win > 0 else 1.0
        
        # Expected value calculation
        expected_value = (sr * avg_win) - ((1 - sr) * avg_loss)
        
        # Win/loss ratio
        wl_ratio = (sr * avg_win) / ((1 - sr) * abs(avg_loss)) if (1 - sr) * avg_loss != 0 else float('inf')
        
        return max(0, wl_ratio)
    
    @staticmethod
    def calculate_sharpe_ratio_proxy(pattern_outcomes, risk_free_rate=0.02):
        """
        Calculate Sharpe ratio proxy based on pattern wins/losses.
        
        Assumes each pattern occurrence = one trade with daily return impact.
        
        Args:
            pattern_outcomes: List of outcomes (1 for win, -1 for loss)
            risk_free_rate: Annual risk-free rate (default 2%)
        
        Returns:
            Sharpe ratio (higher is better)
        """
        if len(pattern_outcomes) < 2:
            return 0
        
        # Convert to returns (assume 1% win, -1% loss)
        returns = np.array(pattern_outcomes)
        
        # Annualize returns (assuming 252 trading days per year)
        annual_return = returns.mean() * 252
        annual_std = returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio
        if annual_std == 0:
            return float('inf') if annual_return > risk_free_rate else 0
        
        sharpe = (annual_return - risk_free_rate) / annual_std
        
        return max(0, sharpe)
    
    @staticmethod
    def calculate_max_drawdown(pattern_outcomes) -> tuple:
        """
        Calculate maximum consecutive losses and drawdown.
        
        Args:
            pattern_outcomes: List of outcomes (1 for win, -1 for loss)
        
        Returns:
            Tuple of (max_consecutive_losses, max_drawdown_pct)
        """
        cumulative = 0
        max_cumulative = 0
        max_drawdown = 0
        
        max_consecutive_losses = 0
        current_loss_streak = 0
        
        for outcome in pattern_outcomes:
            cumulative += outcome
            max_cumulative = max(max_cumulative, cumulative)
            drawdown = max_cumulative - cumulative
            max_drawdown = max(max_drawdown, drawdown)
            
            if outcome == -1:
                current_loss_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_loss_streak)
            else:
                current_loss_streak = 0
        
        # Convert drawdown to percentage
        max_drawdown_pct = (max_drawdown / max(max_cumulative, 1)) * 100
        
        return (max_consecutive_losses, max_drawdown_pct)
    
    @staticmethod
    def score_for_portfolio_selection(pattern, weight_risk=True):
        """
        Create portfolio selection score incorporating risk metrics.
        
        Args:
            pattern: Pattern dict with all metrics
            weight_risk: Whether to penalize high drawdown
        
        Returns:
            Portfolio selection score (0-100)
        """
        # Base composite score
        scoring = PatternScorer.calculate_composite_score(pattern)
        base_score = scoring['composite_score']
        
        # Risk adjustment
        if weight_risk and 'max_drawdown' in pattern:
            # Penalize patterns with high drawdown
            drawdown_pct = pattern['max_drawdown']  # As percentage
            drawdown_penalty = (drawdown_pct / 100) * 20  # Max 20 point penalty
            base_score -= drawdown_penalty
        
        # Consistency bonus (patterns with low p-value variation)
        if 'validation_success_rate' in pattern and 'success_rate' in pattern:
            degradation = abs(pattern['validation_success_rate'] - pattern['success_rate'])
            consistency_bonus = max(0, 10 - degradation)
            base_score += consistency_bonus
        
        return max(0, min(100, base_score))

# Example usage in phase4:
# scorer = PatternScorer()
# score_dict = scorer.calculate_composite_score(pattern)
# pattern['composite_score'] = score_dict['composite_score']
# pattern['score_breakdown'] = score_dict['components']
```

---

## Example 5: Integration into Phase 5

**File**: `src/phase5_pattern_optimization.py`

Add this near the beginning of the class:

```python
from pattern_scorer import PatternScorer

class PatternOptimization:
    """
    Optimize patterns through deduplication, filtering, and scoring.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        # ... existing init code ...
        self.scorer = PatternScorer()
    
    def optimize_all_patterns(self):
        """Optimize patterns with improved scoring"""
        
        logger.info("=" * 80)
        logger.info("PHASE 5: PATTERN OPTIMIZATION")
        logger.info("=" * 80)
        
        # Load patterns
        logger.info("Loading patterns for optimization...")
        with open('data/classified_patterns.json', 'r') as f:
            patterns = json.load(f)
        
        initial_count = len(patterns)
        logger.info(f"Input patterns: {initial_count}")
        
        # Step 1: Apply deduplication filter
        logger.info("\nStep 1: Removing duplicate patterns...")
        from pattern_filter import PatternFilter
        pattern_filter = PatternFilter()
        patterns = pattern_filter.filter_patterns(
            patterns,
            remove_exact_duplicates=True,
            remove_condition_duplicates=True,
            remove_subsets=True
        )
        after_filter = len(patterns)
        logger.info(f"After deduplication: {after_filter} patterns (-{initial_count - after_filter})")
        
        # Step 2: Score all remaining patterns
        logger.info("\nStep 2: Scoring patterns...")
        for i, pattern in enumerate(patterns):
            if 'pattern' in pattern:
                p = pattern['pattern']
            else:
                p = pattern
            
            # Calculate composite score
            score_dict = self.scorer.calculate_composite_score(p)
            p['composite_score'] = score_dict['composite_score']
            p['score_breakdown'] = score_dict['components']
            
            # Calculate portfolio score (risk-adjusted)
            if 'max_drawdown' in p:
                p['portfolio_score'] = self.scorer.score_for_portfolio_selection(p, weight_risk=True)
            else:
                p['portfolio_score'] = score_dict['composite_score']
        
        # Step 3: Sort by composite score
        logger.info("Step 3: Sorting by quality score...")
        patterns.sort(key=lambda p: p.get('pattern', p).get('composite_score', 0), reverse=True)
        
        # Log top patterns
        logger.info("\nTop 10 patterns by composite score:")
        for i, p in enumerate(patterns[:10]):
            pat = p.get('pattern', p)
            logger.info(f"  {i+1}. Score: {pat.get('composite_score', 0):.2f}, "
                       f"Success: {pat.get('success_rate', 0):.1f}%, "
                       f"Occurrences: {pat.get('occurrences', 0)}")
        
        # Save optimized patterns
        logger.info("\nSaving optimized patterns...")
        with open('data/optimized_patterns.json', 'w') as f:
            json.dump(patterns, f, indent=2)
        
        logger.info(f"Optimization complete: {len(patterns)} patterns saved")
        
        return patterns
```

---

## Example 6: How to Use These Changes

### Step 1: Update Config (5 minutes)
```bash
# Edit config.yaml with new thresholds from Example 1
nano config.yaml
```

### Step 2: Update Phase 4 (10 minutes)
```bash
# Apply changes from Example 2 & 3 to phase4_pattern_discovery.py
nano src/phase4_pattern_discovery.py
```

### Step 3: Update Phase 6 (5 minutes)
```bash
# Apply changes from Example 3 to phase6_validation.py
nano src/phase6_validation.py
```

### Step 4: Add Scoring Module (5 minutes)
```bash
# Create new pattern_scorer.py from Example 4
nano src/pattern_scorer.py
```

### Step 5: Test Changes
```bash
# Run phase 4 with new settings
python main.py --phase 4

# Check results
python -c "import json; p=json.load(open('data/optimized_patterns.json')); print(f'Patterns: {len(p)}')"
```

### Step 6: Measure Improvement
```bash
# Compare before/after
python3 << 'EOF'
import json

with open('data/optimized_patterns.json', 'r') as f:
    patterns = json.load(f)

success_rates = [p.get('pattern', p).get('success_rate', 0) for p in patterns]
occurrences = [p.get('pattern', p).get('occurrences', 0) for p in patterns]

print(f"Total patterns: {len(patterns)}")
print(f"Avg success rate: {sum(success_rates)/len(success_rates):.2f}%")
print(f"Avg occurrences: {sum(occurrences)/len(occurrences):.0f}")
print(f"Min occurrences: {min(occurrences)}")
print(f"Max occurrences: {max(occurrences)}")

# Count patterns by quality band
high_quality = sum(1 for p in patterns if p.get('pattern', p).get('success_rate', 0) > 60)
medium_quality = sum(1 for p in patterns if 55 <= p.get('pattern', p).get('success_rate', 0) <= 60)
low_quality = sum(1 for p in patterns if 50 <= p.get('pattern', p).get('success_rate', 0) < 55)

print(f"\nQuality distribution:")
print(f"  High (>60%): {high_quality}")
print(f"  Medium (55-60%): {medium_quality}")
print(f"  Low (50-55%): {low_quality}")
EOF
```

---

## Testing & Validation

After implementing all changes:

```bash
# 1. Verify no errors
python main.py --phase 4 --dry-run

# 2. Check phase 4 output
python -c "import json; p=json.load(open('data/discovered_patterns.json')); print(f'Discovered: {len(p)} patterns')"

# 3. Check phase 5 output
python -c "import json; p=json.load(open('data/optimized_patterns.json')); print(f'Optimized: {len(p)} patterns')"

# 4. Validate operators
python3 << 'EOF'
import json
with open('data/optimized_patterns.json', 'r') as f:
    patterns = json.load(f)

has_tilde = 0
for p in patterns:
    pat = p.get('pattern', p)
    for feat, cond in pat.get('conditions', {}).items():
        if cond.get('operator') == '~':
            has_tilde += 1

print(f"Patterns with ~ operator: {has_tilde} (should be 0)")
EOF

# 5. Check composite scores
python3 << 'EOF'
import json
with open('data/optimized_patterns.json', 'r') as f:
    patterns = json.load(f)

patterns_with_scores = sum(1 for p in patterns if 'composite_score' in p.get('pattern', p))
print(f"Patterns with composite score: {patterns_with_scores}")
EOF
```

---

## Expected Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Patterns discovered | ~500 | ~800 | +60% |
| Min occurrences | 30 | 12 | -60% |
| Min success rate | 55% | 51% | Lower threshold |
| Avg success rate | 57% | 60% | +3% |
| Avg occurrences | ~30 | ~50 | +67% |
| Has composite scores | No | Yes | ✓ |
| Has ~ operator | Yes | No | Fixed ✓ |

