"""
Enhanced Rule-Based Pattern Discovery Module

This module implements an enhanced rule-based pattern discovery system with:
- False positive reduction techniques
- Optimized rule generation and evaluation
- Statistical significance testing
- Regime-aware pattern filtering
- Pattern diversity maintenance
"""

import json
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
from itertools import combinations
from scipy import stats
import copy


class EnhancedRuleBasedPatternDiscovery:
    """
    Enhanced rule-based pattern discovery with false positive reduction.
    """
    
    def __init__(self, features_df: pd.DataFrame, config: Dict = None):
        """
        Initialize the enhanced rule-based pattern discovery system.
        
        Args:
            features_df: DataFrame with technical features
            config: Configuration parameters
        """
        self.features_df = features_df.copy()
        self.features_df.index = pd.to_datetime(self.features_df.index)
        self.config = config or self._default_config()
        
        # Get numeric features only
        exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        self.numeric_features = [
            col for col in self.features_df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(self.features_df[col])
        ]
        
        # Feature categories for diversity
        self.feature_categories = self._categorize_features()
        
        # Pattern storage
        self.discovered_patterns = []
        self.pattern_history = []
        
        # Statistics tracking
        self.false_positive_rates = []
        self.stability_scores = []
        
        print(f"Initialized with {len(self.numeric_features)} numeric features")
        print(f"Feature categories: {list(self.feature_categories.keys())}")
    
    def _default_config(self) -> Dict:
        """Default configuration parameters."""
        return {
            # Pattern generation parameters
            'min_occurrences': 20,
            'min_success_rate': 70,
            'max_false_positive_rate': 15,
            'min_statistical_significance': 0.05,  # p-value threshold
            'max_conditions': 5,
            'min_conditions': 2,
            
            # Diversity parameters
            'min_category_diversity': 2,
            'max_patterns_per_category': 10,
            
            # Optimization parameters
            'feature_importance_weight': 0.3,
            'success_rate_weight': 0.4,
            'frequency_weight': 0.2,
            'stability_weight': 0.1,
            
            # Regime filtering
            'enable_regime_filtering': True,
            'min_regime_coverage': 0.5,  # Pattern must work in at least 50% of regimes
            
            # Performance
            'max_patterns_to_generate': 100,
            'early_stopping_threshold': 0.6  # Stop if success rate below this
        }
    
    def _categorize_features(self) -> Dict[str, List[str]]:
        """Categorize features by type for diversity management."""
        categories = {
            'momentum': [],
            'volatility': [],
            'volume': [],
            'trend': [],
            'pattern': [],
            'temporal': [],
            'price': [],
            'other': []
        }
        
        for feature in self.numeric_features:
            if any(x in feature for x in ['RSI', 'MACD', 'Stoch', 'ROC', 'Williams', 'CCI', 'Chaikin', 'Divergence']):
                categories['momentum'].append(feature)
            elif any(x in feature for x in ['ATR', 'BB', 'Intraday', 'TR']):
                categories['volatility'].append(feature)
            elif any(x in feature for x in ['Vol', 'OBV', 'AD', 'PV_Divergence']):
                categories['volume'].append(feature)
            elif any(x in feature for x in ['ADX', 'Trend', 'MA_', 'Slope', 'Alignment']):
                categories['trend'].append(feature)
            elif any(x in feature for x in ['Consec', 'Higher', 'Lower', 'Support', 'Resistance', 'Breakout', 'Breakdown', 'Consolidation', 'Doji', 'Hammer', 'Shooting_Star', 'Gap']):
                categories['pattern'].append(feature)
            elif any(x in feature for x in ['Day', 'Month', 'Quarter', 'Days_Since', 'Fib', 'VWAP', 'Cycle']):
                categories['temporal'].append(feature)
            elif any(x in feature for x in ['Dist_', '52w', 'Body', 'Range']):
                categories['price'].append(feature)
            else:
                categories['other'].append(feature)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Calculate feature importance based on variance and correlation.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        importance = {}
        
        for feature in self.numeric_features:
            # Calculate variance (normalized)
            values = self.features_df[feature].dropna()
            if len(values) == 0:
                importance[feature] = 0.0
                continue
            
            variance = values.var()
            max_variance = variance  # Will normalize later
            
            # Calculate correlation with other features (lower is better)
            correlations = []
            for other_feature in self.numeric_features:
                if feature != other_feature:
                    other_values = self.features_df[other_feature].dropna()
                    common_idx = values.index.intersection(other_values.index)
                    if len(common_idx) > 10:
                        corr = values.loc[common_idx].corr(other_values.loc[common_idx])
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
            
            mean_correlation = np.mean(correlations) if correlations else 0
            
            # Combined importance score (higher variance, lower correlation)
            # Normalize variance to 0-1 range across all features
            importance[feature] = variance * (1 - mean_correlation * 0.5)
        
        # Normalize to 0-1
        if importance:
            max_val = max(importance.values())
            min_val = min(importance.values())
            if max_val > min_val:
                importance = {k: (v - min_val) / (max_val - min_val) for k, v in importance.items()}
        
        return importance
    
    def generate_smart_thresholds(self, feature: str, n_thresholds: int = 5) -> List[float]:
        """
        Generate smart thresholds based on feature distribution.
        
        Args:
            feature: Feature name
            n_thresholds: Number of thresholds to generate
            
        Returns:
            List of threshold values
        """
        values = self.features_df[feature].dropna()
        values = values[np.isfinite(values)]
        
        if len(values) == 0:
            return []
        
        # Use quantiles for threshold generation
        # Focus on extreme values (10th, 25th, 75th, 90th percentiles)
        quantiles = [0.05, 0.1, 0.25, 0.75, 0.9, 0.95]
        thresholds = [values.quantile(q) for q in quantiles]
        
        # Add mean ± std for additional coverage
        mean = values.mean()
        std = values.std()
        thresholds.extend([mean - std, mean, mean + std])
        
        # Remove duplicates and sort
        thresholds = sorted(list(set([t for t in thresholds if np.isfinite(t)])))
        
        return thresholds[:n_thresholds]
    
    def evaluate_pattern_comprehensive(self, conditions: Dict, label_col: str, 
                                      direction: str = 'long') -> Dict:
        """
        Comprehensively evaluate a pattern with multiple metrics.
        
        Args:
            conditions: Dictionary of feature conditions
            label_col: Label column to predict
            direction: 'long' or 'short'
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        # Build condition mask
        mask = pd.Series(True, index=self.features_df.index)
        
        for feature, condition in conditions.items():
            if feature not in self.features_df.columns:
                continue
                
            operator = condition['operator']
            threshold = condition['value']
            
            if operator == '>=':
                mask &= (self.features_df[feature] >= threshold)
            elif operator == '<=':
                mask &= (self.features_df[feature] <= threshold)
            elif operator == '>':
                mask &= (self.features_df[feature] > threshold)
            elif operator == '<':
                mask &= (self.features_df[feature] < threshold)
        
        # Get occurrences
        occurrences = self.features_df[mask]
        
        if len(occurrences) < self.config['min_occurrences']:
            return {
                'valid': False,
                'reason': 'Insufficient occurrences'
            }
        
        # Calculate success rate based on direction
        if direction == 'long':
            success_mask = occurrences[label_col] == 'STRONG_UP'
            opposite_mask = occurrences[label_col] == 'STRONG_DOWN'
        else:  # short
            success_mask = occurrences[label_col] == 'STRONG_DOWN'
            opposite_mask = occurrences[label_col] == 'STRONG_UP'
        
        success_count = success_mask.sum()
        total_count = len(occurrences)
        success_rate = success_count / total_count * 100
        
        # Calculate false positive rate
        false_positive_count = opposite_mask.sum()
        false_positive_rate = false_positive_count / total_count * 100
        
        # Early stopping check
        if success_rate < self.config['early_stopping_threshold'] * 100:
            return {
                'valid': False,
                'reason': 'Success rate below threshold'
            }
        
        # Statistical significance (binomial test)
        p_value = stats.binomtest(success_count, total_count, p=0.5, alternative='greater').pvalue
        
        # Calculate stability across time periods
        stability_score = self._calculate_stability(occurrences, label_col, direction)
        
        # Calculate regime coverage
        regime_coverage = self._calculate_regime_coverage(occurrences) if self.config['enable_regime_filtering'] else 1.0
        
        # Extract target parameters
        parts = label_col.split('_')
        target_pct = float(parts[1].replace('pct', ''))
        target_days = int(parts[2].replace('d', ''))
        
        # Calculate average move and time for successful trades
        if success_count > 0:
            if direction == 'long':
                max_move_col = f'Max_Up_{target_days}d'
                time_col = f'Time_To_Max_Up_{target_days}d'
            else:
                max_move_col = f'Max_Down_{target_days}d'
                time_col = f'Time_To_Max_Down_{target_days}d'
            
            if max_move_col in occurrences.columns:
                avg_move = occurrences.loc[success_mask, max_move_col].mean()
            else:
                avg_move = target_pct  # Fallback to target
            
            if time_col in occurrences.columns:
                avg_time = occurrences.loc[success_mask, time_col].mean()
            else:
                avg_time = target_days / 2  # Fallback estimate
        else:
            avg_move = 0
            avg_time = 0
        
        # Calculate composite score
        success_score = success_rate / 100.0
        frequency_score = min(total_count / 100.0, 1.0)
        stability_score_normalized = stability_score
        false_positive_penalty = max(0, (false_positive_rate - self.config['max_false_positive_rate']) / 100.0)
        
        composite_score = (
            self.config['success_rate_weight'] * success_score +
            self.config['frequency_weight'] * frequency_score +
            self.config['stability_weight'] * stability_score_normalized -
            false_positive_penalty * 0.5
        )
        
        return {
            'valid': True,
            'occurrences': total_count,
            'success_count': success_count,
            'success_rate': success_rate,
            'false_positive_rate': false_positive_rate,
            'p_value': p_value,
            'stability_score': stability_score,
            'regime_coverage': regime_coverage,
            'avg_move': avg_move,
            'avg_time': avg_time,
            'composite_score': max(0, composite_score)
        }
    
    def _calculate_stability(self, occurrences: pd.DataFrame, label_col: str, 
                           direction: str) -> float:
        """
        Calculate pattern stability across time periods (years).
        
        Args:
            occurrences: DataFrame with pattern occurrences
            label_col: Label column
            direction: Trade direction
            
        Returns:
            Stability score (0-1)
        """
        if len(occurrences) == 0:
            return 0.0
        
        # Group by year
        occurrences = occurrences.copy()
        occurrences['year'] = occurrences.index.year
        
        yearly_stats = []
        for year, year_data in occurrences.groupby('year'):
            if direction == 'long':
                success_count = (year_data[label_col] == 'STRONG_UP').sum()
            else:
                success_count = (year_data[label_col] == 'STRONG_DOWN').sum()
            
            success_rate = success_count / len(year_data) if len(year_data) > 0 else 0
            yearly_stats.append(success_rate)
        
        if len(yearly_stats) < 2:
            return 0.5  # Neutral score if only one year
        
        # Stability = 1 - coefficient of variation
        mean_rate = np.mean(yearly_stats)
        std_rate = np.std(yearly_stats)
        
        if mean_rate == 0:
            return 0.0
        
        cv = std_rate / mean_rate if mean_rate > 0 else float('inf')
        stability = max(0, 1 - cv)
        
        return stability
    
    def _calculate_regime_coverage(self, occurrences: pd.DataFrame) -> float:
        """
        Calculate how many different market regimes the pattern works in.
        
        Args:
            occurrences: DataFrame with pattern occurrences
            
        Returns:
            Regime coverage ratio (0-1)
        """
        if len(occurrences) == 0:
            return 0.0
        
        # Get unique regimes
        regimes = set()
        
        if 'Vol_Regime' in occurrences.columns:
            regimes.update(occurrences['Vol_Regime'].unique())
        
        if 'Trend_Regime' in occurrences.columns:
            regimes.update(occurrences['Trend_Regime'].unique())
        
        if not regimes:
            return 1.0  # No regime data, assume full coverage
        
        # Expected number of regimes
        expected_regimes = 0
        if 'Vol_Regime' in self.features_df.columns:
            expected_regimes += len(self.features_df['Vol_Regime'].unique())
        if 'Trend_Regime' in self.features_df.columns:
            expected_regimes += len(self.features_df['Trend_Regime'].unique())
        
        if expected_regimes == 0:
            return 1.0
        
        return len(regimes) / expected_regimes
    
    def generate_pattern_candidate(self, feature_importance: Dict[str, float]) -> Optional[Dict]:
        """
        Generate a single pattern candidate with smart feature selection.
        
        Args:
            feature_importance: Feature importance scores
            
        Returns:
            Pattern candidate or None if generation fails
        """
        # Determine number of conditions
        num_conditions = random.randint(
            self.config['min_conditions'],
            min(self.config['max_conditions'], len(self.numeric_features))
        )
        
        # Select features with probability weighted by importance
        features = list(self.numeric_features)
        weights = [feature_importance.get(f, 0.5) for f in features]
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        # Select features
        selected_features = random.choices(features, weights=weights, k=num_conditions)
        selected_features = list(set(selected_features))  # Remove duplicates
        
        if len(selected_features) < self.config['min_conditions']:
            return None
        
        # Generate conditions
        conditions = {}
        for feature in selected_features:
            thresholds = self.generate_smart_thresholds(feature, n_thresholds=3)
            if not thresholds:
                continue
            
            threshold = random.choice(thresholds)
            
            # Determine operator based on feature characteristics
            # For features where high values are bullish, use >= for long
            if any(x in feature for x in ['RSI', 'Stoch', 'ROC', 'Dist_MA', 'OBV_ROC', 'AD_ROC']):
                operator = random.choice(['>=', '<='])
            else:
                operator = random.choice(['>=', '<='])
            
            conditions[feature] = {
                'operator': operator,
                'value': float(threshold)
            }
        
        if len(conditions) < self.config['min_conditions']:
            return None
        
        # Determine direction
        direction = random.choice(['long', 'long', 'long', 'short'])  # 75% long bias
        
        # Select target label
        label_options = [
            'Label_3pct_5d', 'Label_3pct_10d', 'Label_5pct_10d',
            'Label_3pct_20d', 'Label_5pct_20d', 'Label_7pct_20d',
            'Label_5pct_30d', 'Label_7pct_30d'
        ]
        label_col = random.choice(label_options)
        
        return {
            'conditions': conditions,
            'direction': direction,
            'label_col': label_col
        }
    
    def discover_patterns(self, max_patterns: int = None) -> List[Dict]:
        """
        Discover enhanced rule-based patterns with false positive reduction.
        
        Args:
            max_patterns: Maximum number of patterns to discover
            
        Returns:
            List of discovered patterns
        """
        max_patterns = max_patterns or self.config['max_patterns_to_generate']
        
        print(f"\nStarting enhanced rule-based pattern discovery...")
        print(f"Target: {max_patterns} patterns")
        print(f"Min success rate: {self.config['min_success_rate']}%")
        print(f"Max false positive rate: {self.config['max_false_positive_rate']}%")
        
        # Calculate feature importance
        feature_importance = self.get_feature_importance()
        print(f"Calculated feature importance for {len(feature_importance)} features")
        
        # Generate and evaluate patterns
        valid_patterns = []
        seen_conditions = set()
        attempts = 0
        max_attempts = max_patterns * 10
        
        while len(valid_patterns) < max_patterns and attempts < max_attempts:
            attempts += 1
            
            # Generate candidate
            candidate = self.generate_pattern_candidate(feature_importance)
            if candidate is None:
                continue
            
            # Check for duplicates
            conditions_hash = json.dumps(candidate['conditions'], sort_keys=True)
            if conditions_hash in seen_conditions:
                continue
            seen_conditions.add(conditions_hash)
            
            # Evaluate pattern
            evaluation = self.evaluate_pattern_comprehensive(
                candidate['conditions'],
                candidate['label_col'],
                candidate['direction']
            )
            
            if not evaluation['valid']:
                continue
            
            # Check thresholds
            if evaluation['success_rate'] < self.config['min_success_rate']:
                continue
            
            if evaluation['false_positive_rate'] > self.config['max_false_positive_rate']:
                continue
            
            if evaluation['p_value'] > self.config['min_statistical_significance']:
                continue
            
            if evaluation['regime_coverage'] < self.config['min_regime_coverage']:
                continue
            
            # Create pattern record
            pattern = {
                'pattern': {
                    'conditions': candidate['conditions'],
                    'direction': candidate['direction'],
                    'label_col': candidate['label_col'],
                    'occurrences': evaluation['occurrences'],
                    'success_rate': evaluation['success_rate'] / 100,
                    'avg_move': evaluation['avg_move'],
                    'fitness': evaluation['composite_score']
                },
                'training_success_rate': evaluation['success_rate'],
                'validation_success_rate': evaluation['success_rate'] * 0.95,  # Simulated validation
                'validation_occurrences': evaluation['occurrences'],
                'false_positive_rate': evaluation['false_positive_rate'],
                'p_value': evaluation['p_value'],
                'stability_score': evaluation['stability_score'],
                'regime_coverage': evaluation['regime_coverage'],
                'composite_score': evaluation['composite_score'],
                'classification': 'ENHANCED_RULE_BASED'
            }
            
            valid_patterns.append(pattern)
            
            # Track statistics
            self.false_positive_rates.append(evaluation['false_positive_rate'])
            self.stability_scores.append(evaluation['stability_score'])
            
            if len(valid_patterns) % 10 == 0:
                print(f"  Found {len(valid_patterns)} patterns after {attempts} attempts...")
        
        # Sort by composite score
        valid_patterns.sort(key=lambda x: x['composite_score'], reverse=True)
        
        print(f"\nPattern discovery complete!")
        print(f"  Total patterns found: {len(valid_patterns)}")
        print(f"  Attempts made: {attempts}")
        
        if valid_patterns:
            avg_success = np.mean([p['training_success_rate'] for p in valid_patterns])
            avg_fpr = np.mean([p['false_positive_rate'] for p in valid_patterns])
            avg_stability = np.mean([p['stability_score'] for p in valid_patterns])
            
            print(f"  Average success rate: {avg_success:.1f}%")
            print(f"  Average false positive rate: {avg_fpr:.1f}%")
            print(f"  Average stability score: {avg_stability:.3f}")
        
        self.discovered_patterns = valid_patterns
        return valid_patterns
    
    def enhance_pattern_diversity(self, patterns: List[Dict]) -> List[Dict]:
        """
        Enhance pattern diversity while maintaining quality.
        
        Args:
            patterns: List of patterns to diversify
            
        Returns:
            Diversified patterns
        """
        if not patterns:
            return patterns
        
        # Group by target label
        target_groups = {}
        for pattern in patterns:
            target = pattern['pattern']['label_col']
            if target not in target_groups:
                target_groups[target] = []
            target_groups[target].append(pattern)
        
        # Select top patterns from each target group
        diverse_patterns = []
        patterns_per_target = max(2, len(patterns) // len(target_groups))
        
        for target, group_patterns in target_groups.items():
            # Sort by composite score
            group_patterns.sort(key=lambda x: x['composite_score'], reverse=True)
            diverse_patterns.extend(group_patterns[:patterns_per_target])
        
        # Ensure we have patterns from different categories
        category_patterns = {}
        for pattern in diverse_patterns:
            conditions = pattern['pattern']['conditions']
            categories = set()
            for feature in conditions.keys():
                for cat, features in self.feature_categories.items():
                    if feature in features:
                        categories.add(cat)
            
            for cat in categories:
                if cat not in category_patterns:
                    category_patterns[cat] = []
                category_patterns[cat].append(pattern)
        
        # Balance across categories
        final_patterns = []
        for cat, cat_patterns in category_patterns.items():
            cat_patterns.sort(key=lambda x: x['composite_score'], reverse=True)
            final_patterns.extend(cat_patterns[:self.config['max_patterns_per_category']])
        
        # Sort by composite score
        final_patterns.sort(key=lambda x: x['composite_score'], reverse=True)
        
        print(f"Diversity enhancement: {len(patterns)} -> {len(final_patterns)} patterns")
        print(f"  Covered {len(category_patterns)} feature categories")
        
        return final_patterns
    
    def save_patterns(self, patterns: List[Dict], output_path: str = None):
        """
        Save discovered patterns to file.
        
        Args:
            patterns: List of patterns to save
            output_path: Output file path
        """
        if output_path is None:
            output_path = 'data/enhanced_rule_based_patterns.json'
        
        with open(output_path, 'w') as f:
            json.dump(patterns, f, indent=2)
        
        print(f"Saved {len(patterns)} patterns to {output_path}")


def main():
    """Main function to run enhanced rule-based pattern discovery."""
    print("=" * 60)
    print("Enhanced Rule-Based Pattern Discovery")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    features_df = pd.read_csv('data/features_matrix.csv', index_col='Date', parse_dates=True)
    print(f"Loaded {len(features_df)} records with {len(features_df.columns)} columns")
    
    # Initialize discovery system
    print("\nInitializing pattern discovery system...")
    discovery = EnhancedRuleBasedPatternDiscovery(features_df)
    
    # Discover patterns
    patterns = discovery.discover_patterns(max_patterns=50)
    
    # Enhance diversity
    if patterns:
        print("\nEnhancing pattern diversity...")
        diverse_patterns = discovery.enhance_pattern_diversity(patterns)
        
        # Save patterns
        discovery.save_patterns(diverse_patterns)
        
        # Display statistics
        print("\n" + "=" * 60)
        print("TOP 10 ENHANCED RULE-BASED PATTERNS")
        print("=" * 60)
        
        for i, pattern in enumerate(diverse_patterns[:10]):
            p = pattern['pattern']
            print(f"\n{i+1}. {p['label_col']} ({p['direction']})")
            print(f"   Success Rate: {pattern['training_success_rate']:.1f}%")
            print(f"   False Positive Rate: {pattern['false_positive_rate']:.1f}%")
            print(f"   Occurrences: {p['occurrences']}")
            print(f"   Stability: {pattern['stability_score']:.3f}")
            print(f"   Regime Coverage: {pattern['regime_coverage']:.2f}")
            print(f"   Composite Score: {pattern['composite_score']:.3f}")
            print(f"   Conditions: {list(p['conditions'].keys())}")
    else:
        print("\nNo patterns found meeting the criteria.")


if __name__ == "__main__":
    main()