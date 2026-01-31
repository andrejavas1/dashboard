"""
False Positive Reduction Validation Module

This module validates false positive reduction techniques implemented in the
enhanced rule-based pattern discovery system. It compares patterns with and
without false positive reduction to measure effectiveness while ensuring
pattern diversity is maintained.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
from pathlib import Path
import copy


class FalsePositiveReductionValidator:
    """
    Validates false positive reduction techniques for pattern discovery.
    """
    
    def __init__(self, features_df: pd.DataFrame, output_dir: str = "data"):
        """
        Initialize the validator.
        
        Args:
            features_df: DataFrame with technical features
            output_dir: Directory for output files
        """
        self.features_df = features_df.copy()
        self.features_df.index = pd.to_datetime(self.features_df.index)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Get numeric features
        exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        self.numeric_features = [
            col for col in self.features_df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(self.features_df[col])
        ]
        
        # Feature categories for diversity
        self.feature_categories = self._categorize_features()
        
        # Storage for results
        self.baseline_patterns = []
        self.enhanced_patterns = []
        self.comparison_results = {}
        
        print(f"Initialized with {len(self.numeric_features)} numeric features")
        print(f"Feature categories: {list(self.feature_categories.keys())}")
    
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
    
    def _get_pattern_categories(self, pattern: Dict) -> Set[str]:
        """
        Get the feature categories used by a pattern.
        
        Args:
            pattern: Pattern dictionary
            
        Returns:
            Set of feature categories
        """
        categories = set()
        conditions = pattern.get('pattern', {}).get('conditions', {})
        
        for feature in conditions.keys():
            for cat, features in self.feature_categories.items():
                if feature in features:
                    categories.add(cat)
        
        return categories
    
    def generate_baseline_patterns(self, n_patterns: int = 50) -> List[Dict]:
        """
        Generate baseline patterns WITHOUT false positive reduction.
        
        Args:
            n_patterns: Number of patterns to generate
            
        Returns:
            List of baseline patterns
        """
        print(f"\nGenerating {n_patterns} baseline patterns (without false positive reduction)...")
        
        baseline_patterns = []
        attempts = 0
        max_attempts = n_patterns * 20
        
        # Relaxed thresholds for baseline (no false positive reduction)
        min_occurrences = 10
        min_success_rate = 60.0
        
        while len(baseline_patterns) < n_patterns and attempts < max_attempts:
            attempts += 1
            
            # Generate simple pattern candidate
            pattern = self._generate_simple_pattern()
            
            if pattern is None:
                continue
            
            # Evaluate without strict false positive checks
            evaluation = self._evaluate_pattern_simple(
                pattern['conditions'],
                pattern['label_col'],
                pattern['direction'],
                min_occurrences=min_occurrences,
                min_success_rate=min_success_rate
            )
            
            if not evaluation['valid']:
                continue
            
            # Create pattern record
            pattern_record = {
                'pattern': {
                    'conditions': pattern['conditions'],
                    'direction': pattern['direction'],
                    'label_col': pattern['label_col'],
                    'occurrences': evaluation['occurrences'],
                    'success_rate': evaluation['success_rate'] / 100,
                    'avg_move': evaluation.get('avg_move', 0),
                    'fitness': evaluation.get('composite_score', 0)
                },
                'training_success_rate': evaluation['success_rate'],
                'false_positive_rate': evaluation['false_positive_rate'],
                'occurrences': evaluation['occurrences'],
                'p_value': evaluation.get('p_value', 1.0),
                'stability_score': evaluation.get('stability_score', 0.5),
                'regime_coverage': evaluation.get('regime_coverage', 1.0),
                'composite_score': evaluation.get('composite_score', 0),
                'classification': 'BASELINE'
            }
            
            baseline_patterns.append(pattern_record)
            
            if len(baseline_patterns) % 10 == 0:
                print(f"  Generated {len(baseline_patterns)} baseline patterns...")
        
        print(f"Baseline generation complete: {len(baseline_patterns)} patterns")
        self.baseline_patterns = baseline_patterns
        return baseline_patterns
    
    def generate_enhanced_patterns(self, n_patterns: int = 50) -> List[Dict]:
        """
        Generate enhanced patterns WITH false positive reduction.
        
        Args:
            n_patterns: Number of patterns to generate
            
        Returns:
            List of enhanced patterns
        """
        print(f"\nGenerating {n_patterns} enhanced patterns (with false positive reduction)...")
        
        enhanced_patterns = []
        attempts = 0
        max_attempts = n_patterns * 20
        
        # Strict thresholds for enhanced (with false positive reduction)
        min_occurrences = 20
        min_success_rate = 70.0
        max_false_positive_rate = 15.0
        min_statistical_significance = 0.05
        min_regime_coverage = 0.5
        
        while len(enhanced_patterns) < n_patterns and attempts < max_attempts:
            attempts += 1
            
            # Generate pattern candidate
            pattern = self._generate_simple_pattern()
            
            if pattern is None:
                continue
            
            # Evaluate with strict false positive checks
            evaluation = self._evaluate_pattern_comprehensive(
                pattern['conditions'],
                pattern['label_col'],
                pattern['direction'],
                min_occurrences=min_occurrences,
                min_success_rate=min_success_rate,
                max_false_positive_rate=max_false_positive_rate,
                min_statistical_significance=min_statistical_significance,
                min_regime_coverage=min_regime_coverage
            )
            
            if not evaluation['valid']:
                continue
            
            # Create pattern record
            pattern_record = {
                'pattern': {
                    'conditions': pattern['conditions'],
                    'direction': pattern['direction'],
                    'label_col': pattern['label_col'],
                    'occurrences': evaluation['occurrences'],
                    'success_rate': evaluation['success_rate'] / 100,
                    'avg_move': evaluation.get('avg_move', 0),
                    'fitness': evaluation.get('composite_score', 0)
                },
                'training_success_rate': evaluation['success_rate'],
                'false_positive_rate': evaluation['false_positive_rate'],
                'occurrences': evaluation['occurrences'],
                'p_value': evaluation.get('p_value', 1.0),
                'stability_score': evaluation.get('stability_score', 0.5),
                'regime_coverage': evaluation.get('regime_coverage', 1.0),
                'composite_score': evaluation.get('composite_score', 0),
                'classification': 'ENHANCED'
            }
            
            enhanced_patterns.append(pattern_record)
            
            if len(enhanced_patterns) % 10 == 0:
                print(f"  Generated {len(enhanced_patterns)} enhanced patterns...")
        
        print(f"Enhanced generation complete: {len(enhanced_patterns)} patterns")
        self.enhanced_patterns = enhanced_patterns
        return enhanced_patterns
    
    def _generate_simple_pattern(self) -> Optional[Dict]:
        """Generate a simple pattern candidate."""
        import random
        from scipy import stats
        
        # Random number of conditions
        num_conditions = random.randint(2, min(5, len(self.numeric_features)))
        
        # Select random features
        selected_features = random.sample(self.numeric_features, num_conditions)
        
        # Generate conditions
        conditions = {}
        for feature in selected_features:
            values = self.features_df[feature].dropna()
            if len(values) == 0:
                continue
            
            # Use quantiles for thresholds
            threshold = values.quantile(random.choice([0.1, 0.25, 0.5, 0.75, 0.9]))
            operator = random.choice(['>=', '<='])
            
            conditions[feature] = {
                'operator': operator,
                'value': float(threshold)
            }
        
        if len(conditions) < 2:
            return None
        
        # Determine direction
        direction = random.choice(['long', 'long', 'long', 'short'])
        
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
    
    def _evaluate_pattern_simple(self, conditions: Dict, label_col: str, direction: str,
                                  min_occurrences: int = 10, min_success_rate: float = 60.0) -> Dict:
        """
        Evaluate pattern with simple criteria (no false positive reduction).
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
        
        # Get occurrences
        occurrences = self.features_df[mask]
        
        if len(occurrences) < min_occurrences:
            return {'valid': False, 'reason': 'Insufficient occurrences'}
        
        # Calculate success rate
        if direction == 'long':
            success_mask = occurrences[label_col] == 'STRONG_UP'
            opposite_mask = occurrences[label_col] == 'STRONG_DOWN'
        else:
            success_mask = occurrences[label_col] == 'STRONG_DOWN'
            opposite_mask = occurrences[label_col] == 'STRONG_UP'
        
        success_count = success_mask.sum()
        total_count = len(occurrences)
        success_rate = success_count / total_count * 100
        
        if success_rate < min_success_rate:
            return {'valid': False, 'reason': 'Success rate below threshold'}
        
        # Calculate false positive rate
        false_positive_count = opposite_mask.sum()
        false_positive_rate = false_positive_count / total_count * 100
        
        # Calculate stability
        stability_score = self._calculate_stability(occurrences, label_col, direction)
        
        # Calculate regime coverage
        regime_coverage = self._calculate_regime_coverage(occurrences)
        
        # Simple composite score
        composite_score = (success_rate / 100.0) * 0.6 + stability_score * 0.4
        
        return {
            'valid': True,
            'occurrences': total_count,
            'success_rate': success_rate,
            'false_positive_rate': false_positive_rate,
            'stability_score': stability_score,
            'regime_coverage': regime_coverage,
            'composite_score': composite_score,
            'p_value': 1.0  # Not calculated in baseline
        }
    
    def _evaluate_pattern_comprehensive(self, conditions: Dict, label_col: str, direction: str,
                                         min_occurrences: int = 20, min_success_rate: float = 70.0,
                                         max_false_positive_rate: float = 15.0,
                                         min_statistical_significance: float = 0.05,
                                         min_regime_coverage: float = 0.5) -> Dict:
        """
        Evaluate pattern with comprehensive criteria (with false positive reduction).
        """
        from scipy import stats
        
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
        
        # Get occurrences
        occurrences = self.features_df[mask]
        
        if len(occurrences) < min_occurrences:
            return {'valid': False, 'reason': 'Insufficient occurrences'}
        
        # Calculate success rate
        if direction == 'long':
            success_mask = occurrences[label_col] == 'STRONG_UP'
            opposite_mask = occurrences[label_col] == 'STRONG_DOWN'
        else:
            success_mask = occurrences[label_col] == 'STRONG_DOWN'
            opposite_mask = occurrences[label_col] == 'STRONG_UP'
        
        success_count = success_mask.sum()
        total_count = len(occurrences)
        success_rate = success_count / total_count * 100
        
        # Early stopping check
        if success_rate < 60.0:
            return {'valid': False, 'reason': 'Success rate below early stopping threshold'}
        
        if success_rate < min_success_rate:
            return {'valid': False, 'reason': 'Success rate below threshold'}
        
        # Calculate false positive rate
        false_positive_count = opposite_mask.sum()
        false_positive_rate = false_positive_count / total_count * 100
        
        # False positive rate check
        if false_positive_rate > max_false_positive_rate:
            return {'valid': False, 'reason': 'False positive rate above threshold'}
        
        # Statistical significance test
        p_value = stats.binomtest(success_count, total_count, p=0.5, alternative='greater').pvalue
        
        if p_value > min_statistical_significance:
            return {'valid': False, 'reason': 'Not statistically significant'}
        
        # Calculate stability
        stability_score = self._calculate_stability(occurrences, label_col, direction)
        
        # Calculate regime coverage
        regime_coverage = self._calculate_regime_coverage(occurrences)
        
        # Regime coverage check
        if regime_coverage < min_regime_coverage:
            return {'valid': False, 'reason': 'Insufficient regime coverage'}
        
        # Extract target parameters
        parts = label_col.split('_')
        target_pct = float(parts[1].replace('pct', ''))
        target_days = int(parts[2].replace('d', ''))
        
        # Calculate average move
        if success_count > 0:
            if direction == 'long':
                max_move_col = f'Max_Up_{target_days}d'
            else:
                max_move_col = f'Max_Down_{target_days}d'
            
            if max_move_col in occurrences.columns:
                avg_move = occurrences.loc[success_mask, max_move_col].mean()
            else:
                avg_move = target_pct
        else:
            avg_move = 0
        
        # Composite score with false positive penalty
        success_score = success_rate / 100.0
        frequency_score = min(total_count / 100.0, 1.0)
        false_positive_penalty = max(0, (false_positive_rate - max_false_positive_rate) / 100.0)
        
        composite_score = (
            0.4 * success_score +
            0.2 * frequency_score +
            0.2 * stability_score +
            0.2 * regime_coverage -
            false_positive_penalty * 0.5
        )
        
        return {
            'valid': True,
            'occurrences': total_count,
            'success_rate': success_rate,
            'false_positive_rate': false_positive_rate,
            'p_value': p_value,
            'stability_score': stability_score,
            'regime_coverage': regime_coverage,
            'avg_move': avg_move,
            'composite_score': max(0, composite_score)
        }
    
    def _calculate_stability(self, occurrences: pd.DataFrame, label_col: str, direction: str) -> float:
        """Calculate pattern stability across time periods."""
        if len(occurrences) == 0:
            return 0.0
        
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
            return 0.5
        
        mean_rate = np.mean(yearly_stats)
        std_rate = np.std(yearly_stats)
        
        if mean_rate == 0:
            return 0.0
        
        cv = std_rate / mean_rate if mean_rate > 0 else float('inf')
        stability = max(0, 1 - cv)
        
        return stability
    
    def _calculate_regime_coverage(self, occurrences: pd.DataFrame) -> float:
        """Calculate regime coverage."""
        if len(occurrences) == 0:
            return 0.0
        
        regimes = set()
        
        if 'Vol_Regime' in occurrences.columns:
            regimes.update(occurrences['Vol_Regime'].unique())
        
        if 'Trend_Regime' in occurrences.columns:
            regimes.update(occurrences['Trend_Regime'].unique())
        
        if not regimes:
            return 1.0
        
        expected_regimes = 0
        if 'Vol_Regime' in self.features_df.columns:
            expected_regimes += len(self.features_df['Vol_Regime'].unique())
        if 'Trend_Regime' in self.features_df.columns:
            expected_regimes += len(self.features_df['Trend_Regime'].unique())
        
        if expected_regimes == 0:
            return 1.0
        
        return len(regimes) / expected_regimes
    
    def compare_patterns(self) -> Dict:
        """
        Compare baseline vs enhanced patterns.
        
        Returns:
            Dictionary with comparison results
        """
        print("\nComparing baseline vs enhanced patterns...")
        
        if not self.baseline_patterns or not self.enhanced_patterns:
            print("Warning: Missing pattern data for comparison")
            return {}
        
        # Calculate metrics for baseline
        baseline_metrics = self._calculate_metrics(self.baseline_patterns, 'BASELINE')
        
        # Calculate metrics for enhanced
        enhanced_metrics = self._calculate_metrics(self.enhanced_patterns, 'ENHANCED')
        
        # Calculate diversity metrics
        baseline_diversity = self._calculate_diversity_metrics(self.baseline_patterns)
        enhanced_diversity = self._calculate_diversity_metrics(self.enhanced_patterns)
        
        # Calculate improvements
        improvements = {}
        for key in baseline_metrics:
            if key in enhanced_metrics:
                baseline_val = baseline_metrics[key]
                enhanced_val = enhanced_metrics[key]
                
                # Skip dictionary values (like category_distribution)
                if isinstance(baseline_val, dict) or isinstance(enhanced_val, dict):
                    continue
                
                if baseline_val != 0:
                    pct_change = ((enhanced_val - baseline_val) / baseline_val) * 100
                else:
                    pct_change = 0 if enhanced_val == 0 else float('inf')
                
                improvements[key] = {
                    'baseline': baseline_val,
                    'enhanced': enhanced_val,
                    'absolute_change': enhanced_val - baseline_val,
                    'percentage_change': pct_change
                }
        
        # Calculate diversity improvements
        diversity_improvements = {}
        for key in baseline_diversity:
            if key in enhanced_diversity:
                baseline_val = baseline_diversity[key]
                enhanced_val = enhanced_diversity[key]
                
                # Skip dictionary values (like category_distribution)
                if isinstance(baseline_val, dict) or isinstance(enhanced_val, dict):
                    continue
                
                if baseline_val != 0:
                    pct_change = ((enhanced_val - baseline_val) / baseline_val) * 100
                else:
                    pct_change = 0 if enhanced_val == 0 else float('inf')
                
                diversity_improvements[key] = {
                    'baseline': baseline_val,
                    'enhanced': enhanced_val,
                    'absolute_change': enhanced_val - baseline_val,
                    'percentage_change': pct_change
                }
        
        self.comparison_results = {
            'baseline_metrics': baseline_metrics,
            'enhanced_metrics': enhanced_metrics,
            'diversity_metrics': {
                'baseline': baseline_diversity,
                'enhanced': enhanced_diversity
            },
            'improvements': improvements,
            'diversity_improvements': diversity_improvements,
            'summary': self._generate_summary(improvements, diversity_improvements)
        }
        
        print("Comparison complete!")
        print(f"  False positive rate reduction: {improvements.get('false_positive_rate', {}).get('percentage_change', 0):.1f}%")
        print(f"  Success rate change: {improvements.get('success_rate', {}).get('percentage_change', 0):.1f}%")
        print(f"  Pattern count change: {improvements.get('pattern_count', {}).get('percentage_change', 0):.1f}%")
        
        return self.comparison_results
    
    def _calculate_metrics(self, patterns: List[Dict], label: str) -> Dict:
        """Calculate metrics for a set of patterns."""
        if not patterns:
            return {}
        
        metrics = {
            'pattern_count': len(patterns),
            'success_rate': np.mean([p['training_success_rate'] for p in patterns]),
            'false_positive_rate': np.mean([p['false_positive_rate'] for p in patterns]),
            'occurrences': np.mean([p['occurrences'] for p in patterns]),
            'stability_score': np.mean([p['stability_score'] for p in patterns]),
            'regime_coverage': np.mean([p['regime_coverage'] for p in patterns]),
            'composite_score': np.mean([p['composite_score'] for p in patterns]),
            'p_value': np.mean([p['p_value'] for p in patterns])
        }
        
        # Calculate percentage of patterns meeting criteria
        metrics['pct_success_rate_above_80'] = np.mean([p['training_success_rate'] >= 80 for p in patterns]) * 100
        metrics['pct_fpr_below_15'] = np.mean([p['false_positive_rate'] <= 15 for p in patterns]) * 100
        metrics['pct_fpr_below_10'] = np.mean([p['false_positive_rate'] <= 10 for p in patterns]) * 100
        metrics['pct_statistically_significant'] = np.mean([p['p_value'] < 0.05 for p in patterns]) * 100
        
        return metrics
    
    def _calculate_diversity_metrics(self, patterns: List[Dict]) -> Dict:
        """Calculate diversity metrics for a set of patterns."""
        if not patterns:
            return {}
        
        # Get feature categories for each pattern
        pattern_categories = [self._get_pattern_categories(p) for p in patterns]
        
        # Count unique categories used
        all_categories = set()
        for cats in pattern_categories:
            all_categories.update(cats)
        
        # Calculate diversity metrics
        avg_categories_per_pattern = np.mean([len(cats) for cats in pattern_categories])
        unique_categories_used = len(all_categories)
        
        # Category distribution
        category_counts = {}
        for cats in pattern_categories:
            for cat in cats:
                category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Calculate diversity index (Shannon entropy)
        total_patterns = len(patterns)
        if total_patterns > 0:
            entropy = 0
            for count in category_counts.values():
                p = count / total_patterns
                if p > 0:
                    entropy -= p * np.log2(p)
            diversity_index = entropy / np.log2(len(category_counts)) if len(category_counts) > 1 else 0
        else:
            diversity_index = 0
        
        # Target label diversity
        target_labels = [p['pattern']['label_col'] for p in patterns]
        unique_targets = len(set(target_labels))
        
        # Direction diversity
        directions = [p['pattern']['direction'] for p in patterns]
        long_pct = np.mean([d == 'long' for d in directions]) * 100
        
        return {
            'avg_categories_per_pattern': avg_categories_per_pattern,
            'unique_categories_used': unique_categories_used,
            'diversity_index': diversity_index,
            'unique_target_labels': unique_targets,
            'long_pattern_percentage': long_pct,
            'category_distribution': category_counts
        }
    
    def _generate_summary(self, improvements: Dict, diversity_improvements: Dict) -> Dict:
        """Generate summary of improvements."""
        summary = {
            'false_positive_reduction_effective': False,
            'diversity_maintained': False,
            'overall_improvement': False
        }
        
        # Check if false positive rate was reduced
        fpr_change = improvements.get('false_positive_rate', {}).get('percentage_change', 0)
        if fpr_change < 0:
            summary['false_positive_reduction_effective'] = True
        
        # Check if diversity was maintained (not significantly reduced)
        diversity_change = diversity_improvements.get('diversity_index', {}).get('percentage_change', 0)
        if diversity_change >= -10:  # Allow up to 10% reduction
            summary['diversity_maintained'] = True
        
        # Check overall improvement
        success_rate_change = improvements.get('success_rate', {}).get('percentage_change', 0)
        if fpr_change < 0 and success_rate_change >= 0:
            summary['overall_improvement'] = True
        
        return summary
    
    def generate_visualizations(self) -> List[str]:
        """
        Generate comparison visualizations.
        
        Returns:
            List of generated file paths
        """
        print("\nGenerating visualizations...")
        
        if not self.baseline_patterns or not self.enhanced_patterns:
            print("Warning: Missing pattern data for visualizations")
            return []
        
        file_paths = []
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle('False Positive Reduction Validation: Baseline vs Enhanced', fontsize=16, fontweight='bold')
        
        # Colors
        baseline_color = '#ff6b6b'
        enhanced_color = '#4ecdc4'
        
        # 1. False Positive Rate Distribution
        ax1 = axes[0, 0]
        baseline_fpr = [p['false_positive_rate'] for p in self.baseline_patterns]
        enhanced_fpr = [p['false_positive_rate'] for p in self.enhanced_patterns]
        ax1.hist([baseline_fpr, enhanced_fpr], bins=20, alpha=0.7, 
                 label=['Baseline', 'Enhanced'], color=[baseline_color, enhanced_color])
        ax1.axvline(x=15, color='red', linestyle='--', label='15% Threshold')
        ax1.set_xlabel('False Positive Rate (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('False Positive Rate Distribution')
        ax1.legend()
        
        # 2. Success Rate Distribution
        ax2 = axes[0, 1]
        baseline_sr = [p['training_success_rate'] for p in self.baseline_patterns]
        enhanced_sr = [p['training_success_rate'] for p in self.enhanced_patterns]
        ax2.hist([baseline_sr, enhanced_sr], bins=20, alpha=0.7,
                 label=['Baseline', 'Enhanced'], color=[baseline_color, enhanced_color])
        ax2.axvline(x=80, color='green', linestyle='--', label='80% Threshold')
        ax2.set_xlabel('Success Rate (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Success Rate Distribution')
        ax2.legend()
        
        # 3. Composite Score Distribution
        ax3 = axes[0, 2]
        baseline_cs = [p['composite_score'] for p in self.baseline_patterns]
        enhanced_cs = [p['composite_score'] for p in self.enhanced_patterns]
        ax3.hist([baseline_cs, enhanced_cs], bins=20, alpha=0.7,
                 label=['Baseline', 'Enhanced'], color=[baseline_color, enhanced_color])
        ax3.set_xlabel('Composite Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Composite Score Distribution')
        ax3.legend()
        
        # 4. Box Plot Comparison
        ax4 = axes[1, 0]
        bp = ax4.boxplot([baseline_fpr, enhanced_fpr, baseline_sr, enhanced_sr,
                          [p['stability_score'] for p in self.baseline_patterns],
                          [p['stability_score'] for p in self.enhanced_patterns]],
                          tick_labels=['Base\nFPR', 'Enh\nFPR', 'Base\nSR', 'Enh\nSR',
                                  'Base\nStab', 'Enh\nStab'],
                          patch_artist=True)
        # Color the boxes
        colors = [baseline_color, enhanced_color, baseline_color, enhanced_color,
                  baseline_color, enhanced_color]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax4.set_ylabel('Value')
        ax4.set_title('Box Plot Comparison')
        ax4.grid(True, alpha=0.3)
        
        # 5. Success Rate vs False Positive Rate
        ax5 = axes[1, 1]
        ax5.scatter(baseline_fpr, baseline_sr, alpha=0.6, c=baseline_color, label='Baseline', s=50)
        ax5.scatter(enhanced_fpr, enhanced_sr, alpha=0.6, c=enhanced_color, label='Enhanced', s=50)
        ax5.axhline(y=80, color='green', linestyle='--', alpha=0.5)
        ax5.axvline(x=15, color='red', linestyle='--', alpha=0.5)
        ax5.set_xlabel('False Positive Rate (%)')
        ax5.set_ylabel('Success Rate (%)')
        ax5.set_title('Success Rate vs False Positive Rate')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Stability vs Composite Score
        ax6 = axes[1, 2]
        baseline_stab = [p['stability_score'] for p in self.baseline_patterns]
        enhanced_stab = [p['stability_score'] for p in self.enhanced_patterns]
        ax6.scatter(baseline_stab, baseline_cs, alpha=0.6, c=baseline_color, label='Baseline', s=50)
        ax6.scatter(enhanced_stab, enhanced_cs, alpha=0.6, c=enhanced_color, label='Enhanced', s=50)
        ax6.set_xlabel('Stability Score')
        ax6.set_ylabel('Composite Score')
        ax6.set_title('Stability vs Composite Score')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Before/After Metrics Comparison
        ax7 = axes[2, 0]
        if 'improvements' in self.comparison_results:
            metrics = ['success_rate', 'false_positive_rate', 'stability_score', 'composite_score']
            baseline_vals = [self.comparison_results['improvements'].get(m, {}).get('baseline', 0) for m in metrics]
            enhanced_vals = [self.comparison_results['improvements'].get(m, {}).get('enhanced', 0) for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = ax7.bar(x - width/2, baseline_vals, width, label='Baseline', color=baseline_color, alpha=0.8)
            bars2 = ax7.bar(x + width/2, enhanced_vals, width, label='Enhanced', color=enhanced_color, alpha=0.8)
            
            ax7.set_ylabel('Value')
            ax7.set_title('Before/After Metrics Comparison')
            ax7.set_xticks(x)
            ax7.set_xticklabels(['Success\nRate', 'FPR', 'Stability', 'Composite'])
            ax7.legend()
            ax7.grid(True, alpha=0.3, axis='y')
        
        # 8. Category Diversity Comparison
        ax8 = axes[2, 1]
        if 'diversity_improvements' in self.comparison_results:
            baseline_div = self.comparison_results['diversity_metrics']['baseline']
            enhanced_div = self.comparison_results['diversity_metrics']['enhanced']
            
            diversity_metrics = ['avg_categories_per_pattern', 'diversity_index', 'unique_target_labels']
            baseline_div_vals = [baseline_div.get(m, 0) for m in diversity_metrics]
            enhanced_div_vals = [enhanced_div.get(m, 0) for m in diversity_metrics]
            
            x = np.arange(len(diversity_metrics))
            width = 0.35
            
            bars1 = ax8.bar(x - width/2, baseline_div_vals, width, label='Baseline', color=baseline_color, alpha=0.8)
            bars2 = ax8.bar(x + width/2, enhanced_div_vals, width, label='Enhanced', color=enhanced_color, alpha=0.8)
            
            ax8.set_ylabel('Value')
            ax8.set_title('Pattern Diversity Comparison')
            ax8.set_xticks(x)
            ax8.set_xticklabels(['Avg\nCategories', 'Diversity\nIndex', 'Unique\nTargets'])
            ax8.legend()
            ax8.grid(True, alpha=0.3, axis='y')
        
        # 9. Improvement Summary
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        if 'improvements' in self.comparison_results:
            summary_text = "FALSE POSITIVE REDUCTION SUMMARY\n\n"
            
            fpr_change = self.comparison_results['improvements'].get('false_positive_rate', {}).get('percentage_change', 0)
            sr_change = self.comparison_results['improvements'].get('success_rate', {}).get('percentage_change', 0)
            stab_change = self.comparison_results['improvements'].get('stability_score', {}).get('percentage_change', 0)
            comp_change = self.comparison_results['improvements'].get('composite_score', {}).get('percentage_change', 0)
            
            summary_text += f"False Positive Rate: {fpr_change:+.1f}%\n"
            summary_text += f"Success Rate: {sr_change:+.1f}%\n"
            summary_text += f"Stability: {stab_change:+.1f}%\n"
            summary_text += f"Composite Score: {comp_change:+.1f}%\n\n"
            
            summary_text += "DIVERSITY METRICS:\n"
            div_change = self.comparison_results['diversity_improvements'].get('diversity_index', {}).get('percentage_change', 0)
            summary_text += f"Diversity Index: {div_change:+.1f}%\n\n"
            
            if 'summary' in self.comparison_results:
                summary_text += "VALIDATION RESULTS:\n"
                summary_text += f"✓ FPR Reduction Effective: {self.comparison_results['summary']['false_positive_reduction_effective']}\n"
                summary_text += f"✓ Diversity Maintained: {self.comparison_results['summary']['diversity_maintained']}\n"
                summary_text += f"✓ Overall Improvement: {self.comparison_results['summary']['overall_improvement']}\n"
            
            ax9.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / 'false_positive_reduction_validation.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        file_paths.append(str(output_path))
        print(f"  Saved: {output_path}")
        
        return file_paths
    
    def generate_report(self) -> str:
        """
        Generate comprehensive validation report.
        
        Returns:
            Path to generated report
        """
        print("\nGenerating validation report...")
        
        output_path = self.output_dir / 'false_positive_reduction_report.md'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Title
            f.write("# False Positive Reduction Validation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            if 'summary' in self.comparison_results:
                summary = self.comparison_results['summary']
                f.write(f"- **False Positive Reduction Effective:** {'✓ Yes' if summary['false_positive_reduction_effective'] else '✗ No'}\n")
                f.write(f"- **Diversity Maintained:** {'✓ Yes' if summary['diversity_maintained'] else '✗ No'}\n")
                f.write(f"- **Overall Improvement:** {'✓ Yes' if summary['overall_improvement'] else '✗ No'}\n\n")
            
            # Test Configuration
            f.write("## Test Configuration\n\n")
            f.write("### Baseline Patterns (Without False Positive Reduction)\n")
            f.write("- Minimum Occurrences: 10\n")
            f.write("- Minimum Success Rate: 60%\n")
            f.write("- No False Positive Rate Limit\n")
            f.write("- No Statistical Significance Test\n")
            f.write("- No Regime Coverage Requirement\n\n")
            
            f.write("### Enhanced Patterns (With False Positive Reduction)\n")
            f.write("- Minimum Occurrences: 20\n")
            f.write("- Minimum Success Rate: 70%\n")
            f.write("- Maximum False Positive Rate: 15%\n")
            f.write("- Statistical Significance: p < 0.05 (binomial test)\n")
            f.write("- Minimum Regime Coverage: 50%\n")
            f.write("- Early Stopping: Success rate < 60%\n\n")
            
            # Results Overview
            f.write("## Results Overview\n\n")
            f.write(f"- **Baseline Patterns Generated:** {len(self.baseline_patterns)}\n")
            f.write(f"- **Enhanced Patterns Generated:** {len(self.enhanced_patterns)}\n\n")
            
            # Performance Metrics Comparison
            f.write("## Performance Metrics Comparison\n\n")
            f.write("| Metric | Baseline | Enhanced | Change | % Change |\n")
            f.write("|--------|----------|----------|--------|----------|\n")
            
            if 'improvements' in self.comparison_results:
                metric_names = {
                    'pattern_count': 'Pattern Count',
                    'success_rate': 'Success Rate (%)',
                    'false_positive_rate': 'False Positive Rate (%)',
                    'occurrences': 'Avg Occurrences',
                    'stability_score': 'Stability Score',
                    'regime_coverage': 'Regime Coverage',
                    'composite_score': 'Composite Score',
                    'p_value': 'Avg P-Value',
                    'pct_success_rate_above_80': '% Patterns > 80% Success',
                    'pct_fpr_below_15': '% Patterns < 15% FPR',
                    'pct_fpr_below_10': '% Patterns < 10% FPR',
                    'pct_statistically_significant': '% Statistically Significant'
                }
                
                for key, name in metric_names.items():
                    if key in self.comparison_results['improvements']:
                        imp = self.comparison_results['improvements'][key]
                        baseline = imp['baseline']
                        enhanced = imp['enhanced']
                        abs_change = imp['absolute_change']
                        pct_change = imp['percentage_change']
                        
                        if isinstance(baseline, float):
                            f.write(f"| {name} | {baseline:.3f} | {enhanced:.3f} | {abs_change:+.3f} | {pct_change:+.1f}% |\n")
                        else:
                            f.write(f"| {name} | {baseline:.1f} | {enhanced:.1f} | {abs_change:+.1f} | {pct_change:+.1f}% |\n")
            
            f.write("\n")
            
            # Diversity Metrics Comparison
            f.write("## Pattern Diversity Comparison\n\n")
            f.write("| Metric | Baseline | Enhanced | Change | % Change |\n")
            f.write("|--------|----------|----------|--------|----------|\n")
            
            if 'diversity_improvements' in self.comparison_results:
                metric_names = {
                    'avg_categories_per_pattern': 'Avg Categories/Pattern',
                    'unique_categories_used': 'Unique Categories Used',
                    'diversity_index': 'Diversity Index (Shannon)',
                    'unique_target_labels': 'Unique Target Labels',
                    'long_pattern_percentage': 'Long Pattern %'
                }
                
                for key, name in metric_names.items():
                    if key in self.comparison_results['diversity_improvements']:
                        imp = self.comparison_results['diversity_improvements'][key]
                        baseline = imp['baseline']
                        enhanced = imp['enhanced']
                        abs_change = imp['absolute_change']
                        pct_change = imp['percentage_change']
                        
                        if isinstance(baseline, float):
                            f.write(f"| {name} | {baseline:.3f} | {enhanced:.3f} | {abs_change:+.3f} | {pct_change:+.1f}% |\n")
                        else:
                            f.write(f"| {name} | {baseline:.1f} | {enhanced:.1f} | {abs_change:+.1f} | {pct_change:+.1f}% |\n")
            
            f.write("\n")
            
            # Category Distribution
            f.write("## Feature Category Distribution\n\n")
            if 'diversity_metrics' in self.comparison_results:
                baseline_dist = self.comparison_results['diversity_metrics']['baseline'].get('category_distribution', {})
                enhanced_dist = self.comparison_results['diversity_metrics']['enhanced'].get('category_distribution', {})
                
                all_categories = sorted(set(list(baseline_dist.keys()) + list(enhanced_dist.keys())))
                
                f.write("| Category | Baseline Count | Enhanced Count | Change |\n")
                f.write("|----------|----------------|----------------|--------|\n")
                
                for cat in all_categories:
                    base = baseline_dist.get(cat, 0)
                    enh = enhanced_dist.get(cat, 0)
                    change = enh - base
                    f.write(f"| {cat} | {base} | {enh} | {change:+d} |\n")
            
            f.write("\n")
            
            # Detailed Analysis
            f.write("## Detailed Analysis\n\n")
            
            # False Positive Reduction Analysis
            f.write("### False Positive Reduction Analysis\n\n")
            if 'improvements' in self.comparison_results:
                fpr_imp = self.comparison_results['improvements'].get('false_positive_rate', {})
                f.write(f"- **Baseline FPR:** {fpr_imp.get('baseline', 0):.2f}%\n")
                f.write(f"- **Enhanced FPR:** {fpr_imp.get('enhanced', 0):.2f}%\n")
                f.write(f"- **Reduction:** {fpr_imp.get('absolute_change', 0):.2f} percentage points ({fpr_imp.get('percentage_change', 0):.1f}%)\n\n")
                
                if fpr_imp.get('percentage_change', 0) < 0:
                    f.write("✓ **False positive reduction is effective.** The enhanced patterns show a significant reduction in false positive rates.\n\n")
                else:
                    f.write("✗ **False positive reduction is not effective.** The enhanced patterns do not show a reduction in false positive rates.\n\n")
            
            # Success Rate Analysis
            f.write("### Success Rate Analysis\n\n")
            if 'improvements' in self.comparison_results:
                sr_imp = self.comparison_results['improvements'].get('success_rate', {})
                f.write(f"- **Baseline Success Rate:** {sr_imp.get('baseline', 0):.2f}%\n")
                f.write(f"- **Enhanced Success Rate:** {sr_imp.get('enhanced', 0):.2f}%\n")
                f.write(f"- **Change:** {sr_imp.get('absolute_change', 0):.2f} percentage points ({sr_imp.get('percentage_change', 0):.1f}%)\n\n")
            
            # Pattern Quality Analysis
            f.write("### Pattern Quality Analysis\n\n")
            if 'improvements' in self.comparison_results:
                comp_imp = self.comparison_results['improvements'].get('composite_score', {})
                f.write(f"- **Baseline Composite Score:** {comp_imp.get('baseline', 0):.3f}\n")
                f.write(f"- **Enhanced Composite Score:** {comp_imp.get('enhanced', 0):.3f}\n")
                f.write(f"- **Change:** {comp_imp.get('absolute_change', 0):.3f} ({comp_imp.get('percentage_change', 0):.1f}%)\n\n")
            
            # Diversity Analysis
            f.write("### Diversity Analysis\n\n")
            if 'diversity_improvements' in self.comparison_results:
                div_imp = self.comparison_results['diversity_improvements'].get('diversity_index', {})
                f.write(f"- **Baseline Diversity Index:** {div_imp.get('baseline', 0):.3f}\n")
                f.write(f"- **Enhanced Diversity Index:** {div_imp.get('enhanced', 0):.3f}\n")
                f.write(f"- **Change:** {div_imp.get('absolute_change', 0):.3f} ({div_imp.get('percentage_change', 0):.1f}%)\n\n")
                
                if div_imp.get('percentage_change', 0) >= -10:
                    f.write("✓ **Pattern diversity is maintained.** The enhanced patterns maintain acceptable diversity levels.\n\n")
                else:
                    f.write("✗ **Pattern diversity is reduced.** The enhanced patterns show significant reduction in diversity.\n\n")
            
            # Statistical Significance Analysis
            f.write("### Statistical Significance Analysis\n\n")
            if 'improvements' in self.comparison_results:
                sig_imp = self.comparison_results['improvements'].get('pct_statistically_significant', {})
                f.write(f"- **Baseline Statistically Significant:** {sig_imp.get('baseline', 0):.1f}%\n")
                f.write(f"- **Enhanced Statistically Significant:** {sig_imp.get('enhanced', 0):.1f}%\n")
                f.write(f"- **Change:** {sig_imp.get('absolute_change', 0):.1f} percentage points ({sig_imp.get('percentage_change', 0):.1f}%)\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            if 'summary' in self.comparison_results:
                summary = self.comparison_results['summary']
                
                if summary['false_positive_reduction_effective']:
                    f.write("1. **Continue using false positive reduction techniques** - The techniques are effective at reducing false positives.\n\n")
                else:
                    f.write("1. **Review false positive reduction thresholds** - Current thresholds may be too strict or need adjustment.\n\n")
                
                if summary['diversity_maintained']:
                    f.write("2. **Pattern diversity is well-maintained** - The enhanced patterns maintain good diversity across feature categories.\n\n")
                else:
                    f.write("2. **Implement diversity preservation techniques** - Consider adjusting category balance requirements.\n\n")
                
                if summary['overall_improvement']:
                    f.write("3. **Deploy enhanced pattern discovery** - The enhanced approach shows overall improvement and should be used in production.\n\n")
                else:
                    f.write("3. **Further optimization needed** - The enhanced approach needs additional tuning before deployment.\n\n")
            
            # Conclusion
            f.write("## Conclusion\n\n")
            f.write("The false positive reduction validation compared pattern discovery with and without enhanced false positive reduction techniques. ")
            
            if 'summary' in self.comparison_results:
                if self.comparison_results['summary']['overall_improvement']:
                    f.write("The results demonstrate that the enhanced approach successfully reduces false positives while maintaining pattern diversity and overall quality. ")
                    f.write("The statistical significance testing, regime coverage filtering, and strict false positive rate thresholds contribute to more reliable patterns.\n\n")
                else:
                    f.write("The results indicate that further optimization is needed to achieve the desired balance between false positive reduction and pattern discovery effectiveness. ")
                    f.write("Consider adjusting thresholds or exploring additional reduction techniques.\n\n")
            
            f.write("---\n\n")
            f.write("*Report generated by FalsePositiveReductionValidator*\n")
        
        print(f"  Saved: {output_path}")
        return str(output_path)
    
    def run_validation(self, n_patterns: int = 50) -> Dict:
        """
        Run complete validation workflow.
        
        Args:
            n_patterns: Number of patterns to generate for each approach
            
        Returns:
            Complete validation results
        """
        print("=" * 60)
        print("False Positive Reduction Validation")
        print("=" * 60)
        
        # Generate baseline patterns
        self.generate_baseline_patterns(n_patterns)
        
        # Generate enhanced patterns
        self.generate_enhanced_patterns(n_patterns)
        
        # Compare patterns
        self.compare_patterns()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Generate report
        report_path = self.generate_report()
        
        print("\n" + "=" * 60)
        print("Validation Complete!")
        print("=" * 60)
        print(f"  Report: {report_path}")
        
        return self.comparison_results


def main():
    """Main function to run false positive reduction validation."""
    print("=" * 60)
    print("False Positive Reduction Validation")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    features_df = pd.read_csv('data/features_matrix.csv', index_col='Date', parse_dates=True)
    print(f"Loaded {len(features_df)} records with {len(features_df.columns)} columns")
    
    # Initialize validator
    print("\nInitializing validator...")
    validator = FalsePositiveReductionValidator(features_df)
    
    # Run validation
    results = validator.run_validation(n_patterns=50)
    
    # Display summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if 'summary' in results:
        summary = results['summary']
        print(f"\nFalse Positive Reduction Effective: {summary['false_positive_reduction_effective']}")
        print(f"Diversity Maintained: {summary['diversity_maintained']}")
        print(f"Overall Improvement: {summary['overall_improvement']}")
    
    if 'improvements' in results:
        print(f"\nKey Metrics:")
        fpr_change = results['improvements'].get('false_positive_rate', {}).get('percentage_change', 0)
        sr_change = results['improvements'].get('success_rate', {}).get('percentage_change', 0)
        print(f"  False Positive Rate Change: {fpr_change:+.1f}%")
        print(f"  Success Rate Change: {sr_change:+.1f}%")


if __name__ == "__main__":
    main()