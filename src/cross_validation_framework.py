"""
Cross-Validation Framework for Pattern Robustness Testing

This module implements comprehensive cross-validation procedures to ensure pattern
robustness across time periods and market conditions. It provides time-series
cross-validation, out-of-sample testing, and stability analysis.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from scipy import stats


@dataclass
class CrossValidationResult:
    """Data class for cross-validation results."""
    pattern_id: str
    pattern_name: str
    
    # In-sample metrics
    in_sample_success_rate: float
    in_sample_occurrences: int
    in_sample_false_positive_rate: float
    
    # Out-of-sample metrics (fold-wise)
    out_sample_success_rates: List[float] = field(default_factory=list)
    out_sample_occurrences: List[int] = field(default_factory=list)
    out_sample_false_positive_rates: List[float] = field(default_factory=list)
    
    # Aggregated out-of-sample metrics
    avg_out_sample_success_rate: float = 0.0
    avg_out_sample_occurrences: float = 0.0
    avg_out_sample_false_positive_rate: float = 0.0
    
    # Stability metrics
    success_rate_std: float = 0.0
    success_rate_cv: float = 0.0  # Coefficient of variation
    stability_score: float = 0.0
    
    # Performance degradation
    performance_degradation: float = 0.0  # In-sample vs out-of-sample difference
    
    # Consistency across folds
    consistent_folds: int = 0
    total_folds: int = 0
    consistency_rate: float = 0.0
    
    # Market condition performance
    market_condition_performance: Dict[str, float] = field(default_factory=dict)
    
    # Overall robustness score
    robustness_score: float = 0.0
    
    # Validation status
    is_robust: bool = False
    validation_notes: List[str] = field(default_factory=list)


@dataclass
class FoldResult:
    """Data class for individual fold results."""
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    
    success_rate: float
    occurrences: int
    false_positive_rate: float
    composite_score: float
    
    # Market conditions
    volatility_regime: Optional[str] = None
    trend_regime: Optional[str] = None


class TimeSeriesCrossValidator:
    """
    Time-series cross-validation for pattern robustness testing.
    
    Implements walk-forward analysis with expanding or rolling windows
    to test pattern performance on out-of-sample data.
    """
    
    def __init__(self, features_df: pd.DataFrame, config: Dict = None):
        """
        Initialize the cross-validator.
        
        Args:
            features_df: DataFrame with technical features
            config: Configuration parameters
        """
        self.features_df = features_df.copy()
        self.features_df.index = pd.to_datetime(self.features_df.index)
        self.config = config or self._default_config()
        
        # Get numeric features
        exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        self.numeric_features = [
            col for col in self.features_df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(self.features_df[col])
        ]
        
        # Get label columns
        self.label_columns = [col for col in self.features_df.columns if col.startswith('Label_')]
        
        # Storage for results
        self.cv_results = {}
        self.fold_results = {}
        
        print(f"Initialized with {len(self.features_df)} records")
        print(f"Label columns: {len(self.label_columns)}")
    
    def _default_config(self) -> Dict:
        """Default configuration parameters."""
        return {
            # Cross-validation parameters
            'n_folds': 5,
            'min_train_size': 0.4,  # Minimum training size (40% of data)
            'test_size': 0.2,       # Test size (20% of data)
            'expanding_window': True,  # Use expanding window vs rolling
            
            # Validation thresholds
            'min_success_rate': 70.0,
            'max_performance_degradation': 15.0,  # Max % drop from in-sample to out-of-sample
            'min_stability_score': 0.6,
            'min_consistency_rate': 0.7,  # Minimum % of folds meeting criteria
            
            # Robustness scoring weights
            'stability_weight': 0.3,
            'consistency_weight': 0.3,
            'performance_weight': 0.2,
            'degradation_weight': 0.2
        }
    
    def create_folds(self) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Create train/validation/test folds for cross-validation.
        
        Returns:
            List of (train_idx, val_idx, test_idx) tuples
        """
        n_folds = self.config['n_folds']
        min_train_size = self.config['min_train_size']
        test_size = self.config['test_size']
        expanding_window = self.config['expanding_window']
        
        n_samples = len(self.features_df)
        folds = []
        
        # Calculate indices
        min_train_samples = int(n_samples * min_train_size)
        test_samples = int(n_samples * test_size)
        
        for i in range(n_folds):
            # Calculate train start
            train_start = 0
            
            # Calculate train end (expanding or rolling)
            if expanding_window:
                train_end = min_train_samples + i * test_samples
            else:
                train_end = min_train_samples + n_folds * test_samples
            
            # Ensure we don't exceed data
            if train_end + test_samples > n_samples:
                break
            
            # Calculate test indices
            test_start = train_end
            test_end = test_start + test_samples
            
            # Create indices
            train_idx = self.features_df.index[train_start:train_end]
            test_idx = self.features_df.index[test_start:test_end]
            
            folds.append((train_idx, test_idx))
        
        print(f"Created {len(folds)} cross-validation folds")
        return folds
    
    def evaluate_pattern_on_fold(self, pattern: Dict, train_idx: pd.DatetimeIndex, 
                                 test_idx: pd.DatetimeIndex) -> FoldResult:
        """
        Evaluate a pattern on a single fold.
        
        Args:
            pattern: Pattern dictionary with conditions
            train_idx: Training data indices
            test_idx: Test data indices
            
        Returns:
            FoldResult with evaluation metrics
        """
        # Get pattern details
        conditions = pattern.get('pattern', {}).get('conditions', {})
        direction = pattern.get('pattern', {}).get('direction', 'long')
        label_col = pattern.get('pattern', {}).get('label_col', 'Label_5pct_10d')
        
        # Split data
        train_df = self.features_df.loc[train_idx]
        test_df = self.features_df.loc[test_idx]
        
        # Calculate success rate on test data
        mask = pd.Series(True, index=test_df.index)
        
        for feature, condition in conditions.items():
            if feature not in test_df.columns:
                continue
            
            operator = condition['operator']
            threshold = condition['value']
            
            if operator == '>=':
                mask &= (test_df[feature] >= threshold)
            elif operator == '<=':
                mask &= (test_df[feature] <= threshold)
            elif operator == '>':
                mask &= (test_df[feature] > threshold)
            elif operator == '<':
                mask &= (test_df[feature] < threshold)
        
        # Get occurrences
        occurrences = test_df[mask]
        
        if len(occurrences) == 0:
            return FoldResult(
                fold_id=0,
                train_start=str(train_idx[0]),
                train_end=str(train_idx[-1]),
                test_start=str(test_idx[0]),
                test_end=str(test_idx[-1]),
                success_rate=0.0,
                occurrences=0,
                false_positive_rate=100.0,
                composite_score=0.0
            )
        
        # Calculate success rate
        if direction == 'long':
            success_mask = occurrences[label_col] == 'STRONG_UP'
            opposite_mask = occurrences[label_col] == 'STRONG_DOWN'
        else:
            success_mask = occurrences[label_col] == 'STRONG_DOWN'
            opposite_mask = occurrences[label_col] == 'STRONG_UP'
        
        success_count = success_mask.sum()
        total_count = len(occurrences)
        success_rate = success_count / total_count * 100 if total_count > 0 else 0
        
        # Calculate false positive rate
        false_positive_count = opposite_mask.sum()
        false_positive_rate = false_positive_count / total_count * 100 if total_count > 0 else 0
        
        # Calculate composite score
        composite_score = (success_rate / 100.0) * 0.7 + (1 - false_positive_rate / 100.0) * 0.3
        
        # Get market conditions
        volatility_regime = occurrences['Vol_Regime'].mode()[0] if 'Vol_Regime' in occurrences.columns else None
        trend_regime = occurrences['Trend_Regime'].mode()[0] if 'Trend_Regime' in occurrences.columns else None
        
        return FoldResult(
            fold_id=0,
            train_start=str(train_idx[0]),
            train_end=str(train_idx[-1]),
            test_start=str(test_idx[0]),
            test_end=str(test_idx[-1]),
            success_rate=success_rate,
            occurrences=total_count,
            false_positive_rate=false_positive_rate,
            composite_score=composite_score,
            volatility_regime=volatility_regime,
            trend_regime=trend_regime
        )
    
    def cross_validate_pattern(self, pattern: Dict, pattern_id: str = None) -> CrossValidationResult:
        """
        Perform cross-validation on a single pattern.
        
        Args:
            pattern: Pattern dictionary
            pattern_id: Optional pattern identifier
            
        Returns:
            CrossValidationResult with comprehensive metrics
        """
        if pattern_id is None:
            pattern_id = f"pattern_{hash(json.dumps(pattern, sort_keys=True)) % 10000}"
        
        pattern_name = pattern.get('pattern', {}).get('label_col', 'Unknown')
        
        # Create folds
        folds = self.create_folds()
        
        # Evaluate on each fold
        fold_results_list = []
        out_sample_success_rates = []
        out_sample_occurrences = []
        out_sample_false_positive_rates = []
        
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            result = self.evaluate_pattern_on_fold(pattern, train_idx, test_idx)
            result.fold_id = fold_id
            fold_results_list.append(result)
            
            out_sample_success_rates.append(result.success_rate)
            out_sample_occurrences.append(result.occurrences)
            out_sample_false_positive_rates.append(result.false_positive_rate)
        
        # Calculate in-sample metrics (use full data)
        full_mask = pd.Series(True, index=self.features_df.index)
        conditions = pattern.get('pattern', {}).get('conditions', {})
        direction = pattern.get('pattern', {}).get('direction', 'long')
        label_col = pattern.get('pattern', {}).get('label_col', 'Label_5pct_10d')
        
        for feature, condition in conditions.items():
            if feature not in self.features_df.columns:
                continue
            
            operator = condition['operator']
            threshold = condition['value']
            
            if operator == '>=':
                full_mask &= (self.features_df[feature] >= threshold)
            elif operator == '<=':
                full_mask &= (self.features_df[feature] <= threshold)
        
        full_occurrences = self.features_df[full_mask]
        
        if direction == 'long':
            success_mask = full_occurrences[label_col] == 'STRONG_UP'
            opposite_mask = full_occurrences[label_col] == 'STRONG_DOWN'
        else:
            success_mask = full_occurrences[label_col] == 'STRONG_DOWN'
            opposite_mask = full_occurrences[label_col] == 'STRONG_UP'
        
        in_sample_success_rate = success_mask.sum() / len(full_occurrences) * 100 if len(full_occurrences) > 0 else 0
        in_sample_occurrences = len(full_occurrences)
        in_sample_false_positive_rate = opposite_mask.sum() / len(full_occurrences) * 100 if len(full_occurrences) > 0 else 0
        
        # Calculate aggregated out-of-sample metrics
        avg_out_sample_success_rate = np.mean(out_sample_success_rates) if out_sample_success_rates else 0
        avg_out_sample_occurrences = np.mean(out_sample_occurrences) if out_sample_occurrences else 0
        avg_out_sample_false_positive_rate = np.mean(out_sample_false_positive_rates) if out_sample_false_positive_rates else 0
        
        # Calculate stability metrics
        success_rate_std = np.std(out_sample_success_rates) if len(out_sample_success_rates) > 1 else 0
        success_rate_cv = success_rate_std / avg_out_sample_success_rate if avg_out_sample_success_rate > 0 else 0
        stability_score = max(0, 1 - success_rate_cv)
        
        # Calculate performance degradation
        performance_degradation = in_sample_success_rate - avg_out_sample_success_rate
        
        # Calculate consistency (folds meeting minimum success rate)
        min_sr = self.config['min_success_rate']
        consistent_folds = sum(1 for sr in out_sample_success_rates if sr >= min_sr)
        total_folds = len(out_sample_success_rates)
        consistency_rate = consistent_folds / total_folds if total_folds > 0 else 0
        
        # Calculate market condition performance
        market_condition_performance = {}
        for result in fold_results_list:
            condition = f"{result.volatility_regime}_{result.trend_regime}"
            if condition not in market_condition_performance:
                market_condition_performance[condition] = []
            market_condition_performance[condition].append(result.success_rate)
        
        # Average market condition performance
        market_condition_performance = {
            k: np.mean(v) for k, v in market_condition_performance.items() if v
        }
        
        # Calculate robustness score
        stability_weight = self.config['stability_weight']
        consistency_weight = self.config['consistency_weight']
        performance_weight = self.config['performance_weight']
        degradation_weight = self.config['degradation_weight']
        
        # Normalize performance degradation (lower is better)
        normalized_degradation = max(0, 1 - performance_degradation / 100)
        
        robustness_score = (
            stability_weight * stability_score +
            consistency_weight * consistency_rate +
            performance_weight * (avg_out_sample_success_rate / 100) +
            degradation_weight * normalized_degradation
        )
        
        # Determine if pattern is robust
        is_robust = (
            avg_out_sample_success_rate >= self.config['min_success_rate'] and
            stability_score >= self.config['min_stability_score'] and
            consistency_rate >= self.config['min_consistency_rate'] and
            performance_degradation <= self.config['max_performance_degradation']
        )
        
        # Generate validation notes
        validation_notes = []
        if avg_out_sample_success_rate < self.config['min_success_rate']:
            validation_notes.append(f"Out-of-sample success rate ({avg_out_sample_success_rate:.1f}%) below threshold ({self.config['min_success_rate']}%)")
        if stability_score < self.config['min_stability_score']:
            validation_notes.append(f"Stability score ({stability_score:.3f}) below threshold ({self.config['min_stability_score']})")
        if consistency_rate < self.config['min_consistency_rate']:
            validation_notes.append(f"Consistency rate ({consistency_rate:.1%}) below threshold ({self.config['min_consistency_rate']:.0%})")
        if performance_degradation > self.config['max_performance_degradation']:
            validation_notes.append(f"Performance degradation ({performance_degradation:.1f}%) exceeds threshold ({self.config['max_performance_degradation']}%)")
        
        if is_robust:
            validation_notes.append("Pattern is robust across all validation criteria")
        
        # Create result
        result = CrossValidationResult(
            pattern_id=pattern_id,
            pattern_name=pattern_name,
            in_sample_success_rate=in_sample_success_rate,
            in_sample_occurrences=in_sample_occurrences,
            in_sample_false_positive_rate=in_sample_false_positive_rate,
            out_sample_success_rates=out_sample_success_rates,
            out_sample_occurrences=out_sample_occurrences,
            out_sample_false_positive_rates=out_sample_false_positive_rates,
            avg_out_sample_success_rate=avg_out_sample_success_rate,
            avg_out_sample_occurrences=avg_out_sample_occurrences,
            avg_out_sample_false_positive_rate=avg_out_sample_false_positive_rate,
            success_rate_std=success_rate_std,
            success_rate_cv=success_rate_cv,
            stability_score=stability_score,
            performance_degradation=performance_degradation,
            consistent_folds=consistent_folds,
            total_folds=total_folds,
            consistency_rate=consistency_rate,
            market_condition_performance=market_condition_performance,
            robustness_score=robustness_score,
            is_robust=is_robust,
            validation_notes=validation_notes
        )
        
        self.cv_results[pattern_id] = result
        self.fold_results[pattern_id] = fold_results_list
        
        return result
    
    def cross_validate_patterns(self, patterns: List[Dict]) -> Dict[str, CrossValidationResult]:
        """
        Perform cross-validation on multiple patterns.
        
        Args:
            patterns: List of pattern dictionaries
            
        Returns:
            Dictionary mapping pattern IDs to CrossValidationResults
        """
        print(f"\nCross-validating {len(patterns)} patterns...")
        print(f"Folds: {self.config['n_folds']}")
        print(f"Min train size: {self.config['min_train_size']:.0%}")
        print(f"Test size: {self.config['test_size']:.0%}")
        
        results = {}
        
        for i, pattern in enumerate(patterns):
            pattern_id = f"pattern_{i}"
            result = self.cross_validate_pattern(pattern, pattern_id)
            results[pattern_id] = result
            
            if (i + 1) % 10 == 0:
                print(f"  Cross-validated {i + 1} patterns...")
        
        print(f"Cross-validation complete: {len(results)} patterns")
        return results
    
    def analyze_robustness(self, results: Dict[str, CrossValidationResult]) -> Dict:
        """
        Analyze overall robustness across all patterns.
        
        Args:
            results: Dictionary of CrossValidationResults
            
        Returns:
            Dictionary with robustness analysis
        """
        if not results:
            return {}
        
        # Calculate aggregate metrics
        robust_patterns = sum(1 for r in results.values() if r.is_robust)
        total_patterns = len(results)
        
        avg_robustness_score = np.mean([r.robustness_score for r in results.values()])
        avg_stability_score = np.mean([r.stability_score for r in results.values()])
        avg_consistency_rate = np.mean([r.consistency_rate for r in results.values()])
        avg_performance_degradation = np.mean([r.performance_degradation for r in results.values()])
        
        # Out-of-sample performance distribution
        out_sample_success_rates = [r.avg_out_sample_success_rate for r in results.values()]
        out_sample_occurrences = [r.avg_out_sample_occurrences for r in results.values()]
        
        return {
            'total_patterns': total_patterns,
            'robust_patterns': robust_patterns,
            'robustness_rate': robust_patterns / total_patterns if total_patterns > 0 else 0,
            'avg_robustness_score': avg_robustness_score,
            'avg_stability_score': avg_stability_score,
            'avg_consistency_rate': avg_consistency_rate,
            'avg_performance_degradation': avg_performance_degradation,
            'out_sample_success_rate_mean': np.mean(out_sample_success_rates) if out_sample_success_rates else 0,
            'out_sample_success_rate_std': np.std(out_sample_success_rates) if out_sample_success_rates else 0,
            'out_sample_success_rate_min': np.min(out_sample_success_rates) if out_sample_success_rates else 0,
            'out_sample_success_rate_max': np.max(out_sample_success_rates) if out_sample_success_rates else 0,
            'out_sample_occurrences_mean': np.mean(out_sample_occurrences) if out_sample_occurrences else 0,
            'out_sample_occurrences_std': np.std(out_sample_occurrences) if out_sample_occurrences else 0
        }
    
    def generate_robustness_report(self, results: Dict[str, CrossValidationResult], 
                                    analysis: Dict, output_path: str = None) -> str:
        """
        Generate comprehensive robustness report.
        
        Args:
            results: Dictionary of CrossValidationResults
            analysis: Robustness analysis
            output_path: Output file path
            
        Returns:
            Path to generated report
        """
        print("\nGenerating robustness report...")
        
        if output_path is None:
            output_path = 'data/cross_validation_report.md'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Title
            f.write("# Cross-Validation Robustness Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Patterns Tested:** {analysis.get('total_patterns', 0)}\n")
            f.write(f"- **Robust Patterns:** {analysis.get('robust_patterns', 0)}\n")
            f.write(f"- **Robustness Rate:** {analysis.get('robustness_rate', 0):.1%}\n\n")
            
            f.write("### Robustness Criteria\n")
            f.write(f"- Minimum Out-of-Sample Success Rate: {self.config['min_success_rate']}%\n")
            f.write(f"- Minimum Stability Score: {self.config['min_stability_score']}\n")
            f.write(f"- Minimum Consistency Rate: {self.config['min_consistency_rate']:.0%}\n")
            f.write(f"- Maximum Performance Degradation: {self.config['max_performance_degradation']}%\n\n")
            
            # Cross-Validation Configuration
            f.write("## Cross-Validation Configuration\n\n")
            f.write(f"- **Number of Folds:** {self.config['n_folds']}\n")
            f.write(f"- **Minimum Training Size:** {self.config['min_train_size']:.0%}\n")
            f.write(f"- **Test Size:** {self.config['test_size']:.0%}\n")
            f.write(f"- **Window Type:** {'Expanding' if self.config['expanding_window'] else 'Rolling'}\n\n")
            
            # Overall Robustness Analysis
            f.write("## Overall Robustness Analysis\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Total Patterns | {analysis.get('total_patterns', 0)} |\n")
            f.write(f"| Robust Patterns | {analysis.get('robust_patterns', 0)} |\n")
            f.write(f"| Robustness Rate | {analysis.get('robustness_rate', 0):.1%} |\n")
            f.write(f"| Average Robustness Score | {analysis.get('avg_robustness_score', 0):.3f} |\n")
            f.write(f"| Average Stability Score | {analysis.get('avg_stability_score', 0):.3f} |\n")
            f.write(f"| Average Consistency Rate | {analysis.get('avg_consistency_rate', 0):.1%} |\n")
            f.write(f"| Average Performance Degradation | {analysis.get('avg_performance_degradation', 0):.2f}% |\n")
            f.write(f"| Out-of-Sample SR Mean | {analysis.get('out_sample_success_rate_mean', 0):.2f}% |\n")
            f.write(f"| Out-of-Sample SR Std | {analysis.get('out_sample_success_rate_std', 0):.2f}% |\n")
            f.write(f"| Out-of-Sample SR Range | {analysis.get('out_sample_success_rate_min', 0):.1f}% - {analysis.get('out_sample_success_rate_max', 0):.1f}% |\n")
            f.write(f"| Out-of-Sample Occurrences Mean | {analysis.get('out_sample_occurrences_mean', 0):.1f} |\n")
            f.write(f"| Out-of-Sample Occurrences Std | {analysis.get('out_sample_occurrences_std', 0):.1f} |\n\n")
            
            # Pattern-by-Pattern Results
            f.write("## Pattern-by-Pattern Results\n\n")
            
            # Sort by robustness score
            sorted_results = sorted(results.items(), key=lambda x: x[1].robustness_score, reverse=True)
            
            f.write("| Pattern ID | Pattern Name | Robust | Robustness Score | Out-Sample SR | Stability | Consistency |\n")
            f.write("|------------|-------------|--------|-----------------|---------------|-----------|-------------|\n")
            
            for pattern_id, result in sorted_results:
                robust = "✓" if result.is_robust else "✗"
                f.write(f"| {pattern_id} | {result.pattern_name} | {robust} | {result.robustness_score:.3f} | {result.avg_out_sample_success_rate:.1f}% | {result.stability_score:.3f} | {result.consistency_rate:.1%} |\n")
            
            f.write("\n")
            
            # Detailed Results for Top 10 Patterns
            f.write("## Detailed Results: Top 10 Patterns\n\n")
            
            for pattern_id, result in sorted_results[:10]:
                f.write(f"### {pattern_id}: {result.pattern_name}\n\n")
                f.write(f"**Robustness Score:** {result.robustness_score:.3f}\n")
                f.write(f"**Is Robust:** {'Yes' if result.is_robust else 'No'}\n\n")
                
                f.write("#### In-Sample Performance\n")
                f.write(f"- Success Rate: {result.in_sample_success_rate:.1f}%\n")
                f.write(f"- Occurrences: {result.in_sample_occurrences}\n")
                f.write(f"- False Positive Rate: {result.in_sample_false_positive_rate:.1f}%\n\n")
                
                f.write("#### Out-of-Sample Performance\n")
                f.write(f"- Average Success Rate: {result.avg_out_sample_success_rate:.1f}%\n")
                f.write(f"- Average Occurrences: {result.avg_out_sample_occurrences:.1f}\n")
                f.write(f"- Average False Positive Rate: {result.avg_out_sample_false_positive_rate:.1f}%\n")
                f.write(f"- Performance Degradation: {result.performance_degradation:.2f}%\n\n")
                
                f.write("#### Stability Metrics\n")
                f.write(f"- Success Rate Std: {result.success_rate_std:.2f}%\n")
                f.write(f"- Success Rate CV: {result.success_rate_cv:.3f}\n")
                f.write(f"- Stability Score: {result.stability_score:.3f}\n\n")
                
                f.write("#### Consistency Metrics\n")
                f.write(f"- Consistent Folds: {result.consistent_folds}/{result.total_folds}\n")
                f.write(f"- Consistency Rate: {result.consistency_rate:.1%}\n\n")
                
                if result.market_condition_performance:
                    f.write("#### Market Condition Performance\n")
                    for condition, sr in result.market_condition_performance.items():
                        f.write(f"- {condition}: {sr:.1f}%\n")
                    f.write("\n")
                
                f.write("#### Validation Notes\n")
                for note in result.validation_notes:
                    f.write(f"- {note}\n")
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            robustness_rate = analysis.get('robustness_rate', 0)
            
            if robustness_rate >= 0.7:
                f.write("1. **High Robustness** - A high percentage of patterns ({robustness_rate:.0%}) are robust across time periods. The pattern discovery approach is producing stable patterns.\n\n")
            elif robustness_rate >= 0.4:
                f.write("1. **Moderate Robustness** - About {robustness_rate:.0%} of patterns are robust. Consider adjusting pattern discovery parameters to improve stability.\n\n")
            else:
                f.write("1. **Low Robustness** - Only {robustness_rate:.0%} of patterns are robust. Pattern discovery needs significant improvement to ensure time-period stability.\n\n")
            
            avg_degradation = analysis.get('avg_performance_degradation', 0)
            if avg_degradation <= 5:
                f.write("2. **Low Performance Degradation** - Patterns maintain performance well on out-of-sample data (avg {avg_degradation:.1f}% degradation).\n\n")
            elif avg_degradation <= 15:
                f.write("2. **Moderate Performance Degradation** - Patterns show some performance degradation on out-of-sample data (avg {avg_degradation:.1f}% degradation). Consider additional regularization.\n\n")
            else:
                f.write("2. **High Performance Degradation** - Patterns show significant performance degradation on out-of-sample data (avg {avg_degradation:.1f}% degradation). Pattern discovery may be overfitting.\n\n")
            
            avg_stability = analysis.get('avg_stability_score', 0)
            if avg_stability >= 0.7:
                f.write("3. **Good Stability** - Patterns show good stability across time periods (avg stability score: {avg_stability:.3f}).\n\n")
            else:
                f.write("3. **Poor Stability** - Patterns show poor stability across time periods (avg stability score: {avg_stability:.3f}). Consider using more fundamental features.\n\n")
            
            # Conclusion
            f.write("## Conclusion\n\n")
            f.write("The cross-validation analysis tested pattern robustness across multiple time periods using ")
            f.write(f"{self.config['n_folds']}-fold time-series cross-validation. ")
            
            if robustness_rate >= 0.7:
                f.write("The results indicate that the pattern discovery approach produces robust patterns that maintain performance across different market conditions. ")
            else:
                f.write("The results indicate that the pattern discovery approach needs improvement to produce more robust patterns. ")
            
            f.write("Key metrics include robustness score, stability score, consistency rate, and performance degradation from in-sample to out-of-sample data.\n\n")
            
            f.write("---\n\n")
            f.write("*Report generated by TimeSeriesCrossValidator*\n")
        
        print(f"  Saved: {output_path}")
        return str(output_path)
    
    def generate_visualizations(self, results: Dict[str, CrossValidationResult], 
                                output_path: str = None) -> str:
        """
        Generate cross-validation visualizations.
        
        Args:
            results: Dictionary of CrossValidationResults
            output_path: Output file path
            
        Returns:
            Path to generated visualization
        """
        print("\nGenerating visualizations...")
        
        if output_path is None:
            output_path = 'data/cross_validation_visualizations.png'
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cross-Validation Robustness Analysis', fontsize=16, fontweight='bold')
        
        # Colors
        robust_color = '#4ecdc4'
        non_robust_color = '#ff6b6b'
        
        # 1. Robustness Score Distribution
        ax1 = axes[0, 0]
        robustness_scores = [r.robustness_score for r in results.values()]
        is_robust = [r.is_robust for r in results.values()]
        colors = [robust_color if r else non_robust_color for r in is_robust]
        ax1.scatter(range(len(robustness_scores)), robustness_scores, c=colors, alpha=0.7, s=50)
        ax1.axhline(y=0.7, color='green', linestyle='--', label='Threshold')
        ax1.set_xlabel('Pattern Index')
        ax1.set_ylabel('Robustness Score')
        ax1.set_title('Robustness Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. In-Sample vs Out-of-Sample Success Rate
        ax2 = axes[0, 1]
        in_sample_sr = [r.in_sample_success_rate for r in results.values()]
        out_sample_sr = [r.avg_out_sample_success_rate for r in results.values()]
        colors = [robust_color if r.is_robust else non_robust_color for r in results.values()]
        ax2.scatter(in_sample_sr, out_sample_sr, c=colors, alpha=0.7, s=50)
        ax2.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Perfect Correlation')
        ax2.set_xlabel('In-Sample Success Rate (%)')
        ax2.set_ylabel('Out-of-Sample Success Rate (%)')
        ax2.set_title('In-Sample vs Out-of-Sample Success Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Stability Score Distribution
        ax3 = axes[0, 2]
        stability_scores = [r.stability_score for r in results.values()]
        # Separate scores by robustness
        robust_scores = [r.stability_score for r in results.values() if r.is_robust]
        non_robust_scores = [r.stability_score for r in results.values() if not r.is_robust]
        ax3.hist([robust_scores, non_robust_scores], bins=20, alpha=0.7,
                 color=[robust_color, non_robust_color], label=['Robust', 'Non-Robust'])
        ax3.axvline(x=0.6, color='green', linestyle='--', label='Threshold')
        ax3.set_xlabel('Stability Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Stability Score Distribution')
        ax3.legend()
        
        # 4. Performance Degradation Distribution
        ax4 = axes[1, 0]
        performance_degradation = [r.performance_degradation for r in results.values()]
        # Separate by robustness
        robust_degradation = [r.performance_degradation for r in results.values() if r.is_robust]
        non_robust_degradation = [r.performance_degradation for r in results.values() if not r.is_robust]
        ax4.hist([robust_degradation, non_robust_degradation], bins=20, alpha=0.7,
                 color=[robust_color, non_robust_color], label=['Robust', 'Non-Robust'])
        ax4.axvline(x=15, color='red', linestyle='--', label='Max Threshold')
        ax4.set_xlabel('Performance Degradation (%)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Performance Degradation Distribution')
        ax4.legend()
        
        # 5. Consistency Rate Distribution
        ax5 = axes[1, 1]
        consistency_rates = [r.consistency_rate for r in results.values()]
        # Separate by robustness
        robust_consistency = [r.consistency_rate for r in results.values() if r.is_robust]
        non_robust_consistency = [r.consistency_rate for r in results.values() if not r.is_robust]
        ax5.hist([robust_consistency, non_robust_consistency], bins=20, alpha=0.7,
                 color=[robust_color, non_robust_color], label=['Robust', 'Non-Robust'])
        ax5.axvline(x=0.7, color='green', linestyle='--', label='Threshold')
        ax5.set_xlabel('Consistency Rate')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Consistency Rate Distribution')
        ax5.legend()
        
        # 6. Summary Metrics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        robust_count = sum(1 for r in results.values() if r.is_robust)
        total_count = len(results)
        avg_robustness = np.mean(robustness_scores) if robustness_scores else 0
        avg_stability = np.mean(stability_scores) if stability_scores else 0
        avg_consistency = np.mean(consistency_rates) if consistency_rates else 0
        avg_degradation = np.mean(performance_degradation) if performance_degradation else 0
        
        summary_text = "CROSS-VALIDATION SUMMARY\n\n"
        summary_text += f"Total Patterns: {total_count}\n"
        summary_text += f"Robust Patterns: {robust_count} ({robust_count/total_count*100:.1f}%)\n\n"
        summary_text += f"Avg Robustness Score: {avg_robustness:.3f}\n"
        summary_text += f"Avg Stability Score: {avg_stability:.3f}\n"
        summary_text += f"Avg Consistency Rate: {avg_consistency:.1%}\n"
        summary_text += f"Avg Performance Degradation: {avg_degradation:.1f}%\n\n"
        summary_text += "CRITERIA:\n"
        summary_text += f"Min Success Rate: {self.config['min_success_rate']}%\n"
        summary_text += f"Min Stability: {self.config['min_stability_score']}\n"
        summary_text += f"Min Consistency: {self.config['min_consistency_rate']:.0%}\n"
        summary_text += f"Max Degradation: {self.config['max_performance_degradation']}%\n"
        
        ax6.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_path}")
        return str(output_path)
    
    def run_cross_validation(self, patterns: List[Dict]) -> Dict:
        """
        Run complete cross-validation workflow.
        
        Args:
            patterns: List of pattern dictionaries
            
        Returns:
            Complete cross-validation results
        """
        print("=" * 60)
        print("Cross-Validation Framework")
        print("=" * 60)
        
        # Cross-validate patterns
        results = self.cross_validate_patterns(patterns)
        
        # Analyze robustness
        analysis = self.analyze_robustness(results)
        
        # Generate visualizations
        self.generate_visualizations(results)
        
        # Generate report
        report_path = self.generate_robustness_report(results, analysis)
        
        print("\n" + "=" * 60)
        print("Cross-Validation Complete!")
        print("=" * 60)
        print(f"  Report: {report_path}")
        
        return {
            'results': results,
            'analysis': analysis
        }


def main():
    """Main function to run cross-validation."""
    print("=" * 60)
    print("Cross-Validation Framework")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    features_df = pd.read_csv('data/features_matrix.csv', index_col='Date', parse_dates=True)
    print(f"Loaded {len(features_df)} records with {len(features_df.columns)} columns")
    
    # Load patterns (use enhanced patterns from false positive reduction)
    print("\nLoading patterns...")
    try:
        with open('data/enhanced_patterns.json', 'r') as f:
            patterns = json.load(f)
        print(f"Loaded {len(patterns)} patterns")
    except FileNotFoundError:
        print("No patterns found, creating dummy patterns for testing...")
        patterns = []
        for i in range(20):
            patterns.append({
                'pattern': {
                    'conditions': {
                        'RSI_14': {'operator': '<=', 'value': 30.0},
                        'MA_Cross_20_50': {'operator': '>', 'value': 0.0}
                    },
                    'direction': 'long',
                    'label_col': 'Label_5pct_10d',
                    'occurrences': 50,
                    'success_rate': 0.8
                }
            })
    
    # Initialize cross-validator
    print("\nInitializing cross-validator...")
    validator = TimeSeriesCrossValidator(features_df)
    
    # Run cross-validation
    cv_results = validator.run_cross_validation(patterns)
    
    # Display summary
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    
    analysis = cv_results['analysis']
    print(f"\nTotal Patterns: {analysis.get('total_patterns', 0)}")
    print(f"Robust Patterns: {analysis.get('robust_patterns', 0)}")
    print(f"Robustness Rate: {analysis.get('robustness_rate', 0):.1%}")
    print(f"Avg Robustness Score: {analysis.get('avg_robustness_score', 0):.3f}")
    print(f"Avg Stability Score: {analysis.get('avg_stability_score', 0):.3f}")
    print(f"Avg Consistency Rate: {analysis.get('avg_consistency_rate', 0):.1%}")
    print(f"Avg Performance Degradation: {analysis.get('avg_performance_degradation', 0):.2f}%")


if __name__ == "__main__":
    main()