"""
Pattern Validation Framework

A comprehensive validation framework for evaluating pattern quality, measuring:
- Success rate and frequency
- False positive rates and pattern diversity metrics
- Automated testing procedures for pattern discovery methods
- Documentation with examples and best practices

Author: Agent_Validation
Date: 2026-01-22
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from scipy import stats
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class PatternType(Enum):
    """Classification of pattern discovery methods."""
    ML_BASED = "ml_based"
    RULE_BASED = "rule_based"
    HYBRID = "hybrid"
    GUARANTEED = "guaranteed"
    CONTEXT7_ENHANCED = "context7_enhanced"


class ValidationStatus(Enum):
    """Status of pattern validation."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    INCONCLUSIVE = "inconclusive"


@dataclass
class PatternMetrics:
    """Comprehensive metrics for a single pattern."""
    # Basic metrics
    pattern_id: str
    pattern_type: PatternType
    direction: str
    label_col: str
    
    # Success metrics
    occurrences: int
    success_count: int
    success_rate: float  # Percentage (0-100)
    
    # False positive metrics
    false_positive_count: int
    false_positive_rate: float  # Percentage (0-100)
    
    # Statistical metrics
    p_value: float
    statistical_significance: bool
    
    # Stability metrics
    stability_score: float  # 0-1
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
    
    # Metadata
    conditions: Dict = field(default_factory=dict)
    creation_date: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type.value,
            'direction': self.direction,
            'label_col': self.label_col,
            'occurrences': self.occurrences,
            'success_count': self.success_count,
            'success_rate': self.success_rate,
            'false_positive_count': self.false_positive_count,
            'false_positive_rate': self.false_positive_rate,
            'p_value': self.p_value,
            'statistical_significance': self.statistical_significance,
            'stability_score': self.stability_score,
            'yearly_success_rates': self.yearly_success_rates,
            'avg_move': self.avg_move,
            'avg_time_to_target': self.avg_time_to_target,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'feature_categories': self.feature_categories,
            'num_conditions': self.num_conditions,
            'composite_score': self.composite_score,
            'validation_status': self.validation_status.value,
            'conditions': self.conditions,
            'creation_date': self.creation_date
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report for a set of patterns."""
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
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'report_id': self.report_id,
            'timestamp': self.timestamp,
            'pattern_type': self.pattern_type.value,
            'summary': {
                'total_patterns': self.total_patterns,
                'passed_patterns': self.passed_patterns,
                'failed_patterns': self.failed_patterns,
                'warning_patterns': self.warning_patterns,
                'pass_rate': self.passed_patterns / self.total_patterns if self.total_patterns > 0 else 0
            },
            'aggregate_metrics': {
                'avg_success_rate': self.avg_success_rate,
                'avg_false_positive_rate': self.avg_false_positive_rate,
                'avg_stability_score': self.avg_stability_score,
                'avg_composite_score': self.avg_composite_score
            },
            'diversity_metrics': {
                'unique_feature_categories': self.unique_feature_categories,
                'pattern_diversity_index': self.pattern_diversity_index
            },
            'performance_metrics': {
                'avg_occurrences': self.avg_occurrences,
                'total_occurrences': self.total_occurrences
            },
            'patterns': [p.to_dict() for p in self.patterns],
            'recommendations': self.recommendations
        }


class PatternValidationFramework:
    """
    Comprehensive validation framework for evaluating pattern quality.
    
    This framework provides tools to:
    - Measure pattern success rate and frequency
    - Track false positive rates and pattern diversity metrics
    - Create automated testing procedures for pattern discovery methods
    - Generate detailed validation reports
    """
    
    # Default thresholds for pattern validation
    DEFAULT_THRESHOLDS = {
        'min_success_rate': 70.0,  # Minimum success rate percentage
        'max_false_positive_rate': 15.0,  # Maximum false positive rate percentage
        'min_p_value': 0.05,  # Maximum p-value for statistical significance
        'min_stability_score': 0.5,  # Minimum stability score (0-1)
        'min_occurrences': 20,  # Minimum number of occurrences
        'min_composite_score': 0.6,  # Minimum composite score (0-1)
        'max_patterns_per_category': 10,  # Maximum patterns per feature category
        'min_feature_categories': 2  # Minimum number of feature categories
    }
    
    def __init__(self, features_df: pd.DataFrame = None, config: Dict = None):
        """
        Initialize the validation framework.
        
        Args:
            features_df: DataFrame with technical features and labels
            config: Configuration parameters
        """
        self.features_df = features_df.copy() if features_df is not None else None
        self.config = config or {}
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        self.thresholds.update(self.config.get('thresholds', {}))
        
        # Feature categories for diversity analysis
        self.feature_categories = self._categorize_features()
        
        # Validation history
        self.validation_history = []
        
        print("Pattern Validation Framework initialized")
        print(f"  Default thresholds: {self.thresholds}")
    
    def _categorize_features(self) -> Dict[str, List[str]]:
        """Categorize features by type for diversity management."""
        if self.features_df is None:
            return {}
        
        exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        numeric_features = [
            col for col in self.features_df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(self.features_df[col])
        ]
        
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
        
        for feature in numeric_features:
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
        
        return {k: v for k, v in categories.items() if v}
    
    def evaluate_pattern(self, pattern: Dict, pattern_id: str = None, 
                        pattern_type: PatternType = PatternType.RULE_BASED) -> PatternMetrics:
        """
        Evaluate a single pattern comprehensively.
        
        Args:
            pattern: Pattern dictionary with conditions and metrics
            pattern_id: Unique identifier for the pattern
            pattern_type: Type of pattern discovery method
            
        Returns:
            PatternMetrics object with all evaluation results
        """
        if pattern_id is None:
            pattern_id = f"pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Extract pattern information
        conditions = pattern.get('conditions', {})
        direction = pattern.get('direction', 'long')
        label_col = pattern.get('label_col', 'Label_3pct_5d')
        
        # Get metrics from pattern
        occurrences = pattern.get('occurrences', 0)
        success_rate = pattern.get('success_rate', 0) * 100 if pattern.get('success_rate') else 0
        avg_move = pattern.get('avg_move', 0)
        
        # Calculate false positive rate
        false_positive_rate = pattern.get('false_positive_rate', 0)
        
        # Calculate p-value using binomial test
        success_count = int(occurrences * success_rate / 100) if occurrences > 0 else 0
        p_value = stats.binomtest(success_count, occurrences, p=0.5, alternative='greater').pvalue if occurrences > 0 else 1.0
        statistical_significance = p_value < self.thresholds['min_p_value']
        
        # Calculate stability score
        stability_score = pattern.get('stability_score', 0.5)
        
        # Calculate yearly success rates
        yearly_success_rates = pattern.get('yearly_success_rates', {})
        
        # Calculate performance metrics
        sharpe_ratio = self._calculate_sharpe_ratio(pattern)
        max_drawdown = self._calculate_max_drawdown(pattern)
        avg_time_to_target = pattern.get('avg_time', pattern.get('avg_time_to_target', 0))
        
        # Get feature categories
        feature_categories = self._get_pattern_categories(conditions)
        
        # Calculate composite score
        composite_score = self._calculate_composite_score(
            success_rate, occurrences, false_positive_rate, 
            stability_score, len(feature_categories)
        )
        
        # Determine validation status
        validation_status = self._determine_validation_status(
            success_rate, false_positive_rate, p_value,
            stability_score, occurrences, composite_score
        )
        
        return PatternMetrics(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            direction=direction,
            label_col=label_col,
            occurrences=occurrences,
            success_count=success_count,
            success_rate=success_rate,
            false_positive_count=int(occurrences * false_positive_rate / 100) if occurrences > 0 else 0,
            false_positive_rate=false_positive_rate,
            p_value=p_value,
            statistical_significance=statistical_significance,
            stability_score=stability_score,
            yearly_success_rates=yearly_success_rates,
            avg_move=avg_move,
            avg_time_to_target=avg_time_to_target,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            feature_categories=feature_categories,
            num_conditions=len(conditions),
            composite_score=composite_score,
            validation_status=validation_status,
            conditions=conditions,
            creation_date=datetime.now().isoformat()
        )
    
    def _calculate_sharpe_ratio(self, pattern: Dict) -> float:
        """Calculate Sharpe ratio for a pattern."""
        # Simplified calculation based on success rate and move
        success_rate = pattern.get('success_rate', 0.5)
        avg_move = pattern.get('avg_move', 0)
        
        # Assume risk-free rate of 2%
        risk_free_rate = 0.02
        
        # Calculate excess return
        excess_return = (success_rate * avg_move / 100) - risk_free_rate
        
        # Estimate volatility (simplified)
        volatility = 0.15  # 15% annual volatility assumption
        
        if volatility > 0:
            return excess_return / volatility
        return 0.0
    
    def _calculate_max_drawdown(self, pattern: Dict) -> float:
        """Calculate maximum drawdown for a pattern."""
        # Simplified calculation
        success_rate = pattern.get('success_rate', 0.5)
        
        # Estimate drawdown based on success rate
        # Lower success rate = higher potential drawdown
        return max(0, (0.5 - success_rate) * 2)
    
    def _get_pattern_categories(self, conditions: Dict) -> List[str]:
        """Get feature categories used in a pattern."""
        categories = []
        for feature in conditions.keys():
            for cat, features in self.feature_categories.items():
                if feature in features:
                    categories.append(cat)
                    break
        return list(set(categories))
    
    def _calculate_composite_score(self, success_rate: float, occurrences: int,
                                   false_positive_rate: float, stability_score: float,
                                   num_categories: int) -> float:
        """Calculate composite quality score for a pattern."""
        # Weights for different components
        weights = {
            'success_rate': 0.35,
            'frequency': 0.15,
            'false_positive': 0.20,
            'stability': 0.15,
            'diversity': 0.15
        }
        
        # Normalize components
        success_score = min(success_rate / 100, 1.0)
        frequency_score = min(occurrences / 100, 1.0)
        false_positive_score = 1 - min(false_positive_rate / 100, 1.0)
        stability_score = min(stability_score, 1.0)
        diversity_score = min(num_categories / self.thresholds['min_feature_categories'], 1.0)
        
        # Calculate weighted composite
        composite = (
            weights['success_rate'] * success_score +
            weights['frequency'] * frequency_score +
            weights['false_positive'] * false_positive_score +
            weights['stability'] * stability_score +
            weights['diversity'] * diversity_score
        )
        
        return max(0, min(composite, 1.0))
    
    def _determine_validation_status(self, success_rate: float, false_positive_rate: float,
                                    p_value: float, stability_score: float,
                                    occurrences: int, composite_score: float) -> ValidationStatus:
        """Determine validation status based on thresholds."""
        failures = 0
        warnings = 0
        
        if success_rate < self.thresholds['min_success_rate']:
            failures += 1
        elif success_rate < self.thresholds['min_success_rate'] + 10:
            warnings += 1
        
        if false_positive_rate > self.thresholds['max_false_positive_rate']:
            failures += 1
        elif false_positive_rate > self.thresholds['max_false_positive_rate'] - 5:
            warnings += 1
        
        if p_value >= self.thresholds['min_p_value']:
            warnings += 1
        
        if stability_score < self.thresholds['min_stability_score']:
            warnings += 1
        
        if occurrences < self.thresholds['min_occurrences']:
            warnings += 1
        
        if composite_score < self.thresholds['min_composite_score']:
            failures += 1
        
        if failures >= 2:
            return ValidationStatus.FAILED
        elif warnings >= 3:
            return ValidationStatus.WARNING
        elif failures == 1:
            return ValidationStatus.WARNING
        else:
            return ValidationStatus.PASSED
    
    def validate_patterns(self, patterns: List[Dict], pattern_type: PatternType = PatternType.RULE_BASED,
                         pattern_prefix: str = "") -> ValidationReport:
        """
        Validate a list of patterns and generate a comprehensive report.
        
        Args:
            patterns: List of pattern dictionaries
            pattern_type: Type of pattern discovery method
            pattern_prefix: Prefix for pattern IDs
            
        Returns:
            ValidationReport with all validation results
        """
        print(f"\n{'='*60}")
        print(f"PATTERN VALIDATION REPORT")
        print(f"{'='*60}")
        print(f"Pattern Type: {pattern_type.value}")
        print(f"Total Patterns: {len(patterns)}")
        
        # Evaluate each pattern
        pattern_metrics = []
        for i, pattern in enumerate(patterns):
            pattern_id = f"{pattern_prefix}{i+1}" if pattern_prefix else f"pattern_{i+1}"
            metrics = self.evaluate_pattern(pattern, pattern_id, pattern_type)
            pattern_metrics.append(metrics)
        
        # Calculate aggregate statistics
        total_patterns = len(pattern_metrics)
        passed_patterns = sum(1 for m in pattern_metrics if m.validation_status == ValidationStatus.PASSED)
        failed_patterns = sum(1 for m in pattern_metrics if m.validation_status == ValidationStatus.FAILED)
        warning_patterns = sum(1 for m in pattern_metrics if m.validation_status == ValidationStatus.WARNING)
        
        # Calculate aggregate metrics
        avg_success_rate = np.mean([m.success_rate for m in pattern_metrics]) if pattern_metrics else 0
        avg_false_positive_rate = np.mean([m.false_positive_rate for m in pattern_metrics]) if pattern_metrics else 0
        avg_stability_score = np.mean([m.stability_score for m in pattern_metrics]) if pattern_metrics else 0
        avg_composite_score = np.mean([m.composite_score for m in pattern_metrics]) if pattern_metrics else 0
        
        # Calculate diversity metrics
        all_categories = set()
        for m in pattern_metrics:
            all_categories.update(m.feature_categories)
        unique_feature_categories = len(all_categories)
        
        # Pattern diversity index (entropy-based)
        category_counts = {}
        for m in pattern_metrics:
            for cat in m.feature_categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1
        
        if category_counts:
            total = sum(category_counts.values())
            probs = [count / total for count in category_counts.values()]
            pattern_diversity_index = -sum(p * np.log(p) for p in probs if p > 0)
            # Normalize to 0-1 range
            max_entropy = np.log(len(category_counts))
            pattern_diversity_index = pattern_diversity_index / max_entropy if max_entropy > 0 else 0
        else:
            pattern_diversity_index = 0
        
        # Calculate performance metrics
        avg_occurrences = np.mean([m.occurrences for m in pattern_metrics]) if pattern_metrics else 0
        total_occurrences = sum(m.occurrences for m in pattern_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(pattern_metrics, pattern_type)
        
        # Create report
        report = ValidationReport(
            report_id=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            pattern_type=pattern_type,
            total_patterns=total_patterns,
            passed_patterns=passed_patterns,
            failed_patterns=failed_patterns,
            warning_patterns=warning_patterns,
            avg_success_rate=avg_success_rate,
            avg_false_positive_rate=avg_false_positive_rate,
            avg_stability_score=avg_stability_score,
            avg_composite_score=avg_composite_score,
            unique_feature_categories=unique_feature_categories,
            pattern_diversity_index=pattern_diversity_index,
            avg_occurrences=avg_occurrences,
            total_occurrences=total_occurrences,
            patterns=pattern_metrics,
            recommendations=recommendations
        )
        
        # Print summary
        print(f"\n--- VALIDATION SUMMARY ---")
        if total_patterns > 0:
            print(f"Passed: {passed_patterns}/{total_patterns} ({passed_patterns/total_patterns*100:.1f}%)")
            print(f"Failed: {failed_patterns}/{total_patterns} ({failed_patterns/total_patterns*100:.1f}%)")
            print(f"Warnings: {warning_patterns}/{total_patterns} ({warning_patterns/total_patterns*100:.1f}%)")
        else:
            print(f"Passed: {passed_patterns}/{total_patterns}")
            print(f"Failed: {failed_patterns}/{total_patterns}")
            print(f"Warnings: {warning_patterns}/{total_patterns}")
        
        print(f"\n--- AGGREGATE METRICS ---")
        print(f"Average Success Rate: {avg_success_rate:.1f}%")
        print(f"Average False Positive Rate: {avg_false_positive_rate:.1f}%")
        print(f"Average Stability Score: {avg_stability_score:.3f}")
        print(f"Average Composite Score: {avg_composite_score:.3f}")
        
        print(f"\n--- DIVERSITY METRICS ---")
        print(f"Unique Feature Categories: {unique_feature_categories}")
        print(f"Pattern Diversity Index: {pattern_diversity_index:.3f}")
        
        print(f"\n--- RECOMMENDATIONS ---")
        for rec in recommendations:
            print(f"  - {rec}")
        
        # Store in history
        self.validation_history.append(report)
        
        return report
    
    def _generate_recommendations(self, pattern_metrics: List[PatternMetrics], 
                                  pattern_type: PatternType) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check success rate
        avg_success = np.mean([m.success_rate for m in pattern_metrics]) if pattern_metrics else 0
        if avg_success < self.thresholds['min_success_rate']:
            recommendations.append(f"Consider increasing minimum success rate threshold (current: {avg_success:.1f}%)")
        
        # Check false positive rate
        avg_fpr = np.mean([m.false_positive_rate for m in pattern_metrics]) if pattern_metrics else 0
        if avg_fpr > self.thresholds['max_false_positive_rate']:
            recommendations.append(f"Implement additional false positive filtering (current: {avg_fpr:.1f}%)")
        
        # Check stability
        avg_stability = np.mean([m.stability_score for m in pattern_metrics]) if pattern_metrics else 0
        if avg_stability < self.thresholds['min_stability_score']:
            recommendations.append("Patterns show low stability across time periods - consider regime filtering")
        
        # Check diversity
        all_categories = set()
        for m in pattern_metrics:
            all_categories.update(m.feature_categories)
        if len(all_categories) < self.thresholds['min_feature_categories']:
            recommendations.append("Pattern diversity is low - consider using more feature categories")
        
        # Check for overfitting indicators
        high_success_low_occurrences = [
            m for m in pattern_metrics 
            if m.success_rate > 90 and m.occurrences < self.thresholds['min_occurrences'] * 2
        ]
        if len(high_success_low_occurrences) > len(pattern_metrics) * 0.3:
            recommendations.append("Warning: Many high-success patterns have low occurrences - potential overfitting")
        
        # Check for patterns needing attention
        failed_patterns = [m for m in pattern_metrics if m.validation_status == ValidationStatus.FAILED]
        if len(failed_patterns) > len(pattern_metrics) * 0.2:
            recommendations.append(f"High failure rate ({len(failed_patterns)} patterns) - review validation thresholds")
        
        if not recommendations:
            recommendations.append("All patterns meet validation criteria - framework is working effectively")
        
        return recommendations
    
    def compare_pattern_sets(self, set1: List[PatternMetrics], set2: List[PatternMetrics],
                            set1_name: str = "Set 1", set2_name: str = "Set 2") -> Dict:
        """
        Compare two sets of patterns to evaluate improvements.
        
        Args:
            set1: First set of pattern metrics
            set2: Second set of pattern metrics
            set1_name: Name for first set
            set2_name: Name for second set
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            'set1_name': set1_name,
            'set2_name': set2_name,
            'metrics_comparison': {},
            'improvements': [],
            'degradations': []
        }
        
        # Define metrics to compare
        metrics_to_compare = [
            ('avg_success_rate', 'Average Success Rate', '%'),
            ('avg_false_positive_rate', 'Average False Positive Rate', '%'),
            ('avg_stability_score', 'Average Stability Score', ''),
            ('avg_composite_score', 'Average Composite Score', ''),
            ('avg_occurrences', 'Average Occurrences', ''),
            ('pass_rate', 'Pass Rate', '%')
        ]
        
        set1_pass_rate = sum(1 for m in set1 if m.validation_status == ValidationStatus.PASSED) / len(set1) * 100 if set1 else 0
        set2_pass_rate = sum(1 for m in set2 if m.validation_status == ValidationStatus.PASSED) / len(set2) * 100 if set2 else 0
        
        set1_metrics = {
            'avg_success_rate': np.mean([m.success_rate for m in set1]) if set1 else 0,
            'avg_false_positive_rate': np.mean([m.false_positive_rate for m in set1]) if set1 else 0,
            'avg_stability_score': np.mean([m.stability_score for m in set1]) if set1 else 0,
            'avg_composite_score': np.mean([m.composite_score for m in set1]) if set1 else 0,
            'avg_occurrences': np.mean([m.occurrences for m in set1]) if set1 else 0,
            'pass_rate': set1_pass_rate
        }
        
        set2_metrics = {
            'avg_success_rate': np.mean([m.success_rate for m in set2]) if set2 else 0,
            'avg_false_positive_rate': np.mean([m.false_positive_rate for m in set2]) if set2 else 0,
            'avg_stability_score': np.mean([m.stability_score for m in set2]) if set2 else 0,
            'avg_composite_score': np.mean([m.composite_score for m in set2]) if set2 else 0,
            'avg_occurrences': np.mean([m.occurrences for m in set2]) if set2 else 0,
            'pass_rate': set2_pass_rate
        }
        
        for metric_key, metric_name, unit in metrics_to_compare:
            val1 = set1_metrics[metric_key]
            val2 = set2_metrics[metric_key]
            diff = val2 - val1
            
            comparison['metrics_comparison'][metric_key] = {
                'name': metric_name,
                'set1_value': val1,
                'set2_value': val2,
                'difference': diff,
                'unit': unit
            }
            
            # Determine if improvement or degradation
            if metric_key == 'avg_false_positive_rate':
                # Lower is better for false positive rate
                if diff < -1:
                    comparison['improvements'].append(f"{metric_name}: {diff:.2f}{unit} improvement")
                elif diff > 1:
                    comparison['degradations'].append(f"{metric_name}: {diff:.2f}{unit} degradation")
            else:
                # Higher is better for other metrics
                if diff > 1:
                    comparison['improvements'].append(f"{metric_name}: +{diff:.2f}{unit}")
                elif diff < -1:
                    comparison['degradations'].append(f"{metric_name}: {diff:.2f}{unit}")
        
        # Overall assessment
        improvement_count = len(comparison['improvements'])
        degradation_count = len(comparison['degradations'])
        
        if improvement_count > degradation_count:
            comparison['overall_assessment'] = f"{set2_name} shows overall improvement over {set1_name}"
        elif degradation_count > improvement_count:
            comparison['overall_assessment'] = f"{set2_name} shows degradation compared to {set1_name}"
        else:
            comparison['overall_assessment'] = f"{set1_name} and {set2_name} show similar performance"
        
        return comparison
    
    def run_automated_tests(self, test_cases: List[Dict]) -> Dict:
        """
        Run automated tests on pattern discovery methods.
        
        Args:
            test_cases: List of test cases with expected inputs and outputs
            
        Returns:
            Dictionary with test results
        """
        print(f"\n{'='*60}")
        print(f"AUTOMATED TESTING PROCEDURES")
        print(f"{'='*60}")
        
        results = {
            'total_tests': len(test_cases),
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'test_results': []
        }
        
        for i, test_case in enumerate(test_cases):
            test_name = test_case.get('name', f"Test {i+1}")
            patterns = test_case.get('patterns', [])
            expected_metrics = test_case.get('expected', {})
            pattern_type = test_case.get('type', PatternType.RULE_BASED)
            
            print(f"\n--- {test_name} ---")
            
            # Run validation
            report = self.validate_patterns(patterns, pattern_type)
            
            # Check against expectations
            test_passed = True
            test_warnings = []
            test_details = []
            
            if 'min_success_rate' in expected_metrics:
                actual = report.avg_success_rate
                expected = expected_metrics['min_success_rate']
                if actual >= expected:
                    test_details.append(f"Success rate: {actual:.1f}% >= {expected}% (expected)")
                else:
                    test_passed = False
                    test_details.append(f"Success rate: {actual:.1f}% < {expected}% (expected)")
            
            if 'max_false_positive_rate' in expected_metrics:
                actual = report.avg_false_positive_rate
                expected = expected_metrics['max_false_positive_rate']
                if actual <= expected:
                    test_details.append(f"False positive rate: {actual:.1f}% <= {expected}% (expected)")
                else:
                    test_passed = False
                    test_details.append(f"False positive rate: {actual:.1f}% > {expected}% (expected)")
            
            if 'min_pass_rate' in expected_metrics:
                actual = report.passed_patterns / report.total_patterns * 100 if report.total_patterns > 0 else 0
                expected = expected_metrics['min_pass_rate']
                if actual >= expected:
                    test_details.append(f"Pass rate: {actual:.1f}% >= {expected}% (expected)")
                else:
                    test_warnings.append(f"Pass rate: {actual:.1f}% < {expected}% (expected)")
            
            # Record results
            if test_passed:
                results['passed'] += 1
                status = "PASSED"
            elif test_warnings:
                results['warnings'] += 1
                status = "WARNING"
            else:
                results['failed'] += 1
                status = "FAILED"
            
            results['test_results'].append({
                'name': test_name,
                'status': status,
                'details': test_details,
                'warnings': test_warnings,
                'metrics': report.to_dict()
            })
            
            print(f"Status: {status}")
            for detail in test_details:
                print(f"  {detail}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed']}")
        print(f"Failed: {results['failed']}")
        print(f"Warnings: {results['warnings']}")
        
        return results
    
    def save_report(self, report: ValidationReport, output_path: str = None):
        """
        Save validation report to file.
        
        Args:
            report: ValidationReport to save
            output_path: Output file path
        """
        if output_path is None:
            output_path = f"data/validation_report_{report.report_id}.json"
        
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        print(f"\nValidation report saved to: {output_path}")
    
    def generate_summary_report(self, reports: List[ValidationReport] = None) -> Dict:
        """
        Generate a summary report across multiple validation runs.
        
        Args:
            reports: List of ValidationReports to summarize (uses history if not provided)
        """
        if reports is None:
            reports = self.validation_history
        
        if not reports:
            return {'error': 'No reports available'}
        
        summary = {
            'total_reports': len(reports),
            'date_range': {
                'start': min(r.timestamp for r in reports),
                'end': max(r.timestamp for r in reports)
            },
            'pattern_types_analyzed': list(set(r.pattern_type.value for r in reports)),
            'aggregate_metrics': {},
            'trends': {}
        }
        
        # Aggregate metrics across all reports
        all_success_rates = []
        all_fpr = []
        all_stability = []
        all_composite = []
        
        for report in reports:
            all_success_rates.append(report.avg_success_rate)
            all_fpr.append(report.avg_false_positive_rate)
            all_stability.append(report.avg_stability_score)
            all_composite.append(report.avg_composite_score)
        
        summary['aggregate_metrics'] = {
            'avg_success_rate': {
                'mean': np.mean(all_success_rates),
                'std': np.std(all_success_rates),
                'min': np.min(all_success_rates),
                'max': np.max(all_success_rates)
            },
            'avg_false_positive_rate': {
                'mean': np.mean(all_fpr),
                'std': np.std(all_fpr),
                'min': np.min(all_fpr),
                'max': np.max(all_fpr)
            },
            'avg_stability_score': {
                'mean': np.mean(all_stability),
                'std': np.std(all_stability),
                'min': np.min(all_stability),
                'max': np.max(all_stability)
            },
            'avg_composite_score': {
                'mean': np.mean(all_composite),
                'std': np.std(all_composite),
                'min': np.min(all_composite),
                'max': np.max(all_composite)
            }
        }
        
        return summary


def main():
    """Main function demonstrating the validation framework."""
    print("=" * 60)
    print("PATTERN VALIDATION FRAMEWORK - DEMONSTRATION")
    print("=" * 60)
    
    # Load features data
    print("\nLoading features data...")
    try:
        features_df = pd.read_csv('data/features_matrix.csv', index_col='Date', parse_dates=True)
        print(f"Loaded {len(features_df)} records with {len(features_df.columns)} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        features_df = None
    
    # Initialize framework
    framework = PatternValidationFramework(features_df)
    
    # Load sample patterns for validation
    print("\nLoading sample patterns...")
    try:
        with open('data/enhanced_patterns.json', 'r') as f:
            patterns = json.load(f)
        print(f"Loaded {len(patterns)} patterns from enhanced_patterns.json")
    except Exception as e:
        print(f"Error loading patterns: {e}")
        patterns = []
    
    if patterns:
        # Run validation
        report = framework.validate_patterns(patterns, PatternType.RULE_BASED, "enhanced_")
        
        # Save report
        framework.save_report(report, 'data/validation_report.json')
        
        # Display top patterns
        print(f"\n{'='*60}")
        print("TOP 5 VALIDATED PATTERNS")
        print(f"{'='*60}")
        
        sorted_patterns = sorted(report.patterns, key=lambda x: x.composite_score, reverse=True)
        for i, pattern in enumerate(sorted_patterns[:5]):
            print(f"\n{i+1}. {pattern.pattern_id} ({pattern.direction})")
            print(f"   Success Rate: {pattern.success_rate:.1f}%")
            print(f"   False Positive Rate: {pattern.false_positive_rate:.1f}%")
            print(f"   Stability: {pattern.stability_score:.3f}")
            print(f"   Composite Score: {pattern.composite_score:.3f}")
            print(f"   Status: {pattern.validation_status.value}")
    else:
        print("No patterns available for validation")
    
    print("\n" + "=" * 60)
    print("VALIDATION FRAMEWORK DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()