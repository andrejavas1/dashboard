"""
Integrated Pattern Discovery System

This module integrates all enhanced components into a cohesive pattern discovery system:
- Machine Learning pattern discovery
- Enhanced Rule-Based pattern discovery
- Pattern Validation Framework
- Cross-Validation Framework
- False Positive Reduction Validator

Provides unified APIs for component communication and data flow.

Author: Agent_CodebaseRefactor
Date: 2026-01-23
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PatternDiscoveryMethod(Enum):
    """Available pattern discovery methods."""
    ML_BASED = "ml_based"
    RULE_BASED = "rule_based"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"


class SystemStatus(Enum):
    """Status of the integrated system."""
    IDLE = "idle"
    DISCOVERING = "discovering"
    VALIDATING = "validating"
    TESTING = "testing"
    READY = "ready"
    ERROR = "error"


@dataclass
class SystemConfig:
    """Configuration for the integrated system."""
    # Data configuration
    features_path: str = "data/features_matrix.csv"
    output_dir: str = "data"
    
    # Pattern discovery settings
    discovery_methods: List[str] = field(default_factory=lambda: ["ml_based", "rule_based"])
    max_patterns_per_method: int = 50
    min_success_rate: float = 70.0
    min_occurrences: int = 20
    
    # Validation settings
    enable_validation: bool = True
    enable_cross_validation: bool = True
    enable_false_positive_reduction: bool = True
    
    # Cross-validation settings
    cv_folds: int = 5
    min_train_size: float = 0.4
    test_size: float = 0.2
    
    # ML settings
    ml_models: List[str] = field(default_factory=lambda: ["random_forest", "gradient_boosting"])
    n_features: int = 50
    feature_selection_method: str = "mutual_info"
    
    # Rule-based settings
    max_conditions: int = 5
    min_conditions: int = 2
    max_false_positive_rate: float = 15.0
    
    # Integration settings
    ensemble_method: str = "weighted"  # weighted, voting, stacking
    pattern_ranking_method: str = "composite"  # composite, success_rate, robustness


@dataclass
class PatternResult:
    """Result of pattern discovery."""
    pattern_id: str
    pattern_name: str
    method: str
    conditions: Dict
    direction: str
    label_col: str
    
    # Performance metrics
    success_rate: float
    occurrences: int
    false_positive_rate: float
    p_value: float
    stability_score: float
    regime_coverage: float
    composite_score: float
    
    # Cross-validation metrics
    cv_success_rate: float = 0.0
    cv_stability: float = 0.0
    cv_robustness: float = 0.0
    is_robust: bool = False
    
    # Validation status
    validation_status: str = "pending"
    
    # Metadata
    creation_date: str = ""
    source_file: str = ""


@dataclass
class SystemState:
    """Current state of the integrated system."""
    status: SystemStatus = SystemStatus.IDLE
    last_update: str = ""
    patterns_discovered: int = 0
    patterns_validated: int = 0
    patterns_robust: int = 0
    current_method: str = ""
    error_message: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'status': self.status.value,
            'last_update': self.last_update,
            'patterns_discovered': self.patterns_discovered,
            'patterns_validated': self.patterns_validated,
            'patterns_robust': self.patterns_robust,
            'current_method': self.current_method,
            'error_message': self.error_message
        }


class IntegratedPatternSystem:
    """
    Integrated pattern discovery system that combines all enhanced components.
    
    This system provides:
    - Unified API for pattern discovery
    - Multiple discovery methods (ML, rule-based, hybrid)
    - Integrated validation and cross-validation
    - Pattern ranking and selection
    - Comprehensive testing capabilities
    """
    
    def __init__(self, config: SystemConfig = None):
        """
        Initialize the integrated pattern discovery system.
        
        Args:
            config: System configuration
        """
        self.config = config or SystemConfig()
        self.state = SystemState()
        self.state.last_update = datetime.now().isoformat()
        
        # Data storage
        self.features_df = None
        self.patterns = []
        self.validated_patterns = []
        self.robust_patterns = []
        
        # Component instances (lazy loaded)
        self._ml_discovery = None
        self._rule_discovery = None
        self._validation_framework = None
        self._cross_validator = None
        self._fpr_validator = None
        
        # Results storage
        self.discovery_results = {}
        self.validation_results = {}
        self.cv_results = {}
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("Integrated Pattern Discovery System initialized")
        logger.info(f"  Discovery methods: {self.config.discovery_methods}")
        logger.info(f"  Output directory: {self.output_dir}")
    
    def load_data(self, features_path: str = None) -> pd.DataFrame:
        """
        Load features data for pattern discovery.
        
        Args:
            features_path: Path to features CSV file
            
        Returns:
            DataFrame with features
        """
        features_path = features_path or self.config.features_path
        
        logger.info(f"Loading features from {features_path}")
        self.features_df = pd.read_csv(features_path, index_col='Date', parse_dates=True)
        logger.info(f"Loaded {len(self.features_df)} records with {len(self.features_df.columns)} columns")
        
        return self.features_df
    
    def discover_patterns(self, methods: List[str] = None) -> Dict[str, List[PatternResult]]:
        """
        Discover patterns using specified methods.
        
        Args:
            methods: List of discovery methods to use
            
        Returns:
            Dictionary mapping method names to pattern results
        """
        if self.features_df is None:
            self.load_data()
        
        methods = methods or self.config.discovery_methods
        
        logger.info("=" * 60)
        logger.info("STARTING PATTERN DISCOVERY")
        logger.info("=" * 60)
        logger.info(f"Methods: {methods}")
        
        self.state.status = SystemStatus.DISCOVERING
        self.state.last_update = datetime.now().isoformat()
        
        all_results = {}
        
        for method in methods:
            logger.info(f"\n{'='*60}")
            logger.info(f"Method: {method}")
            logger.info(f"{'='*60}")
            
            self.state.current_method = method
            
            try:
                if method == "ml_based":
                    results = self._discover_ml_patterns()
                elif method == "rule_based":
                    results = self._discover_rule_based_patterns()
                elif method == "hybrid":
                    results = self._discover_hybrid_patterns()
                elif method == "ensemble":
                    results = self._discover_ensemble_patterns()
                else:
                    logger.warning(f"Unknown method: {method}")
                    continue
                
                all_results[method] = results
                self.discovery_results[method] = results
                
                logger.info(f"Discovered {len(results)} patterns using {method}")
                
            except Exception as e:
                logger.error(f"Error in {method} discovery: {e}")
                self.state.error_message = f"{method} error: {str(e)}"
        
        # Flatten results
        self.patterns = []
        for method_results in all_results.values():
            self.patterns.extend(method_results)
        
        self.state.patterns_discovered = len(self.patterns)
        self.state.status = SystemStatus.IDLE
        self.state.last_update = datetime.now().isoformat()
        
        logger.info(f"\n{'='*60}")
        logger.info("DISCOVERY COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total patterns discovered: {len(self.patterns)}")
        
        return all_results
    
    def _discover_ml_patterns(self) -> List[PatternResult]:
        """Discover patterns using ML-based approach."""
        from src.ml_pattern_discovery import MLPatternDiscovery
        
        if self._ml_discovery is None:
            self._ml_discovery = MLPatternDiscovery()
        
        # Load features
        self._ml_discovery.load_features()
        self._ml_discovery.create_target_variable()
        
        # Run discovery
        patterns = self._ml_discovery.discover_patterns_ml(
            model_type=self.config.ml_models[0],
            feature_selection_method=self.config.feature_selection_method,
            n_features=self.config.n_features
        )
        
        # Convert to PatternResult
        results = []
        for i, pattern in enumerate(patterns):
            result = PatternResult(
                pattern_id=f"ml_{i}",
                pattern_name=pattern.get('pattern_name', f"ML_Pattern_{i}"),
                method="ml_based",
                conditions=pattern.get('conditions', {}),
                direction="long",  # ML patterns default to long
                label_col="Label_5pct_10d",
                success_rate=pattern.get('success_rate', 0),
                occurrences=pattern.get('total_occurrences', 0),
                false_positive_rate=100 - pattern.get('success_rate', 0),
                p_value=0.05,  # Default
                stability_score=0.7,  # Default
                regime_coverage=1.0,  # Default
                composite_score=pattern.get('avg_confidence', 0),
                creation_date=datetime.now().isoformat(),
                source_file="ml_discovery"
            )
            results.append(result)
        
        return results[:self.config.max_patterns_per_method]
    
    def _discover_rule_based_patterns(self) -> List[PatternResult]:
        """Discover patterns using rule-based approach."""
        from src.enhanced_rule_based_patterns import EnhancedRuleBasedPatternDiscovery
        
        if self._rule_discovery is None:
            self._rule_discovery = EnhancedRuleBasedPatternDiscovery(self.features_df)
        
        # Discover patterns
        patterns = self._rule_discovery.discover_patterns(
            max_patterns=self.config.max_patterns_per_method
        )
        
        # Convert to PatternResult
        results = []
        for i, pattern in enumerate(patterns):
            pattern_data = pattern.get('pattern', {})
            result = PatternResult(
                pattern_id=f"rule_{i}",
                pattern_name=f"Rule_Pattern_{i}",
                method="rule_based",
                conditions=pattern_data.get('conditions', {}),
                direction=pattern_data.get('direction', 'long'),
                label_col=pattern_data.get('label_col', 'Label_5pct_10d'),
                success_rate=pattern.get('training_success_rate', 0),
                occurrences=pattern.get('occurrences', 0),
                false_positive_rate=pattern.get('false_positive_rate', 0),
                p_value=pattern.get('p_value', 0.5),
                stability_score=pattern.get('stability_score', 0.5),
                regime_coverage=pattern.get('regime_coverage', 1.0),
                composite_score=pattern.get('composite_score', 0),
                creation_date=datetime.now().isoformat(),
                source_file="rule_discovery"
            )
            results.append(result)
        
        return results
    
    def _discover_hybrid_patterns(self) -> List[PatternResult]:
        """Discover patterns using hybrid ML + rule-based approach."""
        # Get ML patterns
        ml_patterns = self._discover_ml_patterns()
        
        # Get rule-based patterns
        rule_patterns = self._discover_rule_based_patterns()
        
        # Combine and rank
        combined = ml_patterns + rule_patterns
        
        # Rank by composite score
        combined.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Mark as hybrid
        for pattern in combined:
            pattern.method = "hybrid"
            pattern.pattern_name = f"Hybrid_{pattern.pattern_name}"
        
        return combined[:self.config.max_patterns_per_method]
    
    def _discover_ensemble_patterns(self) -> List[PatternResult]:
        """Discover patterns using ensemble approach."""
        # Get patterns from all methods
        all_patterns = []
        
        if "ml_based" in self.config.discovery_methods:
            all_patterns.extend(self._discover_ml_patterns())
        
        if "rule_based" in self.config.discovery_methods:
            all_patterns.extend(self._discover_rule_based_patterns())
        
        # Weighted ensemble ranking
        if self.config.ensemble_method == "weighted":
            # Weight by method-specific metrics
            for pattern in all_patterns:
                if pattern.method == "ml_based":
                    pattern.composite_score *= 1.1  # Boost ML patterns
                elif pattern.method == "rule_based":
                    pattern.composite_score *= 1.0
        
        # Sort and select top patterns
        all_patterns.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Mark as ensemble
        for pattern in all_patterns:
            pattern.method = "ensemble"
            pattern.pattern_name = f"Ensemble_{pattern.pattern_name}"
        
        return all_patterns[:self.config.max_patterns_per_method]
    
    def validate_patterns(self, patterns: List[PatternResult] = None) -> List[PatternResult]:
        """
        Validate discovered patterns.
        
        Args:
            patterns: Patterns to validate (uses discovered patterns if None)
            
        Returns:
            Validated patterns
        """
        if patterns is None:
            patterns = self.patterns
        
        if not patterns:
            logger.warning("No patterns to validate")
            return []
        
        logger.info("=" * 60)
        logger.info("STARTING PATTERN VALIDATION")
        logger.info("=" * 60)
        logger.info(f"Patterns to validate: {len(patterns)}")
        
        self.state.status = SystemStatus.VALIDATING
        self.state.last_update = datetime.now().isoformat()
        
        from src.pattern_validation_framework import (
            PatternValidationFramework, 
            PatternType,
            ValidationStatus
        )
        
        if self._validation_framework is None:
            self._validation_framework = PatternValidationFramework(self.features_df)
        
        # Convert patterns to dict format for validation
        pattern_dicts = []
        for pattern in patterns:
            pattern_dict = {
                'conditions': pattern.conditions,
                'direction': pattern.direction,
                'label_col': pattern.label_col,
                'occurrences': pattern.occurrences,
                'success_rate': pattern.success_rate / 100,
                'false_positive_rate': pattern.false_positive_rate,
                'stability_score': pattern.stability_score,
                'avg_move': 0,  # Will be calculated
                'avg_time': 0
            }
            pattern_dicts.append(pattern_dict)
        
        # Determine pattern type
        pattern_type = PatternType.RULE_BASED
        if patterns and patterns[0].method in ["ml_based", "hybrid", "ensemble"]:
            pattern_type = PatternType.HYBRID
        
        # Run validation
        report = self._validation_framework.validate_patterns(
            pattern_dicts, 
            pattern_type,
            f"{patterns[0].method}_"
        )
        
        self.validation_results = report.to_dict()
        
        # Update patterns with validation results
        for i, pattern in enumerate(patterns):
            if i < len(report.patterns):
                metrics = report.patterns[i]
                pattern.validation_status = metrics.validation_status.value
                pattern.p_value = metrics.p_value
        
        self.validated_patterns = patterns
        self.state.patterns_validated = len(patterns)
        self.state.status = SystemStatus.IDLE
        self.state.last_update = datetime.now().isoformat()
        
        logger.info(f"\n{'='*60}")
        logger.info("VALIDATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Passed: {report.passed_patterns}/{len(patterns)}")
        logger.info(f"Failed: {report.failed_patterns}/{len(patterns)}")
        logger.info(f"Warnings: {report.warning_patterns}/{len(patterns)}")
        
        return patterns
    
    def cross_validate_patterns(self, patterns: List[PatternResult] = None) -> Dict:
        """
        Perform cross-validation on patterns.
        
        Args:
            patterns: Patterns to cross-validate
            
        Returns:
            Cross-validation results
        """
        if not self.config.enable_cross_validation:
            logger.info("Cross-validation disabled in config")
            return {}
        
        if patterns is None:
            patterns = self.validated_patterns if self.validated_patterns else self.patterns
        
        if not patterns:
            logger.warning("No patterns to cross-validate")
            return {}
        
        logger.info("=" * 60)
        logger.info("STARTING CROSS-VALIDATION")
        logger.info("=" * 60)
        logger.info(f"Patterns to cross-validate: {len(patterns)}")
        
        self.state.status = SystemStatus.TESTING
        self.state.last_update = datetime.now().isoformat()
        
        from src.cross_validation_framework import TimeSeriesCrossValidator
        
        if self._cross_validator is None:
            cv_config = {
                'n_folds': self.config.cv_folds,
                'min_train_size': self.config.min_train_size,
                'test_size': self.config.test_size
            }
            self._cross_validator = TimeSeriesCrossValidator(self.features_df, cv_config)
        
        # Convert patterns to dict format
        pattern_dicts = []
        for pattern in patterns:
            pattern_dict = {
                'pattern': {
                    'conditions': pattern.conditions,
                    'direction': pattern.direction,
                    'label_col': pattern.label_col,
                    'occurrences': pattern.occurrences,
                    'success_rate': pattern.success_rate / 100
                }
            }
            pattern_dicts.append(pattern_dict)
        
        # Run cross-validation
        results = self._cross_validator.run_cross_validation(pattern_dicts)
        
        self.cv_results = results
        
        # Update patterns with CV results
        for i, pattern in enumerate(patterns):
            pattern_id = f"pattern_{i}"
            if pattern_id in results['results']:
                cv_result = results['results'][pattern_id]
                pattern.cv_success_rate = cv_result.avg_out_sample_success_rate
                pattern.cv_stability = cv_result.stability_score
                pattern.cv_robustness = cv_result.robustness_score
                pattern.is_robust = cv_result.is_robust
        
        # Count robust patterns
        self.robust_patterns = [p for p in patterns if p.is_robust]
        self.state.patterns_robust = len(self.robust_patterns)
        
        self.state.status = SystemStatus.IDLE
        self.state.last_update = datetime.now().isoformat()
        
        logger.info(f"\n{'='*60}")
        logger.info("CROSS-VALIDATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Robust patterns: {len(self.robust_patterns)}/{len(patterns)}")
        
        return results
    
    def rank_patterns(self, patterns: List[PatternResult] = None, 
                     method: str = None) -> List[PatternResult]:
        """
        Rank patterns using specified method.
        
        Args:
            patterns: Patterns to rank
            method: Ranking method (composite, success_rate, robustness)
            
        Returns:
            Ranked patterns
        """
        if patterns is None:
            patterns = self.patterns
        
        method = method or self.config.pattern_ranking_method
        
        logger.info(f"Ranking {len(patterns)} patterns using method: {method}")
        
        if method == "composite":
            patterns.sort(key=lambda x: x.composite_score, reverse=True)
        elif method == "success_rate":
            patterns.sort(key=lambda x: x.success_rate, reverse=True)
        elif method == "robustness":
            patterns.sort(key=lambda x: x.cv_robustness, reverse=True)
        else:
            logger.warning(f"Unknown ranking method: {method}, using composite")
            patterns.sort(key=lambda x: x.composite_score, reverse=True)
        
        return patterns
    
    def select_top_patterns(self, n: int = 20) -> List[PatternResult]:
        """
        Select top N patterns from all discovered patterns.
        
        Args:
            n: Number of top patterns to select
            
        Returns:
            Top N patterns
        """
        # Rank patterns
        ranked = self.rank_patterns()
        
        # Filter for validated and robust patterns if available
        if self.validated_patterns:
            ranked = [p for p in ranked if p in self.validated_patterns]
        
        if self.robust_patterns:
            ranked = [p for p in ranked if p.is_robust]
        
        return ranked[:n]
    
    def run_full_pipeline(self) -> Dict:
        """
        Run the complete pattern discovery pipeline.
        
        Returns:
            Complete pipeline results
        """
        logger.info("=" * 60)
        logger.info("RUNNING FULL INTEGRATED PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Discover patterns
        discovery_results = self.discover_patterns()
        
        # Step 2: Validate patterns
        if self.config.enable_validation:
            validated_patterns = self.validate_patterns()
        else:
            validated_patterns = self.patterns
        
        # Step 3: Cross-validate patterns
        if self.config.enable_cross_validation:
            cv_results = self.cross_validate_patterns(validated_patterns)
        else:
            cv_results = {}
        
        # Step 4: Select top patterns
        top_patterns = self.select_top_patterns()
        
        # Step 5: Save results
        self.save_results()
        
        # Compile results
        results = {
            'discovery': discovery_results,
            'validation': self.validation_results,
            'cross_validation': self.cv_results,
            'top_patterns': [asdict(p) for p in top_patterns],
            'summary': {
                'total_patterns_discovered': self.state.patterns_discovered,
                'patterns_validated': self.state.patterns_validated,
                'patterns_robust': self.state.patterns_robust,
                'top_patterns_count': len(top_patterns),
                'status': self.state.to_dict()
            }
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total patterns discovered: {self.state.patterns_discovered}")
        logger.info(f"Patterns validated: {self.state.patterns_validated}")
        logger.info(f"Patterns robust: {self.state.patterns_robust}")
        logger.info(f"Top patterns selected: {len(top_patterns)}")
        
        self.state.status = SystemStatus.READY
        self.state.last_update = datetime.now().isoformat()
        
        return results
    
    def save_results(self) -> None:
        """Save all results to files."""
        logger.info("Saving results...")
        
        # Save patterns
        patterns_file = self.output_dir / "integrated_patterns.json"
        with open(patterns_file, 'w') as f:
            json.dump([asdict(p) for p in self.patterns], f, indent=2, default=str)
        logger.info(f"  Saved patterns to {patterns_file}")
        
        # Save validated patterns
        if self.validated_patterns:
            validated_file = self.output_dir / "validated_patterns.json"
            with open(validated_file, 'w') as f:
                json.dump([asdict(p) for p in self.validated_patterns], f, indent=2, default=str)
            logger.info(f"  Saved validated patterns to {validated_file}")
        
        # Save robust patterns
        if self.robust_patterns:
            robust_file = self.output_dir / "robust_patterns.json"
            with open(robust_file, 'w') as f:
                json.dump([asdict(p) for p in self.robust_patterns], f, indent=2, default=str)
            logger.info(f"  Saved robust patterns to {robust_file}")
        
        # Save validation results
        if self.validation_results:
            validation_file = self.output_dir / "validation_results.json"
            with open(validation_file, 'w') as f:
                json.dump(self.validation_results, f, indent=2, default=str)
            logger.info(f"  Saved validation results to {validation_file}")
        
        # Save CV results
        if self.cv_results:
            cv_file = self.output_dir / "cv_results.json"
            # Convert dataclasses to dict
            cv_dict = {}
            for k, v in self.cv_results.items():
                if k == 'results':
                    cv_dict[k] = {pk: asdict(pv) for pk, pv in v.items()}
                elif k == 'analysis':
                    cv_dict[k] = v
            with open(cv_file, 'w') as f:
                json.dump(cv_dict, f, indent=2, default=str)
            logger.info(f"  Saved CV results to {cv_file}")
        
        # Save system state
        state_file = self.output_dir / "system_state.json"
        with open(state_file, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)
        logger.info(f"  Saved system state to {state_file}")
    
    def load_results(self) -> None:
        """Load previously saved results."""
        logger.info("Loading saved results...")
        
        # Load patterns
        patterns_file = self.output_dir / "integrated_patterns.json"
        if patterns_file.exists():
            with open(patterns_file, 'r') as f:
                patterns_data = json.load(f)
                self.patterns = [PatternResult(**p) for p in patterns_data]
            logger.info(f"  Loaded {len(self.patterns)} patterns")
        
        # Load validated patterns
        validated_file = self.output_dir / "validated_patterns.json"
        if validated_file.exists():
            with open(validated_file, 'r') as f:
                validated_data = json.load(f)
                self.validated_patterns = [PatternResult(**p) for p in validated_data]
            logger.info(f"  Loaded {len(self.validated_patterns)} validated patterns")
        
        # Load robust patterns
        robust_file = self.output_dir / "robust_patterns.json"
        if robust_file.exists():
            with open(robust_file, 'r') as f:
                robust_data = json.load(f)
                self.robust_patterns = [PatternResult(**p) for p in robust_data]
            logger.info(f"  Loaded {len(self.robust_patterns)} robust patterns")
    
    def get_system_status(self) -> Dict:
        """Get current system status."""
        return self.state.to_dict()
    
    def get_pattern_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all patterns.
        
        Returns:
            DataFrame with pattern summary
        """
        if not self.patterns:
            return pd.DataFrame()
        
        data = []
        for pattern in self.patterns:
            data.append({
                'Pattern ID': pattern.pattern_id,
                'Pattern Name': pattern.pattern_name,
                'Method': pattern.method,
                'Direction': pattern.direction,
                'Success Rate': f"{pattern.success_rate:.1f}%",
                'Occurrences': pattern.occurrences,
                'FPR': f"{pattern.false_positive_rate:.1f}%",
                'Stability': f"{pattern.stability_score:.3f}",
                'Composite Score': f"{pattern.composite_score:.3f}",
                'Validated': pattern.validation_status,
                'Robust': 'Yes' if pattern.is_robust else 'No'
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('Composite Score', ascending=False)
    
    def export_patterns(self, output_path: str = None, format: str = 'json') -> None:
        """
        Export patterns to file.
        
        Args:
            output_path: Output file path
            format: Export format (json, csv)
        """
        if output_path is None:
            output_path = self.output_dir / f"patterns_export.{format}"
        else:
            output_path = Path(output_path)
        
        patterns_to_export = self.robust_patterns if self.robust_patterns else self.patterns
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump([asdict(p) for p in patterns_to_export], f, indent=2, default=str)
        elif format == 'csv':
            df = self.get_pattern_summary()
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Exported {len(patterns_to_export)} patterns to {output_path}")


def main():
    """Main function to run the integrated system."""
    print("=" * 60)
    print("Integrated Pattern Discovery System")
    print("=" * 60)
    
    # Initialize system
    config = SystemConfig(
        discovery_methods=["ml_based", "rule_based"],
        max_patterns_per_method=30,
        enable_validation=True,
        enable_cross_validation=True
    )
    
    system = IntegratedPatternSystem(config)
    
    # Run full pipeline
    results = system.run_full_pipeline()
    
    # Display summary
    print("\n" + "=" * 60)
    print("SYSTEM SUMMARY")
    print("=" * 60)
    
    summary = results['summary']
    print(f"\nTotal Patterns Discovered: {summary['total_patterns_discovered']}")
    print(f"Patterns Validated: {summary['patterns_validated']}")
    print(f"Patterns Robust: {summary['patterns_robust']}")
    print(f"Top Patterns Selected: {summary['top_patterns_count']}")
    
    # Display top patterns
    print("\n" + "=" * 60)
    print("TOP 10 PATTERNS")
    print("=" * 60)
    
    top_patterns = system.get_pattern_summary().head(10)
    print(top_patterns.to_string(index=False))
    
    # Export results
    system.export_patterns()
    
    print("\n" + "=" * 60)
    print("INTEGRATED SYSTEM COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()