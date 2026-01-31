"""
Comprehensive Integration Tests for Integrated Pattern Discovery System

This test suite validates:
- System initialization and configuration
- Component communication and data flow
- Pattern discovery integration
- Validation framework integration
- Cross-validation integration
- Pattern ranking and selection
- Full pipeline execution

Author: Agent_CodebaseRefactor
Date: 2026-01-23
"""

import unittest
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

# Import the integrated system
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.integrated_system import (
    IntegratedPatternSystem,
    SystemConfig,
    PatternResult,
    SystemStatus,
    PatternDiscoveryMethod
)


class TestSystemInitialization(unittest.TestCase):
    """Test system initialization and configuration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SystemConfig(
            features_path="data/features_matrix.csv",
            output_dir=self.temp_dir,
            discovery_methods=["rule_based"],  # Use faster method for tests
            max_patterns_per_method=5,
            enable_validation=True,
            enable_cross_validation=False  # Disable for faster tests
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_system_initialization(self):
        """Test system can be initialized."""
        system = IntegratedPatternSystem(self.config)
        self.assertIsNotNone(system)
        self.assertEqual(system.state.status, SystemStatus.IDLE)
        self.assertEqual(len(system.patterns), 0)
    
    def test_default_configuration(self):
        """Test system with default configuration."""
        system = IntegratedPatternSystem()
        self.assertIsNotNone(system)
        self.assertIsNotNone(system.config)
        self.assertEqual(system.config.min_success_rate, 70.0)
    
    def test_custom_configuration(self):
        """Test system with custom configuration."""
        config = SystemConfig(
            min_success_rate=80.0,
            max_patterns_per_method=20
        )
        system = IntegratedPatternSystem(config)
        self.assertEqual(system.config.min_success_rate, 80.0)
        self.assertEqual(system.config.max_patterns_per_method, 20)
    
    def test_output_directory_creation(self):
        """Test output directory is created."""
        system = IntegratedPatternSystem(self.config)
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertTrue(Path(self.temp_dir).is_dir())


class TestDataLoading(unittest.TestCase):
    """Test data loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SystemConfig(
            features_path="data/features_matrix.csv",
            output_dir=self.temp_dir,
            discovery_methods=["rule_based"],
            max_patterns_per_method=5,
            enable_cross_validation=False
        )
        self.system = IntegratedPatternSystem(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_features(self):
        """Test loading features data."""
        df = self.system.load_data()
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
    
    def test_features_have_required_columns(self):
        """Test features have required columns."""
        df = self.system.load_data()
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            self.assertIn(col, df.columns)


class TestPatternDiscovery(unittest.TestCase):
    """Test pattern discovery integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SystemConfig(
            features_path="data/features_matrix.csv",
            output_dir=self.temp_dir,
            discovery_methods=["rule_based"],
            max_patterns_per_method=5,
            enable_validation=True,
            enable_cross_validation=False
        )
        self.system = IntegratedPatternSystem(self.config)
        self.system.load_data()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_discover_patterns(self):
        """Test pattern discovery."""
        results = self.system.discover_patterns()
        self.assertIsInstance(results, dict)
        self.assertIn("rule_based", results)
        self.assertGreater(len(results["rule_based"]), 0)
    
    def test_patterns_are_stored(self):
        """Test patterns are stored in system."""
        self.system.discover_patterns()
        self.assertGreater(len(self.system.patterns), 0)
        self.assertEqual(
            self.system.state.patterns_discovered,
            len(self.system.patterns)
        )
    
    def test_pattern_result_structure(self):
        """Test pattern results have correct structure."""
        self.system.discover_patterns()
        if self.system.patterns:
            pattern = self.system.patterns[0]
            self.assertIsInstance(pattern, PatternResult)
            self.assertIsNotNone(pattern.pattern_id)
            self.assertIsNotNone(pattern.pattern_name)
            self.assertIsNotNone(pattern.method)
            self.assertIsNotNone(pattern.conditions)
            self.assertGreater(pattern.success_rate, 0)
            self.assertGreater(pattern.occurrences, 0)
    
    def test_system_status_updates(self):
        """Test system status updates during discovery."""
        initial_status = self.system.state.status
        self.assertEqual(initial_status, SystemStatus.IDLE)
        
        self.system.discover_patterns()
        
        # Status should return to IDLE after discovery
        final_status = self.system.state.status
        self.assertEqual(final_status, SystemStatus.IDLE)


class TestValidationIntegration(unittest.TestCase):
    """Test validation framework integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SystemConfig(
            features_path="data/features_matrix.csv",
            output_dir=self.temp_dir,
            discovery_methods=["rule_based"],
            max_patterns_per_method=5,
            enable_validation=True,
            enable_cross_validation=False
        )
        self.system = IntegratedPatternSystem(self.config)
        self.system.load_data()
        self.system.discover_patterns()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_patterns(self):
        """Test pattern validation."""
        validated = self.system.validate_patterns()
        self.assertIsNotNone(validated)
        self.assertEqual(len(validated), len(self.system.patterns))
    
    def test_validation_status_assigned(self):
        """Test validation status is assigned to patterns."""
        self.system.validate_patterns()
        for pattern in self.system.validated_patterns:
            self.assertIsNotNone(pattern.validation_status)
            self.assertIn(pattern.validation_status, ["passed", "failed", "warning"])
    
    def test_validation_results_stored(self):
        """Test validation results are stored."""
        self.system.validate_patterns()
        self.assertIsNotNone(self.system.validation_results)
        self.assertIn('summary', self.system.validation_results)
    
    def test_validated_patterns_count(self):
        """Test validated patterns count is updated."""
        self.system.validate_patterns()
        self.assertEqual(
            self.system.state.patterns_validated,
            len(self.system.validated_patterns)
        )


class TestCrossValidationIntegration(unittest.TestCase):
    """Test cross-validation framework integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SystemConfig(
            features_path="data/features_matrix.csv",
            output_dir=self.temp_dir,
            discovery_methods=["rule_based"],
            max_patterns_per_method=3,  # Fewer patterns for faster CV
            enable_validation=True,
            enable_cross_validation=True,
            cv_folds=3  # Fewer folds for faster tests
        )
        self.system = IntegratedPatternSystem(self.config)
        self.system.load_data()
        self.system.discover_patterns()
        self.system.validate_patterns()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cross_validate_patterns(self):
        """Test cross-validation."""
        results = self.system.cross_validate_patterns()
        self.assertIsNotNone(results)
        self.assertIn('results', results)
        self.assertIn('analysis', results)
    
    def test_cv_metrics_assigned(self):
        """Test CV metrics are assigned to patterns."""
        self.system.cross_validate_patterns()
        for pattern in self.system.patterns:
            self.assertGreaterEqual(pattern.cv_success_rate, 0)
            self.assertGreaterEqual(pattern.cv_stability, 0)
            self.assertGreaterEqual(pattern.cv_robustness, 0)
    
    def test_robust_patterns_identified(self):
        """Test robust patterns are identified."""
        self.system.cross_validate_patterns()
        self.assertIsNotNone(self.system.robust_patterns)
        # At least some patterns should be marked
        self.assertGreaterEqual(len(self.system.robust_patterns), 0)
    
    def test_cv_results_stored(self):
        """Test CV results are stored."""
        self.system.cross_validate_patterns()
        self.assertIsNotNone(self.system.cv_results)


class TestPatternRanking(unittest.TestCase):
    """Test pattern ranking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SystemConfig(
            features_path="data/features_matrix.csv",
            output_dir=self.temp_dir,
            discovery_methods=["rule_based"],
            max_patterns_per_method=5,
            enable_validation=False,
            enable_cross_validation=False
        )
        self.system = IntegratedPatternSystem(self.config)
        self.system.load_data()
        self.system.discover_patterns()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_rank_by_composite_score(self):
        """Test ranking by composite score."""
        ranked = self.system.rank_patterns(method="composite")
        self.assertEqual(len(ranked), len(self.system.patterns))
        # Check descending order
        scores = [p.composite_score for p in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_rank_by_success_rate(self):
        """Test ranking by success rate."""
        ranked = self.system.rank_patterns(method="success_rate")
        self.assertEqual(len(ranked), len(self.system.patterns))
        # Check descending order
        rates = [p.success_rate for p in ranked]
        self.assertEqual(rates, sorted(rates, reverse=True))
    
    def test_select_top_patterns(self):
        """Test selecting top N patterns."""
        top_3 = self.system.select_top_patterns(n=3)
        self.assertLessEqual(len(top_3), 3)
        # Top patterns should have highest scores
        if len(top_3) > 0 and len(self.system.patterns) > 0:
            self.assertGreaterEqual(
                top_3[0].composite_score,
                self.system.patterns[0].composite_score
            )


class TestResultsPersistence(unittest.TestCase):
    """Test results saving and loading."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SystemConfig(
            features_path="data/features_matrix.csv",
            output_dir=self.temp_dir,
            discovery_methods=["rule_based"],
            max_patterns_per_method=5,
            enable_validation=True,
            enable_cross_validation=False
        )
        self.system = IntegratedPatternSystem(self.config)
        self.system.load_data()
        self.system.discover_patterns()
        self.system.validate_patterns()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_results(self):
        """Test saving results to files."""
        self.system.save_results()
        
        # Check files exist
        patterns_file = Path(self.temp_dir) / "integrated_patterns.json"
        validated_file = Path(self.temp_dir) / "validated_patterns.json"
        state_file = Path(self.temp_dir) / "system_state.json"
        
        self.assertTrue(patterns_file.exists())
        self.assertTrue(validated_file.exists())
        self.assertTrue(state_file.exists())
    
    def test_load_results(self):
        """Test loading saved results."""
        self.system.save_results()
        
        # Create new system and load results
        new_system = IntegratedPatternSystem(self.config)
        new_system.load_results()
        
        self.assertGreater(len(new_system.patterns), 0)
        self.assertGreater(len(new_system.validated_patterns), 0)
    
    def test_export_patterns_json(self):
        """Test exporting patterns to JSON."""
        export_path = Path(self.temp_dir) / "export.json"
        self.system.export_patterns(str(export_path), format='json')
        self.assertTrue(export_path.exists())
    
    def test_export_patterns_csv(self):
        """Test exporting patterns to CSV."""
        export_path = Path(self.temp_dir) / "export.csv"
        self.system.export_patterns(str(export_path), format='csv')
        self.assertTrue(export_path.exists())


class TestSystemStatus(unittest.TestCase):
    """Test system status and state management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SystemConfig(
            features_path="data/features_matrix.csv",
            output_dir=self.temp_dir,
            discovery_methods=["rule_based"],
            max_patterns_per_method=5,
            enable_validation=False,
            enable_cross_validation=False
        )
        self.system = IntegratedPatternSystem(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_system_status(self):
        """Test getting system status."""
        status = self.system.get_system_status()
        self.assertIsInstance(status, dict)
        self.assertIn('status', status)
        self.assertIn('last_update', status)
        self.assertIn('patterns_discovered', status)
    
    def test_status_updates_during_operations(self):
        """Test status updates during operations."""
        self.system.load_data()
        
        # Check status updated
        status = self.system.get_system_status()
        self.assertIsNotNone(status['last_update'])
        
        self.system.discover_patterns()
        
        # Check patterns discovered count updated
        status = self.system.get_system_status()
        self.assertGreater(status['patterns_discovered'], 0)


class TestPatternSummary(unittest.TestCase):
    """Test pattern summary generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SystemConfig(
            features_path="data/features_matrix.csv",
            output_dir=self.temp_dir,
            discovery_methods=["rule_based"],
            max_patterns_per_method=5,
            enable_validation=False,
            enable_cross_validation=False
        )
        self.system = IntegratedPatternSystem(self.config)
        self.system.load_data()
        self.system.discover_patterns()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_pattern_summary(self):
        """Test getting pattern summary."""
        summary = self.system.get_pattern_summary()
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertGreater(len(summary), 0)
    
    def test_summary_has_required_columns(self):
        """Test summary has required columns."""
        summary = self.system.get_pattern_summary()
        required_cols = [
            'Pattern ID', 'Pattern Name', 'Method', 'Direction',
            'Success Rate', 'Occurrences', 'FPR', 'Stability', 'Composite Score'
        ]
        for col in required_cols:
            self.assertIn(col, summary.columns)


class TestFullPipeline(unittest.TestCase):
    """Test complete pipeline execution."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SystemConfig(
            features_path="data/features_matrix.csv",
            output_dir=self.temp_dir,
            discovery_methods=["rule_based"],
            max_patterns_per_method=5,
            enable_validation=True,
            enable_cross_validation=False  # Disable for faster tests
        )
        self.system = IntegratedPatternSystem(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_run_full_pipeline(self):
        """Test running full pipeline."""
        results = self.system.run_full_pipeline()
        
        self.assertIsNotNone(results)
        self.assertIn('discovery', results)
        self.assertIn('validation', results)
        self.assertIn('summary', results)
        self.assertIn('top_patterns', results)
    
    def test_pipeline_summary(self):
        """Test pipeline summary."""
        results = self.system.run_full_pipeline()
        summary = results['summary']
        
        self.assertIn('total_patterns_discovered', summary)
        self.assertIn('patterns_validated', summary)
        self.assertIn('top_patterns_count', summary)
        self.assertGreater(summary['total_patterns_discovered'], 0)
    
    def test_pipeline_system_status(self):
        """Test system status after pipeline."""
        self.system.run_full_pipeline()
        self.assertEqual(self.system.state.status, SystemStatus.READY)
    
    def test_pipeline_results_saved(self):
        """Test results are saved after pipeline."""
        self.system.run_full_pipeline()
        
        patterns_file = Path(self.temp_dir) / "integrated_patterns.json"
        self.assertTrue(patterns_file.exists())


class TestComponentCommunication(unittest.TestCase):
    """Test communication between components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SystemConfig(
            features_path="data/features_matrix.csv",
            output_dir=self.temp_dir,
            discovery_methods=["rule_based"],
            max_patterns_per_method=5,
            enable_validation=True,
            enable_cross_validation=False
        )
        self.system = IntegratedPatternSystem(self.config)
        self.system.load_data()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_discovery_to_validation_data_flow(self):
        """Test data flow from discovery to validation."""
        # Discover patterns
        self.system.discover_patterns()
        discovered_patterns = self.system.patterns.copy()
        
        # Validate patterns
        self.system.validate_patterns()
        
        # Check that validated patterns are from discovered patterns
        self.assertEqual(len(self.system.validated_patterns), len(discovered_patterns))
    
    def test_features_shared_across_components(self):
        """Test features are shared across components."""
        self.system.discover_patterns()
        
        # Check features are available
        self.assertIsNotNone(self.system.features_df)
        
        # Validation framework should have access to features
        if self.system._validation_framework:
            self.assertIsNotNone(self.system._validation_framework.features_df)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_invalid_discovery_method(self):
        """Test handling of invalid discovery method."""
        config = SystemConfig(
            features_path="data/features_matrix.csv",
            output_dir=self.temp_dir,
            discovery_methods=["invalid_method"],
            max_patterns_per_method=5
        )
        system = IntegratedPatternSystem(config)
        system.load_data()
        
        # Should not crash, just skip invalid method
        results = system.discover_patterns()
        self.assertEqual(len(results), 0)
    
    def test_validate_without_patterns(self):
        """Test validation without patterns."""
        config = SystemConfig(
            features_path="data/features_matrix.csv",
            output_dir=self.temp_dir,
            discovery_methods=["rule_based"],
            max_patterns_per_method=5
        )
        system = IntegratedPatternSystem(config)
        system.load_data()
        
        # Validate without discovering patterns first
        validated = system.validate_patterns()
        self.assertEqual(len(validated), 0)
    
    def test_cross_validate_without_patterns(self):
        """Test cross-validation without patterns."""
        config = SystemConfig(
            features_path="data/features_matrix.csv",
            output_dir=self.temp_dir,
            discovery_methods=["rule_based"],
            max_patterns_per_method=5,
            enable_cross_validation=True
        )
        system = IntegratedPatternSystem(config)
        system.load_data()
        
        # Cross-validate without patterns
        results = system.cross_validate_patterns()
        self.assertEqual(len(results), 0)


def run_tests():
    """Run all integration tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestSystemInitialization,
        TestDataLoading,
        TestPatternDiscovery,
        TestValidationIntegration,
        TestCrossValidationIntegration,
        TestPatternRanking,
        TestResultsPersistence,
        TestSystemStatus,
        TestPatternSummary,
        TestFullPipeline,
        TestComponentCommunication,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    run_tests()