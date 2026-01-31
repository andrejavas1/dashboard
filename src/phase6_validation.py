"""
Phase 6: Out-of-Sample Validation Module
Validates patterns on out-of-sample data to prevent overfitting.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OutOfSampleValidation:
    """
    Validates patterns on out-of-sample data.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the validation system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Validation parameters
        self.training_period = self.config['validation_periods']['training_period']
        self.validation_period = self.config['validation_periods']['validation_period']
        self.robust_threshold = self.config['validation_periods']['robust_threshold']
        self.degraded_threshold = self.config['validation_periods']['degraded_threshold']
        
        # Data storage
        self.data = None
        self.patterns = None
        self.validation_results = {}
        self.robust_patterns = []
        self.degraded_patterns = []
        self.failed_patterns = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'validation': {
                'training_period': {'start': '2010-01-01', 'end': '2020-12-31'},
                'validation_period': {'start': '2021-01-01', 'end': '2024-12-31'},
                'robust_threshold': 0.85,
                'degraded_threshold': 0.70
            }
        }
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load features matrix from file.
        
        Args:
            data_path: Path to features matrix CSV file
            
        Returns:
            DataFrame with features and labels
        """
        logger.info(f"Loading data from {data_path}")
        self.data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
        logger.info(f"Loaded {len(self.data)} records")
        return self.data
    
    def load_patterns(self, patterns_path: str) -> List[Dict]:
        """
        Load optimized patterns from file.
        
        Args:
            patterns_path: Path to patterns JSON file
            
        Returns:
            List of patterns
        """
        logger.info(f"Loading patterns from {patterns_path}")
        with open(patterns_path, 'r') as f:
            self.patterns = json.load(f)
        logger.info(f"Loaded {len(self.patterns)} patterns")
        return self.patterns
    
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training, validation, and live periods.
        
        Returns:
            Tuple of (training_data, validation_data, live_data)
        """
        logger.info("Splitting data into periods...")
        
        # Handle "current" keyword in validation period end date
        validation_end = self.validation_period['end']
        if validation_end == 'current':
            validation_end = datetime.now().strftime('%Y-%m-%d')
        
        # Training period (2010-2020)
        training_data = self.data.loc[self.training_period['start']:self.training_period['end']]
        
        # Validation period (2021-current)
        validation_data = self.data.loc[self.validation_period['start']:validation_end]
        
        # Live period (2025-present)
        live_start = '2025-01-01'
        live_data = self.data.loc[live_start:]
        
        logger.info(f"  Training: {len(training_data)} records ({self.training_period['start']} to {self.training_period['end']})")
        logger.info(f"  Validation: {len(validation_data)} records ({self.validation_period['start']} to {validation_end})")
        logger.info(f"  Live: {len(live_data)} records ({live_start} to present)")
        
        return training_data, validation_data, live_data
    
    def evaluate_pattern_on_period(self, pattern: Dict, data: pd.DataFrame) -> Optional[Dict]:
        """
        Evaluate a pattern on a specific time period.
        
        Args:
            pattern: Pattern to evaluate
            data: Data for the period
            
        Returns:
            Pattern evaluation results
        """
        conditions = pattern['conditions']
        label_col = pattern['label_col']
        direction = pattern.get('direction', 'unknown')
        
        # Build condition mask
        mask = pd.Series(True, index=data.index)
        
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
            elif condition['operator'] == 'range':
                # Explicit range operator with documented tolerance
                lower = condition.get('lower', condition['center'] * 0.9)
                upper = condition.get('upper', condition['center'] * 1.1)
                mask &= (data[feature] >= lower) & (data[feature] <= upper)
            elif condition['operator'] == '~':
                # Legacy operator - convert to range for backward compatibility
                mask &= (abs(data[feature] - condition['value']) < condition['value'] * 0.1)
        
        # Get occurrences
        occurrences = data[mask]
        
        if len(occurrences) < 10:
            return None
        
        # Calculate success rate based on direction
        if direction == 'long':
            success_mask = occurrences[label_col] == 'STRONG_UP'
            false_positive_mask = occurrences[label_col] == 'STRONG_DOWN'
        elif direction == 'short':
            success_mask = occurrences[label_col] == 'STRONG_DOWN'
            false_positive_mask = occurrences[label_col] == 'STRONG_UP'
        else:
            # Default to STRONG_UP for backward compatibility
            success_mask = occurrences[label_col] == 'STRONG_UP'
            false_positive_mask = occurrences[label_col] == 'STRONG_DOWN'
        
        success_count = success_mask.sum()
        success_rate = success_count / len(occurrences) * 100 if len(occurrences) > 0 else 0
        
        # Calculate average move and time
        if success_count > 0:
            parts = label_col.split('_')
            threshold = float(parts[1].replace('pct', ''))
            window = int(parts[2].replace('d', ''))
            
            if direction == 'long':
                max_up_col = f'Max_Up_{window}d'
                time_col = f'Time_To_Max_Up_{window}d'
            else:
                max_up_col = f'Max_Down_{window}d'
                time_col = f'Time_To_Max_Down_{window}d'
            
            avg_move = occurrences.loc[success_mask, max_up_col].mean()
            avg_time = occurrences.loc[success_mask, time_col].mean()
        else:
            avg_move = 0
            avg_time = 0
        
        # Calculate false positive rate
        false_positive_rate = false_positive_mask.sum() / len(occurrences) * 100 if len(occurrences) > 0 else 0
        
        return {
            'occurrences': len(occurrences),
            'success_count': success_count,
            'success_rate': success_rate,
            'avg_move': avg_move,
            'avg_time': avg_time,
            'false_positive_rate': false_positive_rate,
            'direction': direction
        }
    
    def validate_pattern(self, pattern: Dict, training_data: pd.DataFrame,
                        validation_data: pd.DataFrame, live_data: pd.DataFrame) -> Dict:
        """
        Validate a pattern on out-of-sample data.
        
        Args:
            pattern: Pattern to validate
            training_data: Training period data
            validation_data: Validation period data
            live_data: Live period data
            
        Returns:
            Validation results
        """
        # Get training success rate (from pattern)
        training_success_rate = pattern['success_rate']
        direction = pattern.get('direction', 'unknown')
        
        # Evaluate on validation data
        validation_results = self.evaluate_pattern_on_period(pattern, validation_data)
        
        # Evaluate on live data
        live_results = self.evaluate_pattern_on_period(pattern, live_data)
        
        if validation_results is None:
            # Pattern didn't occur in validation period
            return {
                'pattern': pattern,
                'training_success_rate': training_success_rate,
                'validation_success_rate': 0,
                'live_success_rate': 0,
                'validation_ratio': 0,
                'classification': 'FAILED',
                'validation_occurrences': 0,
                'live_occurrences': 0,
                'direction': direction
            }
        
        validation_success_rate = validation_results['success_rate']
        validation_ratio = validation_success_rate / training_success_rate if training_success_rate > 0 else 0
        
        # Classify validation result
        if validation_ratio >= self.robust_threshold:
            classification = 'ROBUST'
        elif validation_ratio >= self.degraded_threshold:
            classification = 'DEGRADED'
        else:
            classification = 'FAILED'
        
        # Get live results
        live_success_rate = live_results['success_rate'] if live_results else 0
        live_occurrences = live_results['occurrences'] if live_results else 0
        
        return {
            'pattern': pattern,
            'training_success_rate': training_success_rate,
            'validation_success_rate': validation_success_rate,
            'live_success_rate': live_success_rate,
            'validation_ratio': validation_ratio,
            'classification': classification,
            'validation_occurrences': validation_results['occurrences'],
            'live_occurrences': live_occurrences,
            'validation_avg_move': validation_results.get('avg_move', 0),
            'validation_avg_time': validation_results.get('avg_time', 0),
            'validation_false_positive_rate': validation_results.get('false_positive_rate', 0),
            'direction': direction
        }
    
    def statistical_tests(self, validation_result: Dict) -> Dict:
        """
        Perform statistical tests on validation results.
        
        Args:
            validation_result: Validation result dictionary
            
        Returns:
            Statistical test results
        """
        training_success_rate = validation_result['training_success_rate'] / 100
        validation_occurrences = validation_result['validation_occurrences']
        validation_success_count = int(validation_result['validation_occurrences'] * 
                                       validation_result['validation_success_rate'] / 100)
        
        # Binomial test: Is success rate significantly different from random (50%)?
        if validation_occurrences > 0:
            p_value_binomial = stats.binomtest(
                validation_success_count,
                validation_occurrences,
                p=0.5,
                alternative='greater'
            ).pvalue
        else:
            p_value_binomial = 1.0
        
        # Chi-square test: Compare training vs validation
        if validation_occurrences > 0:
            training_success_count = int(validation_result['pattern']['occurrences'] * 
                                         validation_result['training_success_rate'] / 100)
            training_failures = validation_result['pattern']['occurrences'] - training_success_count
            validation_failures = validation_occurrences - validation_success_count
            
            if training_failures > 0 and validation_failures > 0:
                observed = np.array([[training_success_count, training_failures],
                                    [validation_success_count, validation_failures]])
                
                try:
                    chi2, p_value_chi2, dof, expected = stats.chi2_contingency(observed)
                except:
                    p_value_chi2 = 1.0
            else:
                p_value_chi2 = 1.0
        else:
            p_value_chi2 = 1.0
        
        return {
            'binomial_p_value': p_value_binomial,
            'chi_square_p_value': p_value_chi2,
            'statistically_significant': p_value_binomial < 0.05
        }
    
    def validate_all_patterns(self) -> Dict[str, List[Dict]]:
        """
        Validate all patterns on out-of-sample data.
        
        Returns:
            Dictionary with classified patterns
        """
        logger.info("=" * 60)
        logger.info("PHASE 6: Out-of-Sample Validation")
        logger.info("=" * 60)
        
        if self.data is None or self.patterns is None:
            logger.error("Data or patterns not loaded")
            return {}
        
        # Split data
        training_data, validation_data, live_data = self.split_data()
        
        # Validate each pattern
        logger.info(f"\nValidating {len(self.patterns)} patterns...")
        
        robust = []
        degraded = []
        failed = []
        
        for pattern in tqdm(self.patterns, desc="Validating patterns"):
            # Validate pattern
            result = self.validate_pattern(pattern, training_data, validation_data, live_data)
            
            # Perform statistical tests
            stats_results = self.statistical_tests(result)
            result.update(stats_results)
            
            # Add to validation results
            pattern_id = self._get_pattern_id(pattern)
            self.validation_results[pattern_id] = result
            
            # Classify
            if result['classification'] == 'ROBUST':
                robust.append(result)
            elif result['classification'] == 'DEGRADED':
                degraded.append(result)
            else:
                failed.append(result)
        
        self.robust_patterns = robust
        self.degraded_patterns = degraded
        self.failed_patterns = failed
        
        # Log summary
        logger.info(f"\nValidation Summary:")
        logger.info(f"  ROBUST: {len(robust)} patterns (validation >= {self.robust_threshold * 100:.0f}% of training)")
        logger.info(f"  DEGRADED: {len(degraded)} patterns (validation {self.degraded_threshold * 100:.0f}-{self.robust_threshold * 100:.0f}% of training)")
        logger.info(f"  FAILED: {len(failed)} patterns (validation < {self.degraded_threshold * 100:.0f}% of training)")
        
        # Calculate average metrics
        if robust:
            avg_validation_rate = np.mean([p['validation_success_rate'] for p in robust])
            logger.info(f"\nRobust Patterns Average Validation Success Rate: {avg_validation_rate:.2f}%")
        
        return {
            'robust': robust,
            'degraded': degraded,
            'failed': failed
        }
    
    def _get_pattern_id(self, pattern: Dict) -> str:
        """
        Generate a unique ID for a pattern.
        
        Args:
            pattern: Pattern dictionary
            
        Returns:
            Pattern ID string
        """
        conditions_str = json.dumps(pattern['conditions'], sort_keys=True)
        return str(hash(conditions_str))
    
    def recent_performance_check(self) -> Dict:
        """
        Check recent performance (2024-present) separately.
        
        Returns:
            Recent performance summary
        """
        logger.info("\nChecking recent performance (2024-present)...")
        
        recent_data = self.data.loc['2024-01-01':]
        
        recent_results = []
        
        for pattern in self.robust_patterns + self.degraded_patterns:
            result = self.evaluate_pattern_on_period(pattern['pattern'], recent_data)
            
            if result:
                recent_results.append({
                    'pattern_id': self._get_pattern_id(pattern['pattern']),
                    'recent_success_rate': result['success_rate'],
                    'recent_occurrences': result['occurrences'],
                    'training_success_rate': pattern['training_success_rate'],
                    'validation_success_rate': pattern['validation_success_rate']
                })
        
        # Calculate summary
        if recent_results:
            avg_recent_rate = np.mean([r['recent_success_rate'] for r in recent_results])
            logger.info(f"  Average recent success rate: {avg_recent_rate:.2f}%")
            logger.info(f"  Patterns with recent data: {len(recent_results)}")
        
        return {
            'recent_results': recent_results,
            'average_recent_success_rate': np.mean([r['recent_success_rate'] for r in recent_results]) if recent_results else 0
        }
    
    def save_validation_results(self, output_dir: str = "data"):
        """
        Save validation results to files.
        
        Args:
            output_dir: Directory to save output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all validation results
        results_path = os.path.join(output_dir, "validation_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        logger.info(f"Validation results saved to {results_path}")
        
        # Save classified patterns
        classified = {
            'robust': self.robust_patterns,
            'degraded': self.degraded_patterns,
            'failed': self.failed_patterns
        }
        
        classified_path = os.path.join(output_dir, "validated_patterns.json")
        with open(classified_path, 'w') as f:
            json.dump(classified, f, indent=2, default=str)
        logger.info(f"Validated patterns saved to {classified_path}")
        
        # Save recent performance
        recent = self.recent_performance_check()
        recent_path = os.path.join(output_dir, "recent_performance.json")
        with open(recent_path, 'w') as f:
            json.dump(recent, f, indent=2, default=str)
        logger.info(f"Recent performance saved to {recent_path}")
    
    def run_phase6(self, data_path: str = None, patterns_path: str = None) -> Dict[str, List[Dict]]:
        """
        Run complete Phase 6: Out-of-Sample Validation.
        
        Args:
            data_path: Path to features matrix CSV file
            patterns_path: Path to patterns JSON file
            
        Returns:
            Dictionary with validated patterns
        """
        logger.info("\n" + "=" * 60)
        logger.info("STARTING PHASE 6: OUT-OF-SAMPLE VALIDATION")
        logger.info("=" * 60)
        
        # Load data
        if data_path:
            self.load_data(data_path)
        elif self.data is None:
            default_path = os.path.join("data", "features_matrix.csv")
            if os.path.exists(default_path):
                self.load_data(default_path)
            else:
                logger.error("No data path provided and default file not found")
                return {}
        
        # Load patterns
        if patterns_path:
            self.load_patterns(patterns_path)
        elif self.patterns is None:
            default_path = os.path.join("data", "optimized_patterns.json")
            if os.path.exists(default_path):
                self.load_patterns(default_path)
            else:
                logger.error("No patterns path provided and default file not found")
                return {}
        
        # Validate patterns
        classified = self.validate_all_patterns()
        
        # Check recent performance
        self.recent_performance_check()
        
        # Save results
        self.save_validation_results()
        
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 6 COMPLETE")
        logger.info("=" * 60)
        
        return classified


if __name__ == "__main__":
    # Run Phase 6
    ov = OutOfSampleValidation()
    classified = ov.run_phase6()
    
    print(f"\nFinal Results:")
    print(f"  Robust Patterns: {len(classified['robust'])}")
    print(f"  Degraded Patterns: {len(classified['degraded'])}")
    print(f"  Failed Patterns: {len(classified['failed'])}")