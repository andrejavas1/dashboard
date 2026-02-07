"""
Phase 5: Pattern Optimization & Refinement Module
Optimizes discovered patterns and creates regime-specific pattern sets.
"""

import os
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from itertools import product
from functools import lru_cache
from pattern_filter import PatternFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PatternOptimization:
    """
    Optimizes and refines discovered patterns for better performance.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the pattern optimization system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.run_mode = self.config.get('run_mode', 'full')
        
        # Parameters
        self.time_windows = self.config['movement']['time_windows']
        self.thresholds = self.config['movement']['thresholds']
        
        # Adjust parameters for quick/ultra mode
        if self.run_mode == 'ultra':
            # Skip optimization entirely in ultra mode
            logger.info("Ultra Mode: Pattern optimization will be skipped")
        elif self.run_mode == 'quick':
            # Reduce optimization grid in quick mode
            logger.info("Quick Mode: Reduced pattern optimization")
        
        # Data storage
        self.data = None
        self.patterns = None
        self.optimized_patterns = []
        self.regime_patterns = {}
        
        # Cache for pattern evaluations to avoid repeated calculations
        self._evaluation_cache = {}
        
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
            'movement': {
                'time_windows': [3, 5, 10, 20, 30],
                'thresholds': [1, 2, 3, 5, 7, 10]
            },
            'validation_periods': {
                'training_period': {'start': '2010-01-01', 'end': '2020-12-31'},
                'validation_period': {'start': '2021-01-01', 'end': '2024-12-31'}
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
        Load discovered patterns from file.
        
        Args:
            patterns_path: Path to patterns JSON file
            
        Returns:
            List of patterns
        """
        logger.info(f"Loading patterns from {patterns_path}")
        with open(patterns_path, 'r') as f:
            self.patterns = json.load(f)
        
        # If loaded classified patterns, extract high and medium confidence
        if isinstance(self.patterns, dict):
            high_patterns = self.patterns.get('high_confidence', [])
            medium_patterns = self.patterns.get('medium_confidence', [])
            self.patterns = high_patterns + medium_patterns
        
        logger.info(f"Loaded {len(self.patterns)} patterns")
        return self.patterns
    
    def parameter_optimization(self, pattern: Dict) -> Dict:
        """
        Optimize threshold values for each feature in pattern.
        
        Args:
            pattern: Pattern to optimize
            
        Returns:
            Optimized pattern
        """
        label_col = pattern['label_col']
        conditions = pattern['conditions']
        direction = pattern.get('direction', 'unknown')
        
        best_pattern = pattern.copy()
        best_score = pattern['success_rate']
        
        # Grid search around current thresholds
        for feature, condition in conditions.items():
            if feature not in self.data.columns:
                continue
            
            original_operator = condition['operator']
            
            # Handle different operator types
            if original_operator == 'range':
                # For range operator, use center value
                original_value = condition.get('center', condition.get('value', 0))
                test_values = np.linspace(original_value * 0.9, original_value * 1.1, 3)
                
                for test_value in test_values:
                    new_conditions = conditions.copy()
                    new_conditions[feature] = {
                        'operator': 'range',
                        'lower': test_value * 0.9,
                        'upper': test_value * 1.1,
                        'center': test_value,
                        'tolerance_pct': 10.0
                    }
                    
                    # Evaluate new pattern
                    new_pattern = self._evaluate_pattern(new_conditions, label_col, method=pattern.get('method', 'optimized'), direction=direction)
                    
                    if new_pattern and new_pattern['success_rate'] > best_score:
                        best_score = new_pattern['success_rate']
                        best_pattern = new_pattern
            else:
                # For standard operators (>=, <=, >, <)
                original_value = condition['value']
                test_values = np.linspace(original_value * 0.8, original_value * 1.2, 5)
                
                for test_value in test_values:
                    new_conditions = conditions.copy()
                    new_conditions[feature] = {'operator': original_operator, 'value': test_value}
                    
                    # Evaluate new pattern
                    new_pattern = self._evaluate_pattern(new_conditions, label_col, method=pattern.get('method', 'optimized'), direction=direction)
                    
                    if new_pattern and new_pattern['success_rate'] > best_score:
                        best_score = new_pattern['success_rate']
                        best_pattern = new_pattern
        
        return best_pattern
    
    def _evaluate_pattern(self, conditions: Dict, label_col: str, method: str = 'optimized', direction: str = 'unknown') -> Optional[Dict]:
        """
        Evaluate a pattern's performance with caching.
        
        Args:
            conditions: Dictionary of feature conditions
            label_col: Label column to predict
            method: Method name for the pattern
            direction: Pattern direction (long/short)
            
        Returns:
            Pattern dictionary with performance metrics
        """
        # Create cache key from conditions, label_col, and direction
        # Convert conditions to a hashable format
        conditions_key = []
        for feature, condition in sorted(conditions.items()):
            if isinstance(condition, dict):
                # Convert nested dict to tuple
                condition_key = (feature, tuple(sorted(condition.items())))
            else:
                condition_key = (feature, condition)
            conditions_key.append(condition_key)
        
        cache_key = (
            tuple(conditions_key),
            label_col,
            direction
        )
        
        # Check cache first
        if cache_key in self._evaluation_cache:
            cached_result = self._evaluation_cache[cache_key]
            # Return a copy to avoid modifying cached data
            return cached_result.copy() if cached_result else None
        
        # Build condition mask
        mask = pd.Series(True, index=self.data.index)
        
        for feature, condition in conditions.items():
            if feature not in self.data.columns:
                continue
            
            if condition['operator'] == '>=':
                mask &= (self.data[feature] >= condition['value'])
            elif condition['operator'] == '<=':
                mask &= (self.data[feature] <= condition['value'])
            elif condition['operator'] == '>':
                mask &= (self.data[feature] > condition['value'])
            elif condition['operator'] == '<':
                mask &= (self.data[feature] < condition['value'])
            elif condition['operator'] == 'range':
                # Explicit range operator with documented tolerance
                lower = condition.get('lower', condition['center'] * 0.9)
                upper = condition.get('upper', condition['center'] * 1.1)
                mask &= (self.data[feature] >= lower) & (self.data[feature] <= upper)
            elif condition['operator'] == '~':
                # Legacy operator - convert to range for backward compatibility
                mask &= (abs(self.data[feature] - condition['value']) < condition['value'] * 0.1)
        
        # Get occurrences
        occurrences = self.data[mask]
        
        if len(occurrences) < 30:
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
        success_rate = success_count / len(occurrences) * 100
        
        # Calculate average move and time
        if success_count > 0:
            # Extract window and threshold from label_col
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
        false_positive_rate = false_positive_mask.sum() / len(occurrences) * 100
        
        pattern = {
            'conditions': conditions,
            'occurrences': len(occurrences),
            'success_count': success_count,
            'success_rate': success_rate,
            'avg_move': avg_move,
            'avg_time': avg_time,
            'false_positive_rate': false_positive_rate,
            'p_value': 0,
            'label_col': label_col,
            'method': method,
            'direction': direction
        }
        
        # Cache the result
        self._evaluation_cache[cache_key] = pattern.copy()
        
        return pattern
    
    def time_window_optimization(self, pattern: Dict) -> Dict:
        """
        Test different forward-looking windows to find optimal.
        
        Args:
            pattern: Pattern to optimize
            
        Returns:
            Optimized pattern with best window
        """
        original_label = pattern['label_col']
        conditions = pattern['conditions']
        direction = pattern.get('direction', 'unknown')
        
        best_pattern = pattern.copy()
        best_score = pattern['success_rate']
        
        # Extract original threshold
        parts = original_label.split('_')
        threshold = float(parts[1].replace('pct', ''))
        
        # Test different windows
        for window in self.time_windows:
            new_label = f'Label_{threshold}pct_{window}d'
            
            if new_label not in self.data.columns:
                continue
            
            new_pattern = self._evaluate_pattern(conditions, new_label, method=pattern.get('method', 'optimized'), direction=direction)
            
            if new_pattern and new_pattern['success_rate'] > best_score:
                best_score = new_pattern['success_rate']
                best_pattern = new_pattern
        
        return best_pattern
    
    def threshold_optimization(self, pattern: Dict) -> Dict:
        """
        Test different target movement percentages.
        
        Args:
            pattern: Pattern to optimize
            
        Returns:
            Optimized pattern with best threshold
        """
        original_label = pattern['label_col']
        conditions = pattern['conditions']
        direction = pattern.get('direction', 'unknown')
        
        best_pattern = pattern.copy()
        best_score = pattern['success_rate']
        
        # Extract original window
        parts = original_label.split('_')
        window = int(parts[2].replace('d', ''))
        
        # Test different thresholds
        for threshold in self.thresholds:
            new_label = f'Label_{threshold}pct_{window}d'
            
            if new_label not in self.data.columns:
                continue
            
            new_pattern = self._evaluate_pattern(conditions, new_label, method=pattern.get('method', 'optimized'), direction=direction)
            
            if new_pattern and new_pattern['success_rate'] > best_score:
                best_score = new_pattern['success_rate']
                best_pattern = new_pattern
        
        return best_pattern
    
    def compound_pattern_creation(self, patterns: List[Dict]) -> List[Dict]:
        """
        Combine multiple medium-confidence patterns.
        
        Args:
            patterns: List of patterns to combine
            
        Returns:
            List of compound patterns
        """
        compound_patterns = []
        
        # Only combine patterns with same label
        label_groups = {}
        for pattern in patterns:
            label = pattern['label_col']
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(pattern)
        
        # Create OR combinations
        for label, label_patterns in label_groups.items():
            if len(label_patterns) < 2:
                continue
            
            # Combine top 2 patterns
            sorted_patterns = sorted(label_patterns, key=lambda x: x['success_rate'], reverse=True)
            
            for i in range(min(3, len(sorted_patterns))):
                for j in range(i + 1, min(3, len(sorted_patterns))):
                    # OR combination
                    combined_conditions = {
                        f'Pattern_{i}': sorted_patterns[i]['conditions'],
                        f'Pattern_{j}': sorted_patterns[j]['conditions']
                    }
                    
                    compound_pattern = {
                        'conditions': combined_conditions,
                        'type': 'OR',
                        'component_patterns': [sorted_patterns[i], sorted_patterns[j]],
                        'label_col': label,
                        'method': 'compound'
                    }
                    
                    # Evaluate compound pattern
                    evaluated = self._evaluate_compound_pattern(compound_pattern)
                    if evaluated:
                        compound_patterns.append(evaluated)
        
        return compound_patterns
    
    def _evaluate_compound_pattern(self, pattern: Dict) -> Optional[Dict]:
        """
        Evaluate a compound pattern's performance.
        
        Args:
            pattern: Compound pattern to evaluate
            
        Returns:
            Pattern dictionary with performance metrics
        """
        label_col = pattern['label_col']
        component_patterns = pattern['component_patterns']
        
        # Build condition mask for OR combination
        mask = pd.Series(False, index=self.data.index)
        
        for comp_pattern in component_patterns:
            comp_mask = pd.Series(True, index=self.data.index)
            
            for feature, condition in comp_pattern['conditions'].items():
                if feature not in self.data.columns:
                    continue
                
                if condition['operator'] == '>=':
                    comp_mask &= (self.data[feature] >= condition['value'])
                elif condition['operator'] == '<=':
                    comp_mask &= (self.data[feature] <= condition['value'])
                elif condition['operator'] == '>':
                    comp_mask &= (self.data[feature] > condition['value'])
                elif condition['operator'] == '<':
                    comp_mask &= (self.data[feature] < condition['value'])
            
            mask |= comp_mask
        
        # Get occurrences
        occurrences = self.data[mask]
        
        if len(occurrences) < 30:
            return None
        
        # Calculate success rate
        success_mask = occurrences[label_col] == 'STRONG_UP'
        success_count = success_mask.sum()
        success_rate = success_count / len(occurrences) * 100
        
        # Calculate average move and time
        if success_count > 0:
            parts = label_col.split('_')
            threshold = float(parts[1].replace('pct', ''))
            window = int(parts[2].replace('d', ''))
            
            max_up_col = f'Max_Up_{window}d'
            time_col = f'Time_To_Max_Up_{window}d'
            
            avg_move = occurrences.loc[success_mask, max_up_col].mean()
            avg_time = occurrences.loc[success_mask, time_col].mean()
        else:
            avg_move = 0
            avg_time = 0
        
        pattern['occurrences'] = len(occurrences)
        pattern['success_count'] = success_count
        pattern['success_rate'] = success_rate
        pattern['avg_move'] = avg_move
        pattern['avg_time'] = avg_time
        pattern['false_positive_rate'] = 0
        
        return pattern
    
    def regime_specific_optimization(self, patterns: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Optimize patterns separately for each market regime.
        
        Args:
            patterns: List of patterns to optimize
            
        Returns:
            Dictionary of regime-specific patterns
        """
        regime_patterns = {
            'strong_bull': [],
            'weak_bull': [],
            'sideways': [],
            'weak_bear': [],
            'strong_bear': []
        }
        
        # Split data by regime
        if 'Trend_Regime' not in self.data.columns:
            logger.warning("Trend_Regime column not found, skipping regime optimization")
            return regime_patterns
        
        # Optimize patterns for each regime
        for pattern in patterns:
            conditions = pattern['conditions']
            label_col = pattern['label_col']
            
            for regime in regime_patterns.keys():
                # Filter data by regime
                regime_data = self.data[self.data['Trend_Regime'] == regime.replace('_', ' ').title()]
                
                if len(regime_data) < 100:
                    continue
                
                # Evaluate pattern on regime data
                regime_pattern = self._evaluate_pattern_on_data(conditions, label_col, regime_data)
                
                if regime_pattern and regime_pattern['success_rate'] >= 60:
                    regime_pattern['regime'] = regime
                    regime_patterns[regime].append(regime_pattern)
        
        logger.info("\nRegime-Specific Pattern Counts:")
        for regime, patterns in regime_patterns.items():
            logger.info(f"  {regime}: {len(patterns)} patterns")
        
        self.regime_patterns = regime_patterns
        return regime_patterns
    
    def _evaluate_pattern_on_data(self, conditions: Dict, label_col: str, data: pd.DataFrame) -> Optional[Dict]:
        """
        Evaluate a pattern on specific data subset.
        
        Args:
            conditions: Dictionary of feature conditions
            label_col: Label column to predict
            data: Data subset to evaluate on
            
        Returns:
            Pattern dictionary with performance metrics
        """
        # Build condition mask
        mask = pd.Series(True, index=data.index)
        
        for feature, condition in conditions.items():
            if feature not in data.columns:
                continue
            
            if condition['operator'] == '>=':
                mask &= (data[feature] >= condition['value'])
            elif condition['operator'] == '<=':
                mask &= (data[feature] <= condition['value'])
            elif condition['operator'] == '>':
                mask &= (data[feature] > condition['value'])
            elif condition['operator'] == '<':
                mask &= (data[feature] < condition['value'])
        
        # Get occurrences
        occurrences = data[mask]
        
        if len(occurrences) < 20:
            return None
        
        # Calculate success rate
        success_mask = occurrences[label_col] == 'STRONG_UP'
        success_count = success_mask.sum()
        success_rate = success_count / len(occurrences) * 100 if len(occurrences) > 0 else 0
        
        pattern = {
            'conditions': conditions,
            'occurrences': len(occurrences),
            'success_count': success_count,
            'success_rate': success_rate,
            'avg_move': 0,
            'avg_time': 0,
            'false_positive_rate': 0,
            'label_col': label_col,
            'method': 'regime_optimized'
        }
        
        return pattern
    
    def temporal_stability_testing(self, patterns: List[Dict]) -> List[Dict]:
        """
        Test pattern stability across time periods.
        
        Args:
            patterns: List of patterns to test
            
        Returns:
            List of patterns with stability scores
        """
        stable_patterns = []
        
        # Define time periods
        periods = [
            ('2010-2015', '2010-01-01', '2015-12-31'),
            ('2016-2020', '2016-01-01', '2020-12-31'),
            ('2021-2024', '2021-01-01', '2024-12-31')
        ]
        
        for pattern in patterns:
            conditions = pattern['conditions']
            label_col = pattern['label_col']
            
            success_rates = []
            
            for period_name, start, end in periods:
                period_data = self.data.loc[start:end]
                
                if len(period_data) < 100:
                    continue
                
                period_pattern = self._evaluate_pattern_on_data(conditions, label_col, period_data)
                
                if period_pattern:
                    success_rates.append(period_pattern['success_rate'])
            
            # Calculate stability score (standard deviation of success rates)
            if len(success_rates) >= 2:
                stability_score = np.std(success_rates)
                pattern['stability_score'] = stability_score
                pattern['period_success_rates'] = success_rates
                
                # Keep patterns with reasonable stability
                if stability_score < 15:
                    stable_patterns.append(pattern)
        
        logger.info(f"Temporal stability testing: {len(stable_patterns)} stable patterns")
        return stable_patterns
    
    def optimize_all_patterns(self) -> List[Dict]:
        """
        Run all optimization methods on patterns.

        Returns:
            List of optimized patterns
        """
        logger.info("=" * 60)
        logger.info("PHASE 5: Pattern Optimization & Refinement")
        logger.info("=" * 60)
        
        if self.patterns is None or len(self.patterns) == 0:
            logger.error("No patterns loaded. Call load_patterns() first.")
            return []
        
        # Ultra mode: skip optimization entirely, just pass through patterns
        if self.run_mode == 'ultra':
            logger.info("Ultra Mode: Skipping optimization, passing patterns through")
            self.optimized_patterns = self.patterns.copy()
            return self.optimized_patterns
        
        optimized = []
        
        # Clear cache before optimization
        self._evaluation_cache.clear()
        
        # Limit patterns for optimization in quick mode
        patterns_to_optimize = self.patterns
        if self.run_mode == 'quick' and len(self.patterns) > 50:
            logger.info(f"Quick Mode: Limiting to top 50 patterns (from {len(self.patterns)})")
            # Sort by success rate and take top 50
            patterns_to_optimize = sorted(
                self.patterns, 
                key=lambda p: p.get('success_rate', 0), 
                reverse=True
            )[:50]
        
        # Optimize each pattern
        logger.info(f"\nOptimizing {len(patterns_to_optimize)} patterns...")
        
        for pattern in tqdm(patterns_to_optimize, desc="Optimizing patterns"):
            # Preserve direction from original pattern
            direction = pattern.get('direction', 'unknown')
            
            # Parameter optimization
            pattern = self.parameter_optimization(pattern)
            pattern['direction'] = direction  # Ensure direction is preserved
            
            # Time window optimization
            pattern = self.time_window_optimization(pattern)
            pattern['direction'] = direction  # Ensure direction is preserved
            
            # Threshold optimization
            pattern = self.threshold_optimization(pattern)
            pattern['direction'] = direction  # Ensure direction is preserved
            
            optimized.append(pattern)
        
        # Remove duplicates (basic deduplication based on conditions only)
        optimized = self._deduplicate_patterns(optimized)
        
        logger.info(f"Parameter optimization complete: {len(optimized)} patterns")
        
        # Apply advanced pattern filtering to remove duplicates and subsets
        logger.info("\n" + "=" * 60)
        logger.info("APPLYING PATTERN FILTERING")
        logger.info("=" * 60)
        pattern_filter = PatternFilter()
        optimized = pattern_filter.filter_patterns(
            optimized,
            remove_exact_duplicates=True,
            remove_condition_duplicates=True,
            remove_metric_duplicates=True,
            remove_subsets=True
        )
        
        # Log filter statistics
        stats = pattern_filter.get_filter_stats()
        logger.info(f"\nPattern Filtering Statistics:")
        logger.info(f"  Input patterns: {stats['total_input']}")
        logger.info(f"  Exact duplicates removed: {stats['exact_duplicates_removed']}")
        logger.info(f"  Condition duplicates removed: {stats['condition_duplicates_removed']}")
        logger.info(f"  Metric duplicates removed: {stats['metric_duplicates_removed']}")
        logger.info(f"  Subset patterns removed: {stats['subset_patterns_removed']}")
        logger.info(f"  Output patterns: {stats['total_output']}")
        logger.info(f"  Reduction: {(1 - stats['total_output'] / stats['total_input']) * 100:.1f}%")
        
        # Compound pattern creation
        medium_patterns = [p for p in optimized if 65 <= p['success_rate'] < 75]
        compound_patterns = self.compound_pattern_creation(medium_patterns)
        optimized.extend(compound_patterns)
        
        logger.info(f"Compound patterns created: {len(compound_patterns)}")
        
        # Regime-specific optimization
        self.regime_patterns = self.regime_specific_optimization(optimized)
        
        # Temporal stability testing
        stable_patterns = self.temporal_stability_testing(optimized)
        
        # Combine all optimized patterns
        self.optimized_patterns = optimized + stable_patterns
        
        logger.info(f"\nTotal optimized patterns: {len(self.optimized_patterns)}")
        
        return self.optimized_patterns
    
    def _deduplicate_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """
        Remove duplicate patterns.
        
        Args:
            patterns: List of patterns
            
        Returns:
            Deduplicated patterns
        """
        if not patterns:
            return []
        
        # Sort by success rate
        patterns.sort(key=lambda x: x['success_rate'], reverse=True)
        
        unique_patterns = []
        seen_conditions = set()
        
        for pattern in patterns:
            conditions_str = json.dumps(pattern['conditions'], sort_keys=True)
            conditions_hash = hash(conditions_str)
            
            if conditions_hash not in seen_conditions:
                seen_conditions.add(conditions_hash)
                unique_patterns.append(pattern)
        
        return unique_patterns
    
    def save_optimized_patterns(self, output_dir: str = "data"):
        """
        Save optimized patterns to files with pipeline metadata.
        
        Args:
            output_dir: Directory to save output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate pipeline run ID
        pipeline_run_id = str(uuid.uuid4())[:8]
        pipeline_run_timestamp = datetime.now().isoformat()
        
        # Add pipeline metadata to each pattern
        for pattern in self.optimized_patterns:
            pattern['pipeline_run_id'] = pipeline_run_id
            pattern['pipeline_run_timestamp'] = pipeline_run_timestamp
        
        # Save optimized patterns
        patterns_path = os.path.join(output_dir, "optimized_patterns.json")
        with open(patterns_path, 'w') as f:
            json.dump(self.optimized_patterns, f, indent=2, default=str)
        logger.info(f"Optimized patterns saved to {patterns_path}")
        logger.info(f"Pipeline Run ID: {pipeline_run_id}")
        
        # Save regime patterns with metadata
        regime_path = os.path.join(output_dir, "regime_patterns.json")
        regime_data = {
            'pipeline_run_id': pipeline_run_id,
            'pipeline_run_timestamp': pipeline_run_timestamp,
            'regimes': self.regime_patterns
        }
        with open(regime_path, 'w') as f:
            json.dump(regime_data, f, indent=2, default=str)
        logger.info(f"Regime patterns saved to {regime_path}")
    
    def run_phase5(self, data_path: str = None, patterns_path: str = None) -> List[Dict]:
        """
        Run complete Phase 5: Pattern Optimization & Refinement.
        
        Args:
            data_path: Path to features matrix CSV file
            patterns_path: Path to patterns JSON file
            
        Returns:
            List of optimized patterns
        """
        logger.info("\n" + "=" * 60)
        logger.info("STARTING PHASE 5: PATTERN OPTIMIZATION")
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
                return []
        
        # Load patterns
        if patterns_path:
            self.load_patterns(patterns_path)
        elif self.patterns is None:
            default_path = os.path.join("data", "classified_patterns.json")
            if os.path.exists(default_path):
                self.load_patterns(default_path)
            else:
                logger.error("No patterns path provided and default file not found")
                return []
        
        # Optimize patterns
        optimized = self.optimize_all_patterns()
        
        # Save results
        self.save_optimized_patterns()
        
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 5 COMPLETE")
        logger.info("=" * 60)
        
        return optimized


if __name__ == "__main__":
    # Run Phase 5
    po = PatternOptimization()
    optimized = po.run_phase5()
    
    print(f"\nFinal Results:")
    print(f"  Total Optimized Patterns: {len(optimized)}")
    print(f"  Regime Patterns: {sum(len(v) for v in po.regime_patterns.values())}")