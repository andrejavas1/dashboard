"""
Phase 7: Pattern Ranking & Portfolio Construction Module
Ranks patterns and constructs a diversified portfolio.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from itertools import combinations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PortfolioConstruction:
    """
    Ranks patterns and constructs a diversified portfolio.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the portfolio construction system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.run_mode = self.config.get('run_mode', 'full')
        
        # Portfolio parameters
        self.max_patterns = self.config['portfolio']['max_patterns']
        self.min_patterns = self.config['portfolio']['min_patterns']
        self.long_min = self.config['portfolio'].get('long_patterns_min', 5)
        self.short_min = self.config['portfolio'].get('short_patterns_min', 5)
        self.short_term_min = self.config['portfolio']['short_term_patterns_min']
        self.medium_term_min = self.config['portfolio']['medium_term_patterns_min']
        self.regime_min = self.config['portfolio']['regime_patterns_min']
        
        # Adjust parameters for quick/ultra mode
        if self.run_mode == 'ultra':
            # Lower thresholds for ultra mode
            self.dashboard_min_training_sr = 45
            self.dashboard_min_validation_sr = 40
            self.dashboard_min_occurrences = 3
            logger.info("Ultra Mode: Reduced dashboard pattern thresholds")
        elif self.run_mode == 'quick':
            self.dashboard_min_training_sr = 60
            self.dashboard_min_validation_sr = 55
            self.dashboard_min_occurrences = 5
            logger.info("Quick Mode: Reduced dashboard pattern thresholds")
        else:
            # Full mode - original thresholds
            self.dashboard_min_training_sr = 80
            self.dashboard_min_validation_sr = 80
            self.dashboard_min_occurrences = 30
        
        # Scoring weights
        self.success_weight = self.config['scoring']['success_metrics']
        self.risk_weight = self.config['scoring']['risk_metrics']
        self.practical_weight = self.config['scoring']['practical_metrics']
        
        # Data storage
        self.patterns = None
        self.ranked_patterns = []
        self.final_portfolio = []
        self.correlation_matrix = None
        
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
            'portfolio': {
                'max_patterns': 20,
                'min_patterns': 10,
                'long_patterns_min': 5,
                'short_patterns_min': 5,
                'short_term_patterns_min': 3,
                'medium_term_patterns_min': 3,
                'regime_patterns_min': 2
            },
            'scoring': {
                'success_metrics': 0.30,
                'risk_metrics': 0.50,
                'practical_metrics': 0.20
            }
        }
    
    def load_patterns(self, patterns_path: str) -> List[Dict]:
        """
        Load validated patterns from file.
        For short patterns, also include failed patterns since they have poor validation performance.
        
        Args:
            patterns_path: Path to validated patterns JSON file
            
        Returns:
            List of patterns
        """
        logger.info(f"Loading patterns from {patterns_path}")
        with open(patterns_path, 'r') as f:
            data = json.load(f)
        
        # Use robust patterns + degraded patterns
        robust = data.get('robust', [])
        degraded = data.get('degraded', [])
        failed = data.get('failed', [])
        
        # Separate long and short patterns
        robust_long = [p for p in robust if p.get('pattern', {}).get('direction', 'long') == 'long']
        robust_short = [p for p in robust if p.get('pattern', {}).get('direction', 'long') == 'short']
        degraded_long = [p for p in degraded if p.get('pattern', {}).get('direction', 'long') == 'long']
        degraded_short = [p for p in degraded if p.get('pattern', {}).get('direction', 'long') == 'short']
        failed_long = [p for p in failed if p.get('pattern', {}).get('direction', 'long') == 'long']
        failed_short = [p for p in failed if p.get('pattern', {}).get('direction', 'long') == 'short']
        
        # For long patterns: only use robust + degraded
        long_patterns = robust_long + degraded_long
        
        # For short patterns: also include failed patterns (since they have poor validation performance)
        # But only include failed patterns with decent training success rate (>60%)
        failed_short_filtered = [p for p in failed_short if p.get('training_success_rate', 0) >= 60]
        short_patterns = robust_short + degraded_short + failed_short_filtered
        
        self.patterns = long_patterns + short_patterns
        
        logger.info(f"Loaded {len(self.patterns)} patterns:")
        logger.info(f"  Long patterns (robust + degraded): {len(long_patterns)}")
        logger.info(f"  Short patterns (robust + degraded + failed): {len(short_patterns)}")
        logger.info(f"    - Robust: {len(robust_short)}")
        logger.info(f"    - Degraded: {len(degraded_short)}")
        logger.info(f"    - Failed (filtered): {len(failed_short_filtered)}")
        
        return self.patterns
    
    def calculate_success_score(self, pattern: Dict) -> float:
        """
        Calculate success metrics score (40% weight).
        
        Args:
            pattern: Pattern dictionary
            
        Returns:
            Success score (0-100)
        """
        validation_success_rate = pattern.get('validation_success_rate', 0)
        training_success_rate = pattern.get('training_success_rate', 0)
        
        # Validation success rate (primary factor)
        validation_score = validation_success_rate
        
        # Consistency score (ratio of validation to training)
        consistency_score = pattern.get('validation_ratio', 0) * 100
        
        # Statistical significance
        is_significant = pattern.get('statistically_significant', False)
        significance_score = 100 if is_significant else 50
        
        # Weighted average
        success_score = (validation_score * 0.5 + 
                        consistency_score * 0.3 + 
                        significance_score * 0.2)
        
        return success_score
    
    def calculate_risk_score(self, pattern: Dict) -> float:
        """
        Calculate risk metrics score (30% weight).
        Prioritizes low failure rate (reliability).
        
        Args:
            pattern: Pattern dictionary
            
        Returns:
            Risk score (0-100)
        """
        # False positive rate (lower is better) - PRIMARY RELIABILITY METRIC
        false_positive_rate = pattern.get('validation_false_positive_rate', 100)
        fp_score = max(100 - false_positive_rate * 2, 0)  # Double weight for reliability
        
        # Average winning move vs losing move (higher is better)
        avg_move = pattern.get('validation_avg_move', 0)
        move_score = min(avg_move * 10, 100)  # Cap at 100
        
        # Occurrence count (more occurrences = more robust)
        occurrences = pattern.get('validation_occurrences', 0)
        occurrence_score = min(occurrences, 100)
        
        # Weighted average - prioritize reliability
        risk_score = (fp_score * 0.5 +      # 50% weight on reliability
                     move_score * 0.3 +      # 30% weight on move size
                     occurrence_score * 0.2) # 20% weight on occurrences
        
        return risk_score
    
    def calculate_practical_score(self, pattern: Dict) -> float:
        """
        Calculate practical metrics score (30% weight).
        
        Args:
            pattern: Pattern dictionary
            
        Returns:
            Practical score (0-100)
        """
        # Occurrence frequency (more opportunities = better)
        occurrences = pattern.get('pattern', {}).get('occurrences', 0)
        frequency_score = min(occurrences / 2, 100)  # Scale to 0-100
        
        # Average time to target (faster = better)
        avg_time = pattern.get('validation_avg_time', 30)
        time_score = max(100 - avg_time * 2, 0)  # Lower time = higher score
        
        # Simplicity (fewer conditions = more robust)
        conditions = pattern.get('pattern', {}).get('conditions', {})
        complexity_score = max(100 - len(conditions) * 10, 50)  # Penalize complexity
        
        # Weighted average
        practical_score = (frequency_score * 0.4 + 
                          time_score * 0.3 + 
                          complexity_score * 0.3)
        
        return practical_score
    
    def calculate_composite_score(self, pattern: Dict) -> float:
        """
        Calculate composite score for a pattern.
        
        Args:
            pattern: Pattern dictionary
            
        Returns:
            Composite score (0-100)
        """
        success_score = self.calculate_success_score(pattern)
        risk_score = self.calculate_risk_score(pattern)
        practical_score = self.calculate_practical_score(pattern)
        
        composite_score = (success_score * self.success_weight +
                          risk_score * self.risk_weight +
                          practical_score * self.practical_weight)
        
        return composite_score
    
    def rank_patterns(self) -> List[Dict]:
        """
        Rank patterns by composite score.
        
        Returns:
            List of ranked patterns with scores
        """
        logger.info("=" * 60)
        logger.info("PHASE 7: Pattern Ranking & Portfolio Construction")
        logger.info("=" * 60)
        
        if self.patterns is None or len(self.patterns) == 0:
            logger.error("No patterns loaded")
            return []
        
        logger.info(f"\nRanking {len(self.patterns)} patterns...")
        
        ranked = []
        
        for pattern in tqdm(self.patterns, desc="Calculating scores"):
            # Calculate scores
            success_score = self.calculate_success_score(pattern)
            risk_score = self.calculate_risk_score(pattern)
            practical_score = self.calculate_practical_score(pattern)
            composite_score = self.calculate_composite_score(pattern)
            
            # Add scores to pattern
            pattern_with_scores = pattern.copy()
            pattern_with_scores['scores'] = {
                'success': success_score,
                'risk': risk_score,
                'practical': practical_score,
                'composite': composite_score
            }
            
            ranked.append(pattern_with_scores)
        
        # Sort by composite score
        ranked.sort(key=lambda x: x['scores']['composite'], reverse=True)
        
        self.ranked_patterns = ranked
        
        logger.info(f"\nTop 10 Patterns by Composite Score:")
        for i, pattern in enumerate(ranked[:10], 1):
            logger.info(f"  {i}. Score: {pattern['scores']['composite']:.2f} | "
                       f"Validation: {pattern.get('validation_success_rate', 0):.1f}% | "
                       f"Occurrences: {pattern.get('validation_occurrences', 0)}")
        
        return ranked
    
    def calculate_pattern_correlation(self, pattern1: Dict, pattern2: Dict) -> float:
        """
        Calculate correlation between two patterns based on their occurrence dates.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        # This is a simplified correlation based on pattern characteristics
        # In a full implementation, you'd use actual occurrence data
        
        # Get pattern conditions
        conditions1 = pattern1.get('pattern', {}).get('conditions', {})
        conditions2 = pattern2.get('pattern', {}).get('conditions', {})
        
        # Calculate feature overlap
        features1 = set(conditions1.keys())
        features2 = set(conditions2.keys())
        
        if len(features1) == 0 or len(features2) == 0:
            return 0
        
        overlap = len(features1 & features2)
        total_features = len(features1 | features2)
        
        # Higher overlap = higher correlation
        correlation = overlap / total_features if total_features > 0 else 0
        
        return correlation
    
    def build_correlation_matrix(self, patterns: List[Dict]) -> pd.DataFrame:
        """
        Build correlation matrix for patterns.
        
        Args:
            patterns: List of patterns
            
        Returns:
            Correlation matrix as DataFrame
        """
        n = len(patterns)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    matrix[i, j] = self.calculate_pattern_correlation(patterns[i], patterns[j])
        
        # Create DataFrame
        pattern_ids = [f"P{i}" for i in range(n)]
        self.correlation_matrix = pd.DataFrame(matrix, index=pattern_ids, columns=pattern_ids)
        
        return self.correlation_matrix
    
    def select_diversified_portfolio(self, top_n: int = 20) -> List[Dict]:
        """
        Select a diversified portfolio from top patterns.
        Ensures both long and short patterns are included.
        
        Args:
            top_n: Number of top patterns to consider
            
        Returns:
            Selected portfolio
        """
        logger.info(f"\nSelecting diversified portfolio from top {top_n} patterns...")
        
        # Separate long and short patterns from FULL ranked list (not just top N)
        all_long_patterns = [p for p in self.ranked_patterns if p.get('pattern', {}).get('direction', 'long') == 'long']
        all_short_patterns = [p for p in self.ranked_patterns if p.get('pattern', {}).get('direction', 'long') == 'short']
        
        logger.info(f"  Available long patterns (total): {len(all_long_patterns)}")
        logger.info(f"  Available short patterns (total): {len(all_short_patterns)}")
        
        # Select minimum required from each direction
        selected = []
        selected_patterns_set = set()
        
        # Select top short patterns (minimum required) - from FULL list to ensure we get them
        short_to_select = min(self.short_min, len(all_short_patterns))
        for i in range(short_to_select):
            selected.append(all_short_patterns[i])
            selected_patterns_set.add(id(all_short_patterns[i]))
        
        # Select top long patterns (minimum required)
        long_to_select = min(self.long_min, len(all_long_patterns))
        long_count = 0
        for pattern in all_long_patterns:
            if long_count >= long_to_select:
                break
            if id(pattern) not in selected_patterns_set:
                selected.append(pattern)
                selected_patterns_set.add(id(pattern))
                long_count += 1
        
        # Build correlation matrix for remaining patterns (from top N for efficiency)
        top_patterns = self.ranked_patterns[:top_n]
        remaining_patterns = [p for p in top_patterns if id(p) not in selected_patterns_set]
        
        if remaining_patterns:
            self.build_correlation_matrix(selected + remaining_patterns)
            
            # Greedy selection: add pattern with lowest correlation to selected
            while len(selected) < self.max_patterns and remaining_patterns:
                best_candidate = None
                best_idx_in_remaining = -1
                best_avg_correlation = float('inf')
                
                for i, pattern in enumerate(remaining_patterns):
                    # Calculate average correlation with selected patterns
                    correlations = []
                    for sel_pattern in selected:
                        sel_idx = (selected + remaining_patterns).index(sel_pattern)
                        rem_idx = (selected + remaining_patterns).index(pattern)
                        corr = self.correlation_matrix.iloc[rem_idx, sel_idx]
                        correlations.append(corr)
                    
                    avg_correlation = np.mean(correlations) if correlations else 0
                    
                    if avg_correlation < best_avg_correlation:
                        best_avg_correlation = avg_correlation
                        best_candidate = pattern
                        best_idx_in_remaining = i
                
                if best_candidate:
                    selected.append(best_candidate)
                    remaining_patterns.pop(best_idx_in_remaining)
        
        # Check portfolio composition requirements
        self._validate_portfolio_composition(selected)
        
        self.final_portfolio = selected
        
        logger.info(f"\nSelected {len(selected)} patterns for portfolio")
        
        return selected
    
    def _validate_portfolio_composition(self, portfolio: List[Dict]):
        """
        Validate portfolio meets composition requirements.
        
        Args:
            portfolio: Portfolio to validate
        """
        # Count long vs short patterns using direction field
        long_patterns = 0
        short_patterns = 0
        
        for p in portfolio:
            direction = p.get('pattern', {}).get('direction', 'long')
            if direction == 'long':
                long_patterns += 1
            else:
                short_patterns += 1
        
        # Count short-term vs medium-term patterns
        short_term = 0
        medium_term = 0
        
        for p in portfolio:
            label_col = p.get('pattern', {}).get('label_col', '')
            if '_5d' in label_col or '_3d' in label_col:
                short_term += 1
            elif '_10d' in label_col or '_20d' in label_col or '_30d' in label_col:
                medium_term += 1
        
        logger.info(f"\nPortfolio Composition:")
        logger.info(f"  Long Patterns: {long_patterns} (min: {self.long_min})")
        logger.info(f"  Short Patterns: {short_patterns} (min: {self.short_min})")
        logger.info(f"  Short-Term Patterns: {short_term} (min: {self.short_term_min})")
        logger.info(f"  Medium-Term Patterns: {medium_term} (min: {self.medium_term_min})")
        
        # Warn if requirements not met
        if long_patterns < self.long_min:
            logger.warning(f"  ⚠ Long patterns below minimum ({long_patterns} < {self.long_min})")
        if short_patterns < self.short_min:
            logger.warning(f"  ⚠ Short patterns below minimum ({short_patterns} < {self.short_min})")
        if short_term < self.short_term_min:
            logger.warning(f"  ⚠ Short-term patterns below minimum ({short_term} < {self.short_term_min})")
        if medium_term < self.medium_term_min:
            logger.warning(f"  ⚠ Medium-term patterns below minimum ({medium_term} < {self.medium_term_min})")
    
    def generate_portfolio_summary(self) -> Dict:
        """
        Generate summary statistics for the final portfolio.
        
        Returns:
            Portfolio summary dictionary
        """
        if not self.final_portfolio:
            return {}
        
        # Calculate aggregate metrics
        avg_validation_rate = np.mean([p.get('validation_success_rate', 0) for p in self.final_portfolio])
        avg_training_rate = np.mean([p.get('training_success_rate', 0) for p in self.final_portfolio])
        avg_occurrences = np.mean([p.get('validation_occurrences', 0) for p in self.final_portfolio])
        avg_composite_score = np.mean([p['scores']['composite'] for p in self.final_portfolio])
        
        # Calculate expected opportunity frequency
        total_occurrences = sum(p.get('pattern', {}).get('occurrences', 0) for p in self.final_portfolio)
        days_analyzed = 365 * 14  # Approximate days in training period
        signals_per_month = (total_occurrences / days_analyzed) * 30
        
        summary = {
            'total_patterns': len(self.final_portfolio),
            'average_validation_success_rate': round(avg_validation_rate, 2),
            'average_training_success_rate': round(avg_training_rate, 2),
            'average_occurrences': round(avg_occurrences, 2),
            'average_composite_score': round(avg_composite_score, 2),
            'expected_signals_per_month': round(signals_per_month, 2),
            'patterns': [
                {
                    'id': i,
                    'composite_score': p['scores']['composite'],
                    'validation_success_rate': p.get('validation_success_rate', 0),
                    'training_success_rate': p.get('training_success_rate', 0),
                    'validation_occurrences': p.get('validation_occurrences', 0),
                    'label_column': p.get('pattern', {}).get('label_col', '')
                }
                for i, p in enumerate(self.final_portfolio)
            ]
        }
        
        return summary
    
    def save_portfolio(self, output_dir: str = "data", ticker: str = None):
        """
        Save portfolio to files.
        
        Args:
            output_dir: Directory to save output files
            ticker: Ticker symbol for path construction
        """
        # If ticker provided, use ticker-specific directory
        if ticker:
            output_dir = os.path.join("data", "tickers", ticker)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save ranked patterns
        ranked_path = os.path.join(output_dir, "ranked_patterns.json")
        with open(ranked_path, 'w') as f:
            json.dump(self.ranked_patterns, f, indent=2, default=str)
        logger.info(f"Ranked patterns saved to {ranked_path}")
        
        # Save final portfolio
        portfolio_path = os.path.join(output_dir, "final_portfolio.json")
        with open(portfolio_path, 'w') as f:
            json.dump(self.final_portfolio, f, indent=2, default=str)
        logger.info(f"Final portfolio saved to {portfolio_path}")
        
        # Save correlation matrix
        if self.correlation_matrix is not None:
            corr_path = os.path.join(output_dir, "pattern_correlation_matrix.csv")
            self.correlation_matrix.to_csv(corr_path)
            logger.info(f"Correlation matrix saved to {corr_path}")
        
        # Save portfolio summary
        summary = self.generate_portfolio_summary()
        summary_path = os.path.join(output_dir, "portfolio_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Portfolio summary saved to {summary_path}")
    
    def save_dashboard_patterns(self, output_dir: str = "data"):
        """
        Save filtered patterns for dashboard display.
        Filters depend on run mode:
        - full: >= 80% training, >= 80% validation, >= 30 occurrences
        - quick: >= 60% training, >= 55% validation, >= 5 occurrences
        - ultra: >= 45% training, >= 40% validation, >= 3 occurrences
        
        Args:
            output_dir: Directory to save output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter patterns for dashboard
        filtered_patterns = []
        for pattern_wrapper in self.patterns:
            pattern = pattern_wrapper.get('pattern', pattern_wrapper)
            training_sr = pattern_wrapper.get('training_success_rate', 0)
            validation_sr = pattern_wrapper.get('validation_success_rate', 0)
            occ = pattern.get('occurrences', 0)
            
            # Apply filter based on run mode
            if (training_sr >= self.dashboard_min_training_sr and 
                validation_sr >= self.dashboard_min_validation_sr and 
                occ >= self.dashboard_min_occurrences):
                # Add rates to pattern for dashboard display
                pattern_with_rates = pattern.copy()
                pattern_with_rates['training_success_rate'] = training_sr
                pattern_with_rates['validation_success_rate'] = validation_sr
                filtered_patterns.append(pattern_with_rates)
        
        # Save to patterns.json
        patterns_path = os.path.join(output_dir, "patterns.json")
        with open(patterns_path, 'w') as f:
            json.dump(filtered_patterns, f, indent=2, default=str)
        logger.info(f"Dashboard patterns saved to {patterns_path}: {len(filtered_patterns)} patterns")
        logger.info(f"  (Filtered: training>={self.dashboard_min_training_sr}%, "
                   f"validation>={self.dashboard_min_validation_sr}%, "
                   f"occurrences>={self.dashboard_min_occurrences})")
    
    def run_phase7(self, patterns_path: str = None) -> List[Dict]:
        """
        Run complete Phase 7: Pattern Ranking & Portfolio Construction.
        
        Args:
            patterns_path: Path to validated patterns JSON file
            
        Returns:
            Final portfolio
        """
        logger.info("\n" + "=" * 60)
        logger.info("STARTING PHASE 7: PORTFOLIO CONSTRUCTION")
        logger.info("=" * 60)
        
        # Load patterns
        if patterns_path:
            self.load_patterns(patterns_path)
        elif self.patterns is None:
            default_path = os.path.join("data", "validated_patterns.json")
            if os.path.exists(default_path):
                self.load_patterns(default_path)
            else:
                logger.error("No patterns path provided and default file not found")
                return []
        
        # Rank patterns
        self.rank_patterns()
        
        # Select diversified portfolio
        # Consider top 100 patterns to allow for better diversification
        self.select_diversified_portfolio(top_n=min(100, len(self.ranked_patterns)))
        
        # Generate summary
        summary = self.generate_portfolio_summary()
        if summary:
            logger.info(f"\nPortfolio Summary:")
            logger.info(f"  Total Patterns: {summary['total_patterns']}")
            logger.info(f"  Avg Validation Success Rate: {summary['average_validation_success_rate']:.2f}%")
            logger.info(f"  Expected Signals/Month: {summary['expected_signals_per_month']:.2f}")
        else:
            logger.warning("\nPortfolio Summary: No patterns in portfolio")
        
        # Save results
        self.save_portfolio()
        
        # Save dashboard patterns (filtered for display)
        self.save_dashboard_patterns()
        
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 7 COMPLETE")
        logger.info("=" * 60)
        
        return self.final_portfolio


if __name__ == "__main__":
    # Run Phase 7
    pc = PortfolioConstruction()
    portfolio = pc.run_phase7()
    
    print(f"\nFinal Results:")
    print(f"  Portfolio Size: {len(portfolio)}")
    print(f"  Average Score: {np.mean([p['scores']['composite'] for p in portfolio]):.2f}")
    print(f"  Success Rate: {np.mean([p.get('validation_success_rate', 0) for p in portfolio]):.2f}%")