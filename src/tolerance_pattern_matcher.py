"""
Tolerance-Based Pattern Matcher Module
Matches patterns with ±5% tolerance bands for real-time conditions
"""

import logging
from typing import Dict, List, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TolerancePatternMatcher:
    """
    Matches patterns with tolerance bands for robust real-time detection.
    Scores confidence 0-100% based on condition proximity.
    """
    
    def __init__(self, tolerance_pct: float = 5.0, config_path: str = "config.yaml"):
        """
        Initialize the tolerance-based pattern matcher.
        
        Args:
            tolerance_pct: Percentage tolerance for condition matching (default: 5%)
            config_path: Path to configuration file
        """
        self.tolerance_pct = tolerance_pct
        
        # Load config for thresholds
        import yaml
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.streaming_config = config.get('streaming', {})
            pattern_config = self.streaming_config.get('pattern_matching', {})
            self.min_confidence_threshold = pattern_config.get('min_confidence_threshold', 50)
            self.trigger_threshold = pattern_config.get('trigger_threshold', 90)
        except FileNotFoundError:
            self.streaming_config = {}
            self.min_confidence_threshold = 50
            self.trigger_threshold = 90
        
        # Pattern portfolio
        self.patterns = []
        
        logger.info(f"Tolerance Pattern Matcher initialized with {tolerance_pct}% tolerance")
    
    def load_patterns(self, portfolio_path: str = "data/final_portfolio.json"):
        """
        Load pattern portfolio from file.
        
        Args:
            portfolio_path: Path to portfolio JSON file
        """
        try:
            with open(portfolio_path, 'r') as f:
                portfolio = json.load(f)
            
            # Add IDs to patterns if not present
            for i, pattern in enumerate(portfolio):
                if 'pattern' in pattern:
                    pattern['pattern']['id'] = i
                else:
                    pattern['id'] = i
            
            self.patterns = portfolio
            logger.info(f"Loaded {len(self.patterns)} patterns from {portfolio_path}")
            
        except FileNotFoundError:
            logger.error(f"Portfolio file not found at {portfolio_path}")
            self.patterns = []
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            self.patterns = []
    
    def match_pattern(self, pattern: Dict, features: Dict) -> Dict:
        """
        Match a single pattern against current features.
        
        Args:
            pattern: Pattern dictionary with conditions
            features: Current feature values
            
        Returns:
            Match result with confidence score and details
        """
        # Handle nested pattern structure
        if 'pattern' in pattern:
            pattern_data = pattern['pattern']
            pattern_id = pattern_data.get('id', pattern.get('id', 0))
            validation_rate = pattern.get('validation_success_rate', 0)
        else:
            pattern_data = pattern
            pattern_id = pattern.get('id', 0)
            validation_rate = pattern.get('validation_success_rate', 0)
        
        conditions = pattern_data.get('conditions', {})
        
        if not conditions:
            return {
                'pattern_id': pattern_id,
                'confidence_score': 0,
                'conditions_met': 0,
                'total_conditions': 0,
                'condition_details': {},
                'status': 'NO_CONDITIONS'
            }
        
        total_score = 0.0
        condition_details = {}
        
        for feature, condition in conditions.items():
            required_value = condition.get('value')
            operator = condition.get('operator', '>=')
            current_value = features.get(feature)
            
            if current_value is None:
                condition_details[feature] = {
                    'score': 0,
                    'met': False,
                    'current': None,
                    'required': required_value,
                    'distance_pct': None,
                    'operator': operator
                }
                continue
            
            # Calculate tolerance band
            if required_value != 0:
                tolerance_band = abs(required_value) * (self.tolerance_pct / 100)
                lower_bound = required_value - tolerance_band
                upper_bound = required_value + tolerance_band
                distance_pct = abs(current_value - required_value) / abs(required_value) * 100
            else:
                tolerance_band = 0.01  # Small tolerance for zero values
                lower_bound = -tolerance_band
                upper_bound = tolerance_band
                distance_pct = 100 if current_value != 0 else 0
            
            # Determine if condition is met
            is_met = self._check_operator(current_value, required_value, operator)
            
            # Calculate score based on proximity
            if is_met:
                # Exact match = 100, at tolerance boundary = 70
                distance = abs(current_value - required_value)
                if distance == 0:
                    score = 100.0
                elif tolerance_band > 0:
                    score = 100.0 - (distance / tolerance_band) * 30.0
                else:
                    score = 100.0
            else:
                # Check if within tolerance band (near match)
                if lower_bound <= current_value <= upper_bound:
                    # Near match: score 50-70 based on proximity
                    distance = abs(current_value - required_value)
                    score = 70.0 - (distance / tolerance_band) * 20.0 if tolerance_band > 0 else 50.0
                else:
                    # Outside tolerance = 0
                    score = 0.0
            
            condition_details[feature] = {
                'score': min(100.0, max(0.0, score)),
                'met': is_met,
                'current': current_value,
                'required': required_value,
                'distance_pct': distance_pct,
                'operator': operator
            }
            
            total_score += score
        
        # Calculate overall confidence
        avg_score = total_score / len(conditions) if conditions else 0.0
        
        # Bonus for all conditions met within tolerance
        all_within_tolerance = all(
            d['met'] or (d['distance_pct'] is not None and d['distance_pct'] <= self.tolerance_pct)
            for d in condition_details.values()
        )
        if all_within_tolerance:
            avg_score = min(100.0, avg_score + 10.0)
        
        # Determine status
        status = self._get_status(avg_score)
        
        return {
            'pattern_id': pattern_id,
            'confidence_score': round(avg_score, 2),
            'conditions_met': sum(1 for d in condition_details.values() if d['met']),
            'total_conditions': len(conditions),
            'condition_details': condition_details,
            'status': status,
            'validation_success_rate': validation_rate,
            'pattern_data': pattern_data
        }
    
    def match_all_patterns(self, features: Dict) -> List[Dict]:
        """
        Match all patterns against current features.
        
        Args:
            features: Current feature values
            
        Returns:
            List of match results sorted by confidence score
        """
        if not self.patterns:
            logger.warning("No patterns loaded")
            return []
        
        results = []
        
        for pattern in self.patterns:
            result = self.match_pattern(pattern, features)
            
            # Only include patterns above minimum threshold
            if result['confidence_score'] >= self.min_confidence_threshold:
                results.append(result)
        
        # Sort by confidence score (descending)
        results.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        logger.debug(f"Matched {len(results)} patterns above {self.min_confidence_threshold}% threshold")
        
        return results
    
    def _check_operator(self, current: float, required: float, operator: str) -> bool:
        """Check if condition is met based on operator."""
        if operator == '>=':
            return current >= required
        elif operator == '<=':
            return current <= required
        elif operator == '>':
            return current > required
        elif operator == '<':
            return current < required
        elif operator == '==':
            return abs(current - required) < 0.0001
        elif operator == '!=':
            return abs(current - required) >= 0.0001
        return False
    
    def _get_status(self, confidence: float) -> str:
        """Get status based on confidence score."""
        if confidence >= self.trigger_threshold:
            return 'TRIGGERED'
        elif confidence >= self.trigger_threshold - 10:
            return 'NEAR_TRIGGER'
        elif confidence >= self.min_confidence_threshold:
            return 'WATCHING'
        return 'NOT_TRIGGERED'
    
    def get_triggered_patterns(self, features: Dict) -> List[Dict]:
        """
        Get only triggered patterns (confidence >= trigger_threshold).
        
        Args:
            features: Current feature values
            
        Returns:
            List of triggered patterns
        """
        all_matches = self.match_all_patterns(features)
        return [m for m in all_matches if m['status'] == 'TRIGGERED']
    
    def get_near_triggered_patterns(self, features: Dict) -> List[Dict]:
        """
        Get patterns that are near triggering (NEAR_TRIGGER status).
        
        Args:
            features: Current feature values
            
        Returns:
            List of near-triggered patterns
        """
        all_matches = self.match_all_patterns(features)
        return [m for m in all_matches if m['status'] == 'NEAR_TRIGGER']
    
    def get_watching_patterns(self, features: Dict) -> List[Dict]:
        """
        Get patterns being watched (WATCHING status).
        
        Args:
            features: Current feature values
            
        Returns:
            List of watching patterns
        """
        all_matches = self.match_all_patterns(features)
        return [m for m in all_matches if m['status'] == 'WATCHING']
    
    def set_tolerance(self, tolerance_pct: float):
        """Update the tolerance percentage."""
        self.tolerance_pct = max(0, min(100, tolerance_pct))
        logger.info(f"Tolerance updated to {self.tolerance_pct}%")
    
    def get_pattern_count(self) -> int:
        """Get the number of loaded patterns."""
        return len(self.patterns)


if __name__ == "__main__":
    # Demo usage
    matcher = TolerancePatternMatcher(tolerance_pct=5.0)
    
    # Load patterns
    matcher.load_patterns("data/final_portfolio.json")
    
    # Simulate features
    features = {
        'Close': 106.00,
        'RSI_14': 65.4,
        'ATR_14_Pct': 1.8,
        'Vol_Ratio_20': 1.2,
        'Dist_MA_50': 2.3,
        'MACD_Histogram': 0.12,
        'Vol_Regime': 'Medium',
        'Trend_Regime': 'Strong Bull'
    }
    
    # Match all patterns
    matches = matcher.match_all_patterns(features)
    
    print(f"\nPattern Matching Results:")
    print(f"Total patterns: {matcher.get_pattern_count()}")
    print(f"Matches above threshold: {len(matches)}")
    
    for match in matches[:5]:  # Show top 5
        print(f"\nPattern #{match['pattern_id']}:")
        print(f"  Confidence: {match['confidence_score']:.2f}%")
        print(f"  Status: {match['status']}")
        print(f"  Conditions: {match['conditions_met']}/{match['total_conditions']}")
        print(f"  Validation Rate: {match['validation_success_rate']:.2f}%")