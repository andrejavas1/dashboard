"""
Probability Estimator Module
Synthesizes historical data with current regime to output probability distributions
"""

import logging
from typing import Dict, List, Optional
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProbabilityEstimator:
    """
    Estimates probability distributions for price targets based on:
    - Historical success rates from portfolio
    - Current market regime
    - Pattern match confidence
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the probability estimator.
        
        Args:
            config_path: Path to configuration file
        """
        import yaml
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.streaming_config = config.get('streaming', {})
            estimation_config = self.streaming_config.get('estimation', {})
        except FileNotFoundError:
            self.streaming_config = {}
            estimation_config = {}
        
        # Weighting for probability calculation
        self.base_weight = estimation_config.get('base_weight', 0.4)
        self.regime_weight = estimation_config.get('regime_weight', 0.3)
        self.confidence_weight = estimation_config.get('confidence_weight', 0.3)
        
        logger.info("Probability Estimator initialized")
    
    def estimate(self, pattern: Dict, match_result: Dict, regime: Dict, current_price: float) -> Dict:
        """
        Estimate probability distribution for a matched pattern.
        
        Args:
            pattern: Pattern dictionary with validation metrics
            match_result: Pattern match result with confidence score
            regime: Current market regime information
            current_price: Current price of the asset
            
        Returns:
            Probability estimation with price targets and actionable signals
        """
        # Handle nested pattern structure
        if 'pattern' in pattern:
            pattern_data = pattern['pattern']
            wrapper = pattern
        else:
            pattern_data = pattern
            wrapper = pattern
        
        # Extract base metrics from portfolio
        base_rate = wrapper.get('validation_success_rate', 0)
        avg_move = wrapper.get('validation_avg_move', 0)
        avg_time = wrapper.get('validation_avg_time', 0)
        fpr = wrapper.get('validation_false_positive_rate', 0)
        
        # Extract pattern info
        pattern_id = match_result.get('pattern_id', 0)
        label_col = pattern_data.get('label_col', '')
        direction = pattern_data.get('direction', 'long')
        
        # Parse label for target and timeframe
        target_pct, timeframe_days = self._parse_label(label_col)
        
        # Apply regime adjustment
        regime_adj = self._get_regime_adjustment(pattern, regime)
        regime_adjusted_rate = base_rate * regime_adj
        
        # Apply confidence adjustment
        confidence = match_result.get('confidence_score', 0)
        confidence_adj = 0.5 + (confidence / 100) * 0.5  # 0.5 to 1.0
        confidence_adjusted_rate = regime_adjusted_rate * confidence_adj
        
        # Calculate final probability (weighted average)
        final_prob = (
            base_rate * self.base_weight +
            regime_adjusted_rate * self.regime_weight +
            confidence_adjusted_rate * self.confidence_weight
        )
        
        # Generate price targets
        targets = self._generate_price_targets(
            final_prob, target_pct, timeframe_days, direction, current_price
        )
        
        # Risk assessment
        risk = self._assess_risk(fpr, regime, final_prob)
        
        # Actionable signals
        signals = self._generate_signals(
            pattern_id, label_col, direction, final_prob, targets, risk, current_price
        )
        
        return {
            'pattern_id': pattern_id,
            'pattern_name': f"Pattern {pattern_id}: {label_col}",
            'direction': direction,
            'base_success_rate': round(base_rate, 2),
            'regime_adjusted_rate': round(regime_adjusted_rate, 2),
            'confidence_adjusted_rate': round(confidence_adjusted_rate, 2),
            'final_probability': round(final_prob, 2),
            'regime_adjustment': round(regime_adj, 2),
            'match_confidence': round(confidence, 2),
            'price_targets': targets,
            'risk_assessment': risk,
            'actionable_signals': signals
        }
    
    def _parse_label(self, label_col: str) -> tuple:
        """
        Parse label column to extract target percentage and timeframe.
        
        Args:
            label_col: Label string (e.g., "Label_2.0pct_10d")
            
        Returns:
            Tuple of (target_pct, timeframe_days)
        """
        try:
            parts = label_col.split('_')
            if len(parts) >= 3:
                target_pct = float(parts[1].replace('pct', ''))
                timeframe_days = int(parts[2].replace('d', ''))
                return target_pct, timeframe_days
        except (ValueError, IndexError):
            pass
        
        # Default values
        return 2.0, 10
    
    def _get_regime_adjustment(self, pattern: Dict, regime: Dict) -> float:
        """
        Calculate regime adjustment factor.
        
        Args:
            pattern: Pattern dictionary
            regime: Current regime information
            
        Returns:
            Adjustment factor (0.5 to 1.5)
        """
        # Import here to avoid circular dependency
        from market_regime_detector import MarketRegimeDetector
        
        # Create a temporary detector instance to use its method
        detector = MarketRegimeDetector()
        detector.current_regime = regime
        
        return detector.get_regime_adjustment(pattern)
    
    def _generate_price_targets(self, probability: float, base_target_pct: float, 
                               timeframe_days: int, direction: str, current_price: float) -> List[Dict]:
        """
        Generate price targets with probabilities.
        
        Args:
            probability: Base probability
            base_target_pct: Base target percentage from label
            timeframe_days: Timeframe in days
            direction: 'long' or 'short'
            current_price: Current price
            
        Returns:
            List of price target dictionaries
        """
        targets = []
        
        # Convert days to hours (assuming 6.5 trading hours per day)
        hours_per_day = 6.5
        base_hours = timeframe_days * hours_per_day
        
        # Generate multiple targets
        target_multipliers = [1.0, 1.5, 2.5]  # 1x, 1.5x, 2.5x the base target
        
        for i, multiplier in enumerate(target_multipliers):
            target_pct = base_target_pct * multiplier
            
            # Adjust probability based on target difficulty
            target_prob = probability * (1.0 - (i * 0.15))  # Reduce probability for harder targets
            target_prob = max(10, target_prob)  # Minimum 10%
            
            # Calculate expected time (longer for larger targets)
            expected_hours = base_hours * multiplier
            time_range = [
                int(expected_hours * 0.75),
                int(expected_hours * 1.25)
            ]
            
            # Calculate target price
            if direction == 'long':
                target_price = current_price * (1 + target_pct / 100)
            else:  # short
                target_price = current_price * (1 - target_pct / 100)
            
            targets.append({
                'target_id': i + 1,
                'move_pct': round(target_pct, 2),
                'probability': round(target_prob, 2),
                'expected_time_hours': round(expected_hours, 1),
                'time_range_hours': time_range,
                'target_price': round(target_price, 2),
                'direction': direction
            })
        
        return targets
    
    def _assess_risk(self, fpr: float, regime: Dict, probability: float) -> Dict:
        """
        Assess risk for the pattern.
        
        Args:
            fpr: False positive rate
            regime: Current regime
            probability: Final probability
            
        Returns:
            Risk assessment dictionary
        """
        # False positive probability
        fp_prob = 100 - probability
        
        # Max drawdown based on direction and regime
        vol_regime = regime.get('volatility_regime', 'Medium')
        
        if vol_regime == 'Low':
            max_drawdown_pct = 1.5
        elif vol_regime == 'Medium':
            max_drawdown_pct = 2.5
        else:  # High
            max_drawdown_pct = 4.0
        
        # Volatility risk level
        volatility_score = regime.get('volatility_score', 50)
        if volatility_score < 33:
            vol_risk = 'low'
        elif volatility_score < 66:
            vol_risk = 'medium'
        else:
            vol_risk = 'high'
        
        return {
            'false_positive_probability': round(fp_prob, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'volatility_risk': vol_risk,
            'risk_level': self._get_risk_level(probability, fpr, vol_risk)
        }
    
    def _get_risk_level(self, probability: float, fpr: float, vol_risk: str) -> str:
        """Determine overall risk level."""
        if probability >= 80 and fpr < 15 and vol_risk == 'low':
            return 'LOW'
        elif probability >= 70 and fpr < 25:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _generate_signals(self, pattern_id: int, label_col: str, direction: str,
                         probability: float, targets: List[Dict], 
                         risk: Dict, current_price: float) -> Dict:
        """
        Generate actionable trading signals.
        
        Args:
            pattern_id: Pattern ID
            label_col: Label column
            direction: 'long' or 'short'
            probability: Final probability
            targets: Price targets
            risk: Risk assessment
            current_price: Current price
            
        Returns:
            Actionable signals dictionary
        """
        # Determine entry signal
        if probability >= 90:
            entry_signal = 'STRONG BUY' if direction == 'long' else 'STRONG SELL'
        elif probability >= 75:
            entry_signal = 'BUY' if direction == 'long' else 'SELL'
        elif probability >= 60:
            entry_signal = 'MODERATE BUY' if direction == 'long' else 'MODERATE SELL'
        else:
            entry_signal = 'WATCH'
        
        # Calculate stop loss
        max_drawdown = risk['max_drawdown_pct']
        if direction == 'long':
            stop_loss = current_price * (1 - max_drawdown / 100)
        else:
            stop_loss = current_price * (1 + max_drawdown / 100)
        
        # Position size based on probability and risk
        if probability >= 85 and risk['risk_level'] == 'LOW':
            position_size_pct = 10.0
        elif probability >= 75 and risk['risk_level'] != 'HIGH':
            position_size_pct = 7.5
        elif probability >= 65:
            position_size_pct = 5.0
        else:
            position_size_pct = 2.5
        
        # Time horizon (from primary target)
        time_horizon_hours = targets[0]['expected_time_hours'] if targets else 16.7
        
        # Take profit levels
        take_profits = []
        for target in targets[:3]:  # Top 3 targets
            take_profits.append({
                'level': target['target_id'],
                'price': target['target_price'],
                'move_pct': target['move_pct'],
                'probability': target['probability']
            })
        
        return {
            'entry_signal': entry_signal,
            'entry_price': round(current_price, 2),
            'stop_loss': round(stop_loss, 2),
            'stop_loss_pct': round(max_drawdown, 2),
            'take_profits': take_profits,
            'position_size_pct': round(position_size_pct, 1),
            'time_horizon_hours': round(time_horizon_hours, 1),
            'risk_reward_ratio': self._calculate_risk_reward_ratio(
                current_price, stop_loss, targets[0]['target_price'] if targets else current_price
            )
        }
    
    def _calculate_risk_reward_ratio(self, entry: float, stop_loss: float, target: float) -> float:
        """Calculate risk/reward ratio."""
        risk = abs(entry - stop_loss)
        reward = abs(target - entry)
        
        if risk == 0:
            return 0.0
        
        return round(reward / risk, 2)
    
    def batch_estimate(self, matches: List[Dict], regime: Dict, current_price: float) -> List[Dict]:
        """
        Estimate probabilities for multiple pattern matches.
        
        Args:
            matches: List of pattern match results
            regime: Current market regime
            current_price: Current price
            
        Returns:
            List of probability estimations
        """
        estimations = []
        
        for match in matches:
            # Get pattern data from match
            pattern_data = match.get('pattern_data', {})
            
            # Create minimal pattern wrapper for estimation
            pattern_wrapper = {
                'pattern': pattern_data,
                'validation_success_rate': match.get('validation_success_rate', 0),
                'validation_avg_move': pattern_data.get('avg_move', 0),
                'validation_avg_time': pattern_data.get('avg_time', 0),
                'validation_false_positive_rate': pattern_data.get('false_positive_rate', 0)
            }
            
            # Estimate probability
            estimation = self.estimate(pattern_wrapper, match, regime, current_price)
            estimations.append(estimation)
        
        # Sort by final probability (descending)
        estimations.sort(key=lambda x: x['final_probability'], reverse=True)
        
        return estimations


if __name__ == "__main__":
    # Demo usage
    estimator = ProbabilityEstimator()
    
    # Sample pattern
    pattern = {
        'pattern': {
            'conditions': {'RSI_14': {'operator': '<=', 'value': 30}},
            'volatility_regime': 'All',
            'trend_regime': 'All',
            'label_col': 'Label_2.0pct_10d',
            'direction': 'long',
            'avg_move': 4.5,
            'avg_time': 8.5,
            'false_positive_rate': 10.0
        },
        'validation_success_rate': 85.0,
        'validation_avg_move': 4.5,
        'validation_avg_time': 8.5,
        'validation_false_positive_rate': 10.0
    }
    
    # Sample match result
    match_result = {
        'pattern_id': 0,
        'confidence_score': 92.5,
        'conditions_met': 1,
        'total_conditions': 1,
        'status': 'TRIGGERED',
        'validation_success_rate': 85.0,
        'pattern_data': pattern['pattern']
    }
    
    # Sample regime
    regime = {
        'volatility_regime': 'Medium',
        'trend_regime': 'Strong Bull',
        'volatility_score': 45.0,
        'trend_score': 75.0,
        'atr_pct': 2.2,
        'ma_slope': 0.3,
        'rsi': 65.0,
        'volume_ratio': 1.2
    }
    
    # Estimate probability
    estimation = estimator.estimate(pattern, match_result, regime, current_price=106.00)
    
    print("\nProbability Estimation:")
    print(f"Pattern: {estimation['pattern_name']}")
    print(f"Direction: {estimation['direction']}")
    print(f"\nSuccess Rates:")
    print(f"  Base: {estimation['base_success_rate']:.2f}%")
    print(f"  Regime Adjusted: {estimation['regime_adjusted_rate']:.2f}%")
    print(f"  Confidence Adjusted: {estimation['confidence_adjusted_rate']:.2f}%")
    print(f"  Final Probability: {estimation['final_probability']:.2f}%")
    
    print(f"\nPrice Targets:")
    for target in estimation['price_targets']:
        print(f"  Target {target['target_id']}: {target['move_pct']}% @ ${target['target_price']:.2f}")
        print(f"    Probability: {target['probability']:.2f}%")
        print(f"    Time: {target['expected_time_hours']:.1f}h ({target['time_range_hours'][0]}-{target['time_range_hours'][1]}h)")
    
    print(f"\nRisk Assessment:")
    print(f"  False Positive Probability: {estimation['risk_assessment']['false_positive_probability']:.2f}%")
    print(f"  Max Drawdown: {estimation['risk_assessment']['max_drawdown_pct']:.2f}%")
    print(f"  Risk Level: {estimation['risk_assessment']['risk_level']}")
    
    print(f"\nActionable Signals:")
    print(f"  Entry Signal: {estimation['actionable_signals']['entry_signal']}")
    print(f"  Entry Price: ${estimation['actionable_signals']['entry_price']:.2f}")
    print(f"  Stop Loss: ${estimation['actionable_signals']['stop_loss']:.2f} (-{estimation['actionable_signals']['stop_loss_pct']:.2f}%)")
    print(f"  Position Size: {estimation['actionable_signals']['position_size_pct']:.1f}%")
    print(f"  Risk/Reward Ratio: {estimation['actionable_signals']['risk_reward_ratio']:.2f}")