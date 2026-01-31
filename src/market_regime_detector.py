"""
Market Regime Detector Module
Detects real-time market regimes and adjusts pattern weights dynamically
"""

import logging
from typing import Dict, List, Optional
from collections import deque
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Detects market regimes (volatility and trend) in real-time.
    Provides regime adjustments for pattern weighting.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the market regime detector.
        
        Args:
            config_path: Path to configuration file
        """
        import yaml
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.streaming_config = config.get('streaming', {})
            regime_config = self.streaming_config.get('regime_detection', {})
        except FileNotFoundError:
            self.streaming_config = {}
            regime_config = {}
        
        # Window size for regime calculation
        self.window_size = regime_config.get('window_size', 20)
        
        # Volatility thresholds
        vol_thresholds = regime_config.get('volatility_thresholds', {})
        self.vol_low_threshold = vol_thresholds.get('low', 1.5)
        self.vol_medium_threshold = vol_thresholds.get('medium', 3.0)
        
        # Trend thresholds
        trend_thresholds = regime_config.get('trend_thresholds', {})
        self.trend_strong_up = trend_thresholds.get('strong_up', 0.2)
        self.trend_weak_up = trend_thresholds.get('weak_up', 0.05)
        self.trend_neutral = trend_thresholds.get('neutral', -0.05)
        self.trend_weak_down = trend_thresholds.get('weak_down', -0.2)
        
        # Feature history for regime detection
        self.feature_history = deque(maxlen=self.window_size)
        
        # Current regime state
        self.current_regime = {
            'volatility_regime': 'Medium',
            'trend_regime': 'Sideways',
            'volatility_score': 50.0,
            'trend_score': 50.0,
            'atr_pct': 0.0,
            'ma_slope': 0.0,
            'rsi': 50.0,
            'volume_ratio': 1.0
        }
        
        # Previous regime for change detection
        self.previous_regime = None
        
        logger.info("Market Regime Detector initialized")
    
    def update(self, features: Dict) -> Dict:
        """
        Update regime detection with new features.
        
        Args:
            features: Current feature values
            
        Returns:
            Current regime information
        """
        self.feature_history.append(features)
        
        # Only calculate if we have enough data
        if len(self.feature_history) < 5:
            return self.current_regime
        
        # Calculate regime metrics
        regime = self._detect_regime()
        
        # Check for regime change
        regime_changed = self._check_regime_change(regime)
        
        if regime_changed:
            logger.info(f"Regime changed: {self.previous_regime} -> {regime}")
        
        self.previous_regime = self.current_regime.copy()
        self.current_regime = regime
        
        return regime
    
    def _detect_regime(self) -> Dict:
        """Detect current market regime from feature history."""
        recent = list(self.feature_history)
        
        # Calculate ATR% average
        atr_values = [f.get('ATR_14_Pct', 0) for f in recent if 'ATR_14_Pct' in f]
        avg_atr = sum(atr_values) / len(atr_values) if atr_values else 0
        
        # Calculate MA slope (price trend)
        close_values = [f.get('Close', 0) for f in recent if 'Close' in f]
        if len(close_values) >= 5:
            ma_slope = ((close_values[-1] - close_values[0]) / close_values[0]) * 100 if close_values[0] > 0 else 0
        else:
            ma_slope = 0
        
        # Get current RSI
        current_rsi = recent[-1].get('RSI_14', 50) if recent else 50
        
        # Get current volume ratio
        current_vol_ratio = recent[-1].get('Vol_Ratio_20', 1.0) if recent else 1.0
        
        # Determine volatility regime
        vol_regime = self._classify_volatility(avg_atr)
        
        # Determine trend regime
        trend_regime = self._classify_trend(ma_slope)
        
        # Calculate scores (0-100)
        vol_score = min(100, (avg_atr / 5.0) * 100)  # Normalize to 0-100
        trend_score = self._normalize_trend(ma_slope)
        
        return {
            'volatility_regime': vol_regime,
            'trend_regime': trend_regime,
            'volatility_score': round(vol_score, 2),
            'trend_score': round(trend_score, 2),
            'atr_pct': round(avg_atr, 2),
            'ma_slope': round(ma_slope, 2),
            'rsi': round(current_rsi, 2),
            'volume_ratio': round(current_vol_ratio, 2)
        }
    
    def _classify_volatility(self, atr_pct: float) -> str:
        """Classify volatility regime based on ATR%."""
        if atr_pct < self.vol_low_threshold:
            return 'Low'
        elif atr_pct < self.vol_medium_threshold:
            return 'Medium'
        else:
            return 'High'
    
    def _classify_trend(self, ma_slope: float) -> str:
        """Classify trend regime based on MA slope."""
        if ma_slope > self.trend_strong_up:
            return 'Strong Bull'
        elif ma_slope > self.trend_weak_up:
            return 'Weak Bull'
        elif ma_slope > self.trend_neutral:
            return 'Sideways'
        elif ma_slope > self.trend_weak_down:
            return 'Weak Bear'
        else:
            return 'Strong Bear'
    
    def _normalize_trend(self, ma_slope: float) -> float:
        """Normalize trend slope to 0-100 score."""
        # Map -0.5 to +0.5 range to 0-100
        normalized = (ma_slope + 0.5) * 100
        return max(0, min(100, normalized))
    
    def _check_regime_change(self, new_regime: Dict) -> bool:
        """Check if regime has changed."""
        if self.previous_regime is None:
            return False
        
        return (
            self.previous_regime['volatility_regime'] != new_regime['volatility_regime'] or
            self.previous_regime['trend_regime'] != new_regime['trend_regime']
        )
    
    def get_regime_adjustment(self, pattern: Dict) -> float:
        """
        Calculate pattern weight adjustment based on regime compatibility.
        
        Args:
            pattern: Pattern dictionary with volatility_regime and trend_regime
            
        Returns:
            Adjustment factor (0.5 to 1.5)
        """
        # Handle nested pattern structure
        if 'pattern' in pattern:
            pattern_data = pattern['pattern']
        else:
            pattern_data = pattern
        
        pattern_vol_regime = pattern_data.get('volatility_regime', 'All')
        pattern_trend_regime = pattern_data.get('trend_regime', 'All')
        
        current_vol = self.current_regime['volatility_regime']
        current_trend = self.current_regime['trend_regime']
        
        adjustment = 1.0
        
        # Volatility regime adjustment
        if pattern_vol_regime != 'All':
            if pattern_vol_regime == current_vol:
                adjustment *= 1.2  # +20% boost
            elif pattern_vol_regime == 'Low' and current_vol == 'High':
                adjustment *= 0.5  # -50% penalty
            elif pattern_vol_regime == 'High' and current_vol == 'Low':
                adjustment *= 0.7  # -30% penalty
            elif pattern_vol_regime == 'Medium' and current_vol in ['Low', 'High']:
                adjustment *= 0.85  # -15% penalty
        
        # Trend regime adjustment
        if pattern_trend_regime != 'All':
            if 'Bull' in pattern_trend_regime and 'Bull' in current_trend:
                adjustment *= 1.15
            elif 'Bear' in pattern_trend_regime and 'Bear' in current_trend:
                adjustment *= 1.15
            elif 'Bull' in pattern_trend_regime and 'Bear' in current_trend:
                adjustment *= 0.6
            elif 'Bear' in pattern_trend_regime and 'Bull' in current_trend:
                adjustment *= 0.6
            elif 'Sideways' in pattern_trend_regime and current_trend == 'Sideways':
                adjustment *= 1.1
        
        return max(0.5, min(1.5, adjustment))
    
    def get_current_regime(self) -> Dict:
        """Get the current regime information."""
        return self.current_regime.copy()
    
    def get_regime_summary(self) -> str:
        """Get a human-readable regime summary."""
        return (
            f"Volatility: {self.current_regime['volatility_regime']} "
            f"(ATR: {self.current_regime['atr_pct']}%), "
            f"Trend: {self.current_regime['trend_regime']} "
            f"(Slope: {self.current_regime['ma_slope']}%), "
            f"RSI: {self.current_regime['rsi']}"
        )
    
    def reset(self):
        """Reset the regime detector state."""
        self.feature_history.clear()
        self.current_regime = {
            'volatility_regime': 'Medium',
            'trend_regime': 'Sideways',
            'volatility_score': 50.0,
            'trend_score': 50.0,
            'atr_pct': 0.0,
            'ma_slope': 0.0,
            'rsi': 50.0,
            'volume_ratio': 1.0
        }
        self.previous_regime = None
        logger.info("Regime detector state reset")


if __name__ == "__main__":
    # Demo usage
    detector = MarketRegimeDetector()
    
    # Simulate some feature updates
    print("Simulating regime detection...\n")
    
    # Low volatility, strong uptrend
    for i in range(10):
        features = {
            'Close': 100 + i * 0.5,  # Rising price
            'ATR_14_Pct': 1.2,  # Low volatility
            'RSI_14': 60 + i,  # Rising RSI
            'Vol_Ratio_20': 1.1
        }
        regime = detector.update(features)
        print(f"Step {i+1}: {detector.get_regime_summary()}")
    
    print(f"\nRegime changed: {detector._check_regime_change(regime)}")
    
    # High volatility, downtrend
    for i in range(10):
        features = {
            'Close': 105 - i * 0.3,  # Falling price
            'ATR_14_Pct': 3.5,  # High volatility
            'RSI_14': 70 - i * 2,  # Falling RSI
            'Vol_Ratio_20': 1.5
        }
        regime = detector.update(features)
        print(f"Step {i+11}: {detector.get_regime_summary()}")
    
    # Test regime adjustment
    print("\n\nRegime Adjustment Examples:")
    
    # Pattern that works well in low volatility, bull market
    pattern_bull_low = {
        'pattern': {
            'volatility_regime': 'Low',
            'trend_regime': 'Strong Bull'
        }
    }
    
    # Pattern that works well in high volatility, bear market
    pattern_bear_high = {
        'pattern': {
            'volatility_regime': 'High',
            'trend_regime': 'Strong Bear'
        }
    }
    
    print(f"Current regime: {detector.get_regime_summary()}")
    print(f"Bull/Low pattern adjustment: {detector.get_regime_adjustment(pattern_bull_low):.2f}x")
    print(f"Bear/High pattern adjustment: {detector.get_regime_adjustment(pattern_bear_high):.2f}x")