"""
Incremental Feature Calculator Module
Updates technical indicators incrementally without recalculating entire history
"""

import logging
from datetime import datetime
from typing import Dict, Optional
from collections import deque
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IncrementalFeatureCalculator:
    """
    Calculates and updates market state features incrementally.
    Maintains rolling window state and uses incremental formulas for efficiency.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the incremental feature calculator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.streaming_config = self.config.get('streaming', {})
        
        # Feature parameters from config
        features_config = self.config.get('features', {})
        self.ma_periods = features_config.get('moving_averages', [20, 50, 100, 200])
        self.atr_periods = features_config.get('atr_periods', [14])
        self.rsi_periods = features_config.get('rsi_periods', [14])
        self.volume_avg_periods = features_config.get('volume_avg_periods', [10, 20])
        
        # MACD parameters
        macd_params = features_config.get('macd_params', {})
        self.macd_fast = macd_params.get('fast', 12)
        self.macd_slow = macd_params.get('slow', 26)
        self.macd_signal = macd_params.get('signal', 9)
        
        # Alpha for EMA calculations
        self.ema_alpha_fast = 2 / (self.macd_fast + 1)
        self.ema_alpha_slow = 2 / (self.macd_slow + 1)
        self.ema_alpha_signal = 2 / (self.macd_signal + 1)
        
        # Initialize state
        self._initialize_state()
        
        logger.info("Incremental Feature Calculator initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        import yaml
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return {}
    
    def _initialize_state(self):
        """Initialize the feature calculation state."""
        self.state = {
            # Rolling windows (for features that need full window)
            'price_history': deque(maxlen=252),
            'volume_history': deque(maxlen=252),
            'high_history': deque(maxlen=252),
            'low_history': deque(maxlen=252),
            
            # EMA states
            'ema_12': None,
            'ema_26': None,
            'ema_50': None,
            'ema_100': None,
            'ema_200': None,
            
            # RSI states (14-period)
            'rsi_gains_14': deque(maxlen=14),
            'rsi_losses_14': deque(maxlen=14),
            'prev_close_rsi': None,
            
            # ATR states (14-period)
            'tr_history': deque(maxlen=14),
            'atr_14': None,
            
            # MACD states
            'macd_line': None,
            'macd_signal': None,
            'macd_histogram': None,
            
            # Volume states
            'vol_ma_20': None,
            'vol_ma_50': None,
            'obv': 0,
            'obv_history': deque(maxlen=252),
            
            # AD (Accumulation/Distribution)
            'ad': 0,
            'ad_history': deque(maxlen=252),
            
            # Stochastic
            'stoch_k_14': None,
            'stoch_d_14': None,
            'stoch_k_history': deque(maxlen=3),
            
            # CCI
            'tp_history': deque(maxlen=20),
            'tp_ma_history': deque(maxlen=20),
            
            # Williams %R
            'high_14_history': deque(maxlen=14),
            'low_14_history': deque(maxlen=14),
            
            # ADX
            'plus_dm_14': deque(maxlen=14),
            'minus_dm_14': deque(maxlen=14),
            'atr_for_adx': deque(maxlen=14),
            
            # Regime states
            'vol_regime': 'Medium',
            'trend_regime': 'Sideways',
        }
        
        # Track if we have enough data for each feature
        self.data_points = 0
    
    def _update_ema(self, current_value: float, prev_ema: Optional[float], alpha: float) -> float:
        """Update EMA incrementally."""
        if prev_ema is None:
            return current_value
        return prev_ema + alpha * (current_value - prev_ema)
    
    def _calculate_true_range(self, high: float, low: float, prev_close: float) -> float:
        """Calculate True Range."""
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        return max(tr1, tr2, tr3)
    
    def _update_rsi(self, close: float) -> Optional[float]:
        """Update RSI incrementally."""
        if self.state['prev_close_rsi'] is None:
            self.state['prev_close_rsi'] = close
            return 50.0
        
        delta = close - self.state['prev_close_rsi']
        gain = max(delta, 0)
        loss = abs(min(delta, 0))
        
        self.state['rsi_gains_14'].append(gain)
        self.state['rsi_losses_14'].append(loss)
        self.state['prev_close_rsi'] = close
        
        if len(self.state['rsi_gains_14']) < 14:
            return 50.0
        
        avg_gain = sum(self.state['rsi_gains_14']) / 14
        avg_loss = sum(self.state['rsi_losses_14']) / 14
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _update_atr(self, high: float, low: float, prev_close: float) -> Optional[float]:
        """Update ATR incrementally."""
        tr = self._calculate_true_range(high, low, prev_close)
        self.state['tr_history'].append(tr)
        
        if len(self.state['tr_history']) < 14:
            return None
        
        # Use simple average for initialization, then EMA
        if self.state['atr_14'] is None:
            self.state['atr_14'] = sum(self.state['tr_history']) / 14
        else:
            self.state['atr_14'] = self._update_ema(tr, self.state['atr_14'], 2/15)
        
        return self.state['atr_14']
    
    def _update_macd(self, close: float) -> tuple:
        """Update MACD incrementally."""
        # Update EMAs
        self.state['ema_12'] = self._update_ema(close, self.state['ema_12'], self.ema_alpha_fast)
        self.state['ema_26'] = self._update_ema(close, self.state['ema_26'], self.ema_alpha_slow)
        
        if self.state['ema_12'] is None or self.state['ema_26'] is None:
            return None, None, None
        
        # Calculate MACD line
        macd_line = self.state['ema_12'] - self.state['ema_26']
        self.state['macd_line'] = macd_line
        
        # Update signal line
        self.state['macd_signal'] = self._update_ema(macd_line, self.state['macd_signal'], self.ema_alpha_signal)
        
        # Calculate histogram
        if self.state['macd_signal'] is not None:
            self.state['macd_histogram'] = macd_line - self.state['macd_signal']
        
        return self.state['ema_12'], self.state['ema_26'], self.state['macd_histogram']
    
    def _update_volume_features(self, close: float, volume: float) -> dict:
        """Update volume-based features."""
        features = {}
        
        # Volume MA
        self.state['volume_history'].append(volume)
        
        if len(self.state['volume_history']) >= 20:
            if self.state['vol_ma_20'] is None:
                self.state['vol_ma_20'] = sum(list(self.state['volume_history'])[-20:]) / 20
            else:
                self.state['vol_ma_20'] = self._update_ema(volume, self.state['vol_ma_20'], 1/20)
            
            features['Vol_Ratio_20'] = volume / self.state['vol_ma_20'] if self.state['vol_ma_20'] > 0 else 1.0
        
        if len(self.state['volume_history']) >= 50:
            if self.state['vol_ma_50'] is None:
                self.state['vol_ma_50'] = sum(list(self.state['volume_history'])[-50:]) / 50
            else:
                self.state['vol_ma_50'] = self._update_ema(volume, self.state['vol_ma_50'], 1/50)
            
            features['Vol_Ratio_50'] = volume / self.state['vol_ma_50'] if self.state['vol_ma_50'] > 0 else 1.0
        
        # OBV
        prev_close = self.state['price_history'][-1] if len(self.state['price_history']) > 0 else close
        obv_change = np.sign(close - prev_close) * volume
        self.state['obv'] += obv_change
        self.state['obv_history'].append(self.state['obv'])
        
        if len(self.state['obv_history']) >= 5:
            features['OBV_ROC_5d'] = (self.state['obv'] - list(self.state['obv_history'])[-5]) / abs(list(self.state['obv_history'])[-5]) * 100 if list(self.state['obv_history'])[-5] != 0 else 0
        
        if len(self.state['obv_history']) >= 20:
            features['OBV_ROC_20d'] = (self.state['obv'] - list(self.state['obv_history'])[-20]) / abs(list(self.state['obv_history'])[-20]) * 100 if list(self.state['obv_history'])[-20] != 0 else 0
        
        # AD (Accumulation/Distribution)
        high = self.state['high_history'][-1] if self.state['high_history'] else close
        low = self.state['low_history'][-1] if self.state['low_history'] else close
        mfm = ((close - low) - (high - close)) / (high - low) if (high - low) > 0 else 0.5
        mfv = mfm * volume
        self.state['ad'] += mfv
        self.state['ad_history'].append(self.state['ad'])
        
        if len(self.state['ad_history']) >= 5:
            features['AD_ROC_5d'] = (self.state['ad'] - list(self.state['ad_history'])[-5]) / abs(list(self.state['ad_history'])[-5]) * 100 if list(self.state['ad_history'])[-5] != 0 else 0
        
        if len(self.state['ad_history']) >= 20:
            features['AD_ROC_20d'] = (self.state['ad'] - list(self.state['ad_history'])[-20]) / abs(list(self.state['ad_history'])[-20]) * 100 if list(self.state['ad_history'])[-20] != 0 else 0
        
        return features
    
    def _update_stochastic(self, high: float, low: float, close: float) -> dict:
        """Update Stochastic Oscillator."""
        features = {}
        
        self.state['high_14_history'].append(high)
        self.state['low_14_history'].append(low)
        
        if len(self.state['high_14_history']) >= 14:
            highest_high = max(self.state['high_14_history'])
            lowest_low = min(self.state['low_14_history'])
            
            if highest_high - lowest_low > 0:
                stoch_k = (close - lowest_low) / (highest_high - lowest_low) * 100
            else:
                stoch_k = 50.0
            
            self.state['stoch_k_history'].append(stoch_k)
            
            if len(self.state['stoch_k_history']) >= 3:
                self.state['stoch_d_14'] = sum(list(self.state['stoch_k_history'])[-3:]) / 3
            
            features['Stoch_14_K'] = stoch_k
            if self.state['stoch_d_14'] is not None:
                features['Stoch_14_D'] = self.state['stoch_d_14']
        
        return features
    
    def _update_regime_features(self, close: float, atr_pct: float) -> dict:
        """Update market regime features."""
        features = {}
        
        # Volatility regime
        if atr_pct is not None:
            if atr_pct < 1.5:
                self.state['vol_regime'] = 'Low'
            elif atr_pct < 3.0:
                self.state['vol_regime'] = 'Medium'
            else:
                self.state['vol_regime'] = 'High'
        
        features['Vol_Regime'] = self.state['vol_regime']
        
        # Trend regime (based on MA slopes)
        if len(self.state['price_history']) >= 50:
            recent_prices = list(self.state['price_history'])[-50:]
            ma_slope = ((recent_prices[-1] - recent_prices[0]) / recent_prices[0]) * 100
            
            if ma_slope > 0.2:
                self.state['trend_regime'] = 'Strong Bull'
            elif ma_slope > 0.05:
                self.state['trend_regime'] = 'Weak Bull'
            elif ma_slope > -0.05:
                self.state['trend_regime'] = 'Sideways'
            elif ma_slope > -0.2:
                self.state['trend_regime'] = 'Weak Bear'
            else:
                self.state['trend_regime'] = 'Strong Bear'
        
        features['Trend_Regime'] = self.state['trend_regime']
        
        return features
    
    def update(self, bar: Dict) -> Dict:
        """
        Update all features for a new bar.
        
        Args:
            bar: OHLCV bar dictionary with keys: Open, High, Low, Close, Volume, timestamp
            
        Returns:
            Dictionary with all calculated features
        """
        open_price = bar['Open']
        high = bar['High']
        low = bar['Low']
        close = bar['Close']
        volume = bar['Volume']
        timestamp = bar['timestamp']
        
        # Update rolling windows
        self.state['price_history'].append(close)
        self.state['volume_history'].append(volume)
        self.state['high_history'].append(high)
        self.state['low_history'].append(low)
        
        self.data_points += 1
        
        # Initialize features with basic OHLCV
        features = {
            'timestamp': timestamp,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume,
        }
        
        prev_close = self.state['price_history'][-2] if len(self.state['price_history']) > 1 else close
        
        # Update EMA-based features
        self.state['ema_50'] = self._update_ema(close, self.state['ema_50'], 2/51)
        self.state['ema_100'] = self._update_ema(close, self.state['ema_100'], 2/101)
        self.state['ema_200'] = self._update_ema(close, self.state['ema_200'], 2/201)
        
        if self.state['ema_50'] is not None:
            features['Dist_MA_50'] = (close - self.state['ema_50']) / self.state['ema_50'] * 100
        if self.state['ema_100'] is not None:
            features['Dist_MA_100'] = (close - self.state['ema_100']) / self.state['ema_100'] * 100
        if self.state['ema_200'] is not None:
            features['Dist_MA_200'] = (close - self.state['ema_200']) / self.state['ema_200'] * 100
        
        # Update RSI
        rsi = self._update_rsi(close)
        if rsi is not None:
            features['RSI_14'] = rsi
        
        # Update ATR
        atr = self._update_atr(high, low, prev_close)
        if atr is not None and close > 0:
            atr_pct = atr / close * 100
            features['ATR_14_Pct'] = atr_pct
        else:
            atr_pct = None
        
        # Update MACD
        ema_12, ema_26, macd_hist = self._update_macd(close)
        if macd_hist is not None:
            features['MACD_Histogram'] = macd_hist
            features['MACD_Signal_Pct'] = (self.state['macd_signal'] / close * 100) if self.state['macd_signal'] is not None else 0
        
        # Update volume features
        vol_features = self._update_volume_features(close, volume)
        features.update(vol_features)
        
        # Update Stochastic
        stoch_features = self._update_stochastic(high, low, close)
        features.update(stoch_features)
        
        # Update regime features
        regime_features = self._update_regime_features(close, atr_pct)
        features.update(regime_features)
        
        # Additional derived features
        if len(self.state['price_history']) >= 5:
            features['Price_Change_5d'] = (close - list(self.state['price_history'])[-5]) / list(self.state['price_history'])[-5] * 100
        
        if len(self.state['volume_history']) >= 5:
            features['Vol_Change_5d'] = (volume - list(self.state['volume_history'])[-5]) / list(self.state['volume_history'])[-5] * 100
        
        # Price-volume divergence
        if 'Price_Change_5d' in features and 'Vol_Change_5d' in features:
            features['PV_Divergence'] = int(
                (features['Price_Change_5d'] > 0 and features['Vol_Change_5d'] < 0) or
                (features['Price_Change_5d'] < 0 and features['Vol_Change_5d'] > 0)
            )
        
        logger.debug(f"Updated {len(features)} features for bar {timestamp}")
        
        return features
    
    def get_state(self) -> Dict:
        """Get the current calculation state."""
        return self.state.copy()
    
    def reset(self):
        """Reset the calculator state."""
        self._initialize_state()
        logger.info("Feature calculator state reset")


if __name__ == "__main__":
    # Demo usage
    import time
    
    calculator = IncrementalFeatureCalculator()
    
    # Simulate some bars
    base_price = 100.0
    for i in range(20):
        # Generate synthetic bar
        change = np.random.normal(0, 0.5)
        open_price = base_price
        close_price = open_price + change
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.2))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.2))
        volume = int(1000000 * np.random.uniform(0.8, 1.2))
        
        bar = {
            'timestamp': datetime.now(),
            'Open': round(open_price, 2),
            'High': round(high_price, 2),
            'Low': round(low_price, 2),
            'Close': round(close_price, 2),
            'Volume': volume
        }
        
        features = calculator.update(bar)
        
        print(f"\nBar {i+1}:")
        print(f"  Close: ${close_price:.2f}")
        print(f"  RSI_14: {features.get('RSI_14', 'N/A'):.2f}" if 'RSI_14' in features else "  RSI_14: N/A")
        print(f"  ATR_14_Pct: {features.get('ATR_14_Pct', 'N/A'):.2f}%" if 'ATR_14_Pct' in features else "  ATR_14_Pct: N/A")
        print(f"  MACD_Histogram: {features.get('MACD_Histogram', 'N/A'):.4f}" if 'MACD_Histogram' in features else "  MACD_Histogram: N/A")
        print(f"  Vol_Regime: {features.get('Vol_Regime', 'N/A')}")
        print(f"  Trend_Regime: {features.get('Trend_Regime', 'N/A')}")
        
        base_price = close_price
        time.sleep(0.1)