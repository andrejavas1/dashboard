"""
Phase 3: Market State Feature Engineering Module
Calculates comprehensive market state features for pattern discovery.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    Calculates comprehensive market state features for each historical candle.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the feature engineering system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.run_mode = self.config.get('run_mode', 'full')
        
        # Feature parameters
        self.ma_periods = self.config['features']['moving_averages']
        self.roc_periods = self.config['features']['rate_of_change_periods']
        self.atr_periods = self.config['features']['atr_periods']
        self.rsi_periods = self.config['features']['rsi_periods']
        self.adx_periods = self.config['features']['adx_periods']
        self.volume_avg_periods = self.config['features']['volume_avg_periods']
        self.high_low_periods = self.config['features']['high_low_periods']
        self.sr_range = self.config['features']['support_resistance_range']
        
        # MACD parameters
        self.macd_fast = self.config['features']['macd_params']['fast']
        self.macd_slow = self.config['features']['macd_params']['slow']
        self.macd_signal = self.config['features']['macd_params']['signal']
        
        # Adjust parameters for quick/ultra mode
        if self.run_mode == 'ultra':
            # Ultra mode: minimal features for maximum speed
            self.ma_periods = [20]  # Single MA
            self.roc_periods = [5]  # Single ROC
            self.atr_periods = [14]
            self.rsi_periods = [14]
            self.adx_periods = [14]
            self.volume_avg_periods = [10]
            self.high_low_periods = [10]
            logger.info("Ultra Mode: Minimal feature calculation parameters")
        elif self.run_mode == 'quick':
            self.ma_periods = [20, 50]  # Reduced from [10, 20, 50, 100, 200]
            self.roc_periods = [5, 10]  # Reduced from [1, 3, 5, 10, 20]
            self.atr_periods = [14]  # Reduced from [5, 10, 14, 20]
            self.rsi_periods = [14]  # Reduced from [7, 14, 21, 28]
            self.adx_periods = [14]  # Reduced from [14, 20]
            self.volume_avg_periods = [10, 20]  # Reduced from [5, 10, 20, 50]
            self.high_low_periods = [10, 20]  # Reduced from [5, 10, 20, 50, 100]
            logger.info("Quick Mode: Reduced feature calculation parameters")
        
        logger.info(f"Run Mode: {self.run_mode.upper()}")
        
        # Data storage
        self.data = None
        self.features = None
        self.feature_list = []
        
        # Caching for expensive calculations
        self._cache = {}
        
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
            'features': {
                'moving_averages': [10, 20, 50, 100, 200],
                'rate_of_change_periods': [1, 3, 5, 10, 20],
                'atr_periods': [5, 10, 14, 20],
                'rsi_periods': [7, 14, 21, 28],
                'macd_params': {'fast': 12, 'slow': 26, 'signal': 9},
                'adx_periods': [14, 20],
                'volume_avg_periods': [5, 10, 20, 50],
                'high_low_periods': [5, 10, 20, 50, 100],
                'support_resistance_range': 2
            }
        }
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load movement labeled data from file.
        
        Args:
            data_path: Path to movement labeled data CSV file
            
        Returns:
            DataFrame with OHLCV and movement data
        """
        logger.info(f"Loading data from {data_path}")
        self.data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
        logger.info(f"Loaded {len(self.data)} records with {len(self.data.columns)} columns")
        return self.data
    
    def calculate_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price-based features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with price features added
        """
        logger.info("Calculating price-based features...")
        
        # Distance from moving averages
        for period in self.ma_periods:
            ma = data['Close'].rolling(window=period).mean()
            data[f'Dist_MA_{period}'] = (data['Close'] - ma) / ma * 100
        
        # Rate of change (returns)
        for period in self.roc_periods:
            data[f'ROC_{period}d'] = data['Close'].pct_change(period) * 100
        
        # 52-week range percentile (only relative percentage)
        week_52_high = data['High'].rolling(window=252).max()
        week_52_low = data['Low'].rolling(window=252).min()
        # Only store relative position in range, not absolute high/low values
        data['52w_Range_Pct'] = (data['Close'] - week_52_low) / (week_52_high - week_52_low) * 100
        
        # Proximity to recent highs/lows (only relative features, no absolute values)
        for period in self.high_low_periods:
            # Calculate temporary highs/lows for relative distance calculation
            period_high = data['High'].rolling(window=period).max()
            period_low = data['Low'].rolling(window=period).min()
            # Only store relative distance features (percentage-based)
            data[f'Dist_{period}d_High'] = (data['Close'] - period_high) / period_high * 100
            data[f'Dist_{period}d_Low'] = (data['Close'] - period_low) / period_low * 100
        
        # Candle patterns (only percentage-based, no absolute values)
        body = abs(data['Close'] - data['Open'])
        range_val = data['High'] - data['Low']
        data['Body_Range_Ratio'] = body / range_val * 100
        data['Upper_Shadow_Ratio'] = (data['High'] - data[['Open', 'Close']].max(axis=1)) / range_val * 100
        data['Lower_Shadow_Ratio'] = (data[['Open', 'Close']].min(axis=1) - data['Low']) / range_val * 100
        # Doji detection (boolean)
        data['Doji'] = (data['Body_Range_Ratio'] < 10).astype(int)
        # Hammer/Hanging Man detection (boolean)
        data['Hammer'] = ((data['Lower_Shadow_Ratio'] > 60) & (data['Upper_Shadow_Ratio'] < 10)).astype(int)
        data['Shooting_Star'] = ((data['Upper_Shadow_Ratio'] > 60) & (data['Lower_Shadow_Ratio'] < 10)).astype(int)
        
        # Gap characteristics (only relative percentage)
        data['Gap_Pct'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1) * 100
        data['Gap_Up'] = (data['Gap_Pct'] > 0).astype(int)
        data['Gap_Down'] = (data['Gap_Pct'] < 0).astype(int)
        
        return data
    
    def calculate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility features added
        """
        logger.info("Calculating volatility features...")
        
        # True Range (only used for ATR calculation, not stored as absolute value)
        tr = np.maximum(
            data['High'] - data['Low'],
            np.maximum(
                abs(data['High'] - data['Close'].shift(1)),
                abs(data['Low'] - data['Close'].shift(1))
            )
        )
        data['TR'] = tr  # Only used internally for ATR calculation
        
        # ATR (only percentage-based, no absolute values)
        for period in self.atr_periods:
            atr = tr.rolling(window=period).mean()
            # Only store percentage-based ATR, not absolute values
            data[f'ATR_{period}_Pct'] = atr / data['Close'] * 100
        
        # ATR percentile rank (vs last 100 days)
        for period in self.atr_periods:
            atr_pct_col = f'ATR_{period}_Pct'
            if atr_pct_col in data.columns:
                data[f'ATR_{period}_Percentile'] = data[atr_pct_col].rolling(window=100).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
                )
        
        # Bollinger Bands (only relative features, no absolute values)
        for period in [self.ma_periods[0]]:  # Use first MA period
            bb_middle = data['Close'].rolling(window=period).mean()
            bb_std = data['Close'].rolling(window=period).std()
            # Only store relative width and position, not absolute band values
            data[f'BB_Width_{period}'] = (4 * bb_std) / bb_middle * 100
            data[f'BB_Position_{period}'] = (data['Close'] - (bb_middle - 2 * bb_std)) / (4 * bb_std) * 100
            # BB squeeze detection (boolean)
            data[f'BB_Squeeze_{period}'] = (data[f'BB_Width_{period}'] < data[f'BB_Width_{period}'].rolling(50).quantile(0.2)).astype(int)
        else:
            # Fallback if no MA periods available
            data['BB_Width_20'] = 0
            data['BB_Position_20'] = 50
            data['BB_Squeeze_20'] = 0
        
        # Intraday range (already percentage-based)
        data['Intraday_Range'] = (data['High'] - data['Low']) / data['Close'] * 100
        
        # Volatility trend (already percentage-based)
        for period in self.atr_periods:
            atr_pct_col = f'ATR_{period}_Pct'
            if atr_pct_col in data.columns:
                data[f'ATR_{period}_Trend'] = data[atr_pct_col] - data[atr_pct_col].shift(10)
        
        return data
    
    def calculate_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with momentum features added
        """
        logger.info("Calculating momentum features...")
        
        # RSI
        for period in self.rsi_periods:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD (only use relative features, not absolute values)
        exp1 = data['Close'].ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = data['Close'].ewm(span=self.macd_slow, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        # Only store histogram (relative difference) and signal line (normalized to price)
        data['MACD_Histogram'] = macd - macd_signal
        data['MACD_Signal_Pct'] = (macd_signal / data['Close']) * 100
        # MACD crossover signal (boolean)
        data['MACD_Bull_Cross'] = ((macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))).astype(int)
        data['MACD_Bear_Cross'] = ((macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1))).astype(int)
        
        # Stochastic Oscillator
        for period in [self.rsi_periods[0]]:  # Use first RSI period
            data[f'Stoch_{period}_K'] = (data['Close'] - data['Low'].rolling(period).min()) / \
                                       (data['High'].rolling(period).max() - data['Low'].rolling(period).min()) * 100
            data[f'Stoch_{period}_D'] = data[f'Stoch_{period}_K'].rolling(3).mean()
        
        # Rate of Change oscillators
        for period in self.roc_periods:
            data[f'ROC_Oscillator_{period}'] = data['Close'].pct_change(period) * 100
        
        # Momentum divergence flags (use percentage-based price momentum)
        data['Price_Momentum_Pct'] = data['Close'].pct_change(5) * 100
        rsi_col = f'RSI_{self.rsi_periods[0]}'
        if rsi_col in data.columns:
            data['RSI_Momentum'] = data[rsi_col] - data[rsi_col].shift(5)
        else:
            data['RSI_Momentum'] = 0
        data['Divergence_Flag'] = ((data['Price_Momentum_Pct'] > 0) & (data['RSI_Momentum'] < 0) |
                                   (data['Price_Momentum_Pct'] < 0) & (data['RSI_Momentum'] > 0)).astype(int)
        
        # Williams %R
        for period in [self.rsi_periods[0]]:  # Use first RSI period
            highest_high = data['High'].rolling(window=period).max()
            lowest_low = data['Low'].rolling(window=period).min()
            data[f'Williams_R_{period}'] = (highest_high - data['Close']) / (highest_high - lowest_low) * -100
        
        # CCI (Commodity Channel Index)
        for period in [self.ma_periods[0]]:  # Use first MA period
            tp = (data['High'] + data['Low'] + data['Close']) / 3
            sma_tp = tp.rolling(window=period).mean()
            mean_deviation = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
            data[f'CCI_{period}'] = (tp - sma_tp) / (0.015 * mean_deviation)
        
        # Chaikin Oscillator
        # Money Flow Multiplier
        mfm = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
        mfm = mfm.fillna(0)
        # Money Flow Volume
        mfv = mfm * data['Volume']
        # Accumulation Distribution Line
        adl = mfv.cumsum()
        # Chaikin Oscillator (3-day EMA - 10-day EMA of ADL)
        chaikin_ema3 = adl.ewm(span=3, adjust=False).mean()
        chaikin_ema10 = adl.ewm(span=10, adjust=False).mean()
        data['Chaikin_Oscillator'] = chaikin_ema3 - chaikin_ema10
        
        return data
    
    def calculate_enhanced_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate enhanced momentum features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with enhanced momentum features added
        """
        logger.info("Calculating enhanced momentum features...")
        
        # Rate of change of RSI (momentum of momentum)
        for period in self.rsi_periods:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            data[f'RSI_{period}_ROC'] = rsi.diff(5)  # 5-day ROC of RSI
        
        # MACD slope (rate of change of MACD)
        exp1 = data['Close'].ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = data['Close'].ewm(span=self.macd_slow, adjust=False).mean()
        macd = exp1 - exp2
        data['MACD_Slope'] = macd.diff(3)  # 3-day slope
        
        # Price acceleration (second derivative)
        data['Price_Acceleration'] = data['Close'].pct_change().diff(3)
        
        # Volume momentum
        data['Volume_Momentum'] = data['Volume'].pct_change(5)
        
        return data
    
    def calculate_cycle_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cycle-based features using a simplified approach.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with cycle features added
        """
        logger.info("Calculating cycle features...")
        
        # Dominant cycle period estimation using autocorrelation
        # This is a simplified approach to cycle analysis
        price_changes = data['Close'].pct_change()
        
        # Calculate autocorrelation at different lags (1-20 days)
        cycle_periods = []
        for i in range(len(data)):
            if i < 20:  # Need at least 20 data points
                cycle_periods.append(0)
                continue
                
            # Get recent 50 data points
            recent_data = price_changes.iloc[max(0, i-50):i]
            
            # Calculate autocorrelation at different lags
            autocorr_values = []
            for lag in range(1, min(21, len(recent_data))):
                if len(recent_data) > lag:
                    autocorr = recent_data.autocorr(lag=lag)
                    autocorr_values.append(abs(autocorr) if not np.isnan(autocorr) else 0)
                else:
                    autocorr_values.append(0)
            
            # Find the lag with highest autocorrelation (dominant cycle)
            if autocorr_values:
                dominant_cycle = np.argmax(autocorr_values) + 1
                cycle_periods.append(dominant_cycle)
            else:
                cycle_periods.append(0)
        
        data['Dominant_Cycle'] = cycle_periods
        
        # Cycle phase (normalized to 0-100)
        # This is a simplified representation of where we are in the cycle
        data['Cycle_Phase'] = (data['Dominant_Cycle'].rolling(10).apply(
            lambda x: (x.iloc[-1] / 20 * 100) if len(x) > 0 and not np.isnan(x.iloc[-1]) else 50
        )).fillna(50)
        
        return data
    
    def calculate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume features added
        """
        logger.info("Calculating volume features...")
        
        # Volume vs average (only relative ratio, no absolute volume values)
        for period in self.volume_avg_periods:
            vol_ma = data['Volume'].rolling(window=period).mean()
            # Only store relative ratio, not absolute volume MA values
            data[f'Vol_Ratio_{period}'] = data['Volume'] / vol_ma
        
        # Volume percentile rank
        for period in self.volume_avg_periods:
            data[f'Vol_Percentile_{period}'] = data['Volume'].rolling(window=100).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
            )
        
        # On-Balance Volume (OBV) - only use relative trend, not absolute values
        obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        obv_ma = obv.rolling(self.ma_periods[0]).mean()
        # Only store relative trend (boolean), not absolute OBV values
        data['OBV_Trend'] = (obv > obv_ma).astype(int)
        # Store OBV rate of change (relative) instead of absolute values
        # Cap extreme values to prevent outliers when OBV crosses zero
        obv_roc_5d = obv.pct_change(5) * 100
        obv_roc_20d = obv.pct_change(20) * 100
        data['OBV_ROC_5d'] = obv_roc_5d.clip(-500, 500)
        data['OBV_ROC_20d'] = obv_roc_20d.clip(-500, 500)
        
        # Accumulation/Distribution Line
        mfv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
        mfv = mfv.fillna(0.5)
        ad = (mfv * data['Volume']).cumsum()
        
        # Chaikin A/D Oscillator (industry standard) - EMA difference instead of percentage change
        # This avoids the percentage change problem on cumulative values
        ad_ema_fast = ad.ewm(span=3, adjust=False).mean()
        ad_ema_slow = ad.ewm(span=10, adjust=False).mean()
        data['Chaikin_Oscillator'] = ad_ema_fast - ad_ema_slow
        
        # Keep backward compatibility: use absolute difference instead of percentage
        data['AD_ROC_5d'] = ad.diff(5)  # Absolute change, not percentage
        data['AD_ROC_20d'] = ad.diff(20)  # Absolute change, not percentage
        
        # Store AD trend direction (boolean)
        data['AD_Trend'] = (ad.diff(5) > 0).astype(int)
        
        # Volume spike detection - use first volume ratio period
        vol_ratio_col = f'Vol_Ratio_{self.volume_avg_periods[0]}'
        if vol_ratio_col in data.columns:
            data['Vol_Spike_2x'] = (data[vol_ratio_col] > 2).astype(int)
            data['Vol_Spike_3x'] = (data[vol_ratio_col] > 3).astype(int)
        else:
            data['Vol_Spike_2x'] = 0
            data['Vol_Spike_3x'] = 0
        
        # Price-volume divergence (only percentage-based)
        data['Price_Change_5d'] = data['Close'].pct_change(5) * 100
        data['Vol_Change_5d'] = data['Volume'].pct_change(5) * 100
        data['PV_Divergence'] = ((data['Price_Change_5d'] > 0) & (data['Vol_Change_5d'] < 0) |
                                 (data['Price_Change_5d'] < 0) & (data['Vol_Change_5d'] > 0)).astype(int)
        
        return data
    
    def calculate_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with trend features added
        """
        logger.info("Calculating trend features...")
        
        # ADX (Average Directional Index) - only use relative indicators
        for period in self.atr_periods:  # Use atr_periods since they're the same
            # Calculate +DM, -DM
            up_move = data['High'] - data['High'].shift(1)
            down_move = data['Low'].shift(1) - data['Low']
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            plus_dm_ma = pd.Series(plus_dm).rolling(window=period).mean()
            minus_dm_ma = pd.Series(minus_dm).rolling(window=period).mean()
            
            # Calculate +DI, -DI (relative indicators)
            atr_pct = data[f'ATR_{period}_Pct']
            # Use close-based ATR approximation for DI calculation
            atr_approx = data['Close'] * atr_pct / 100
            data[f'Plus_DI_{period}'] = (plus_dm_ma / atr_approx) * 100
            data[f'Minus_DI_{period}'] = (minus_dm_ma / atr_approx) * 100
            
            # Calculate ADX (relative indicator)
            dx = abs(data[f'Plus_DI_{period}'] - data[f'Minus_DI_{period}']) / \
                 (data[f'Plus_DI_{period}'] + data[f'Minus_DI_{period}']) * 100
            data[f'ADX_{period}'] = dx.rolling(window=period).mean()
        
        # Trend direction classification
        for period in self.atr_periods:  # Use atr_periods since ADX uses the same periods
            adx_col = f'ADX_{period}'
            plus_di_col = f'Plus_DI_{period}'
            minus_di_col = f'Minus_DI_{period}'
            
            if adx_col in data.columns and plus_di_col in data.columns and minus_di_col in data.columns:
                data[f'Trend_Dir_{period}'] = 'Neutral'
                data.loc[data[adx_col] > 25, f'Trend_Dir_{period}'] = 'Sideways'
                data.loc[(data[adx_col] > 25) & (data[plus_di_col] > data[minus_di_col]), f'Trend_Dir_{period}'] = 'Up'
                data.loc[(data[adx_col] > 25) & (data[minus_di_col] > data[plus_di_col]), f'Trend_Dir_{period}'] = 'Down'
        
        # Trend strength score - use first ADX period
        adx_col = f'ADX_{self.adx_periods[0]}'
        if adx_col in data.columns:
            data['Trend_Strength'] = data[adx_col] / 100
        else:
            data['Trend_Strength'] = 0
        
        # Moving average alignment
        ma_cols = [f'Dist_MA_{p}' for p in self.ma_periods if f'Dist_MA_{p}' in data.columns]
        data['MA_Alignment'] = 0
        for i in range(len(ma_cols) - 1):
            data['MA_Alignment'] += (data[ma_cols[i]] > data[ma_cols[i+1]]).astype(int)
        data['MA_Alignment_Score'] = data['MA_Alignment'] / (len(ma_cols) - 1) * 100
        
        # Slope of key moving averages (already relative as percentage diff)
        for period in self.ma_periods:
            data[f'MA_{period}_Slope'] = data[f'Dist_MA_{period}'].diff(5)
        
        return data
    
    def calculate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market regime features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with regime features added
        """
        logger.info("Calculating market regime features...")
        
        # Volatility regime (based on ATR terciles) - use first ATR period
        atr_col = f'ATR_{self.atr_periods[0]}_Pct'
        if atr_col in data.columns:
            atr_pct = data[atr_col].dropna()
            if len(atr_pct) > 0:
                low_vol = atr_pct.quantile(0.33)
                high_vol = atr_pct.quantile(0.67)
                
                data['Vol_Regime'] = 'Medium'
                data.loc[data[atr_col] <= low_vol, 'Vol_Regime'] = 'Low'
                data.loc[data[atr_col] >= high_vol, 'Vol_Regime'] = 'High'
        else:
            # Fallback if no ATR column exists
            data['Vol_Regime'] = 'Medium'
        
        # Trend regime - use available MA slopes
        ma_slopes = [f'MA_{period}_Slope' for period in self.ma_periods if f'MA_{period}_Slope' in data.columns]
        
        if len(ma_slopes) >= 1:
            ma_short_slope = data[ma_slopes[0]]
            if len(ma_slopes) >= 2:
                ma_long_slope = data[ma_slopes[1]]
            else:
                ma_long_slope = ma_short_slope
            
            data['Trend_Regime'] = 'Sideways'
            data.loc[(ma_short_slope > 0.1) & (ma_long_slope > 0.1), 'Trend_Regime'] = 'Strong Bull'
            data.loc[(ma_short_slope > 0) & (ma_short_slope <= 0.1) & (ma_long_slope > 0), 'Trend_Regime'] = 'Weak Bull'
            data.loc[(ma_short_slope < -0.1) & (ma_long_slope < -0.1), 'Trend_Regime'] = 'Strong Bear'
            data.loc[(ma_short_slope < 0) & (ma_short_slope >= -0.1) & (ma_long_slope < 0), 'Trend_Regime'] = 'Weak Bear'
        else:
            data['Trend_Regime'] = 'Sideways'
        
        # Volume regime - use first volume ratio period
        vol_col = f'Vol_Ratio_{self.volume_avg_periods[0]}'
        if vol_col in data.columns:
            vol_ratio = data[vol_col].dropna()
            if len(vol_ratio) > 0:
                low_vol = vol_ratio.quantile(0.33)
                high_vol = vol_ratio.quantile(0.67)
                
                data['Vol_Regime_Level'] = 'Normal'
                data.loc[data[vol_col] <= low_vol, 'Vol_Regime_Level'] = 'Low'
                data.loc[data[vol_col] >= high_vol, 'Vol_Regime_Level'] = 'High'
        else:
            data['Vol_Regime_Level'] = 'Normal'
        
        return data
    
    def calculate_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pattern recognition features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with pattern features added
        """
        logger.info("Calculating pattern recognition features...")
        
        # Consecutive up/down days counter
        data['Price_Up'] = (data['Close'] > data['Close'].shift(1)).astype(int)
        data['Consec_Up'] = data['Price_Up'].groupby((data['Price_Up'] != data['Price_Up'].shift()).cumsum()).cumsum()
        data['Price_Down'] = (data['Close'] < data['Close'].shift(1)).astype(int)
        data['Consec_Down'] = data['Price_Down'].groupby((data['Price_Down'] != data['Price_Down'].shift()).cumsum()).cumsum()
        
        # Higher highs / lower lows sequences
        data['Higher_High'] = (data['High'] > data['High'].shift(1)).astype(int)
        data['Higher_Low'] = (data['Low'] > data['Low'].shift(1)).astype(int)
        data['Lower_High'] = (data['High'] < data['High'].shift(1)).astype(int)
        data['Lower_Low'] = (data['Low'] < data['Low'].shift(1)).astype(int)
        
        # Support/resistance proximity (only relative features)
        support_level = data['Low'].rolling(window=self.ma_periods[0]).min()
        resistance_level = data['High'].rolling(window=self.ma_periods[0]).max()
        # Only store relative distance as percentage
        data['Dist_Support'] = (data['Close'] - support_level) / support_level * 100
        data['Dist_Resistance'] = (data['Close'] - resistance_level) / resistance_level * 100
        data['Near_Support'] = (abs(data['Dist_Support']) <= self.sr_range).astype(int)
        data['Near_Resistance'] = (abs(data['Dist_Resistance']) <= self.sr_range).astype(int)
        
        # Breakout/breakdown signals (using relative distance features)
        # Breakout: close crosses above resistance (dist_resistance goes from negative to positive)
        data['Breakout'] = ((data['Dist_Resistance'] > 0) &
                           (data['Dist_Resistance'].shift(1) <= 0)).astype(int)
        # Breakdown: close crosses below support (dist_support goes from positive to negative)
        data['Breakdown'] = ((data['Dist_Support'] < 0) &
                            (data['Dist_Support'].shift(1) >= 0)).astype(int)
        
        # Consolidation detection (low volatility periods) - use first ATR period
        atr_col = f'ATR_{self.atr_periods[0]}_Pct'
        if atr_col in data.columns:
            data['Consolidation'] = (data[atr_col] < data[atr_col].rolling(60).quantile(0.25)).astype(int)
        else:
            data['Consolidation'] = 0
        
        return data
    
    def calculate_fibonacci_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci retracement levels.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Fibonacci features added
        """
        logger.info("Calculating Fibonacci retracement features...")
        
        # Calculate recent high and low (using 20-day period)
        period_high = data['High'].rolling(window=20).max()
        period_low = data['Low'].rolling(window=20).min()
        price_range = period_high - period_low
        
        # Calculate Fibonacci levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
        # These are relative to current price position within the range
        data['Fib_23.6'] = (period_low + price_range * 0.236 - data['Close']) / data['Close'] * 100
        data['Fib_38.2'] = (period_low + price_range * 0.382 - data['Close']) / data['Close'] * 100
        data['Fib_50.0'] = (period_low + price_range * 0.500 - data['Close']) / data['Close'] * 100
        data['Fib_61.8'] = (period_low + price_range * 0.618 - data['Close']) / data['Close'] * 100
        data['Fib_78.6'] = (period_low + price_range * 0.786 - data['Close']) / data['Close'] * 100
        
        return data
    
    def calculate_vwap_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP (Volume Weighted Average Price) features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with VWAP features added
        """
        logger.info("Calculating VWAP features...")
        
        # Typical price = (High + Low + Close) / 3
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        
        # VWAP calculation using rolling window (use first MA period)
        vwap_period = self.ma_periods[0]
        vwap = (typical_price * data['Volume']).rolling(window=vwap_period).sum() / data['Volume'].rolling(window=vwap_period).sum()
        
        # Distance from VWAP as percentage
        data['VWAP_Distance'] = (data['Close'] - vwap) / vwap * 100
        
        # VWAP trend (slope over 5 periods)
        data['VWAP_Slope'] = vwap.diff(5)
        
        return data
    
    def calculate_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate temporal features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with temporal features added
        """
        logger.info("Calculating temporal features...")
        
        # Ensure index is datetime (handle timezone-aware datetimes)
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index, utc=True)
        elif data.index.tz is not None:
            # Convert timezone-aware to UTC, then remove timezone
            data.index = data.index.tz_convert('UTC').tz_localize(None)
        
        # Day of week
        data['Day_of_Week'] = data.index.dayofweek
        data['Day_of_Week_Name'] = data.index.day_name()
        
        # Month of year
        data['Month'] = data.index.month
        data['Month_Name'] = data.index.month_name()
        
        # Quarter
        data['Quarter'] = data.index.quarter
        
        # Days since last major move (>5%)
        daily_return = data['Close'].pct_change() * 100
        data['Major_Move'] = (abs(daily_return) > 5).astype(int)
        data['Days_Since_Major_Move'] = data['Major_Move'].groupby(
            (data['Major_Move'] != data['Major_Move'].shift()).cumsum()
        ).cumsum()
        
        # Time since 52-week high/low
        data['Days_Since_52w_High'] = 0
        data['Days_Since_52w_Low'] = 0
        
        # Vectorized calculation of days since 52-week high/low
        # Using rolling window approach for better performance
        high_52w = data['High'].rolling(window=252, min_periods=1).max()
        low_52w = data['Low'].rolling(window=252, min_periods=1).min()
        
        # Find the index of the last occurrence of each 52-week high/low
        # Using a more efficient approach with rolling apply
        def get_days_since_high(x):
            if len(x) == 0:
                return 0
            current_idx = x.index[-1]
            high_idx = x.idxmax()
            return (current_idx - high_idx).days
        
        def get_days_since_low(x):
            if len(x) == 0:
                return 0
            current_idx = x.index[-1]
            low_idx = x.idxmin()
            return (current_idx - low_idx).days
        
        # Calculate days since high/low using rolling window
        data['Days_Since_52w_High'] = data['High'].rolling(window=252, min_periods=1).apply(
            get_days_since_high, raw=False
        )
        data['Days_Since_52w_Low'] = data['Low'].rolling(window=252, min_periods=1).apply(
            get_days_since_low, raw=False
        )
        
        return data
    
    def clean_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean features by replacing infinity values and handling NaN.
        
        Args:
            data: DataFrame with features
            
        Returns:
            DataFrame with cleaned features
        """
        logger.info("Cleaning features...")
        
        # Replace infinity with NaN
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # For numeric columns, fill NaN with forward fill then backward fill
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Forward fill
            data[col] = data[col].ffill()
            # Backward fill for remaining NaN
            data[col] = data[col].bfill()
            # Fill any remaining NaN with 0 (for features that should be 0 when undefined)
            data[col] = data[col].fillna(0)
        
        return data
    
    def calculate_all_features(self) -> pd.DataFrame:
        """
        Calculate all features for the dataset.
        
        Returns:
            DataFrame with all features calculated
        """
        logger.info("=" * 60)
        logger.info("PHASE 3: Market State Feature Engineering")
        logger.info("=" * 60)
        
        if self.data is None or self.data.empty:
            logger.error("No data loaded. Call load_data() first.")
            return pd.DataFrame()
        
        # Make a copy to avoid modifying original data
        features = self.data.copy()
        
        # Calculate all feature groups
        features = self.calculate_price_features(features)
        features = self.calculate_volatility_features(features)
        features = self.calculate_momentum_features(features)
        features = self.calculate_volume_features(features)
        features = self.calculate_trend_features(features)
        features = self.calculate_regime_features(features)
        features = self.calculate_pattern_features(features)
        features = self.calculate_temporal_features(features)
        
        # Calculate enhanced features (skip some in quick mode for speed)
        features = self.calculate_fibonacci_features(features)
        features = self.calculate_vwap_features(features)
        
        # Skip these expensive features in quick/ultra mode
        if self.run_mode not in ['quick', 'ultra']:
            features = self.calculate_enhanced_momentum_features(features)
            features = self.calculate_cycle_features(features)
        else:
            logger.info(f"{self.run_mode.upper()} Mode: Skipping enhanced momentum and cycle features")
        
        # Clean features (replace infinity, handle NaN)
        features = self.clean_features(features)
        
        self.features = features
        self.feature_list = list(features.columns)
        
        logger.info(f"\nFeature Engineering Complete:")
        logger.info(f"  Total Features: {len(self.feature_list)}")
        logger.info(f"  Records: {len(features)}")
        
        # Log feature categories
        feature_categories = {
            'Price': [c for c in self.feature_list if any(x in c for x in ['Dist_MA', 'ROC', '52w', 'Body', 'Gap'])],
            'Volatility': [c for c in self.feature_list if any(x in c for x in ['ATR', 'BB', 'TR', 'Intraday'])],
            'Momentum': [c for c in self.feature_list if any(x in c for x in ['RSI', 'MACD', 'Stoch', 'Divergence', 'Williams_R', 'CCI', 'Chaikin'])],
            'Volume': [c for c in self.feature_list if any(x in c for x in ['Vol', 'OBV', 'AD', 'PV_Divergence'])],
            'Trend': [c for c in self.feature_list if any(x in c for x in ['ADX', 'Trend', 'MA_', 'Slope'])],
            'Regime': [c for c in self.feature_list if any(x in c for x in ['Regime'])],
            'Pattern': [c for c in self.feature_list if any(x in c for x in ['Consec', 'Higher', 'Lower', 'Support', 'Resistance', 'Breakout', 'Breakdown', 'Consolidation'])],
            'Temporal': [c for c in self.feature_list if any(x in c for x in ['Day', 'Month', 'Quarter', 'Days_Since'])],
            'Fibonacci': [c for c in self.feature_list if 'Fib' in c],
            'VWAP': [c for c in self.feature_list if 'VWAP' in c],
            'Cycle': [c for c in self.feature_list if any(x in c for x in ['Cycle', 'Dominant'])]
        }
        
        logger.info("\nFeature Categories:")
        for category, cols in feature_categories.items():
            logger.info(f"  {category}: {len(cols)} features")
        
        return features
    
    def save_features(self, output_dir: str = "data"):
        """
        Save features to files.
        
        Args:
            output_dir: Directory to save output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.features is None:
            logger.error("No features to save")
            return
        
        # Save features
        features_path = os.path.join(output_dir, "features_matrix.csv")
        self.features.to_csv(features_path)
        logger.info(f"Features saved to {features_path}")
        
        # Save feature list
        feature_list_path = os.path.join(output_dir, "feature_list.txt")
        with open(feature_list_path, 'w') as f:
            for feature in self.feature_list:
                f.write(f"{feature}\n")
        logger.info(f"Feature list saved to {feature_list_path}")
    
    def run_phase3(self, data_path: str = None) -> pd.DataFrame:
        """
        Run complete Phase 3: Market State Feature Engineering.
        
        Args:
            data_path: Path to movement labeled data CSV file
            
        Returns:
            DataFrame with all features
        """
        logger.info("\n" + "=" * 60)
        logger.info("STARTING PHASE 3: FEATURE ENGINEERING")
        logger.info("=" * 60)
        
        # Load data
        if data_path:
            self.load_data(data_path)
        elif self.data is None:
            # Try default path
            default_path = os.path.join("data", "movement_labeled_data.csv")
            if os.path.exists(default_path):
                self.load_data(default_path)
            else:
                logger.error("No data path provided and default file not found")
                return pd.DataFrame()
        
        # Calculate all features
        self.calculate_all_features()
        
        # Save results
        self.save_features()
        
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3 COMPLETE")
        logger.info("=" * 60)
        
        return self.features
    
    def calculate_feature_importance(self, features: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Calculate feature importance using correlation analysis.
        
        Args:
            features: DataFrame with features
            target_col: Target column for supervised importance (optional)
            
        Returns:
            DataFrame with feature importance scores
        """
        logger.info("Calculating feature importance...")
        
        # Get numeric columns only
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Calculate variance for each feature (higher variance = more informative)
        variance_scores = numeric_features.var()
        
        # Calculate correlation matrix
        correlation_matrix = numeric_features.corr().abs()
        
        # Calculate mean absolute correlation for each feature (lower = better, less redundant)
        mean_correlations = correlation_matrix.mean()
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'variance': variance_scores,
            'mean_correlation': mean_correlations
        })
        
        # Normalize scores to 0-1 range
        importance_df['variance_norm'] = (importance_df['variance'] - importance_df['variance'].min()) / \
                                           (importance_df['variance'].max() - importance_df['variance'].min())
        importance_df['correlation_norm'] = 1 - (importance_df['mean_correlation'] - importance_df['mean_correlation'].min()) / \
                                              (importance_df['mean_correlation'].max() - importance_df['mean_correlation'].min())
        
        # Combined score (higher is better)
        importance_df['importance_score'] = (importance_df['variance_norm'] + importance_df['correlation_norm']) / 2
        
        # Sort by importance score
        importance_df = importance_df.sort_values('importance_score', ascending=False)
        
        return importance_df
    
    def select_top_features(self, features: pd.DataFrame, n_features: int = 100) -> List[str]:
        """
        Select top features based on importance scores.
        
        Args:
            features: DataFrame with features
            n_features: Number of top features to select
            
        Returns:
            List of top feature names
        """
        # Calculate feature importance
        importance_df = self.calculate_feature_importance(features)
        
        # Select top features
        top_features = importance_df.head(n_features).index.tolist()
        
        logger.info(f"Selected top {len(top_features)} features based on importance scores")
        
        return top_features


if __name__ == "__main__":
    # Run Phase 3
    fe = FeatureEngineering()
    features = fe.run_phase3()
    
    print(f"\nFinal Results:")
    print(f"  Total Records: {len(features)}")
    print(f"  Total Features: {len(features.columns)}")
    print(f"\nFeature Categories:")
    print(f"  Price-based: {len([c for c in features.columns if any(x in c for x in ['Dist_MA', 'ROC', '52w', 'Body', 'Gap'])])}")
    print(f"  Volatility: {len([c for c in features.columns if any(x in c for x in ['ATR', 'BB', 'TR', 'Intraday'])])}")
    print(f"  Momentum: {len([c for c in features.columns if any(x in c for x in ['RSI', 'MACD', 'Stoch', 'Divergence', 'Williams_R', 'CCI', 'Chaikin'])])}")
    print(f"  Volume: {len([c for c in features.columns if any(x in c for x in ['Vol', 'OBV', 'AD', 'PV_Divergence'])])}")
    print(f"  Trend: {len([c for c in features.columns if any(x in c for x in ['ADX', 'Trend', 'MA_', 'Slope'])])}")
    print(f"  Regime: {len([c for c in features.columns if any(x in c for x in ['Regime'])])}")
    print(f"  Pattern: {len([c for c in features.columns if any(x in c for x in ['Consec', 'Higher', 'Lower', 'Support', 'Resistance', 'Breakout', 'Breakdown', 'Consolidation'])])}")
    print(f"  Temporal: {len([c for c in features.columns if any(x in c for x in ['Day', 'Month', 'Quarter', 'Days_Since'])])}")
    print(f"  Fibonacci: {len([c for c in features.columns if 'Fib' in c])}")
    print(f"  VWAP: {len([c for c in features.columns if 'VWAP' in c])}")
    print(f"  Cycle: {len([c for c in features.columns if any(x in c for x in ['Cycle', 'Dominant'])])}")