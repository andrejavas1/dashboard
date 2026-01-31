"""
Real-Time Feature Calculator Module
Calculates all 140+ technical indicators on daily OHLCV bars.

This is the KEY component: patterns were discovered on daily bars using
50-day MAs, 14-period RSI, etc. So we must calculate features on the
reconstructed daily bars (not 15m bars) to match patterns correctly.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealtimeFeatureCalculator:
    """
    Calculates comprehensive market state features on daily OHLCV bars.
    
    This uses the same logic as Phase 3 feature engineering, but works
    on the updating daily bar reconstruction from 15m intraday data.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the real-time feature calculator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Feature parameters (same as Phase 3)
        features_config = self.config.get('features', {})
        self.ma_periods = features_config.get('moving_averages', [10, 20, 50, 100, 200])
        self.roc_periods = features_config.get('rate_of_change_periods', [1, 3, 5, 10, 20])
        self.atr_periods = features_config.get('atr_periods', [5, 10, 14, 20])
        self.rsi_periods = features_config.get('rsi_periods', [7, 14, 21, 28])
        self.adx_periods = features_config.get('adx_periods', [14, 20])
        self.volume_avg_periods = features_config.get('volume_avg_periods', [5, 10, 20, 50])
        self.high_low_periods = features_config.get('high_low_periods', [5, 10, 20, 50, 100])
        self.sr_range = features_config.get('support_resistance_range', 2)
        
        # MACD parameters
        macd_config = features_config.get('macd_params', {})
        self.macd_fast = macd_config.get('fast', 12)
        self.macd_slow = macd_config.get('slow', 26)
        self.macd_signal = macd_config.get('signal', 9)
        
        # Stochastic parameters
        self.stoch_k = features_config.get('stoch_k_period', 14)
        self.stoch_d = features_config.get('stoch_d_period', 3)
        
        # CCI period
        self.cci_period = features_config.get('cci_period', 20)
        
        logger.info(f"Realtime Feature Calculator initialized")
        logger.info(f"MA periods: {self.ma_periods}")
        logger.info(f"RSI periods: {self.rsi_periods}")
        logger.info(f"ROC periods: {self.roc_periods}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return {}
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all features on daily OHLCV data.
        
        Args:
            df: DataFrame with Date, Open, High, Low, Close, Volume columns
            
        Returns:
            DataFrame with all calculated features
        """
        if df.empty or len(df) < 50:
            logger.warning(f"Insufficient data: {len(df)} rows. Need at least 50.")
            return None
        
        # Work on a copy
        data = df.copy()
        
        # Ensure we have required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in data.columns:
                logger.error(f"Missing required column: {col}")
                return None
        
        # Calculate all feature groups
        self._calculate_price_features(data)
        self._calculate_moving_averages(data)
        self._calculate_rsi(data)
        self._calculate_macd(data)
        self._calculate_bollinger_bands(data)
        self._calculate_atr(data)
        self._calculate_roc(data)
        self._calculate_volume_features(data)
        self._calculate_volatility_features(data)
        self._calculate_trend_features(data)
        self._calculate_oscillators(data)
        self._calculate_candlestick_features(data)
        
        logger.debug(f"Calculated {len(data.columns)} features on {len(data)} rows")
        return data
    
    def _calculate_price_features(self, data: pd.DataFrame):
        """Calculate basic price-based features."""
        # Daily returns
        data['Daily_Return'] = data['Close'].pct_change()
        data['Daily_Return_Pct'] = data['Daily_Return'] * 100
        
        # Log returns
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Price position within day's range
        data['Price_Position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'] + 1e-10)
        
        # Body size and characteristics
        data['Body_Size'] = abs(data['Close'] - data['Open'])
        data['Body_Size_Pct'] = data['Body_Size'] / data['Open'] * 100
        data['Upper_Shadow'] = data['High'] - data[['Open', 'Close']].max(axis=1)
        data['Lower_Shadow'] = data[['Open', 'Close']].min(axis=1) - data['Low']
        data['Total_Range'] = data['High'] - data['Low']
        data['Total_Range_Pct'] = data['Total_Range'] / data['Open'] * 100
    
    def _calculate_moving_averages(self, data: pd.DataFrame):
        """Calculate moving averages and related features."""
        for period in self.ma_periods:
            # Simple Moving Average
            data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
            
            # Exponential Moving Average
            data[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
            
            # Distance from MA
            data[f'Close_to_SMA{period}'] = (data['Close'] - data[f'SMA_{period}']) / data[f'SMA_{period}'] * 100
            data[f'Close_to_EMA{period}'] = (data['Close'] - data[f'EMA_{period}']) / data[f'EMA_{period}'] * 100
            
            # MA slope (trend)
            data[f'SMA{period}_Slope'] = data[f'SMA_{period}'].diff()
            data[f'EMA{period}_Slope'] = data[f'EMA_{period}'].diff()
        
        # MA crossovers
        if 20 in self.ma_periods and 50 in self.ma_periods:
            data['SMA20_50_Cross'] = data['SMA_20'] - data['SMA_50']
            data['EMA20_50_Cross'] = data['EMA_20'] - data['EMA_50']
        
        if 50 in self.ma_periods and 200 in self.ma_periods:
            data['SMA50_200_Cross'] = data['SMA_50'] - data['SMA_200']
            data['Golden_Cross'] = (data['SMA_50'] > data['SMA_200']).astype(int)
    
    def _calculate_rsi(self, data: pd.DataFrame):
        """Calculate RSI for multiple periods."""
        for period in self.rsi_periods:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # RSI momentum
        if 14 in self.rsi_periods:
            data['RSI_14_Momentum'] = data['RSI_14'].diff()
            data['RSI_14_Overbought'] = (data['RSI_14'] > 70).astype(int)
            data['RSI_14_Oversold'] = (data['RSI_14'] < 30).astype(int)
    
    def _calculate_macd(self, data: pd.DataFrame):
        """Calculate MACD indicator."""
        ema_fast = data['Close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=self.macd_slow, adjust=False).mean()
        
        data['MACD_Line'] = ema_fast - ema_slow
        data['MACD_Signal'] = data['MACD_Line'].ewm(span=self.macd_signal, adjust=False).mean()
        data['MACD_Histogram'] = data['MACD_Line'] - data['MACD_Signal']
        data['MACD_Cross'] = (data['MACD_Line'] > data['MACD_Signal']).astype(int)
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame):
        """Calculate Bollinger Bands."""
        for period in [20, 50]:
            if period in self.ma_periods:
                sma = data[f'SMA_{period}']
                std = data['Close'].rolling(window=period).std()
                
                data[f'BB_Upper_{period}'] = sma + (std * 2)
                data[f'BB_Lower_{period}'] = sma - (std * 2)
                data[f'BB_Width_{period}'] = (data[f'BB_Upper_{period}'] - data[f'BB_Lower_{period}']) / sma
                data[f'BB_Position_{period}'] = (data['Close'] - data[f'BB_Lower_{period}']) / (data[f'BB_Upper_{period}'] - data[f'BB_Lower_{period}'] + 1e-10)
                data[f'BB_PctB_{period}'] = (data['Close'] - data[f'BB_Lower_{period}']) / (data[f'BB_Upper_{period}'] - data[f'BB_Lower_{period}'] + 1e-10)
    
    def _calculate_atr(self, data: pd.DataFrame):
        """Calculate Average True Range."""
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift(1))
        low_close = abs(data['Low'] - data['Close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        for period in self.atr_periods:
            data[f'ATR_{period}'] = true_range.rolling(window=period).mean()
            data[f'ATR_{period}_Pct'] = data[f'ATR_{period}'] / data['Close'] * 100
    
    def _calculate_roc(self, data: pd.DataFrame):
        """Calculate Rate of Change for multiple periods."""
        for period in self.roc_periods:
            data[f'ROC_{period}d'] = data['Close'].pct_change(periods=period) * 100
            data[f'Momentum_{period}d'] = data['Close'] - data['Close'].shift(period)
    
    def _calculate_volume_features(self, data: pd.DataFrame):
        """Calculate volume-based features."""
        # Volume moving averages
        for period in self.volume_avg_periods:
            data[f'Vol_MA_{period}'] = data['Volume'].rolling(window=period).mean()
            data[f'Vol_Ratio_{period}'] = data['Volume'] / data[f'Vol_MA_{period}']
        
        # On-Balance Volume
        obv = [0]
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        data['OBV'] = obv
        
        # OBV moving average
        data['OBV_MA_20'] = data['OBV'].rolling(window=20).mean()
        data['OBV_Trend'] = (data['OBV'] > data['OBV_MA_20']).astype(int)
        
        # Volume Rate of Change
        data['Vol_ROC_10d'] = data['Volume'].pct_change(periods=10) * 100
        
        # Accumulation/Distribution Line
        ad = []
        ad_val = 0
        for i in range(len(data)):
            mfm = ((data['Close'].iloc[i] - data['Low'].iloc[i]) - 
                   (data['High'].iloc[i] - data['Close'].iloc[i])) / (data['High'].iloc[i] - data['Low'].iloc[i] + 1e-10)
            mfv = mfm * data['Volume'].iloc[i]
            ad_val += mfv
            ad.append(ad_val)
        data['AD_Line'] = ad
        
        # Chaikin Oscillator
        data['AD_3d_EMA'] = data['AD_Line'].ewm(span=3, adjust=False).mean()
        data['AD_10d_EMA'] = data['AD_Line'].ewm(span=10, adjust=False).mean()
        data['Chaikin_Oscillator'] = data['AD_3d_EMA'] - data['AD_10d_EMA']
        
        # ADX for volume trends
        for period in [14, 20]:
            data[f'AD_ROC_{period}d'] = data['AD_Line'].pct_change(periods=period) * 100
    
    def _calculate_volatility_features(self, data: pd.DataFrame):
        """Calculate volatility features."""
        # Standard deviation of returns
        for period in [10, 20, 50]:
            data[f'Volatility_{period}d'] = data['Daily_Return'].rolling(window=period).std() * np.sqrt(252) * 100
        
        # Historical volatility
        data['Hist_Vol_20d'] = data['Daily_Return'].rolling(window=20).std() * np.sqrt(252) * 100
        
        # Average directional movement
        for period in self.adx_periods:
            data = self._calculate_adx(data, period)
    
    def _calculate_adx(self, data: pd.DataFrame, period: int):
        """Calculate ADX (Average Directional Index)."""
        plus_dm = data['High'].diff()
        minus_dm = data['Low'].diff().abs()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr1 = data['High'] - data['Low']
        tr2 = abs(data['High'] - data['Close'].shift(1))
        tr3 = abs(data['Low'] - data['Close'].shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = true_range.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        data[f'ADX_{period}'] = dx.rolling(window=period).mean()
        data[f'Plus_DI_{period}'] = plus_di
        data[f'Minus_DI_{period}'] = minus_di
        
        return data
    
    def _calculate_trend_features(self, data: pd.DataFrame):
        """Calculate trend-related features."""
        # Price relative to highs/lows
        for period in self.high_low_periods:
            data[f'High_{period}d'] = data['High'].rolling(window=period).max()
            data[f'Low_{period}d'] = data['Low'].rolling(window=period).min()
            data[f'Close_to_High{period}'] = (data['Close'] - data[f'High_{period}d']) / data[f'High_{period}d'] * 100
            data[f'Close_to_Low{period}'] = (data['Close'] - data[f'Low_{period}d']) / data[f'Low_{period}d'] * 100
            data[f'Position_in_Range_{period}'] = (data['Close'] - data[f'Low_{period}d']) / (data[f'High_{period}d'] - data[f'Low_{period}d'] + 1e-10)
        
        # Trend strength
        if 50 in self.ma_periods:
            data['Trend_Strength'] = (data['Close'] > data['SMA_50']).astype(int)
    
    def _calculate_oscillators(self, data: pd.DataFrame):
        """Calculate various oscillators."""
        # Stochastic Oscillator
        for period in [14]:
            lowest_low = data['Low'].rolling(window=period).min()
            highest_high = data['High'].rolling(window=period).max()
            data[f'Stoch_K_{period}'] = 100 * (data['Close'] - lowest_low) / (highest_high - lowest_low + 1e-10)
            data[f'Stoch_D_{period}'] = data[f'Stoch_K_{period}'].rolling(window=3).mean()
        
        # Commodity Channel Index (CCI)
        tp = (data['High'] + data['Low'] + data['Close']) / 3
        tp_ma = tp.rolling(window=self.cci_period).mean()
        tp_std = tp.rolling(window=self.cci_period).std()
        data['CCI'] = (tp - tp_ma) / (0.015 * tp_std + 1e-10)
        
        # Williams %R
        for period in [14]:
            highest_high = data['High'].rolling(window=period).max()
            lowest_low = data['Low'].rolling(window=period).min()
            data[f'Williams_R_{period}'] = -100 * (highest_high - data['Close']) / (highest_high - lowest_low + 1e-10)
        
        # Money Flow Index (MFI)
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        raw_money_flow = typical_price * data['Volume']
        
        positive_flow = []
        negative_flow = []
        
        for i in range(len(typical_price)):
            if i == 0:
                positive_flow.append(0)
                negative_flow.append(0)
            else:
                if typical_price.iloc[i] > typical_price.iloc[i-1]:
                    positive_flow.append(raw_money_flow.iloc[i])
                    negative_flow.append(0)
                else:
                    positive_flow.append(0)
                    negative_flow.append(raw_money_flow.iloc[i])
        
        data['Positive_Flow'] = positive_flow
        data['Negative_Flow'] = negative_flow
        
        positive_mf = data['Positive_Flow'].rolling(window=14).sum()
        negative_mf = data['Negative_Flow'].rolling(window=14).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
        data['MFI'] = mfi
    
    def _calculate_candlestick_features(self, data: pd.DataFrame):
        """Calculate candlestick pattern features."""
        # Doji (small body)
        data['Doji'] = (data['Body_Size_Pct'] < 0.1).astype(int)
        
        # Hammer (small body at top, long lower shadow)
        data['Hammer'] = (
            (data['Body_Size_Pct'] < data['Total_Range_Pct'] * 0.3) &
            (data['Lower_Shadow'] > data['Body_Size'] * 2) &
            (data['Upper_Shadow'] < data['Body_Size'] * 0.5)
        ).astype(int)
        
        # Shooting Star (small body at bottom, long upper shadow)
        data['Shooting_Star'] = (
            (data['Body_Size_Pct'] < data['Total_Range_Pct'] * 0.3) &
            (data['Upper_Shadow'] > data['Body_Size'] * 2) &
            (data['Lower_Shadow'] < data['Body_Size'] * 0.5)
        ).astype(int)
        
        # Marubozu (no shadows)
        data['Marubozu'] = (
            (data['Upper_Shadow'] < data['Body_Size'] * 0.1) &
            (data['Lower_Shadow'] < data['Body_Size'] * 0.1)
        ).astype(int)
        
        # Engulfing patterns
        data['Bullish_Engulfing'] = (
            (data['Close'] > data['Open']) &
            (data['Close'].shift(1) < data['Open'].shift(1)) &
            (data['Open'] < data['Close'].shift(1)) &
            (data['Close'] > data['Open'].shift(1))
        ).astype(int)
        
        data['Bearish_Engulfing'] = (
            (data['Close'] < data['Open']) &
            (data['Close'].shift(1) > data['Open'].shift(1)) &
            (data['Open'] > data['Close'].shift(1)) &
            (data['Close'] < data['Open'].shift(1))
        ).astype(int)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all calculated feature names."""
        # Return common feature names (actual features depend on data)
        features = []
        
        # Price features
        features.extend(['Daily_Return', 'Daily_Return_Pct', 'Price_Position', 'Body_Size_Pct'])
        
        # Moving averages
        for period in self.ma_periods:
            features.extend([f'SMA_{period}', f'EMA_{period}', f'Close_to_SMA{period}'])
        
        # RSI
        for period in self.rsi_periods:
            features.append(f'RSI_{period}')
        
        # MACD
        features.extend(['MACD_Line', 'MACD_Signal', 'MACD_Histogram'])
        
        # Bollinger Bands
        features.extend(['BB_Position_20', 'BB_Width_20'])
        
        # ATR
        for period in self.atr_periods:
            features.append(f'ATR_{period}')
        
        # Volume
        features.extend(['Vol_Ratio_20', 'OBV_Trend', 'Chaikin_Oscillator'])
        
        # Volatility
        features.extend(['Volatility_20d', 'Hist_Vol_20d'])
        
        # Trend
        for period in [20, 50]:
            features.extend([f'Close_to_High{period}', f'Close_to_Low{period}'])
        
        # Oscillators
        features.extend(['Stoch_K_14', 'CCI', 'Williams_R_14', 'MFI'])
        
        return features
