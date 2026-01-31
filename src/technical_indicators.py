"""
Technical Indicator Library
A comprehensive library of technical analysis indicators for financial data.

This module provides both traditional and advanced technical indicators with
consistent interfaces, optimized for use with OHLCV data from the data acquisition pipeline.

Author: Agent_DataEngineering
Date: 2025-01-22
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, List
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings('ignore')


class IndicatorBase(ABC):
    """
    Base class for all technical indicators.
    Provides consistent interface and common functionality.
    """
    
    def __init__(self, name: str):
        """
        Initialize indicator base class.
        
        Args:
            name: Name of the indicator
        """
        self.name = name
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate the indicator values.
        
        Args:
            data: DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)
            **kwargs: Additional parameters specific to the indicator
            
        Returns:
            DataFrame with indicator values
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data has required columns.
        
        Args:
            data: DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def _get_close(self, data: pd.DataFrame) -> pd.Series:
        """Get close prices from data."""
        return data['Close']
    
    def _get_high(self, data: pd.DataFrame) -> pd.Series:
        """Get high prices from data."""
        return data['High']
    
    def _get_low(self, data: pd.DataFrame) -> pd.Series:
        """Get low prices from data."""
        return data['Low']
    
    def _get_volume(self, data: pd.DataFrame) -> pd.Series:
        """Get volume from data."""
        return data['Volume']


# ============================================================================
# TREND INDICATORS
# ============================================================================

class SimpleMovingAverage(IndicatorBase):
    """
    Simple Moving Average (SMA)
    
    The SMA is calculated by taking the arithmetic mean of a given set of prices
    over a specified number of periods.
    
    Formula: SMA = (Sum of prices over n periods) / n
    
    Usage: Identifying trend direction, support/resistance levels
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize SMA indicator.
        
        Args:
            period: Number of periods for the moving average (default: 20)
        """
        super().__init__("SMA")
        self.period = period
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate Simple Moving Average.
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Optional 'period' parameter to override instance default
            
        Returns:
            DataFrame with SMA values
        """
        self.validate_data(data)
        period = kwargs.get('period', self.period)
        
        result = pd.DataFrame(index=data.index)
        result[f'SMA_{period}'] = self._get_close(data).rolling(window=period).mean()
        
        return result


class ExponentialMovingAverage(IndicatorBase):
    """
    Exponential Moving Average (EMA)
    
    The EMA gives more weight to recent prices, making it more responsive
    to new information than the SMA.
    
    Formula: EMA = (Current Price * k) + (Previous EMA * (1 - k))
    Where k = 2 / (n + 1)
    
    Usage: Trend following, crossover signals, momentum analysis
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize EMA indicator.
        
        Args:
            period: Number of periods for the moving average (default: 20)
        """
        super().__init__("EMA")
        self.period = period
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate Exponential Moving Average.
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Optional 'period' parameter to override instance default
            
        Returns:
            DataFrame with EMA values
        """
        self.validate_data(data)
        period = kwargs.get('period', self.period)
        
        result = pd.DataFrame(index=data.index)
        result[f'EMA_{period}'] = self._get_close(data).ewm(span=period, adjust=False).mean()
        
        return result


class WeightedMovingAverage(IndicatorBase):
    """
    Weighted Moving Average (WMA)
    
    The WMA assigns more weight to recent data points, with the weighting
    decreasing linearly for older data.
    
    Formula: WMA = (Sum of (price * weight)) / (Sum of weights)
    Where weight = n, n-1, n-2, ..., 1 for n periods
    
    Usage: Trend analysis with emphasis on recent prices
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize WMA indicator.
        
        Args:
            period: Number of periods for the moving average (default: 20)
        """
        super().__init__("WMA")
        self.period = period
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate Weighted Moving Average.
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Optional 'period' parameter to override instance default
            
        Returns:
            DataFrame with WMA values
        """
        self.validate_data(data)
        period = kwargs.get('period', self.period)
        
        close = self._get_close(data)
        weights = np.arange(1, period + 1)
        
        result = pd.DataFrame(index=data.index)
        result[f'WMA_{period}'] = close.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
        
        return result


class MACD(IndicatorBase):
    """
    Moving Average Convergence Divergence (MACD)
    
    The MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of prices.
    
    Formula:
        MACD Line = EMA(12) - EMA(26)
        Signal Line = EMA(9) of MACD Line
        Histogram = MACD Line - Signal Line
    
    Usage: Trend identification, momentum signals, divergence analysis
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initialize MACD indicator.
        
        Args:
            fast_period: Period for fast EMA (default: 12)
            slow_period: Period for slow EMA (default: 26)
            signal_period: Period for signal line EMA (default: 9)
        """
        super().__init__("MACD")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate MACD indicator.
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Optional parameters to override instance defaults
            
        Returns:
            DataFrame with MACD, Signal, and Histogram values
        """
        self.validate_data(data)
        fast_period = kwargs.get('fast_period', self.fast_period)
        slow_period = kwargs.get('slow_period', self.slow_period)
        signal_period = kwargs.get('signal_period', self.signal_period)
        
        close = self._get_close(data)
        
        ema_fast = close.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        result = pd.DataFrame(index=data.index)
        result['MACD'] = macd_line
        result['MACD_Signal'] = signal_line
        result['MACD_Histogram'] = histogram
        
        return result


class ADX(IndicatorBase):
    """
    Average Directional Index (ADX)
    
    The ADX measures the strength of a trend without indicating direction.
    It is derived from the Directional Movement System.
    
    Formula:
        +DM = Current High - Previous High (if positive, else 0)
        -DM = Previous Low - Current Low (if positive, else 0)
        TR = max(High-Low, |High-Previous Close|, |Low-Previous Close|)
        +DI = 100 * smoothed +DM / smoothed TR
        -DI = 100 * smoothed -DM / smoothed TR
        DX = 100 * |+DI - -DI| / (+DI + -DI)
        ADX = smoothed DX
    
    Usage: Trend strength measurement (values > 25 indicate strong trend)
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize ADX indicator.
        
        Args:
            period: Period for ADX calculation (default: 14)
        """
        super().__init__("ADX")
        self.period = period
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate ADX indicator.
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Optional 'period' parameter to override instance default
            
        Returns:
            DataFrame with ADX, +DI, and -DI values
        """
        self.validate_data(data)
        period = kwargs.get('period', self.period)
        
        high = self._get_high(data)
        low = self._get_low(data)
        close = self._get_close(data)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        plus_dm = high - high.shift()
        minus_dm = low.shift() - low
        
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0.0)
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0.0)
        
        # Smooth TR and DM using Wilder's smoothing
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        result = pd.DataFrame(index=data.index)
        result['ADX'] = adx
        result['Plus_DI'] = plus_di
        result['Minus_DI'] = minus_di
        
        return result


class ParabolicSAR(IndicatorBase):
    """
    Parabolic Stop and Reverse (Parabolic SAR)
    
    The Parabolic SAR is a trend-following indicator that is used to set
    trailing price stops. It is designed to find potential reversals in price.
    
    Formula:
        SAR_new = SAR_prev + AF * (EP - SAR_prev)
        Where AF starts at 0.02 and increases by 0.02 up to 0.20
        EP is the extreme point (highest high for uptrend, lowest low for downtrend)
    
    Usage: Setting stop-loss levels, identifying trend reversals
    """
    
    def __init__(self, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2):
        """
        Initialize Parabolic SAR indicator.
        
        Args:
            af_start: Starting acceleration factor (default: 0.02)
            af_increment: Acceleration factor increment (default: 0.02)
            af_max: Maximum acceleration factor (default: 0.2)
        """
        super().__init__("ParabolicSAR")
        self.af_start = af_start
        self.af_increment = af_increment
        self.af_max = af_max
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate Parabolic SAR.
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Optional parameters to override instance defaults
            
        Returns:
            DataFrame with Parabolic SAR values
        """
        self.validate_data(data)
        af_start = kwargs.get('af_start', self.af_start)
        af_increment = kwargs.get('af_increment', self.af_increment)
        af_max = kwargs.get('af_max', self.af_max)
        
        high = self._get_high(data).values
        low = self._get_low(data).values
        
        n = len(data)
        sar = np.zeros(n)
        ep = np.zeros(n)
        af = np.zeros(n)
        is_uptrend = np.zeros(n, dtype=bool)
        
        # Initialize
        sar[0] = low[0]
        ep[0] = high[0]
        af[0] = af_start
        is_uptrend[0] = True
        
        for i in range(1, n):
            if is_uptrend[i-1]:
                # Uptrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                sar[i] = min(sar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
                
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
                
                if low[i] < sar[i]:
                    is_uptrend[i] = False
                    sar[i] = ep[i]
                    ep[i] = low[i]
                    af[i] = af_start
                else:
                    is_uptrend[i] = True
            else:
                # Downtrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                sar[i] = max(sar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
                
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
                
                if high[i] > sar[i]:
                    is_uptrend[i] = True
                    sar[i] = ep[i]
                    ep[i] = high[i]
                    af[i] = af_start
                else:
                    is_uptrend[i] = False
        
        result = pd.DataFrame(index=data.index)
        result['ParabolicSAR'] = sar
        
        return result


# ============================================================================
# MOMENTUM INDICATORS
# ============================================================================

class RSI(IndicatorBase):
    """
    Relative Strength Index (RSI)
    
    The RSI is a momentum oscillator that measures the speed and change of
    price movements. It oscillates between 0 and 100.
    
    Formula:
        RSI = 100 - (100 / (1 + RS))
        Where RS = Average Gain / Average Loss over n periods
    
    Usage: Overbought/oversold conditions, divergence analysis, trend confirmation
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize RSI indicator.
        
        Args:
            period: Period for RSI calculation (default: 14)
        """
        super().__init__("RSI")
        self.period = period
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate RSI indicator.
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Optional 'period' parameter to override instance default
            
        Returns:
            DataFrame with RSI values
        """
        self.validate_data(data)
        period = kwargs.get('period', self.period)
        
        close = self._get_close(data)
        delta = close.diff()
        
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        result = pd.DataFrame(index=data.index)
        result[f'RSI_{period}'] = rsi
        
        return result


class StochasticOscillator(IndicatorBase):
    """
    Stochastic Oscillator
    
    The Stochastic Oscillator compares a particular closing price of a security
    to a range of its prices over a certain period of time.
    
    Formula:
        %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
        %D = SMA of %K over smoothing_period
    
    Usage: Overbought/oversold conditions, momentum signals, divergence
    """
    
    def __init__(self, k_period: int = 14, d_period: int = 3, smooth_k: int = 3):
        """
        Initialize Stochastic Oscillator indicator.
        
        Args:
            k_period: Period for %K calculation (default: 14)
            d_period: Period for %D calculation (default: 3)
            smooth_k: Period for smoothing %K (default: 3)
        """
        super().__init__("Stochastic")
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Optional parameters to override instance defaults
            
        Returns:
            DataFrame with %K and %D values
        """
        self.validate_data(data)
        k_period = kwargs.get('k_period', self.k_period)
        d_period = kwargs.get('d_period', self.d_period)
        smooth_k = kwargs.get('smooth_k', self.smooth_k)
        
        close = self._get_close(data)
        high = self._get_high(data)
        low = self._get_low(data)
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k_smooth = k_percent.rolling(window=smooth_k).mean()
        d_percent = k_smooth.rolling(window=d_period).mean()
        
        result = pd.DataFrame(index=data.index)
        result['Stochastic_K'] = k_smooth
        result['Stochastic_D'] = d_percent
        
        return result


class WilliamsR(IndicatorBase):
    """
    Williams %R
    
    Williams %R is a momentum indicator that measures overbought and oversold
    levels. It is similar to the Stochastic Oscillator but with inverted scale.
    
    Formula:
        %R = -100 * (Highest High - Close) / (Highest High - Lowest Low)
    
    Usage: Overbought/oversold conditions, potential reversals
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize Williams %R indicator.
        
        Args:
            period: Period for calculation (default: 14)
        """
        super().__init__("WilliamsR")
        self.period = period
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate Williams %R.
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Optional 'period' parameter to override instance default
            
        Returns:
            DataFrame with Williams %R values
        """
        self.validate_data(data)
        period = kwargs.get('period', self.period)
        
        close = self._get_close(data)
        high = self._get_high(data)
        low = self._get_low(data)
        
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        result = pd.DataFrame(index=data.index)
        result[f'WilliamsR_{period}'] = williams_r
        
        return result


class CCI(IndicatorBase):
    """
    Commodity Channel Index (CCI)
    
    The CCI measures the deviation of the price from its statistical mean.
    It is used to identify cyclical trends in commodities and other instruments.
    
    Formula:
        Typical Price = (High + Low + Close) / 3
        SMA_TP = SMA of Typical Price over n periods
        Mean Deviation = Average of |TP - SMA_TP| over n periods
        CCI = (TP - SMA_TP) / (0.015 * Mean Deviation)
    
    Usage: Identifying overbought/oversold conditions, trend reversals
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize CCI indicator.
        
        Args:
            period: Period for CCI calculation (default: 20)
        """
        super().__init__("CCI")
        self.period = period
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate CCI indicator.
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Optional 'period' parameter to override instance default
            
        Returns:
            DataFrame with CCI values
        """
        self.validate_data(data)
        period = kwargs.get('period', self.period)
        
        high = self._get_high(data)
        low = self._get_low(data)
        close = self._get_close(data)
        
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        result = pd.DataFrame(index=data.index)
        result[f'CCI_{period}'] = cci
        
        return result


class ROC(IndicatorBase):
    """
    Rate of Change (ROC)
    
    The ROC measures the percentage change in price between the current price
    and the price n periods ago.
    
    Formula:
        ROC = ((Close - Close_n_periods_ago) / Close_n_periods_ago) * 100
    
    Usage: Momentum measurement, overbought/oversold identification
    """
    
    def __init__(self, period: int = 12):
        """
        Initialize ROC indicator.
        
        Args:
            period: Period for ROC calculation (default: 12)
        """
        super().__init__("ROC")
        self.period = period
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate ROC indicator.
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Optional 'period' parameter to override instance default
            
        Returns:
            DataFrame with ROC values
        """
        self.validate_data(data)
        period = kwargs.get('period', self.period)
        
        close = self._get_close(data)
        roc = ((close - close.shift(period)) / close.shift(period)) * 100
        
        result = pd.DataFrame(index=data.index)
        result[f'ROC_{period}'] = roc
        
        return result


# ============================================================================
# VOLATILITY INDICATORS
# ============================================================================

class BollingerBands(IndicatorBase):
    """
    Bollinger Bands
    
    Bollinger Bands are a volatility indicator that consists of a middle band
    (SMA) and two outer bands that are standard deviations away from the middle band.
    
    Formula:
        Middle Band = SMA(Close, n)
        Upper Band = Middle Band + (k * Standard Deviation)
        Lower Band = Middle Band - (k * Standard Deviation)
        Bandwidth = (Upper - Lower) / Middle
        %B = (Close - Lower) / (Upper - Lower)
    
    Usage: Volatility measurement, overbought/oversold conditions, squeeze patterns
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands indicator.
        
        Args:
            period: Period for middle band calculation (default: 20)
            std_dev: Number of standard deviations for bands (default: 2.0)
        """
        super().__init__("BollingerBands")
        self.period = period
        self.std_dev = std_dev
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Optional parameters to override instance defaults
            
        Returns:
            DataFrame with Upper, Middle, Lower bands, Bandwidth, and %B
        """
        self.validate_data(data)
        period = kwargs.get('period', self.period)
        std_dev = kwargs.get('std_dev', self.std_dev)
        
        close = self._get_close(data)
        
        middle_band = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        bandwidth = (upper_band - lower_band) / middle_band
        percent_b = (close - lower_band) / (upper_band - lower_band)
        
        result = pd.DataFrame(index=data.index)
        result['BB_Upper'] = upper_band
        result['BB_Middle'] = middle_band
        result['BB_Lower'] = lower_band
        result['BB_Bandwidth'] = bandwidth
        result['BB_PercentB'] = percent_b
        
        return result


class ATR(IndicatorBase):
    """
    Average True Range (ATR)
    
    The ATR measures market volatility by decomposing the entire range of an
    asset price for that period.
    
    Formula:
        True Range = max(High-Low, |High-Previous Close|, |Low-Previous Close|)
        ATR = SMA of True Range over n periods
    
    Usage: Volatility measurement, setting stop-loss levels, position sizing
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize ATR indicator.
        
        Args:
            period: Period for ATR calculation (default: 14)
        """
        super().__init__("ATR")
        self.period = period
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate ATR indicator.
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Optional 'period' parameter to override instance default
            
        Returns:
            DataFrame with ATR values
        """
        self.validate_data(data)
        period = kwargs.get('period', self.period)
        
        high = self._get_high(data)
        low = self._get_low(data)
        close = self._get_close(data)
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        result = pd.DataFrame(index=data.index)
        result[f'ATR_{period}'] = atr
        
        return result


class KeltnerChannels(IndicatorBase):
    """
    Keltner Channels
    
    Keltner Channels are volatility-based envelopes set above and below an
    exponential moving average.
    
    Formula:
        Middle Line = EMA(Close, n)
        Upper Channel = Middle Line + (k * ATR)
        Lower Channel = Middle Line - (k * ATR)
    
    Usage: Trend identification, volatility measurement, breakout signals
    """
    
    def __init__(self, period: int = 20, multiplier: float = 2.0, atr_period: int = 10):
        """
        Initialize Keltner Channels indicator.
        
        Args:
            period: Period for EMA calculation (default: 20)
            multiplier: Multiplier for ATR (default: 2.0)
            atr_period: Period for ATR calculation (default: 10)
        """
        super().__init__("KeltnerChannels")
        self.period = period
        self.multiplier = multiplier
        self.atr_period = atr_period
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate Keltner Channels.
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Optional parameters to override instance defaults
            
        Returns:
            DataFrame with Upper, Middle, Lower channels
        """
        self.validate_data(data)
        period = kwargs.get('period', self.period)
        multiplier = kwargs.get('multiplier', self.multiplier)
        atr_period = kwargs.get('atr_period', self.atr_period)
        
        close = self._get_close(data)
        high = self._get_high(data)
        low = self._get_low(data)
        
        middle_line = close.ewm(span=period, adjust=False).mean()
        
        # Calculate ATR
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_period).mean()
        
        upper_channel = middle_line + (multiplier * atr)
        lower_channel = middle_line - (multiplier * atr)
        
        result = pd.DataFrame(index=data.index)
        result['KC_Upper'] = upper_channel
        result['KC_Middle'] = middle_line
        result['KC_Lower'] = lower_channel
        
        return result


# ============================================================================
# VOLUME INDICATORS
# ============================================================================

class VolumeSMA(IndicatorBase):
    """
    Volume Simple Moving Average
    
    The Volume SMA is the average volume over a specified number of periods.
    
    Formula: Volume SMA = SMA(Volume, n)
    
    Usage: Identifying volume trends, confirming price movements
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize Volume SMA indicator.
        
        Args:
            period: Period for volume SMA (default: 20)
        """
        super().__init__("VolumeSMA")
        self.period = period
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate Volume SMA.
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Optional 'period' parameter to override instance default
            
        Returns:
            DataFrame with Volume SMA values
        """
        self.validate_data(data)
        period = kwargs.get('period', self.period)
        
        volume = self._get_volume(data)
        volume_sma = volume.rolling(window=period).mean()
        
        result = pd.DataFrame(index=data.index)
        result[f'Volume_SMA_{period}'] = volume_sma
        
        return result


class OnBalanceVolume(IndicatorBase):
    """
    On Balance Volume (OBV)
    
    OBV is a momentum indicator that uses volume flow to predict changes in
    stock price. It adds volume on up days and subtracts volume on down days.
    
    Formula:
        If Close > Previous Close: OBV = Previous OBV + Volume
        If Close < Previous Close: OBV = Previous OBV - Volume
        If Close = Previous Close: OBV = Previous OBV
    
    Usage: Trend confirmation, divergence analysis, momentum measurement
    """
    
    def __init__(self):
        """Initialize OBV indicator."""
        super().__init__("OBV")
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate OBV indicator.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with OBV values
        """
        self.validate_data(data)
        
        close = self._get_close(data)
        volume = self._get_volume(data)
        
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        
        result = pd.DataFrame(index=data.index)
        result['OBV'] = obv
        
        return result


class MoneyFlowIndex(IndicatorBase):
    """
    Money Flow Index (MFI)
    
    The MFI is a momentum indicator that incorporates both price and volume data.
    It is similar to RSI but uses money flow instead of price.
    
    Formula:
        Typical Price = (High + Low + Close) / 3
        Money Flow = Typical Price * Volume
        Positive MF = Money Flow if TP > Previous TP, else 0
        Negative MF = Money Flow if TP < Previous TP, else 0
        Money Ratio = (14-period sum of Positive MF) / (14-period sum of Negative MF)
        MFI = 100 - (100 / (1 + Money Ratio))
    
    Usage: Overbought/oversold conditions, divergence analysis
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize MFI indicator.
        
        Args:
            period: Period for MFI calculation (default: 14)
        """
        super().__init__("MFI")
        self.period = period
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate MFI indicator.
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Optional 'period' parameter to override instance default
            
        Returns:
            DataFrame with MFI values
        """
        self.validate_data(data)
        period = kwargs.get('period', self.period)
        
        high = self._get_high(data)
        low = self._get_low(data)
        close = self._get_close(data)
        volume = self._get_volume(data)
        
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        tp_diff = typical_price.diff()
        
        positive_mf = money_flow.where(tp_diff > 0, 0)
        negative_mf = money_flow.where(tp_diff < 0, 0)
        
        positive_mf_sum = positive_mf.rolling(window=period).sum()
        negative_mf_sum = negative_mf.rolling(window=period).sum()
        
        money_ratio = positive_mf_sum / negative_mf_sum
        mfi = 100 - (100 / (1 + money_ratio))
        
        result = pd.DataFrame(index=data.index)
        result[f'MFI_{period}'] = mfi
        
        return result


class ChaikinOscillator(IndicatorBase):
    """
    Chaikin Oscillator
    
    The Chaikin Oscillator is a momentum indicator that monitors the flow of
    money into and out of a security. It is based on the Accumulation/Distribution Line.
    
    Formula:
        Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        Money Flow Volume = Money Flow Multiplier * Volume
        ADL = Cumulative sum of Money Flow Volume
        Chaikin Oscillator = EMA(3) of ADL - EMA(10) of ADL
    
    Usage: Momentum measurement, confirming trends, divergence analysis
    """
    
    def __init__(self, fast_period: int = 3, slow_period: int = 10):
        """
        Initialize Chaikin Oscillator indicator.
        
        Args:
            fast_period: Period for fast EMA (default: 3)
            slow_period: Period for slow EMA (default: 10)
        """
        super().__init__("ChaikinOscillator")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate Chaikin Oscillator.
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Optional parameters to override instance defaults
            
        Returns:
            DataFrame with Chaikin Oscillator values
        """
        self.validate_data(data)
        fast_period = kwargs.get('fast_period', self.fast_period)
        slow_period = kwargs.get('slow_period', self.slow_period)
        
        high = self._get_high(data)
        low = self._get_low(data)
        close = self._get_close(data)
        volume = self._get_volume(data)
        
        # Calculate Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.replace([np.inf, -np.inf], 0)
        
        # Calculate Money Flow Volume
        mfv = mfm * volume
        
        # Calculate Accumulation/Distribution Line
        adl = mfv.cumsum()
        
        # Calculate Chaikin Oscillator
        co = adl.ewm(span=fast_period, adjust=False).mean() - adl.ewm(span=slow_period, adjust=False).mean()
        
        result = pd.DataFrame(index=data.index)
        result['Chaikin_Oscillator'] = co
        
        return result


# ============================================================================
# ADVANCED INDICATORS
# ============================================================================

class IchimokuCloud(IndicatorBase):
    """
    Ichimoku Cloud (Ichimoku Kinko Hyo)
    
    The Ichimoku Cloud is a comprehensive indicator that defines support and
    resistance, identifies trend direction, gauges momentum, and provides trading signals.
    
    Formula:
        Tenkan-sen (Conversion Line) = (Highest High + Lowest Low) / 2 over 9 periods
        Kijun-sen (Base Line) = (Highest High + Lowest Low) / 2 over 26 periods
        Senkou Span A (Leading Span A) = (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods ahead
        Senkou Span B (Leading Span B) = (Highest High + Lowest Low) / 2 over 52 periods, shifted 26 periods ahead
        Chikou Span (Lagging Span) = Close, shifted 26 periods behind
    
    Usage: Trend identification, support/resistance, momentum, trading signals
    """
    
    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26, senkou_period: int = 52):
        """
        Initialize Ichimoku Cloud indicator.
        
        Args:
            tenkan_period: Period for Tenkan-sen (default: 9)
            kijun_period: Period for Kijun-sen (default: 26)
            senkou_period: Period for Senkou Span B (default: 52)
        """
        super().__init__("IchimokuCloud")
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_period = senkou_period
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud.
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Optional parameters to override instance defaults
            
        Returns:
            DataFrame with Ichimoku Cloud components
        """
        self.validate_data(data)
        tenkan_period = kwargs.get('tenkan_period', self.tenkan_period)
        kijun_period = kwargs.get('kijun_period', self.kijun_period)
        senkou_period = kwargs.get('senkou_period', self.senkou_period)
        
        high = self._get_high(data)
        low = self._get_low(data)
        close = self._get_close(data)
        
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(window=tenkan_period).max() + 
                      low.rolling(window=tenkan_period).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(window=kijun_period).max() + 
                     low.rolling(window=kijun_period).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(window=senkou_period).max() + 
                          low.rolling(window=senkou_period).min()) / 2).shift(kijun_period)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-kijun_period)
        
        result = pd.DataFrame(index=data.index)
        result['Ichimoku_Tenkan'] = tenkan_sen
        result['Ichimoku_Kijun'] = kijun_sen
        result['Ichimoku_Senkou_A'] = senkou_span_a
        result['Ichimoku_Senkou_B'] = senkou_span_b
        result['Ichimoku_Chikou'] = chikou_span
        
        return result


class FibonacciRetracement(IndicatorBase):
    """
    Fibonacci Retracement
    
    Fibonacci retracement levels are horizontal lines that indicate where support
    and resistance are likely to occur. They are based on Fibonacci ratios.
    
    Formula:
        High = Highest price over lookback period
        Low = Lowest price over lookback period
        Diff = High - Low
        Levels = Low + (Diff * ratio) where ratio ∈ {0, 0.236, 0.382, 0.5, 0.618, 0.786, 1}
    
    Usage: Support/resistance levels, entry/exit points, trend continuation
    """
    
    def __init__(self, lookback: int = 100):
        """
        Initialize Fibonacci Retracement indicator.
        
        Args:
            lookback: Period to look back for high/low (default: 100)
        """
        super().__init__("FibonacciRetracement")
        self.lookback = lookback
        self.ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate Fibonacci Retracement levels.
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Optional 'lookback' parameter to override instance default
            
        Returns:
            DataFrame with Fibonacci retracement levels
        """
        self.validate_data(data)
        lookback = kwargs.get('lookback', self.lookback)
        
        high = self._get_high(data)
        low = self._get_low(data)
        
        rolling_high = high.rolling(window=lookback).max()
        rolling_low = low.rolling(window=lookback).min()
        diff = rolling_high - rolling_low
        
        result = pd.DataFrame(index=data.index)
        
        ratio_names = ['0%', '23.6%', '38.2%', '50%', '61.8%', '78.6%', '100%']
        
        for ratio, name in zip(self.ratios, ratio_names):
            result[f'Fib_{name}'] = rolling_low + (diff * ratio)
        
        return result


class PivotPoints(IndicatorBase):
    """
    Pivot Points
    
    Pivot Points are used to identify potential support and resistance levels.
    They are calculated based on the previous day's high, low, and close.
    
    Formula:
        Pivot Point (P) = (High + Low + Close) / 3
        Resistance 1 (R1) = (2 * P) - Low
        Resistance 2 (R2) = P + (High - Low)
        Resistance 3 (R3) = High + 2 * (P - Low)
        Support 1 (S1) = (2 * P) - High
        Support 2 (S2) = P - (High - Low)
        Support 3 (S3) = Low - 2 * (High - P)
    
    Usage: Intraday support/resistance, entry/exit points
    """
    
    def __init__(self):
        """Initialize Pivot Points indicator."""
        super().__init__("PivotPoints")
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate Pivot Points.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Pivot Point levels
        """
        self.validate_data(data)
        
        high = self._get_high(data)
        low = self._get_low(data)
        close = self._get_close(data)
        
        # Use previous day's values
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)
        
        p = (prev_high + prev_low + prev_close) / 3
        
        r1 = (2 * p) - prev_low
        r2 = p + (prev_high - prev_low)
        r3 = prev_high + 2 * (p - prev_low)
        
        s1 = (2 * p) - prev_high
        s2 = p - (prev_high - prev_low)
        s3 = prev_low - 2 * (prev_high - p)
        
        result = pd.DataFrame(index=data.index)
        result['Pivot_P'] = p
        result['Pivot_R1'] = r1
        result['Pivot_R2'] = r2
        result['Pivot_R3'] = r3
        result['Pivot_S1'] = s1
        result['Pivot_S2'] = s2
        result['Pivot_S3'] = s3
        
        return result


class VWAP(IndicatorBase):
    """
    Volume Weighted Average Price (VWAP)
    
    VWAP is the average price a security has traded at throughout the day,
    based on both volume and price. It is important because it provides
    traders with insight into both the trend and value of a security.
    
    Formula:
        Typical Price = (High + Low + Close) / 3
        VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
    
    Usage: Intraday trading, identifying fair value, trend confirmation
    """
    
    def __init__(self):
        """Initialize VWAP indicator."""
        super().__init__("VWAP")
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate VWAP.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with VWAP values
        """
        self.validate_data(data)
        
        high = self._get_high(data)
        low = self._get_low(data)
        close = self._get_close(data)
        volume = self._get_volume(data)
        
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        result = pd.DataFrame(index=data.index)
        result['VWAP'] = vwap
        
        return result


class DonchianChannels(IndicatorBase):
    """
    Donchian Channels
    
    Donchian Channels are volatility bands used to identify potential breakouts.
    They consist of three lines: the upper channel, lower channel, and middle line.
    
    Formula:
        Upper Channel = Highest High over n periods
        Lower Channel = Lowest Low over n periods
        Middle Channel = (Upper Channel + Lower Channel) / 2
    
    Usage: Breakout trading, trend following, volatility measurement
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize Donchian Channels indicator.
        
        Args:
            period: Period for channel calculation (default: 20)
        """
        super().__init__("DonchianChannels")
        self.period = period
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate Donchian Channels.
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Optional 'period' parameter to override instance default
            
        Returns:
            DataFrame with Donchian Channel values
        """
        self.validate_data(data)
        period = kwargs.get('period', self.period)
        
        high = self._get_high(data)
        low = self._get_low(data)
        
        upper_channel = high.rolling(window=period).max()
        lower_channel = low.rolling(window=period).min()
        middle_channel = (upper_channel + lower_channel) / 2
        
        result = pd.DataFrame(index=data.index)
        result['Donchian_Upper'] = upper_channel
        result['Donchian_Middle'] = middle_channel
        result['Donchian_Lower'] = lower_channel
        
        return result


# ============================================================================
# FACTORY CLASS
# ============================================================================

class IndicatorFactory:
    """
    Factory class for creating and managing technical indicators.
    Provides a unified interface for calculating multiple indicators.
    """
    
    _indicators = {
        # Trend Indicators
        'SMA': SimpleMovingAverage,
        'EMA': ExponentialMovingAverage,
        'WMA': WeightedMovingAverage,
        'MACD': MACD,
        'ADX': ADX,
        'ParabolicSAR': ParabolicSAR,
        
        # Momentum Indicators
        'RSI': RSI,
        'Stochastic': StochasticOscillator,
        'WilliamsR': WilliamsR,
        'CCI': CCI,
        'ROC': ROC,
        
        # Volatility Indicators
        'BollingerBands': BollingerBands,
        'ATR': ATR,
        'KeltnerChannels': KeltnerChannels,
        
        # Volume Indicators
        'VolumeSMA': VolumeSMA,
        'OBV': OnBalanceVolume,
        'MFI': MoneyFlowIndex,
        'ChaikinOscillator': ChaikinOscillator,
        
        # Advanced Indicators
        'IchimokuCloud': IchimokuCloud,
        'FibonacciRetracement': FibonacciRetracement,
        'PivotPoints': PivotPoints,
        'VWAP': VWAP,
        'DonchianChannels': DonchianChannels,
    }
    
    @classmethod
    def create_indicator(cls, indicator_name: str, **kwargs) -> IndicatorBase:
        """
        Create an indicator instance by name.
        
        Args:
            indicator_name: Name of the indicator to create
            **kwargs: Parameters to pass to the indicator constructor
            
        Returns:
            Indicator instance
            
        Raises:
            ValueError: If indicator name is not recognized
        """
        indicator_class = cls._indicators.get(indicator_name)
        if indicator_class is None:
            available = ', '.join(cls._indicators.keys())
            raise ValueError(f"Unknown indicator: {indicator_name}. Available: {available}")
        return indicator_class(**kwargs)
    
    @classmethod
    def calculate_indicator(cls, data: pd.DataFrame, indicator_name: str, **kwargs) -> pd.DataFrame:
        """
        Calculate an indicator by name.
        
        Args:
            data: DataFrame with OHLCV data
            indicator_name: Name of the indicator to calculate
            **kwargs: Parameters for the indicator calculation
            
        Returns:
            DataFrame with indicator values
        """
        indicator = cls.create_indicator(indicator_name, **kwargs)
        return indicator.calculate(data, **kwargs)
    
    @classmethod
    def calculate_multiple(cls, data: pd.DataFrame, indicators: List[Dict]) -> pd.DataFrame:
        """
        Calculate multiple indicators at once.
        
        Args:
            data: DataFrame with OHLCV data
            indicators: List of dictionaries with 'name' and optional 'params' keys
                Example: [{'name': 'RSI', 'params': {'period': 14}}, {'name': 'MACD'}]
            
        Returns:
            DataFrame with all indicator values
        """
        result = pd.DataFrame(index=data.index)
        
        for indicator_config in indicators:
            name = indicator_config['name']
            params = indicator_config.get('params', {})
            
            indicator_data = cls.calculate_indicator(data, name, **params)
            result = pd.concat([result, indicator_data], axis=1)
        
        return result
    
    @classmethod
    def list_indicators(cls) -> List[str]:
        """
        Get a list of all available indicators.
        
        Returns:
            List of indicator names
        """
        return list(cls._indicators.keys())
    
    @classmethod
    def get_indicator_info(cls, indicator_name: str) -> Dict:
        """
        Get information about an indicator.
        
        Args:
            indicator_name: Name of the indicator
            
        Returns:
            Dictionary with indicator information
        """
        indicator_class = cls._indicators.get(indicator_name)
        if indicator_class is None:
            return {}
        
        return {
            'name': indicator_name,
            'class': indicator_class.__name__,
            'docstring': indicator_class.__doc__
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all available indicators with default parameters.
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        DataFrame with all indicator values
    """
    factory = IndicatorFactory()
    indicators = [{'name': name} for name in factory.list_indicators()]
    return factory.calculate_multiple(data, indicators)


def get_indicator_summary() -> pd.DataFrame:
    """
    Get a summary of all available indicators.
    
    Returns:
        DataFrame with indicator information
    """
    factory = IndicatorFactory()
    indicators = factory.list_indicators()
    
    summary_data = []
    for name in indicators:
        info = factory.get_indicator_info(name)
        summary_data.append({
            'Name': name,
            'Class': info['class'],
            'Description': info['docstring'].split('\n')[1].strip() if info['docstring'] else ''
        })
    
    return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Example usage
    print("Technical Indicator Library")
    print("=" * 60)
    print("\nAvailable Indicators:")
    print(IndicatorFactory.list_indicators())
    
    print("\nIndicator Summary:")
    print(get_indicator_summary())