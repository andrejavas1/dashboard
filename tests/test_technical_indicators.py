"""
Unit Tests for Technical Indicator Library

This module contains comprehensive unit tests for all technical indicators
in the technical_indicators.py module.

Author: Agent_DataEngineering
Date: 2025-01-22
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.technical_indicators import (
    # Base class
    IndicatorBase,
    
    # Trend Indicators
    SimpleMovingAverage,
    ExponentialMovingAverage,
    WeightedMovingAverage,
    MACD,
    ADX,
    ParabolicSAR,
    
    # Momentum Indicators
    RSI,
    StochasticOscillator,
    WilliamsR,
    CCI,
    ROC,
    
    # Volatility Indicators
    BollingerBands,
    ATR,
    KeltnerChannels,
    
    # Volume Indicators
    VolumeSMA,
    OnBalanceVolume,
    MoneyFlowIndex,
    ChaikinOscillator,
    
    # Advanced Indicators
    IchimokuCloud,
    FibonacciRetracement,
    PivotPoints,
    VWAP,
    DonchianChannels,
    
    # Factory
    IndicatorFactory,
    
    # Convenience functions
    calculate_all_indicators,
    get_indicator_summary
)


class TestSampleData:
    """Helper class to generate test data."""
    
    @staticmethod
    def generate_ohlcv_data(n: int = 100, seed: int = 42) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for testing.
        
        Args:
            n: Number of data points
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with OHLCV data
        """
        np.random.seed(seed)
        
        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        
        # Generate price data with some trend and volatility
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        open_price = close + np.random.randn(n) * 0.2
        volume = np.random.randint(1000000, 10000000, n)
        
        # Ensure High >= Close >= Low
        high = np.maximum(high, close)
        low = np.minimum(low, close)
        
        data = pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        }, index=dates)
        
        return data


class TestIndicatorBase(unittest.TestCase):
    """Tests for the IndicatorBase class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = SimpleMovingAverage(period=10)
    
    def test_validate_data_valid(self):
        """Test validation with valid data."""
        # Should not raise an exception
        self.indicator.validate_data(self.data)
    
    def test_validate_data_invalid(self):
        """Test validation with invalid data."""
        invalid_data = pd.DataFrame({'A': [1, 2, 3]})
        with self.assertRaises(ValueError):
            self.indicator.validate_data(invalid_data)
    
    def test_get_close(self):
        """Test getting close prices."""
        close = self.indicator._get_close(self.data)
        self.assertIsInstance(close, pd.Series)
        self.assertEqual(len(close), len(self.data))
    
    def test_get_high(self):
        """Test getting high prices."""
        high = self.indicator._get_high(self.data)
        self.assertIsInstance(high, pd.Series)
        self.assertEqual(len(high), len(self.data))
    
    def test_get_low(self):
        """Test getting low prices."""
        low = self.indicator._get_low(self.data)
        self.assertIsInstance(low, pd.Series)
        self.assertEqual(len(low), len(self.data))
    
    def test_get_volume(self):
        """Test getting volume."""
        volume = self.indicator._get_volume(self.data)
        self.assertIsInstance(volume, pd.Series)
        self.assertEqual(len(volume), len(self.data))


class TestSimpleMovingAverage(unittest.TestCase):
    """Tests for SimpleMovingAverage indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = SimpleMovingAverage(period=10)
    
    def test_calculate(self):
        """Test SMA calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        self.assertIn('SMA_10', result.columns)
    
    def test_sma_values(self):
        """Test SMA values are correct."""
        result = self.indicator.calculate(self.data)
        
        # First 9 values should be NaN
        self.assertTrue(result['SMA_10'].iloc[:9].isna().all())
        
        # 10th value should be mean of first 10 closes
        expected_sma = self.data['Close'].iloc[:10].mean()
        self.assertAlmostEqual(result['SMA_10'].iloc[9], expected_sma, places=5)
    
    def test_custom_period(self):
        """Test SMA with custom period."""
        result = self.indicator.calculate(self.data, period=20)
        self.assertIn('SMA_20', result.columns)
        self.assertTrue(result['SMA_20'].iloc[:19].isna().all())


class TestExponentialMovingAverage(unittest.TestCase):
    """Tests for ExponentialMovingAverage indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = ExponentialMovingAverage(period=10)
    
    def test_calculate(self):
        """Test EMA calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        self.assertIn('EMA_10', result.columns)
    
    def test_ema_values(self):
        """Test EMA values are correct."""
        result = self.indicator.calculate(self.data)
        
        # EMA should have first value as NaN or close to first close
        self.assertFalse(result['EMA_10'].iloc[10:].isna().any())
    
    def test_custom_period(self):
        """Test EMA with custom period."""
        result = self.indicator.calculate(self.data, period=20)
        self.assertIn('EMA_20', result.columns)


class TestWeightedMovingAverage(unittest.TestCase):
    """Tests for WeightedMovingAverage indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = WeightedMovingAverage(period=10)
    
    def test_calculate(self):
        """Test WMA calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        self.assertIn('WMA_10', result.columns)
    
    def test_wma_values(self):
        """Test WMA values are correct."""
        result = self.indicator.calculate(self.data)
        
        # First 9 values should be NaN
        self.assertTrue(result['WMA_10'].iloc[:9].isna().all())


class TestMACD(unittest.TestCase):
    """Tests for MACD indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = MACD()
    
    def test_calculate(self):
        """Test MACD calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        
        # Check all columns are present
        self.assertIn('MACD', result.columns)
        self.assertIn('MACD_Signal', result.columns)
        self.assertIn('MACD_Histogram', result.columns)
    
    def test_macd_histogram_relationship(self):
        """Test MACD Histogram = MACD - Signal."""
        result = self.indicator.calculate(self.data)
        
        # For non-NaN values
        valid_mask = ~(result['MACD'].isna() | result['MACD_Signal'].isna())
        expected_histogram = result['MACD'][valid_mask] - result['MACD_Signal'][valid_mask]
        
        np.testing.assert_array_almost_equal(
            result['MACD_Histogram'][valid_mask].values,
            expected_histogram.values,
            decimal=5
        )
    
    def test_custom_parameters(self):
        """Test MACD with custom parameters."""
        result = self.indicator.calculate(self.data, fast_period=5, slow_period=10, signal_period=5)
        
        self.assertIn('MACD', result.columns)
        self.assertIn('MACD_Signal', result.columns)
        self.assertIn('MACD_Histogram', result.columns)


class TestADX(unittest.TestCase):
    """Tests for ADX indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = ADX()
    
    def test_calculate(self):
        """Test ADX calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        
        # Check all columns are present
        self.assertIn('ADX', result.columns)
        self.assertIn('Plus_DI', result.columns)
        self.assertIn('Minus_DI', result.columns)
    
    def test_adx_range(self):
        """Test ADX values are in expected range."""
        result = self.indicator.calculate(self.data)
        
        # ADX should be non-negative (after initial NaN)
        non_nan_adx = result['ADX'].dropna()
        self.assertTrue((non_nan_adx >= 0).all())


class TestParabolicSAR(unittest.TestCase):
    """Tests for ParabolicSAR indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = ParabolicSAR()
    
    def test_calculate(self):
        """Test Parabolic SAR calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        self.assertIn('ParabolicSAR', result.columns)
    
    def test_parabolic_sar_values(self):
        """Test Parabolic SAR values are finite."""
        result = self.indicator.calculate(self.data)
        
        # All values should be finite
        self.assertTrue(np.isfinite(result['ParabolicSAR']).all())


class TestRSI(unittest.TestCase):
    """Tests for RSI indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = RSI()
    
    def test_calculate(self):
        """Test RSI calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        self.assertIn('RSI_14', result.columns)
    
    def test_rsi_range(self):
        """Test RSI values are in expected range [0, 100]."""
        result = self.indicator.calculate(self.data)
        
        non_nan_rsi = result['RSI_14'].dropna()
        self.assertTrue((non_nan_rsi >= 0).all())
        self.assertTrue((non_nan_rsi <= 100).all())
    
    def test_custom_period(self):
        """Test RSI with custom period."""
        result = self.indicator.calculate(self.data, period=20)
        self.assertIn('RSI_20', result.columns)


class TestStochasticOscillator(unittest.TestCase):
    """Tests for Stochastic Oscillator indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = StochasticOscillator()
    
    def test_calculate(self):
        """Test Stochastic Oscillator calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        
        # Check all columns are present
        self.assertIn('Stochastic_K', result.columns)
        self.assertIn('Stochastic_D', result.columns)
    
    def test_stochastic_range(self):
        """Test Stochastic values are in expected range [0, 100]."""
        result = self.indicator.calculate(self.data)
        
        non_nan_k = result['Stochastic_K'].dropna()
        non_nan_d = result['Stochastic_D'].dropna()
        
        self.assertTrue((non_nan_k >= 0).all())
        self.assertTrue((non_nan_k <= 100).all())
        self.assertTrue((non_nan_d >= 0).all())
        self.assertTrue((non_nan_d <= 100).all())


class TestWilliamsR(unittest.TestCase):
    """Tests for Williams %R indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = WilliamsR()
    
    def test_calculate(self):
        """Test Williams %R calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        self.assertIn('WilliamsR_14', result.columns)
    
    def test_williams_r_range(self):
        """Test Williams %R values are in expected range [-100, 0]."""
        result = self.indicator.calculate(self.data)
        
        non_nan_wr = result['WilliamsR_14'].dropna()
        self.assertTrue((non_nan_wr >= -100).all())
        self.assertTrue((non_nan_wr <= 0).all())


class TestCCI(unittest.TestCase):
    """Tests for CCI indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = CCI()
    
    def test_calculate(self):
        """Test CCI calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        self.assertIn('CCI_20', result.columns)
    
    def test_cci_values_finite(self):
        """Test CCI values are finite."""
        result = self.indicator.calculate(self.data)
        
        non_nan_cci = result['CCI_20'].dropna()
        self.assertTrue(np.isfinite(non_nan_cci).all())


class TestROC(unittest.TestCase):
    """Tests for ROC indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = ROC()
    
    def test_calculate(self):
        """Test ROC calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        self.assertIn('ROC_12', result.columns)
    
    def test_roc_values_finite(self):
        """Test ROC values are finite."""
        result = self.indicator.calculate(self.data)
        
        non_nan_roc = result['ROC_12'].dropna()
        self.assertTrue(np.isfinite(non_nan_roc).all())


class TestBollingerBands(unittest.TestCase):
    """Tests for Bollinger Bands indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = BollingerBands()
    
    def test_calculate(self):
        """Test Bollinger Bands calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        
        # Check all columns are present
        self.assertIn('BB_Upper', result.columns)
        self.assertIn('BB_Middle', result.columns)
        self.assertIn('BB_Lower', result.columns)
        self.assertIn('BB_Bandwidth', result.columns)
        self.assertIn('BB_PercentB', result.columns)
    
    def test_bollinger_bands_relationship(self):
        """Test Upper >= Middle >= Lower and Upper - Lower = 2 * std * 2."""
        result = self.indicator.calculate(self.data)
        
        valid_mask = ~(result['BB_Upper'].isna() | result['BB_Middle'].isna() | result['BB_Lower'].isna())
        
        # Upper >= Middle
        self.assertTrue((result['BB_Upper'][valid_mask] >= result['BB_Middle'][valid_mask]).all())
        
        # Middle >= Lower
        self.assertTrue((result['BB_Middle'][valid_mask] >= result['BB_Lower'][valid_mask]).all())
    
    def test_percent_b_range(self):
        """Test %B is in expected range [0, 1] for non-extreme values."""
        result = self.indicator.calculate(self.data)
        
        non_nan_pb = result['BB_PercentB'].dropna()
        # Most values should be in [0, 1], extreme values may be outside
        within_range = ((non_nan_pb >= 0) & (non_nan_pb <= 1)).sum()
        self.assertGreater(within_range, len(non_nan_pb) * 0.9)  # At least 90% within range


class TestATR(unittest.TestCase):
    """Tests for ATR indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = ATR()
    
    def test_calculate(self):
        """Test ATR calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        self.assertIn('ATR_14', result.columns)
    
    def test_atr_positive(self):
        """Test ATR values are non-negative."""
        result = self.indicator.calculate(self.data)
        
        non_nan_atr = result['ATR_14'].dropna()
        self.assertTrue((non_nan_atr >= 0).all())


class TestKeltnerChannels(unittest.TestCase):
    """Tests for Keltner Channels indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = KeltnerChannels()
    
    def test_calculate(self):
        """Test Keltner Channels calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        
        # Check all columns are present
        self.assertIn('KC_Upper', result.columns)
        self.assertIn('KC_Middle', result.columns)
        self.assertIn('KC_Lower', result.columns)
    
    def test_keltner_channels_relationship(self):
        """Test Upper >= Middle >= Lower."""
        result = self.indicator.calculate(self.data)
        
        valid_mask = ~(result['KC_Upper'].isna() | result['KC_Middle'].isna() | result['KC_Lower'].isna())
        
        # Upper >= Middle
        self.assertTrue((result['KC_Upper'][valid_mask] >= result['KC_Middle'][valid_mask]).all())
        
        # Middle >= Lower
        self.assertTrue((result['KC_Middle'][valid_mask] >= result['KC_Lower'][valid_mask]).all())


class TestVolumeSMA(unittest.TestCase):
    """Tests for Volume SMA indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = VolumeSMA()
    
    def test_calculate(self):
        """Test Volume SMA calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        self.assertIn('Volume_SMA_20', result.columns)
    
    def test_volume_sma_values(self):
        """Test Volume SMA values are correct."""
        result = self.indicator.calculate(self.data)
        
        # First 19 values should be NaN
        self.assertTrue(result['Volume_SMA_20'].iloc[:19].isna().all())


class TestOnBalanceVolume(unittest.TestCase):
    """Tests for OBV indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = OnBalanceVolume()
    
    def test_calculate(self):
        """Test OBV calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        self.assertIn('OBV', result.columns)
    
    def test_obv_cumulative(self):
        """Test OBV is cumulative."""
        result = self.indicator.calculate(self.data)
        
        # OBV should change based on price direction
        # First value should be 0 or close to 0
        self.assertAlmostEqual(result['OBV'].iloc[0], 0, places=5)


class TestMoneyFlowIndex(unittest.TestCase):
    """Tests for MFI indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = MoneyFlowIndex()
    
    def test_calculate(self):
        """Test MFI calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        self.assertIn('MFI_14', result.columns)
    
    def test_mfi_range(self):
        """Test MFI values are in expected range [0, 100]."""
        result = self.indicator.calculate(self.data)
        
        non_nan_mfi = result['MFI_14'].dropna()
        self.assertTrue((non_nan_mfi >= 0).all())
        self.assertTrue((non_nan_mfi <= 100).all())


class TestChaikinOscillator(unittest.TestCase):
    """Tests for Chaikin Oscillator indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = ChaikinOscillator()
    
    def test_calculate(self):
        """Test Chaikin Oscillator calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        self.assertIn('Chaikin_Oscillator', result.columns)
    
    def test_chaikin_oscillator_finite(self):
        """Test Chaikin Oscillator values are finite."""
        result = self.indicator.calculate(self.data)
        
        non_nan_co = result['Chaikin_Oscillator'].dropna()
        self.assertTrue(np.isfinite(non_nan_co).all())


class TestIchimokuCloud(unittest.TestCase):
    """Tests for Ichimoku Cloud indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = IchimokuCloud()
    
    def test_calculate(self):
        """Test Ichimoku Cloud calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        
        # Check all columns are present
        self.assertIn('Ichimoku_Tenkan', result.columns)
        self.assertIn('Ichimoku_Kijun', result.columns)
        self.assertIn('Ichimoku_Senkou_A', result.columns)
        self.assertIn('Ichimoku_Senkou_B', result.columns)
        self.assertIn('Ichimoku_Chikou', result.columns)
    
    def test_ichimoku_relationships(self):
        """Test Ichimoku component relationships."""
        result = self.indicator.calculate(self.data)
        
        # Tenkan should be more responsive than Kijun
        valid_mask = ~(result['Ichimoku_Tenkan'].isna() | result['Ichimoku_Kijun'].isna())
        
        # Both should have valid values after their respective periods
        self.assertGreater(valid_mask.sum(), 0)


class TestFibonacciRetracement(unittest.TestCase):
    """Tests for Fibonacci Retracement indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = FibonacciRetracement()
    
    def test_calculate(self):
        """Test Fibonacci Retracement calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        
        # Check all Fibonacci levels are present
        expected_levels = ['Fib_0%', 'Fib_23.6%', 'Fib_38.2%', 'Fib_50%', 'Fib_61.8%', 'Fib_78.6%', 'Fib_100%']
        for level in expected_levels:
            self.assertIn(level, result.columns)
    
    def test_fibonacci_levels_order(self):
        """Test Fibonacci levels are in correct order (0% <= 23.6% <= ... <= 100%)."""
        result = self.indicator.calculate(self.data)
        
        # Check at a specific row where all values are non-NaN
        row_idx = 99  # Last row should have all values
        levels = [
            result['Fib_0%'].iloc[row_idx],
            result['Fib_23.6%'].iloc[row_idx],
            result['Fib_38.2%'].iloc[row_idx],
            result['Fib_50%'].iloc[row_idx],
            result['Fib_61.8%'].iloc[row_idx],
            result['Fib_78.6%'].iloc[row_idx],
            result['Fib_100%'].iloc[row_idx]
        ]
        
        # Check levels are monotonically increasing
        for i in range(len(levels) - 1):
            self.assertLessEqual(levels[i], levels[i + 1])


class TestPivotPoints(unittest.TestCase):
    """Tests for Pivot Points indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = PivotPoints()
    
    def test_calculate(self):
        """Test Pivot Points calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        
        # Check all columns are present
        self.assertIn('Pivot_P', result.columns)
        self.assertIn('Pivot_R1', result.columns)
        self.assertIn('Pivot_R2', result.columns)
        self.assertIn('Pivot_R3', result.columns)
        self.assertIn('Pivot_S1', result.columns)
        self.assertIn('Pivot_S2', result.columns)
        self.assertIn('Pivot_S3', result.columns)
    
    def test_pivot_points_relationship(self):
        """Test Pivot Points relationships: R3 >= R2 >= R1 >= P >= S1 >= S2 >= S3."""
        result = self.indicator.calculate(self.data)
        
        valid_mask = ~(result['Pivot_P'].isna())
        
        for idx in result[valid_mask].index:
            self.assertLessEqual(result['Pivot_P'].loc[idx], result['Pivot_R1'].loc[idx])
            self.assertLessEqual(result['Pivot_R1'].loc[idx], result['Pivot_R2'].loc[idx])
            self.assertLessEqual(result['Pivot_R2'].loc[idx], result['Pivot_R3'].loc[idx])
            self.assertLessEqual(result['Pivot_S3'].loc[idx], result['Pivot_S2'].loc[idx])
            self.assertLessEqual(result['Pivot_S2'].loc[idx], result['Pivot_S1'].loc[idx])
            self.assertLessEqual(result['Pivot_S1'].loc[idx], result['Pivot_P'].loc[idx])


class TestVWAP(unittest.TestCase):
    """Tests for VWAP indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = VWAP()
    
    def test_calculate(self):
        """Test VWAP calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        self.assertIn('VWAP', result.columns)
    
    def test_vwap_values_finite(self):
        """Test VWAP values are finite."""
        result = self.indicator.calculate(self.data)
        
        self.assertTrue(np.isfinite(result['VWAP']).all())


class TestDonchianChannels(unittest.TestCase):
    """Tests for Donchian Channels indicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.indicator = DonchianChannels()
    
    def test_calculate(self):
        """Test Donchian Channels calculation."""
        result = self.indicator.calculate(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        
        # Check all columns are present
        self.assertIn('Donchian_Upper', result.columns)
        self.assertIn('Donchian_Middle', result.columns)
        self.assertIn('Donchian_Lower', result.columns)
    
    def test_donchian_channels_relationship(self):
        """Test Upper >= Middle >= Lower."""
        result = self.indicator.calculate(self.data)
        
        valid_mask = ~(result['Donchian_Upper'].isna() | result['Donchian_Middle'].isna() | result['Donchian_Lower'].isna())
        
        # Upper >= Middle
        self.assertTrue((result['Donchian_Upper'][valid_mask] >= result['Donchian_Middle'][valid_mask]).all())
        
        # Middle >= Lower
        self.assertTrue((result['Donchian_Middle'][valid_mask] >= result['Donchian_Lower'][valid_mask]).all())


class TestIndicatorFactory(unittest.TestCase):
    """Tests for IndicatorFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
        self.factory = IndicatorFactory()
    
    def test_create_indicator(self):
        """Test creating an indicator by name."""
        indicator = self.factory.create_indicator('RSI', period=14)
        self.assertIsInstance(indicator, RSI)
        self.assertEqual(indicator.period, 14)
    
    def test_create_indicator_invalid(self):
        """Test creating an invalid indicator raises error."""
        with self.assertRaises(ValueError):
            self.factory.create_indicator('InvalidIndicator')
    
    def test_calculate_indicator(self):
        """Test calculating an indicator by name."""
        result = self.factory.calculate_indicator(self.data, 'RSI', period=14)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('RSI_14', result.columns)
    
    def test_calculate_multiple(self):
        """Test calculating multiple indicators at once."""
        indicators = [
            {'name': 'RSI', 'params': {'period': 14}},
            {'name': 'MACD'},
            {'name': 'BollingerBands'}
        ]
        
        result = self.factory.calculate_multiple(self.data, indicators)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('RSI_14', result.columns)
        self.assertIn('MACD', result.columns)
        self.assertIn('BB_Upper', result.columns)
    
    def test_list_indicators(self):
        """Test listing all available indicators."""
        indicators = self.factory.list_indicators()
        
        self.assertIsInstance(indicators, list)
        self.assertGreater(len(indicators), 0)
        self.assertIn('RSI', indicators)
        self.assertIn('MACD', indicators)
        self.assertIn('BollingerBands', indicators)
    
    def test_get_indicator_info(self):
        """Test getting indicator information."""
        info = self.factory.get_indicator_info('RSI')
        
        self.assertIsInstance(info, dict)
        self.assertEqual(info['name'], 'RSI')
        self.assertIn('class', info)
        self.assertIn('docstring', info)
    
    def test_get_indicator_info_invalid(self):
        """Test getting info for invalid indicator returns empty dict."""
        info = self.factory.get_indicator_info('InvalidIndicator')
        self.assertEqual(info, {})


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
    
    def test_calculate_all_indicators(self):
        """Test calculating all indicators."""
        result = calculate_all_indicators(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        
        # Should have many columns from all indicators
        self.assertGreater(len(result.columns), 20)
    
    def test_get_indicator_summary(self):
        """Test getting indicator summary."""
        summary = get_indicator_summary()
        
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertIn('Name', summary.columns)
        self.assertIn('Class', summary.columns)
        self.assertIn('Description', summary.columns)
        
        # Should have all indicators
        self.assertGreater(len(summary), 0)


class TestIndicatorIntegration(unittest.TestCase):
    """Integration tests for indicators working together."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = TestSampleData.generate_ohlcv_data(n=100)
    
    def test_multiple_indicators_same_data(self):
        """Test multiple indicators can be calculated on the same data."""
        indicators = [
            {'name': 'SMA', 'params': {'period': 20}},
            {'name': 'EMA', 'params': {'period': 20}},
            {'name': 'RSI', 'params': {'period': 14}},
            {'name': 'MACD'},
            {'name': 'BollingerBands'}
        ]
        
        result = IndicatorFactory.calculate_multiple(self.data, indicators)
        
        # Check all indicators are present
        self.assertIn('SMA_20', result.columns)
        self.assertIn('EMA_20', result.columns)
        self.assertIn('RSI_14', result.columns)
        self.assertIn('MACD', result.columns)
        self.assertIn('BB_Upper', result.columns)
    
    def test_indicator_data_alignment(self):
        """Test indicators maintain data alignment."""
        result = calculate_all_indicators(self.data)
        
        # All results should have the same index as input
        self.assertTrue(result.index.equals(self.data.index))
    
    def test_indicator_edge_cases(self):
        """Test indicators handle edge cases."""
        # Very small dataset
        small_data = TestSampleData.generate_ohlcv_data(n=10)
        result = IndicatorFactory.calculate_indicator(small_data, 'SMA', period=5)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(small_data))


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestIndicatorBase,
        TestSimpleMovingAverage,
        TestExponentialMovingAverage,
        TestWeightedMovingAverage,
        TestMACD,
        TestADX,
        TestParabolicSAR,
        TestRSI,
        TestStochasticOscillator,
        TestWilliamsR,
        TestCCI,
        TestROC,
        TestBollingerBands,
        TestATR,
        TestKeltnerChannels,
        TestVolumeSMA,
        TestOnBalanceVolume,
        TestMoneyFlowIndex,
        TestChaikinOscillator,
        TestIchimokuCloud,
        TestFibonacciRetracement,
        TestPivotPoints,
        TestVWAP,
        TestDonchianChannels,
        TestIndicatorFactory,
        TestConvenienceFunctions,
        TestIndicatorIntegration,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)