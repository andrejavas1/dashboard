# Technical Indicator Library Documentation

## Overview

The Technical Indicator Library provides a comprehensive collection of technical analysis indicators for financial data. It includes both traditional indicators (RSI, MACD, Bollinger Bands) and advanced indicators, all with consistent interfaces and optimized for use with OHLCV data from the data acquisition pipeline.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Indicator Categories](#indicator-categories)
4. [Detailed Indicator Reference](#detailed-indicator-reference)
5. [Indicator Factory](#indicator-factory)
6. [Usage Examples](#usage-examples)
7. [Testing](#testing)
8. [API Reference](#api-reference)

---

## Installation

The library is part of the project and can be imported directly:

```python
from src.technical_indicators import (
    IndicatorFactory,
    # Individual indicators
    RSI, MACD, BollingerBands,
    # ... and more
)
```

### Dependencies

- `numpy` >= 1.20.0
- `pandas` >= 1.3.0

---

## Quick Start

### Basic Usage

```python
import pandas as pd
from src.technical_indicators import IndicatorFactory

# Load your OHLCV data
data = pd.read_csv('your_data.csv', index_col='Date', parse_dates=True)

# Calculate a single indicator
rsi = IndicatorFactory.calculate_indicator(data, 'RSI', period=14)
print(rsi.head())

# Calculate multiple indicators at once
indicators = [
    {'name': 'RSI', 'params': {'period': 14}},
    {'name': 'MACD'},
    {'name': 'BollingerBands', 'params': {'period': 20, 'std_dev': 2}}
]
results = IndicatorFactory.calculate_multiple(data, indicators)
print(results.head())
```

### Using Individual Indicator Classes

```python
from src.technical_indicators import RSI, MACD, BollingerBands

# Create indicator instances
rsi = RSI(period=14)
macd = MACD(fast_period=12, slow_period=26, signal_period=9)
bb = BollingerBands(period=20, std_dev=2)

# Calculate indicators
rsi_values = rsi.calculate(data)
macd_values = macd.calculate(data)
bb_values = bb.calculate(data)
```

---

## Indicator Categories

### 1. Trend Indicators

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| **SMA** | Simple Moving Average | `period` (default: 20) |
| **EMA** | Exponential Moving Average | `period` (default: 20) |
| **WMA** | Weighted Moving Average | `period` (default: 20) |
| **MACD** | Moving Average Convergence Divergence | `fast_period`, `slow_period`, `signal_period` |
| **ADX** | Average Directional Index | `period` (default: 14) |
| **ParabolicSAR** | Parabolic Stop and Reverse | `af_start`, `af_increment`, `af_max` |

### 2. Momentum Indicators

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| **RSI** | Relative Strength Index | `period` (default: 14) |
| **Stochastic** | Stochastic Oscillator | `k_period`, `d_period`, `smooth_k` |
| **WilliamsR** | Williams %R | `period` (default: 14) |
| **CCI** | Commodity Channel Index | `period` (default: 20) |
| **ROC** | Rate of Change | `period` (default: 12) |

### 3. Volatility Indicators

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| **BollingerBands** | Bollinger Bands | `period` (default: 20), `std_dev` (default: 2.0) |
| **ATR** | Average True Range | `period` (default: 14) |
| **KeltnerChannels** | Keltner Channels | `period`, `multiplier`, `atr_period` |

### 4. Volume Indicators

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| **VolumeSMA** | Volume Simple Moving Average | `period` (default: 20) |
| **OBV** | On Balance Volume | None |
| **MFI** | Money Flow Index | `period` (default: 14) |
| **ChaikinOscillator** | Chaikin Oscillator | `fast_period`, `slow_period` |

### 5. Advanced Indicators

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| **IchimokuCloud** | Ichimoku Cloud | `tenkan_period`, `kijun_period`, `senkou_period` |
| **FibonacciRetracement** | Fibonacci Retracement Levels | `lookback` (default: 100) |
| **PivotPoints** | Pivot Points | None |
| **VWAP** | Volume Weighted Average Price | None |
| **DonchianChannels** | Donchian Channels | `period` (default: 20) |

---

## Detailed Indicator Reference

### Trend Indicators

#### Simple Moving Average (SMA)

**Formula:**
```
SMA = (Sum of prices over n periods) / n
```

**Usage:** Identifying trend direction, support/resistance levels

**Example:**
```python
from src.technical_indicators import SimpleMovingAverage

sma = SimpleMovingAverage(period=20)
result = sma.calculate(data)
# Returns DataFrame with column 'SMA_20'
```

#### Exponential Moving Average (EMA)

**Formula:**
```
EMA = (Current Price × k) + (Previous EMA × (1 - k))
where k = 2 / (n + 1)
```

**Usage:** Trend following, crossover signals, momentum analysis

**Example:**
```python
from src.technical_indicators import ExponentialMovingAverage

ema = ExponentialMovingAverage(period=20)
result = ema.calculate(data)
# Returns DataFrame with column 'EMA_20'
```

#### MACD (Moving Average Convergence Divergence)

**Formula:**
```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
Histogram = MACD Line - Signal Line
```

**Usage:** Trend identification, momentum signals, divergence analysis

**Example:**
```python
from src.technical_indicators import MACD

macd = MACD(fast_period=12, slow_period=26, signal_period=9)
result = macd.calculate(data)
# Returns DataFrame with columns: 'MACD', 'MACD_Signal', 'MACD_Histogram'

# Signal interpretation:
# - MACD crosses above Signal: Bullish signal
# - MACD crosses below Signal: Bearish signal
# - Histogram increasing: Momentum strengthening
```

#### ADX (Average Directional Index)

**Formula:**
```
+DM = Current High - Previous High (if positive, else 0)
-DM = Previous Low - Current Low (if positive, else 0)
TR = max(High-Low, |High-Previous Close|, |Low-Previous Close|)
+DI = 100 × smoothed +DM / smoothed TR
-DI = 100 × smoothed -DM / smoothed TR
DX = 100 × |+DI - -DI| / (+DI + -DI)
ADX = smoothed DX
```

**Usage:** Trend strength measurement
- ADX > 25: Strong trend
- ADX < 20: Weak trend or ranging market

**Example:**
```python
from src.technical_indicators import ADX

adx = ADX(period=14)
result = adx.calculate(data)
# Returns DataFrame with columns: 'ADX', 'Plus_DI', 'Minus_DI'

# Trend direction:
# - Plus_DI > Minus_DI: Uptrend
# - Minus_DI > Plus_DI: Downtrend
```

### Momentum Indicators

#### RSI (Relative Strength Index)

**Formula:**
```
RSI = 100 - (100 / (1 + RS))
where RS = Average Gain / Average Loss over n periods
```

**Usage:** Overbought/oversold conditions, divergence analysis
- RSI > 70: Overbought
- RSI < 30: Oversold

**Example:**
```python
from src.technical_indicators import RSI

rsi = RSI(period=14)
result = rsi.calculate(data)
# Returns DataFrame with column 'RSI_14'

# Signal interpretation:
# - RSI crosses above 70: Potential sell signal
# - RSI crosses below 30: Potential buy signal
```

#### Stochastic Oscillator

**Formula:**
```
%K = 100 × (Close - Lowest Low) / (Highest High - Lowest Low)
%D = SMA of %K over smoothing_period
```

**Usage:** Overbought/oversold conditions, momentum signals
- %K > 80: Overbought
- %K < 20: Oversold

**Example:**
```python
from src.technical_indicators import StochasticOscillator

stoch = StochasticOscillator(k_period=14, d_period=3, smooth_k=3)
result = stoch.calculate(data)
# Returns DataFrame with columns: 'Stochastic_K', 'Stochastic_D'

# Signal interpretation:
# - %K crosses above %D: Bullish signal
# - %K crosses below %D: Bearish signal
```

### Volatility Indicators

#### Bollinger Bands

**Formula:**
```
Middle Band = SMA(Close, n)
Upper Band = Middle Band + (k × Standard Deviation)
Lower Band = Middle Band - (k × Standard Deviation)
Bandwidth = (Upper - Lower) / Middle
%B = (Close - Lower) / (Upper - Lower)
```

**Usage:** Volatility measurement, overbought/oversold conditions, squeeze patterns

**Example:**
```python
from src.technical_indicators import BollingerBands

bb = BollingerBands(period=20, std_dev=2.0)
result = bb.calculate(data)
# Returns DataFrame with columns: 'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Bandwidth', 'BB_PercentB'

# Signal interpretation:
# - Price touches Upper Band: Potential overbought
# - Price touches Lower Band: Potential oversold
# - Bandwidth contraction: Squeeze pattern (potential breakout)
# - %B > 1: Price above upper band
# - %B < 0: Price below lower band
```

#### ATR (Average True Range)

**Formula:**
```
True Range = max(High-Low, |High-Previous Close|, |Low-Previous Close|)
ATR = SMA of True Range over n periods
```

**Usage:** Volatility measurement, setting stop-loss levels, position sizing

**Example:**
```python
from src.technical_indicators import ATR

atr = ATR(period=14)
result = atr.calculate(data)
# Returns DataFrame with column 'ATR_14'

# Position sizing example:
# position_size = account_value × risk_per_trade / (ATR × multiplier)
```

### Volume Indicators

#### OBV (On Balance Volume)

**Formula:**
```
If Close > Previous Close: OBV = Previous OBV + Volume
If Close < Previous Close: OBV = Previous OBV - Volume
If Close = Previous Close: OBV = Previous OBV
```

**Usage:** Trend confirmation, divergence analysis, momentum measurement

**Example:**
```python
from src.technical_indicators import OnBalanceVolume

obv = OnBalanceVolume()
result = obv.calculate(data)
# Returns DataFrame with column 'OBV'

# Signal interpretation:
# - OBV rising with prices: Confirmed uptrend
# - OBV falling with prices: Confirmed downtrend
# - Price rising but OBV falling: Bearish divergence
```

#### MFI (Money Flow Index)

**Formula:**
```
Typical Price = (High + Low + Close) / 3
Money Flow = Typical Price × Volume
Positive MF = Money Flow if TP > Previous TP, else 0
Negative MF = Money Flow if TP < Previous TP, else 0
Money Ratio = (14-period sum of Positive MF) / (14-period sum of Negative MF)
MFI = 100 - (100 / (1 + Money Ratio))
```

**Usage:** Overbought/oversold conditions, divergence analysis
- MFI > 80: Overbought
- MFI < 20: Oversold

**Example:**
```python
from src.technical_indicators import MoneyFlowIndex

mfi = MoneyFlowIndex(period=14)
result = mfi.calculate(data)
# Returns DataFrame with column 'MFI_14'
```

### Advanced Indicators

#### Ichimoku Cloud

**Formula:**
```
Tenkan-sen = (Highest High + Lowest Low) / 2 over 9 periods
Kijun-sen = (Highest High + Lowest Low) / 2 over 26 periods
Senkou Span A = (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods ahead
Senkou Span B = (Highest High + Lowest Low) / 2 over 52 periods, shifted 26 periods ahead
Chikou Span = Close, shifted 26 periods behind
```

**Usage:** Trend identification, support/resistance, momentum, trading signals

**Example:**
```python
from src.technical_indicators import IchimokuCloud

ichimoku = IchimokuCloud(tenkan_period=9, kijun_period=26, senkou_period=52)
result = ichimoku.calculate(data)
# Returns DataFrame with columns: 'Ichimoku_Tenkan', 'Ichimoku_Kijun', 
# 'Ichimoku_Senkou_A', 'Ichimoku_Senkou_B', 'Ichimoku_Chikou'

# Signal interpretation:
# - Price above cloud: Bullish trend
# - Price below cloud: Bearish trend
# - Tenkan crosses above Kijun: Bullish signal
# - Tenkan crosses below Kijun: Bearish signal
```

#### Fibonacci Retracement

**Formula:**
```
High = Highest price over lookback period
Low = Lowest price over lookback period
Diff = High - Low
Levels = Low + (Diff × ratio) where ratio ∈ {0, 0.236, 0.382, 0.5, 0.618, 0.786, 1}
```

**Usage:** Support/resistance levels, entry/exit points, trend continuation

**Example:**
```python
from src.technical_indicators import FibonacciRetracement

fib = FibonacciRetracement(lookback=100)
result = fib.calculate(data)
# Returns DataFrame with columns: 'Fib_0%', 'Fib_23.6%', 'Fib_38.2%', 
# 'Fib_50%', 'Fib_61.8%', 'Fib_78.6%', 'Fib_100%'

# Common support/resistance levels:
# - 38.2% and 61.8%: Primary retracement levels
# - 50%: Psychological level
```

#### VWAP (Volume Weighted Average Price)

**Formula:**
```
Typical Price = (High + Low + Close) / 3
VWAP = Cumulative(Typical Price × Volume) / Cumulative(Volume)
```

**Usage:** Intraday trading, identifying fair value, trend confirmation

**Example:**
```python
from src.technical_indicators import VWAP

vwap = VWAP()
result = vwap.calculate(data)
# Returns DataFrame with column 'VWAP'

# Signal interpretation:
# - Price above VWAP: Bullish bias
# - Price below VWAP: Bearish bias
# - Price returns to VWAP: Potential mean reversion
```

---

## Indicator Factory

The `IndicatorFactory` provides a unified interface for creating and calculating indicators.

### List Available Indicators

```python
from src.technical_indicators import IndicatorFactory

# Get list of all available indicators
indicators = IndicatorFactory.list_indicators()
print(indicators)
# Output: ['SMA', 'EMA', 'WMA', 'MACD', 'ADX', 'ParabolicSAR', 'RSI', ...]
```

### Get Indicator Information

```python
# Get information about a specific indicator
info = IndicatorFactory.get_indicator_info('RSI')
print(info)
# Output: {'name': 'RSI', 'class': 'RSI', 'docstring': '...'}
```

### Create Indicator Instance

```python
# Create an indicator by name
rsi = IndicatorFactory.create_indicator('RSI', period=14)
result = rsi.calculate(data)
```

### Calculate Indicator Directly

```python
# Calculate an indicator without creating an instance
result = IndicatorFactory.calculate_indicator(data, 'RSI', period=14)
```

### Calculate Multiple Indicators

```python
# Calculate multiple indicators at once
indicators = [
    {'name': 'RSI', 'params': {'period': 14}},
    {'name': 'MACD', 'params': {'fast_period': 12, 'slow_period': 26}},
    {'name': 'BollingerBands', 'params': {'period': 20, 'std_dev': 2}}
]
results = IndicatorFactory.calculate_multiple(data, indicators)
```

---

## Usage Examples

### Example 1: Complete Technical Analysis

```python
import pandas as pd
from src.technical_indicators import IndicatorFactory

# Load data
data = pd.read_csv('data/XOM_verified_ohlcv.csv', index_col='Date', parse_dates=True)

# Define indicators for analysis
indicators = [
    # Trend indicators
    {'name': 'SMA', 'params': {'period': 20}},
    {'name': 'SMA', 'params': {'period': 50}},
    {'name': 'EMA', 'params': {'period': 20}},
    {'name': 'MACD'},
    
    # Momentum indicators
    {'name': 'RSI', 'params': {'period': 14}},
    {'name': 'Stochastic'},
    
    # Volatility indicators
    {'name': 'BollingerBands'},
    {'name': 'ATR'},
    
    # Volume indicators
    {'name': 'OBV'},
    {'name': 'MFI'},
    
    # Advanced indicators
    {'name': 'IchimokuCloud'},
    {'name': 'VWAP'}
]

# Calculate all indicators
results = IndicatorFactory.calculate_multiple(data, indicators)

# Combine with original data
analysis_data = pd.concat([data, results], axis=1)

# Save results
analysis_data.to_csv('data/technical_analysis.csv')

print(f"Analysis complete. {len(analysis_data.columns)} indicators calculated.")
```

### Example 2: Trading Signal Generation

```python
import pandas as pd
from src.technical_indicators import IndicatorFactory

# Load data
data = pd.read_csv('data/XOM_verified_ohlcv.csv', index_col='Date', parse_dates=True)

# Calculate indicators
indicators = [
    {'name': 'RSI', 'params': {'period': 14}},
    {'name': 'MACD'},
    {'name': 'BollingerBands'}
]
results = IndicatorFactory.calculate_multiple(data, indicators)

# Generate signals
signals = pd.DataFrame(index=data.index)
signals['Close'] = data['Close']

# RSI signals
signals['RSI_Buy'] = (results['RSI_14'] < 30) & (results['RSI_14'].shift() >= 30)
signals['RSI_Sell'] = (results['RSI_14'] > 70) & (results['RSI_14'].shift() <= 70)

# MACD signals
signals['MACD_Buy'] = (results['MACD'] > results['MACD_Signal']) & \
                      (results['MACD'].shift() <= results['MACD_Signal'].shift())
signals['MACD_Sell'] = (results['MACD'] < results['MACD_Signal']) & \
                       (results['MACD'].shift() >= results['MACD_Signal'].shift())

# Bollinger Bands signals
signals['BB_Buy'] = (data['Close'] < results['BB_Lower']) & \
                    (data['Close'].shift() >= results['BB_Lower'].shift())
signals['BB_Sell'] = (data['Close'] > results['BB_Upper']) & \
                     (data['Close'].shift() <= results['BB_Upper'].shift())

# Combined signals
signals['Combined_Buy'] = signals['RSI_Buy'] | signals['MACD_Buy'] | signals['BB_Buy']
signals['Combined_Sell'] = signals['RSI_Sell'] | signals['MACD_Sell'] | signals['BB_Sell']

# Print recent signals
print("Recent Buy Signals:")
print(signals[signals['Combined_Buy']].tail())

print("\nRecent Sell Signals:")
print(signals[signals['Combined_Sell']].tail())
```

### Example 3: Pattern Recognition

```python
import pandas as pd
from src.technical_indicators import IndicatorFactory

# Load data
data = pd.read_csv('data/XOM_verified_ohlcv.csv', index_col='Date', parse_dates=True)

# Calculate indicators
results = IndicatorFactory.calculate_multiple(data, [
    {'name': 'RSI', 'params': {'period': 14}},
    {'name': 'BollingerBands'},
    {'name': 'ATR'}
])

# Identify patterns
patterns = pd.DataFrame(index=data.index)

# Oversold bounce pattern
patterns['Oversold_Bounce'] = (
    (results['RSI_14'] < 30) &
    (data['Close'] < results['BB_Lower']) &
    (data['Close'] > data['Close'].shift())
)

# Volatility squeeze (Bollinger Band contraction)
patterns['Volatility_Squeeze'] = (
    (results['BB_Bandwidth'] < results['BB_Bandwidth'].rolling(20).mean() * 0.8)
)

# Trend confirmation
patterns['Uptrend_Confirmed'] = (
    (data['Close'] > data['Close'].rolling(50).mean()) &
    (results['RSI_14'] > 50) &
    (results['ATR_14'] > results['ATR_14'].rolling(20).mean())
)

# Count patterns
print("Pattern Counts:")
print(patterns.sum())
```

### Example 4: Backtesting Framework Integration

```python
import pandas as pd
from src.technical_indicators import IndicatorFactory

class SimpleStrategy:
    def __init__(self, data):
        self.data = data
        self.indicators = IndicatorFactory.calculate_multiple(data, [
            {'name': 'RSI', 'params': {'period': 14}},
            {'name': 'MACD'},
            {'name': 'BollingerBands'}
        ])
        self.position = 0
        self.trades = []
    
    def generate_signals(self):
        """Generate trading signals based on indicators."""
        signals = pd.DataFrame(index=self.data.index)
        
        # RSI oversold/overbought
        signals['RSI_Signal'] = 0
        signals.loc[self.indicators['RSI_14'] < 30, 'RSI_Signal'] = 1  # Buy
        signals.loc[self.indicators['RSI_14'] > 70, 'RSI_Signal'] = -1  # Sell
        
        # MACD crossover
        signals['MACD_Signal'] = 0
        macd_cross = (self.indicators['MACD'] > self.indicators['MACD_Signal']).astype(int)
        signals.loc[macd_cross.diff() == 1, 'MACD_Signal'] = 1
        signals.loc[macd_cross.diff() == -1, 'MACD_Signal'] = -1
        
        return signals
    
    def backtest(self):
        """Run simple backtest."""
        signals = self.generate_signals()
        
        for i in range(1, len(signals)):
            if signals.iloc[i]['RSI_Signal'] == 1 and self.position == 0:
                # Buy signal
                self.position = 1
                self.trades.append({
                    'date': signals.index[i],
                    'action': 'BUY',
                    'price': self.data['Close'].iloc[i]
                })
            elif signals.iloc[i]['RSI_Signal'] == -1 and self.position == 1:
                # Sell signal
                self.position = 0
                self.trades.append({
                    'date': signals.index[i],
                    'action': 'SELL',
                    'price': self.data['Close'].iloc[i]
                })
        
        return pd.DataFrame(self.trades)

# Run backtest
data = pd.read_csv('data/XOM_verified_ohlcv.csv', index_col='Date', parse_dates=True)
strategy = SimpleStrategy(data)
trades = strategy.backtest()

print(f"Total Trades: {len(trades)}")
print(trades.head())
```

---

## Testing

The library includes comprehensive unit tests for all indicators.

### Running Tests

```bash
# Run all tests
cd tests
python test_technical_indicators.py

# Run specific test class
python -m unittest tests.test_technical_indicators.TestRSI

# Run with verbose output
python -m unittest tests.test_technical_indicators -v
```

### Test Coverage

The test suite includes:
- 69 unit tests covering all indicators
- Tests for mathematical correctness
- Tests for edge cases and boundary conditions
- Integration tests for multiple indicators
- Factory pattern tests

### Test Results Summary

```
======================================================================
TEST SUMMARY
======================================================================
Tests run: 69
Successes: 69
Failures: 0
Errors: 0
======================================================================
```

---

## API Reference

### IndicatorBase

Base class for all technical indicators.

**Methods:**
- `validate_data(data: pd.DataFrame) -> None`: Validate input data
- `calculate(data: pd.DataFrame, **kwargs) -> pd.DataFrame`: Calculate indicator values
- `_get_close(data: pd.DataFrame) -> pd.Series`: Get close prices
- `_get_high(data: pd.DataFrame) -> pd.Series`: Get high prices
- `_get_low(data: pd.DataFrame) -> pd.Series`: Get low prices
- `_get_volume(data: pd.DataFrame) -> pd.Series`: Get volume

### IndicatorFactory

Factory class for creating and managing indicators.

**Class Methods:**
- `create_indicator(indicator_name: str, **kwargs) -> IndicatorBase`: Create indicator instance
- `calculate_indicator(data: pd.DataFrame, indicator_name: str, **kwargs) -> pd.DataFrame`: Calculate indicator
- `calculate_multiple(data: pd.DataFrame, indicators: List[Dict]) -> pd.DataFrame`: Calculate multiple indicators
- `list_indicators() -> List[str]`: List all available indicators
- `get_indicator_info(indicator_name: str) -> Dict`: Get indicator information

### Convenience Functions

- `calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame`: Calculate all indicators
- `get_indicator_summary() -> pd.DataFrame`: Get summary of all indicators

---

## Best Practices

1. **Data Validation**: Always ensure your data has the required columns (Open, High, Low, Close, Volume) before calculating indicators.

2. **Parameter Selection**: Choose appropriate parameters based on your trading timeframe and strategy:
   - Short-term: Use smaller periods (e.g., RSI 7, SMA 10)
   - Long-term: Use larger periods (e.g., RSI 21, SMA 50)

3. **Signal Confirmation**: Use multiple indicators to confirm signals and reduce false positives.

4. **Backtesting**: Always backtest your strategies before live trading.

5. **Performance**: When calculating many indicators, use `calculate_multiple()` for better performance.

6. **NaN Handling**: Be aware that most indicators produce NaN values for initial periods equal to their period length.

---

## Troubleshooting

### Common Issues

**Issue: `ValueError: Missing required columns`**
- **Solution**: Ensure your DataFrame has columns: Open, High, Low, Close, Volume

**Issue: All NaN values in results**
- **Solution**: Check that you have enough data points (at least equal to the indicator period)

**Issue: Division by zero warnings**
- **Solution**: These are expected for initial calculations and are handled internally

**Issue: Slow performance with many indicators**
- **Solution**: Use `calculate_multiple()` instead of calling `calculate()` multiple times

---

## Contributing

When adding new indicators:

1. Inherit from `IndicatorBase`
2. Implement the `calculate()` method
3. Add comprehensive docstring with formula and usage
4. Add unit tests in `tests/test_technical_indicators.py`
5. Register in `IndicatorFactory._indicators` dictionary
6. Update this documentation

---

## License

This library is part of the project and follows the same license terms.

---

## Changelog

### Version 1.0.0 (2025-01-22)
- Initial release
- 22 technical indicators
- Comprehensive unit tests (69 tests)
- Factory pattern for indicator management
- Full documentation with examples