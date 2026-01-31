# Feature Engineering Documentation

## Overview

This document provides comprehensive documentation for all features calculated in the Market State Feature Engineering module. The feature engineering process transforms raw OHLCV (Open, High, Low, Close, Volume) data into meaningful indicators that support data-driven pattern discovery.

## Feature Categories

### 1. Price-Based Features

These features capture price movements and relationships to moving averages and historical price levels.

#### Distance from Moving Averages
- **Dist_MA_{period}**: Distance from {period}-day moving average, expressed as percentage
  - Formula: `((Current Price - MA) / MA) * 100`
  - Range: Typically -50% to +50%
  - Interpretation: Positive values indicate price above MA (bullish), negative values indicate price below MA (bearish)

#### Rate of Change
- **ROC_{period}d**: {period}-day rate of change
  - Formula: `((Price_now - Price_{period}_days_ago) / Price_{period}_days_ago) * 100`
  - Range: Typically -20% to +20%
  - Interpretation: Measures price momentum over specified period

#### 52-Week Range
- **52w_Range_Pct**: Position within 52-week range
  - Formula: `((Current Price - 52w_Low) / (52w_High - 52w_Low)) * 100`
  - Range: 0% to 100%
  - Interpretation: 0% = at 52-week low, 100% = at 52-week high

#### Distance from Recent Highs/Lows
- **Dist_{period}d_High**: Distance from {period}-day high
  - Formula: `((Current Price - Period_High) / Period_High) * 100`
- **Dist_{period}d_Low**: Distance from {period}-day low
  - Formula: `((Current Price - Period_Low) / Period_Low) * 100`

#### Candlestick Pattern Features
- **Body_Range_Ratio**: Ratio of candle body to total range
  - Formula: `abs(Close - Open) / (High - Low) * 100`
- **Upper_Shadow_Ratio**: Ratio of upper shadow to total range
  - Formula: `((High - max(Open, Close)) / (High - Low)) * 100`
- **Lower_Shadow_Ratio**: Ratio of lower shadow to total range
  - Formula: `((min(Open, Close) - Low) / (High - Low)) * 100`
- **Doji**: Boolean indicator for doji candles (body < 10% of range)
- **Hammer**: Boolean indicator for hammer candles
- **Shooting_Star**: Boolean indicator for shooting star candles

#### Gap Features
- **Gap_Pct**: Gap percentage from previous close
  - Formula: `((Open - Previous_Close) / Previous_Close) * 100`
- **Gap_Up**: Boolean indicator for positive gaps
- **Gap_Down**: Boolean indicator for negative gaps

### 2. Volatility Features

These features measure market volatility and price dispersion.

#### Average True Range (ATR)
- **ATR_{period}_Pct**: {period}-day ATR as percentage of price
  - Formula: `(ATR / Close) * 100`
- **ATR_{period}_Percentile**: Percentile rank of ATR vs. last 100 days
- **ATR_{period}_Trend**: Change in ATR over 10 periods

#### Bollinger Bands
- **BB_Width_{period}**: Bollinger Band width
  - Formula: `((Upper_BB - Lower_BB) / Middle_BB) * 100`
- **BB_Position_{period}**: Position within Bollinger Bands (0-100 scale)
- **BB_Squeeze_{period}**: Boolean indicator for low volatility periods

#### Intraday Range
- **Intraday_Range**: Daily range as percentage of closing price
  - Formula: `((High - Low) / Close) * 100`

### 3. Momentum Features

These features capture price momentum and trend strength.

#### Relative Strength Index (RSI)
- **RSI_{period}**: {period}-day RSI
  - Range: 0 to 100
  - Interpretation: <30 = oversold, >70 = overbought

#### MACD (Moving Average Convergence Divergence)
- **MACD_Histogram**: MACD histogram values
- **MACD_Signal_Pct**: MACD signal line as percentage of price
- **MACD_Bull_Cross**: Boolean indicator for bullish crossover
- **MACD_Bear_Cross**: Boolean indicator for bearish crossover

#### Stochastic Oscillator
- **Stoch_{period}_K**: %K line of Stochastic oscillator
- **Stoch_{period}_D**: %D line (3-period moving average of %K)

#### Williams %R
- **Williams_R_{period}**: {period}-day Williams %R
  - Range: -100 to 0
  - Interpretation: Values near -100 indicate oversold conditions, near 0 indicate overbought

#### Commodity Channel Index (CCI)
- **CCI_{period}**: {period}-day CCI
  - Range: Typically -300 to +300
  - Interpretation: Values >100 indicate overbought, <-100 indicate oversold

#### Chaikin Oscillator
- **Chaikin_Oscillator**: Difference between 3-day and 10-day EMA of ADL

#### Enhanced Momentum Features
- **RSI_{period}_ROC**: Rate of change of RSI (momentum of momentum)
- **MACD_Slope**: 3-day slope of MACD line
- **Price_Acceleration**: Second derivative of price
- **Volume_Momentum**: 5-day rate of change of volume

### 4. Volume Features

These features analyze trading volume patterns and relationships.

#### Volume Ratios
- **Vol_Ratio_{period}**: Current volume vs. {period}-day average
  - Formula: `Volume / MA_Volume_{period}`
- **Vol_Percentile_{period}**: Volume percentile rank vs. last 100 days

#### On-Balance Volume (OBV)
- **OBV_Trend**: Boolean indicator for OBV trend vs. 20-day MA
- **OBV_ROC_5d**: 5-day rate of change of OBV
- **OBV_ROC_20d**: 20-day rate of change of OBV

#### Accumulation/Distribution
- **AD_ROC_5d**: 5-day rate of change of A/D line
- **AD_ROC_20d**: 20-day rate of change of A/D line
- **AD_Trend**: Boolean indicator for A/D line trend

#### Volume Spike Detection
- **Vol_Spike_2x**: Boolean indicator for 2x average volume
- **Vol_Spike_3x**: Boolean indicator for 3x average volume

#### Price-Volume Divergence
- **PV_Divergence**: Boolean indicator for price-volume divergence

### 5. Trend Features

These features identify and measure trend direction and strength.

#### Average Directional Index (ADX)
- **Plus_DI_{period}**: Positive Directional Indicator
- **Minus_DI_{period}**: Negative Directional Indicator
- **ADX_{period}**: Average Directional Index

#### Trend Direction Classification
- **Trend_Dir_{period}**: Categorical trend direction (Up/Down/Sideways/Neutral)

#### Trend Strength
- **Trend_Strength**: Normalized ADX value (0-1 scale)

#### Moving Average Alignment
- **MA_Alignment**: Count of aligned moving averages
- **MA_Alignment_Score**: Percentage of aligned moving averages
- **MA_{period}_Slope**: 5-day slope of distance from {period}-day MA

### 6. Regime Features

These features classify market conditions into different regimes.

#### Volatility Regime
- **Vol_Regime**: Categorical volatility regime (Low/Medium/High)

#### Trend Regime
- **Trend_Regime**: Categorical trend regime (Strong Bull/Weak Bull/Sideways/Weak Bear/Strong Bear)

#### Volume Regime
- **Vol_Regime_Level**: Categorical volume regime (Low/Normal/High)

### 7. Pattern Recognition Features

These features identify common chart patterns and price behaviors.

#### Consecutive Moves
- **Consec_Up**: Count of consecutive up days
- **Consec_Down**: Count of consecutive down days

#### Price Sequences
- **Higher_High**: Boolean indicator for higher high
- **Higher_Low**: Boolean indicator for higher low
- **Lower_High**: Boolean indicator for lower high
- **Lower_Low**: Boolean indicator for lower low

#### Support/Resistance Proximity
- **Dist_Support**: Distance from support level
- **Dist_Resistance**: Distance from resistance level
- **Near_Support**: Boolean indicator for proximity to support
- **Near_Resistance**: Boolean indicator for proximity to resistance

#### Breakout/Breakdown Signals
- **Breakout**: Boolean indicator for breakout above resistance
- **Breakdown**: Boolean indicator for breakdown below support

#### Consolidation Detection
- **Consolidation**: Boolean indicator for low volatility periods

### 8. Temporal Features

These features capture time-based patterns and cycles.

#### Time-Based Features
- **Day_of_Week**: Numeric day of week (0=Monday, 6=Sunday)
- **Month**: Numeric month (1-12)
- **Quarter**: Quarter of year (1-4)

#### Time Since Events
- **Days_Since_Major_Move**: Days since >5% price move
- **Days_Since_52w_High**: Days since 52-week high
- **Days_Since_52w_Low**: Days since 52-week low

### 9. Fibonacci Features

These features use Fibonacci retracement levels to identify potential support/resistance areas.

#### Fibonacci Retracement Levels
- **Fib_23.6**: Distance from 23.6% Fibonacci level
- **Fib_38.2**: Distance from 38.2% Fibonacci level
- **Fib_50.0**: Distance from 50.0% Fibonacci level
- **Fib_61.8**: Distance from 61.8% Fibonacci level
- **Fib_78.6**: Distance from 78.6% Fibonacci level

### 10. VWAP Features

These features use Volume Weighted Average Price to identify institutional trading levels.

#### VWAP Metrics
- **VWAP_Distance**: Distance from VWAP as percentage
- **VWAP_Slope**: 5-day slope of VWAP

### 11. Cycle Features

These features identify market cycles and their phases.

#### Cycle Indicators
- **Dominant_Cycle**: Estimated dominant cycle period in days
- **Cycle_Phase**: Normalized cycle phase (0-100)

## Feature Selection

### Importance Scoring

Features are scored based on two criteria:
1. **Variance**: Higher variance indicates more information content
2. **Correlation**: Lower mean correlation with other features indicates less redundancy

### Selection Process

The feature selection mechanism ranks all features by importance score and can select the top N features for pattern discovery.

## Performance Considerations

### Optimizations

1. **Vectorized Operations**: All calculations use vectorized pandas operations for efficiency
2. **Caching**: Expensive calculations are cached to avoid recomputation
3. **Memory Management**: NaN values are handled efficiently to minimize memory usage

### Computational Complexity

- Most features: O(n) where n is the number of data points
- Temporal features: O(n) with optimized vectorized operations
- Cycle features: O(n) with simplified autocorrelation approach

## Usage Examples

### For Pattern Discovery
```python
# Initialize feature engineering
fe = FeatureEngineering()

# Load data
fe.load_data("data/movement_labeled_data.csv")

# Calculate all features
features = fe.calculate_all_features()

# Select top features
top_features = fe.select_top_features(features, n_features=50)
```

### For Feature Analysis
```python
# Calculate feature importance
importance_df = fe.calculate_feature_importance(features)

# View top 10 most important features
print(importance_df.head(10))
```

## Best Practices

1. **Normalization**: All features are normalized to percentage or categorical values for consistency
2. **Redundancy Reduction**: Correlation analysis helps identify and remove redundant features
3. **Regime Awareness**: Features are designed to work across different market regimes
4. **Robustness**: Features use relative measures to adapt to different price levels and market conditions

## Feature Engineering Pipeline

1. **Data Loading**: Load movement-labeled OHLCV data
2. **Feature Calculation**: Calculate all feature categories
3. **Cleaning**: Handle NaN and infinity values
4. **Selection**: (Optional) Select top features based on importance
5. **Output**: Save features to CSV and feature list to text file

This comprehensive feature engineering approach provides a rich set of indicators for data-driven pattern discovery while maintaining computational efficiency and robustness across different market conditions.