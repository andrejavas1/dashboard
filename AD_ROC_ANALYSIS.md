# AD_ROC_20d Extreme Value Analysis

## Problem Summary
Pattern #0 shows `AD_ROC_20d` values exceeding 49,613% on 2025-06-25. This is mathematically extreme and indicates a fundamental issue with the calculation method.

## Root Cause Analysis

### The Calculation
```python
# Accumulation/Distribution Line
mfv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
ad = (mfv * data['Volume']).cumsum()

# Rate of Change
AD_ROC_20d = ad.pct_change(20) * 100
```

### What Happened on 2025-06-25
- **AD 20 days ago (2025-05-27)**: 92,438.76
- **AD today (2025-06-25)**: 45,954,785.11
- **AD_ROC_20d**: ((45,954,785.11 - 92,438.76) / 92,438.76) * 100 = **49,613.76%**

### The Core Problem
The percentage change formula `((new - old) / old) * 100` produces extreme values when:
1. The base value (`old`) is very small compared to the new value
2. The base value crosses zero (changes from negative to positive or vice versa)

In this case:
- AD 20 days ago was a small positive number (92,438)
- AD today is a large positive number (45,954,785)
- The ratio is 497x, hence 49,613%

## Comparison with Standard Libraries

### TA-Lib
TA-Lib's `AD` function calculates the Accumulation/Distribution Line but does NOT provide a rate-of-change version. The standard approach is to use:
- Raw AD Line values
- AD trend direction (up/down)
- AD momentum (absolute difference, not percentage)

### Pandas-TA
Pandas-TA calculates AD (Accumulation/Distribution) but does not provide `AD_ROC`. Instead, they use:
- `adosc()` - Chaikin A/D Oscillator (difference between fast and slow EMAs of AD)
- Trend direction indicators

## Why Percentage Change on AD is Wrong

AD is a **cumulative oscillator** - it's designed to accumulate over time. Unlike price which stays in a somewhat bounded range, AD grows indefinitely as volume accumulates. 

### Mathematical Issues:
1. **Scale dependency**: Early AD values are small, later values are huge
2. **Crossing zero**: Percentage change is undefined or inverted when crossing zero
3. **No normalization**: Unlike price, AD has no natural reference point

### Example of the Problem:
```
Day 1: AD = 100
Day 21: AD = 10,000
AD_ROC_20d = ((10,000 - 100) / 100) * 100 = 9,900%

But this just means volume accumulated - not necessarily a 100x "change" in any meaningful sense.
```

## Better Alternatives

### 1. Chaikin A/D Oscillator (ADOSC)
Standard industry approach:
```python
# Calculate AD Line
mfv = ((Close - Low) - (High - Close)) / (High - Low)
ad = (mfv * Volume).cumsum()

# Chaikin Oscillator = EMA(fast) - EMA(slow) of AD
adosc = ad.ewm(span=3).mean() - ad.ewm(span=10).mean()
```
This measures momentum relative to recent history, not from a fixed point.

### 2. AD Rate of Change (Absolute)
Use absolute difference instead of percentage:
```python
AD_ROC_20d = ad.diff(20)  # Raw difference, not percentage
```

### 3. Normalized AD
Divide by rolling standard deviation:
```python
AD_ROC_20d = ad.diff(20) / ad.rolling(50).std()
```

### 4. AD Trend Only
Just track direction:
```python
AD_Trend = (ad.diff(20) > 0).astype(int)  # 1 if rising, 0 if falling
```

## Recommendation

**Replace `AD_ROC_20d` with `ADOSC` (Chaikin A/D Oscillator)**

This is the industry-standard approach used by:
- TradingView
- ThinkOrSwim
- Bloomberg
- Most professional trading platforms

The oscillator measures whether AD momentum is increasing or decreasing relative to recent history, avoiding the percentage-change-on-cumulative-values problem entirely.

## Implementation Note

Since existing patterns use `AD_ROC_20d` conditions, a migration strategy would be needed:
1. Calculate both indicators during transition
2. Map old thresholds to new scale
3. Or use absolute difference instead of percentage
