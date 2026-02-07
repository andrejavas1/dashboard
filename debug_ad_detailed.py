"""Detailed debug of AD_ROC calculation to find the 49613% value."""
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/features_matrix.csv', index_col='Date', parse_dates=True)

# Focus on the extreme value date
extreme_date = '2025-06-25'
idx = df.index.get_loc(extreme_date)

print("=" * 80)
print(f"DETAILED AD_ROC DEBUG FOR {extreme_date}")
print("=" * 80)
print()

# Get the exact AD_ROC value
ad_roc_value = df.loc[extreme_date, 'AD_ROC_20d']
print(f"AD_ROC_20d from features_matrix: {ad_roc_value:.2f}%")
print()

# Calculate AD step by step exactly as in the code
df_calc = df.copy()

# Step 1: Calculate MFV exactly as in phase3_feature_engineering.py
# mfv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
df_calc['mfv'] = ((df_calc['Close'] - df_calc['Low']) - (df_calc['High'] - df_calc['Close'])) / (df_calc['High'] - df_calc['Low'])
df_calc['mfv'] = df_calc['mfv'].fillna(0.5)

# Step 2: Calculate AD line - cumulative sum of MFV * Volume
# ad = (mfv * data['Volume']).cumsum()
df_calc['ad'] = (df_calc['mfv'] * df_calc['Volume']).cumsum()

# Step 3: Calculate AD_ROC_20d
# data['AD_ROC_20d'] = ad.pct_change(20) * 100
df_calc['ad_roc_20d_calc'] = df_calc['ad'].pct_change(20) * 100

# Compare
calculated_roc = df_calc.loc[extreme_date, 'ad_roc_20d_calc']
print(f"AD_ROC_20d recalculated: {calculated_roc:.2f}%")
print()

# Show AD values around the extreme date
print("AD values around the extreme date:")
slice_df = df_calc.iloc[idx-25:idx+1][['ad', 'ad_roc_20d_calc']]
for date, row in slice_df.iterrows():
    marker = " <-- EXTREME" if date.strftime('%Y-%m-%d') == extreme_date else ""
    print(f"  {date.strftime('%Y-%m-%d')}: AD={row['ad']:15,.2f}, AD_ROC_20d={row['ad_roc_20d_calc']:10.2f}%{marker}")
print()

# Check if the formula is using percentage correctly
print("Checking AD 20 days before:")
ad_today = df_calc.loc[extreme_date, 'ad']
ad_20_before = df_calc.iloc[idx-20]['ad']
print(f"  AD today ({extreme_date}): {ad_today:,.2f}")
print(f"  AD 20 days ago ({df_calc.index[idx-20].strftime('%Y-%m-%d')}): {ad_20_before:,.2f}")
print()

# Manual calculation
if ad_20_before != 0:
    manual_calc = ((ad_today - ad_20_before) / ad_20_before) * 100
    print(f"  Manual calculation: (({ad_today:,.2f} - {ad_20_before:,.2f}) / {ad_20_before:,.2f}) * 100")
    print(f"                     = {manual_calc:.2f}%")
else:
    print("  AD 20 days ago is ZERO - division by zero!")

print()
print("=" * 80)
print("CHECKING IF AD 20 DAYS AGO IS NEGATIVE")
print("=" * 80)
if ad_20_before < 0:
    print(f"AD 20 days ago is NEGATIVE: {ad_20_before:,.2f}")
    print()
    print("This explains the extreme value!")
    print("When base is negative, percentage change formula gives wrong results.")
    print()
    print("Example: AD goes from -100 to +100")
    print("  Actual change: +200 (from -100 to +100)")
    print("  Formula: (100 - (-100)) / (-100) * 100 = 200 / -100 * 100 = -200%")
    print("  But should be: +200% (doubled in magnitude and changed sign)")
    print()
    print("The percentage change formula is mathematically incorrect for")
    print("cumulative oscillators that cross zero!")
