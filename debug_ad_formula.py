"""Debug AD calculation formula and compare with standard implementation."""
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/features_matrix.csv', index_col='Date', parse_dates=True)

# Focus on the extreme value date
extreme_date = '2025-06-25'
idx = df.index.get_loc(extreme_date)

print("=" * 80)
print(f"DEBUGGING AD_ROC CALCULATION FOR {extreme_date}")
print("=" * 80)
print()

# Show the row
row = df.loc[extreme_date]
print(f"Data for {extreme_date}:")
print(f"  Open: {row['Open']:.2f}")
print(f"  High: {row['High']:.2f}")
print(f"  Low: {row['Low']:.2f}")
print(f"  Close: {row['Close']:.2f}")
print(f"  Volume: {row['Volume']}")
print(f"  AD_ROC_20d (from file): {row['AD_ROC_20d']:.2f}")
print()

# Recalculate manually
close = row['Close']
high = row['High']
low = row['Low']
volume = row['Volume']

# Money Flow Volume (MFV) = ((Close - Low) - (High - Close)) / (High - Low)
# = (2*Close - Low - High) / (High - Low)
numerator = (close - low) - (high - close)
denominator = high - low
mfv = numerator / denominator

print("Step 1: Calculate Money Flow Volume (MFV)")
print(f"  Formula: ((Close - Low) - (High - Close)) / (High - Low)")
print(f"  = (({close:.2f} - {low:.2f}) - ({high:.2f} - {close:.2f})) / ({high:.2f} - {low:.2f})")
print(f"  = ({close - low:.2f} - {high - close:.2f}) / {denominator:.2f}")
print(f"  = {numerator:.2f} / {denominator:.2f}")
print(f"  = {mfv:.6f}")
print()

# AD is cumulative sum of MFV * Volume
df_slice = df.iloc[max(0, idx-25):idx+1].copy()
df_slice['mfv'] = ((df_slice['Close'] - df_slice['Low']) - (df_slice['High'] - df_slice['Close'])) / (df_slice['High'] - df_slice['Low'])
df_slice['mfv'] = df_slice['mfv'].fillna(0.5)
df_slice['ad'] = (df_slice['mfv'] * df_slice['Volume']).cumsum()

print("Step 2: Calculate AD Line (cumulative)")
print(f"  AD = cumsum(MFV * Volume)")
print()
print("Last 5 days of AD values:")
for date, row_data in df_slice.tail(5).iterrows():
    print(f"  {date.strftime('%Y-%m-%d')}: MFV={row_data['mfv']:+.6f}, Vol={row_data['Volume']}, AD={row_data['ad']:,.2f}")
print()

# Calculate AD_ROC_20d
ad_current = df_slice.iloc[-1]['ad']
ad_20_days_ago = df_slice.iloc[-21]['ad'] if len(df_slice) > 20 else df_slice.iloc[0]['ad']

print("Step 3: Calculate AD_ROC_20d")
print(f"  AD today ({extreme_date}): {ad_current:,.2f}")
print(f"  AD 20 days ago ({df_slice.index[-21].strftime('%Y-%m-%d')}): {ad_20_days_ago:,.2f}")
print()

if ad_20_days_ago != 0:
    ad_roc = ((ad_current - ad_20_days_ago) / abs(ad_20_days_ago)) * 100
    print(f"  AD_ROC_20d = (({ad_current:,.2f} - {ad_20_days_ago:,.2f}) / {abs(ad_20_days_ago):,.2f}) * 100")
    print(f"             = ({ad_current - ad_20_days_ago:,.2f} / {abs(ad_20_days_ago):,.2f}) * 100")
    print(f"             = {ad_roc:.2f}%")
else:
    print("  ERROR: AD 20 days ago is zero - cannot calculate percentage change!")
    ad_roc = float('inf')

print()
print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print()

if abs(ad_20_days_ago) < 1000:
    print(f"ISSUE FOUND: AD 20 days ago is very small ({ad_20_days_ago:,.2f})")
    print("When the base value is close to zero, percentage change becomes extreme.")
    print()
    print("This is a mathematical artifact of using percentage change on cumulative")
    print("oscillators that cross zero or start from small values.")
    print()
    print("SOLUTION OPTIONS:")
    print("  1. Use absolute difference instead of percentage: ad.diff(20)")
    print("  2. Add a small offset before calculating %: (ad + offset).pct_change(20)")
    print("  3. Use normalized AD: ad / ad.rolling(100).std()")
    print("  4. Use the direction only: np.sign(ad.diff(20))")
else:
    print("The AD 20 days ago has a reasonable value.")
    print("The percentage change calculation should work correctly.")
