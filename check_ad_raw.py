import pandas as pd
import numpy as np

df = pd.read_csv('data/features_matrix.csv')

# Recalculate AD to see raw values
mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
mfv = mfv.fillna(0.5)
ad = (mfv * df['Volume']).cumsum()

print('Raw AD Line Statistics:')
print('  Min:', ad.min())
print('  Max:', ad.max())
print('  First value:', ad.iloc[0])
print('  Last value:', ad.iloc[-1])
print()
print('AD differences (what AD_ROC_20d now shows):')
print('  Min diff:', ad.diff(20).min())
print('  Max diff:', ad.diff(20).max())
print()
print('AD changes over time:')
print('  First 5 AD values:', ad.head(5).values)
print('  Last 5 AD values:', ad.tail(5).values)
