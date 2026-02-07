"""Debug OBV_ROC_5d calculation."""
import pandas as pd
import numpy as np

# Load features matrix
df = pd.read_csv('data/features_matrix.csv', index_col='Date', parse_dates=True)

# Check OBV_ROC_5d
obv_roc = df['OBV_ROC_5d']

print("OBV_ROC_5d Statistics:")
print(f"  Min: {obv_roc.min():.2f}")
print(f"  Max: {obv_roc.max():.2f}")
print(f"  Mean: {obv_roc.mean():.2f}")
print(f"  Std: {obv_roc.std():.2f}")
print()

# Find extreme values
print("Top 10 highest OBV_ROC_5d values:")
top10 = obv_roc.nlargest(10)
for date, value in top10.items():
    print(f"  {date.strftime('%Y-%m-%d')}: {value:.2f}")

print()
print("Top 10 lowest OBV_ROC_5d values:")
bottom10 = obv_roc.nsmallest(10)
for date, value in bottom10.items():
    print(f"  {date.strftime('%Y-%m-%d')}: {value:.2f}")

# Check for any patterns with extreme OBV_ROC_5d conditions
print()
print("Checking patterns with OBV_ROC_5d conditions...")
import json
with open('data/patterns.json', 'r') as f:
    patterns = json.load(f)

for i, pattern in enumerate(patterns):
    conditions = pattern.get('conditions', {})
    if 'OBV_ROC_5d' in conditions:
        cond = conditions['OBV_ROC_5d']
        print(f"  Pattern #{i}: OBV_ROC_5d {cond['operator']} {cond['value']:.2f}")
