"""Debug AD_ROC calculation for Pattern #0."""
import pandas as pd
import numpy as np
import json

# Load data
df = pd.read_csv('data/features_matrix.csv', index_col='Date', parse_dates=True)

# Check Pattern #0 conditions
with open('data/patterns.json', 'r') as f:
    patterns = json.load(f)

pattern = patterns[0]
print("Pattern #0:")
print(f"  Conditions: {pattern.get('conditions', {})}")
print(f"  Label: {pattern.get('label_col')}")
print()

# Check AD_ROC_20d values
ad_roc = df['AD_ROC_20d']
print("AD_ROC_20d Statistics:")
print(f"  Min: {ad_roc.min():.2f}")
print(f"  Max: {ad_roc.max():.2f}")
print(f"  Mean: {ad_roc.mean():.2f}")
print(f"  Std: {ad_roc.std():.2f}")
print()

# Find extreme values
print("Top 10 highest AD_ROC_20d values:")
top10 = ad_roc.nlargest(10)
for date, value in top10.items():
    row = df.loc[date]
    print(f"  {date.strftime('%Y-%m-%d')}: {value:.2f}")
    print(f"    Close={row['Close']:.2f}, High={row['High']:.2f}, Low={row['Low']:.2f}, Volume={row['Volume']}")
print()

# Check for Pattern #0 matching dates
conditions = pattern.get('conditions', {})
matching_dates = []
for date, row in df.iterrows():
    match = True
    for feature, condition in conditions.items():
        if feature not in row:
            match = False
            break
        value = row[feature]
        op = condition.get('operator', '~')
        target = condition['value']
        
        if op == '>=':
            if value < target: match = False
        elif op == '<=':
            if value > target: match = False
        elif op == '>':
            if value <= target: match = False
        elif op == '<':
            if value >= target: match = False
    
    if match:
        matching_dates.append((date, row))

print(f"Pattern #0 matches {len(matching_dates)} dates:")
for date, row in matching_dates[:10]:
    print(f"  {date.strftime('%Y-%m-%d')}: AD_ROC_20d={row['AD_ROC_20d']:.2f}")
