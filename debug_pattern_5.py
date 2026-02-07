"""Debug script to check Pattern #5's first occurrence on 2024-03-07."""
import json
import pandas as pd

# Load patterns
with open('data/patterns.json', 'r') as f:
    patterns = json.load(f)

pattern_5 = patterns[5]
print("Pattern #5 Conditions:")
for cond, details in pattern_5['conditions'].items():
    print(f"  {cond} {details['operator']} {details['value']}")

# Load features matrix
df = pd.read_csv('data/features_matrix.csv', index_col='Date', parse_dates=True)

# Check the specific date
date = '2024-03-07'
if date in df.index:
    row = df.loc[date]
    print(f"\nValues on {date}:")
    for cond in pattern_5['conditions'].keys():
        if cond in row:
            print(f"  {cond}: {row[cond]}")
    
    print("\nChecking conditions:")
    all_match = True
    for cond, details in pattern_5['conditions'].items():
        if cond in row:
            value = row[cond]
            op = details['operator']
            target = details['value']
            
            if op == '>=':
                match = value >= target
            elif op == '<=':
                match = value <= target
            elif op == '>':
                match = value > target
            elif op == '<':
                match = value < target
            else:
                match = False
            
            status = "PASS" if match else "FAIL"
            print(f"  [{status}] {cond} {op} {target}: actual={value}")
            if not match:
                all_match = False
    
    print(f"\nAll conditions match: {all_match}")
else:
    print(f"\nDate {date} not found in data")
