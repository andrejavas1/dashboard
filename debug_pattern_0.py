"""Debug script to check why Pattern #0 has invalid occurrences."""
import json
import pandas as pd

# Load patterns
with open('data/patterns.json', 'r') as f:
    patterns = json.load(f)

pattern_0 = patterns[0]
print("Pattern #0 Conditions:")
for cond, details in pattern_0['conditions'].items():
    print(f"  {cond} {details['operator']} {details['value']}")

# Load features matrix
df = pd.read_csv('data/features_matrix.csv', index_col='Date', parse_dates=True)

# Check the specific date
date = '2024-12-12'
if date in df.index:
    row = df.loc[date]
    print(f"\nValues on {date}:")
    for cond in pattern_0['conditions'].keys():
        if cond in row:
            print(f"  {cond}: {row[cond]}")
    
    print("\nChecking conditions:")
    all_match = True
    for cond, details in pattern_0['conditions'].items():
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
            
            status = "✓" if match else "✗"
            print(f"  {status} {cond} {op} {target}: actual={value}")
            if not match:
                all_match = False
    
    print(f"\nAll conditions match: {all_match}")
else:
    print(f"\nDate {date} not found in data")

# Find all actual occurrences
print("\n\nFinding ALL dates that match Pattern #0 conditions...")
matches = []
for date_idx, (date, row) in enumerate(df.iterrows()):
    match = True
    for cond, details in pattern_0['conditions'].items():
        if cond not in row or pd.isna(row[cond]):
            match = False
            break
        
        value = row[cond]
        op = details['operator']
        target = details['value']
        
        if op == '>=':
            if value < target: match = False
        elif op == '<=':
            if value > target: match = False
        elif op == '>':
            if value <= target: match = False
        elif op == '<':
            if value >= target: match = False
    
    if match:
        matches.append(date.strftime('%Y-%m-%d'))

print(f"Found {len(matches)} valid occurrences:")
for m in matches[:10]:
    print(f"  {m}")
if len(matches) > 10:
    print(f"  ... and {len(matches)-10} more")
