"""Debug script to check why occurrence generation is not matching patterns."""
import json
import pandas as pd
import os

# Load patterns from patterns.json
with open('data/patterns.json', 'r') as f:
    patterns = json.load(f)

print(f"Loaded {len(patterns)} patterns from patterns.json")
print()

# Check pattern #0
pattern_0 = patterns[0]
print("Pattern #0 from patterns.json:")
print(f"  Direction: {pattern_0.get('direction')}")
print(f"  Label: {pattern_0.get('label_col')}")
print("  Conditions:")
for cond, details in pattern_0['conditions'].items():
    print(f"    {cond} {details['operator']} {details['value']}")
print()

# Load occurrence file for pattern #0
with open('data/pattern_0_occurrences.json', 'r') as f:
    occurrences = json.load(f)

print(f"Occurrence file has {len(occurrences)} entries")
print(f"First occurrence: {occurrences[0]['Date']}")
print()

# Now let's manually check if the first occurrence matches pattern #0 conditions
df = pd.read_csv('data/features_matrix.csv', index_col='Date', parse_dates=True)

# Check first occurrence date
date = occurrences[0]['Date']
if date in df.index:
    row = df.loc[date]
    print(f"Checking if {date} matches Pattern #0 conditions:")
    
    for feature, condition in pattern_0['conditions'].items():
        if feature in row:
            value = row[feature]
            op = condition['operator']
            target = condition['value']
            
            if op == '>=':
                matches = value >= target
            elif op == '<=':
                matches = value <= target
            elif op == '>':
                matches = value > target
            elif op == '<':
                matches = value < target
            else:
                matches = False
            
            status = "PASS" if matches else "FAIL"
            print(f"  [{status}] {feature}: actual={value:.2f}, need {op} {target}")
        else:
            print(f"  [MISSING] {feature} not in data")
else:
    print(f"Date {date} not found in features matrix")

print()
print("="*60)
print("Now let's see what pattern the occurrence file ACTUALLY matches...")
print("="*60)

# Find which pattern in patterns.json actually matches the occurrence dates
occurrence_dates = set(occ['Date'] for occ in occurrences)
print(f"Occurrence file has {len(occurrence_dates)} unique dates")
print(f"First 5 dates: {list(occurrence_dates)[:5]}")
print()

# Check each pattern to see which one matches these dates
for i, pattern in enumerate(patterns[:5]):
    conditions = pattern['conditions']
    matching_dates = []
    
    for date_str in occurrence_dates:
        if date_str in df.index:
            row = df.loc[date_str]
            match = True
            for feature, condition in conditions.items():
                if feature not in row or pd.isna(row[feature]):
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
                matching_dates.append(date_str)
    
    match_pct = len(matching_dates) / len(occurrence_dates) * 100 if occurrence_dates else 0
    print(f"Pattern #{i}: {len(matching_dates)}/{len(occurrence_dates)} dates match ({match_pct:.1f}%)")
    if match_pct > 50:
        print(f"  THIS LOOKS LIKE THE CORRECT PATTERN!")
        print(f"  Conditions: {list(conditions.keys())}")
