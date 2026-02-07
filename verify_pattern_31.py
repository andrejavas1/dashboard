"""Verify Pattern #31 data."""
import json
import pandas as pd

# Load pattern #31
with open('data/patterns.json', 'r') as f:
    patterns = json.load(f)

pattern = patterns[31]
print("Pattern #31:")
print(f"  Direction: {pattern.get('direction')}")
print(f"  Success Rate: {pattern.get('success_rate', 0):.1f}%")
print(f"  Occurrences: {pattern.get('occurrences', 0)}")
print(f"  Avg Move: {pattern.get('avg_move', 0):.2f}%")
print(f"  Label: {pattern.get('label_col')}")
print("  Conditions:")
for cond, details in pattern['conditions'].items():
    print(f"    {cond} {details['operator']} {details['value']}")

# Load occurrence file
with open('data/pattern_31_occurrences.json', 'r') as f:
    occurrences = json.load(f)

print(f"\nOccurrence file has {len(occurrences)} entries")

# Verify first occurrence matches conditions
df = pd.read_csv('data/features_matrix.csv', index_col='Date', parse_dates=True)

if occurrences:
    first_date = occurrences[0]['Date']
    print(f"\nVerifying first occurrence: {first_date}")
    
    if first_date in df.index:
        row = df.loc[first_date]
        all_match = True
        
        for feature, condition in pattern['conditions'].items():
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
                if not matches:
                    all_match = False
            else:
                print(f"  [MISSING] {feature}")
                all_match = False
        
        print(f"\nAll conditions match: {all_match}")
    else:
        print(f"Date {first_date} not found in data")
