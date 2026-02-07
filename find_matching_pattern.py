"""Find which pattern in patterns.json matches the occurrence file."""
import json
import pandas as pd

# Load patterns and occurrences
with open('data/patterns.json', 'r') as f:
    patterns = json.load(f)

with open('data/pattern_0_occurrences.json', 'r') as f:
    occurrences = json.load(f)

df = pd.read_csv('data/features_matrix.csv', index_col='Date', parse_dates=True)

occurrence_dates = [occ['Date'] for occ in occurrences]
print(f"Occurrence file has {len(occurrence_dates)} dates")
print(f"First 5 dates: {occurrence_dates[:5]}")
print()

# Check each pattern to find the best match
best_match = None
best_score = 0

for i, pattern in enumerate(patterns):
    conditions = pattern['conditions']
    direction = pattern.get('direction', 'long')
    
    matching = 0
    for date_str in occurrence_dates:
        if date_str not in df.index:
            continue
        
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
            matching += 1
    
    score = matching / len(occurrence_dates) * 100
    
    if score > best_score:
        best_score = score
        best_match = (i, pattern, matching, score)
    
    if score > 50:  # Print any pattern with >50% match
        print(f"Pattern #{i}: {matching}/{len(occurrence_dates)} match ({score:.1f}%)")
        print(f"  Direction: {direction}")
        print(f"  Label: {pattern.get('label_col')}")
        print(f"  Conditions:")
        for cond, details in conditions.items():
            print(f"    {cond} {details['operator']} {details['value']}")
        print()

if best_match:
    i, pattern, matching, score = best_match
    print("="*60)
    print(f"BEST MATCH: Pattern #{i} with {score:.1f}% match")
    print("="*60)
    print(f"This occurrence file was likely generated for Pattern #{i}")
    print(f"But the dashboard thinks it's for Pattern #0")
else:
    print("No pattern matches >50% of the occurrence dates!")
    print("The occurrence file may be completely out of sync.")
