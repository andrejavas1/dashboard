"""
Test the enhanced pattern filter to detect metric duplicates.
"""
import json
import sys
sys.path.insert(0, 'src')

from pattern_filter import PatternFilter

# Load patterns.json
print("Loading patterns.json...")
with open('data/patterns.json', 'r') as f:
    patterns = json.load(f)

print(f"Loaded {len(patterns)} patterns")

# Create filter
filter = PatternFilter()

# Test on patterns with similar metrics
print("\n" + "="*80)
print("Testing metric duplicate detection...")
print("="*80)

# Extract pattern objects (some may be wrapped in 'pattern' key)
pattern_objects = []
for item in patterns:
    if 'pattern' in item:
        # Merge pattern data with validation data
        pattern_data = item['pattern'].copy()
        # Add validation metrics at top level
        for key in ['training_success_rate', 'validation_success_rate', 
                     'statistically_significant', 'classification']:
            if key in item:
                pattern_data[key] = item[key]
        pattern_objects.append(pattern_data)
    else:
        pattern_objects.append(item)

print(f"Extracted {len(pattern_objects)} pattern objects")

# Apply filtering
filtered_patterns = filter.filter_patterns(
    pattern_objects,
    remove_exact_duplicates=True,
    remove_condition_duplicates=True,
    remove_metric_duplicates=True,
    remove_subsets=True
)

print("\n" + "="*80)
print("FILTERING RESULTS")
print("="*80)
print(f"Input patterns: {filter.filter_stats['total_input']}")
print(f"Exact duplicates removed: {filter.filter_stats['exact_duplicates_removed']}")
print(f"Condition duplicates removed: {filter.filter_stats['condition_duplicates_removed']}")
print(f"Metric duplicates removed: {filter.filter_stats['metric_duplicates_removed']}")
print(f"Subset patterns removed: {filter.filter_stats['subset_patterns_removed']}")
print(f"Output patterns: {filter.filter_stats['total_output']}")
print(f"Reduction: {(1 - filter.filter_stats['total_output'] / filter.filter_stats['total_input']) * 100:.1f}%")

# Save filtered patterns
print("\nSaving filtered patterns...")
with open('data/filtered_patterns.json', 'w') as f:
    json.dump(filtered_patterns, f, indent=2)

print("Saved to data/filtered_patterns.json")

# Show some examples of removed duplicates
print("\n" + "="*80)
print("EXAMPLES OF DETECTED DUPLICATES")
print("="*80)

# Find patterns with similar conditions but different statistical significance
seen_conditions = {}
duplicates_found = []

for idx, pattern in enumerate(pattern_objects):
    conditions = pattern.get('conditions', {})
    # Create a simplified key for grouping
    cond_key = tuple(sorted([
        (k, v.get('operator'), round(v.get('value', 0), 2))
        for k, v in conditions.items()
    ]))
    
    if cond_key in seen_conditions:
        duplicates_found.append((seen_conditions[cond_key], idx, pattern))
    else:
        seen_conditions[cond_key] = idx

if duplicates_found:
    for dup_idx, (idx1, idx2, pattern2) in enumerate(duplicates_found[:5]):
        pattern1 = pattern_objects[idx1]
        print(f"\nDuplicate #{dup_idx + 1}:")
        print(f"  Pattern #{idx1}:")
        print(f"    Conditions: {list(pattern1.get('conditions', {}).keys())}")
        print(f"    Success Rate: {pattern1.get('success_rate', 'N/A'):.1f}%")
        print(f"    Statistically Significant: {pattern1.get('statistically_significant', 'N/A')}")
        print(f"  Pattern #{idx2}:")
        print(f"    Conditions: {list(pattern2.get('conditions', {}).keys())}")
        print(f"    Success Rate: {pattern2.get('success_rate', 'N/A'):.1f}%")
        print(f"    Statistically Significant: {pattern2.get('statistically_significant', 'N/A')}")
else:
    print("No duplicates found with identical conditions.")