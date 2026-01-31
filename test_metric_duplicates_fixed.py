"""
Test the updated metric duplicate filter.
Verifies that Pattern #74 and #75 are now treated as duplicates.
"""

import json
import sys
sys.path.append('src')
from pattern_filter import PatternFilter

# Load patterns from the actual patterns.json file
with open('data/patterns.json', 'r', encoding='utf-8') as f:
    patterns = json.load(f)

# Find Pattern #74 and #75
pattern_74 = None
pattern_75 = None

for p in patterns:
    if p.get('id') == 74:
        pattern_74 = p
    elif p.get('id') == 75:
        pattern_75 = p

if pattern_74 is None or pattern_75 is None:
    print("ERROR: Could not find Pattern #74 or #75")
    sys.exit(1)

print("=" * 60)
print("Pattern #74:")
print("=" * 60)
print(f"Conditions: {pattern_74.get('conditions', {})}")
print(f"Success Rate: {pattern_74.get('success_rate', 0):.2f}%")
print(f"Avg Move: {pattern_74.get('avg_move', 0):.2f}%")
print(f"Occurrences: {pattern_74.get('occurrences', 0)}")
print(f"Statistically Significant: {pattern_74.get('statistically_significant')}")
print(f"Classification: {pattern_74.get('classification')}")

print("\n" + "=" * 60)
print("Pattern #75:")
print("=" * 60)
print(f"Conditions: {pattern_75.get('conditions', {})}")
print(f"Success Rate: {pattern_75.get('success_rate', 0):.2f}%")
print(f"Avg Move: {pattern_75.get('avg_move', 0):.2f}%")
print(f"Occurrences: {pattern_75.get('occurrences', 0)}")
print(f"Statistically Significant: {pattern_75.get('statistically_significant')}")
print(f"Classification: {pattern_75.get('classification')}")

# Test the filter
print("\n" + "=" * 60)
print("Testing Pattern Filter")
print("=" * 60)

filter_obj = PatternFilter()

# Check if metrics are similar
metrics_similar = filter_obj.metrics_similar(pattern_74, pattern_75)
print(f"\nMetrics similar (Pattern #74 vs #75): {metrics_similar}")

# Calculate scores
score_74 = filter_obj.get_pattern_score(pattern_74)
score_75 = filter_obj.get_pattern_score(pattern_75)
print(f"Pattern #74 score: {score_74:.1f}")
print(f"Pattern #75 score: {score_75:.1f}")

# Run the full filter
print("\n" + "=" * 60)
print("Running Full Filter on All Patterns")
print("=" * 60)
filtered_patterns = filter_obj.filter_patterns(patterns)

stats = filter_obj.get_filter_stats()
print(f"\nInput patterns: {stats['total_input']}")
print(f"Output patterns: {stats['total_output']}")
print(f"Exact duplicates removed: {stats['exact_duplicates_removed']}")
print(f"Condition duplicates removed: {stats['condition_duplicates_removed']}")
print(f"Metric duplicates removed: {stats['metric_duplicates_removed']}")
print(f"Subset patterns removed: {stats['subset_patterns_removed']}")
print(f"Total reduction: {(1 - stats['total_output'] / stats['total_input']) * 100:.1f}%")

# Check if Pattern #74 or #75 is in the filtered results
pattern_74_in_filtered = any(p.get('id') == 74 for p in filtered_patterns)
pattern_75_in_filtered = any(p.get('id') == 75 for p in filtered_patterns)

print("\n" + "=" * 60)
print("Pattern #74 and #75 Status After Filtering")
print("=" * 60)
print(f"Pattern #74 in filtered results: {pattern_74_in_filtered}")
print(f"Pattern #75 in filtered results: {pattern_75_in_filtered}")

if not pattern_74_in_filtered and not pattern_75_in_filtered:
    print("\nERROR: Both patterns were removed (should keep one)")
elif pattern_74_in_filtered and pattern_75_in_filtered:
    print("\nERROR: Both patterns are still present (should remove one)")
elif pattern_74_in_filtered:
    print("\nSUCCESS: Pattern #74 kept (higher score)")
    print(f"  Pattern #74 score: {score_74:.1f} > Pattern #75 score: {score_75:.1f}")
else:
    print("\nSUCCESS: Pattern #75 kept (higher score)")
    print(f"  Pattern #75 score: {score_75:.1f} > Pattern #74 score: {score_74:.1f}")