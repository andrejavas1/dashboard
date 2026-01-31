"""
Test the updated metric duplicate filter on discovered_patterns.json.
"""

import json
import sys
sys.path.append('src')
from pattern_filter import PatternFilter

# Load patterns from discovered_patterns.json
with open('data/discovered_patterns.json', 'r', encoding='utf-8') as f:
    patterns = json.load(f)

print("=" * 60)
print("Testing Updated Pattern Filter on discovered_patterns.json")
print("=" * 60)
print(f"Total patterns in file: {len(patterns)}")

# Run the filter
filter_obj = PatternFilter()
filtered_patterns = filter_obj.filter_patterns(patterns)

stats = filter_obj.get_filter_stats()
print("\n" + "=" * 60)
print("Filter Results")
print("=" * 60)
print(f"Input patterns: {stats['total_input']}")
print(f"Output patterns: {stats['total_output']}")
print(f"Exact duplicates removed: {stats['exact_duplicates_removed']}")
print(f"Condition duplicates removed: {stats['condition_duplicates_removed']}")
print(f"Metric duplicates removed: {stats['metric_duplicates_removed']}")
print(f"Subset patterns removed: {stats['subset_patterns_removed']}")
print(f"Total reduction: {(1 - stats['total_output'] / stats['total_input']) * 100:.1f}%")

# Find patterns with the same occurrences count
occurrence_groups = {}
for p in patterns:
    occ = p.get('occurrences', 0)
    if occ not in occurrence_groups:
        occurrence_groups[occ] = []
    occurrence_groups[occ].append(p)

# Find groups with multiple patterns having the same occurrences
duplicate_occurrences = {k: v for k, v in occurrence_groups.items() if len(v) > 1}

print("\n" + "=" * 60)
print(f"Patterns with Same Occurrences Count (potential metric duplicates)")
print("=" * 60)
print(f"Found {len(duplicate_occurrences)} occurrence counts with multiple patterns")

# Show top 5 groups with most patterns
sorted_groups = sorted(duplicate_occurrences.items(), key=lambda x: len(x[1]), reverse=True)
for occ, group in sorted_groups[:5]:
    print(f"\nOccurrences: {occ} ({len(group)} patterns)")
    for i, p in enumerate(group[:3]):  # Show first 3
        print(f"  Pattern {i+1}: SR={p.get('success_rate', 0):.2f}%, Avg={p.get('avg_move', 0):.2f}%, Conditions={list(p.get('conditions', {}).keys())[:3]}")
    if len(group) > 3:
        print(f"  ... and {len(group) - 3} more")

# Check if any patterns with same occurrences also have same success rate and avg move
print("\n" + "=" * 60)
print("Patterns with Identical Metrics (occurrences + success_rate + avg_move)")
print("=" * 60)

metric_groups = {}
for p in patterns:
    key = (p.get('occurrences', 0), round(p.get('success_rate', 0), 2), round(p.get('avg_move', 0), 2))
    if key not in metric_groups:
        metric_groups[key] = []
    metric_groups[key].append(p)

identical_metric_groups = {k: v for k, v in metric_groups.items() if len(v) > 1}
print(f"Found {len(identical_metric_groups)} groups with identical metrics")

for key, group in list(identical_metric_groups.items())[:5]:
    occ, sr, avg = key
    print(f"\nMetrics: occ={occ}, SR={sr}%, Avg={avg}% ({len(group)} patterns)")
    for i, p in enumerate(group[:3]):
        print(f"  Pattern {i+1}: Conditions={list(p.get('conditions', {}).keys())}")
    if len(group) > 3:
        print(f"  ... and {len(group) - 3} more")