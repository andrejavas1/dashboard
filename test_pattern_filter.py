"""
Test script to demonstrate pattern filtering on existing patterns.json
"""

import json
import sys
sys.path.insert(0, 'src')
from pattern_filter import PatternFilter


def main():
    print("=" * 80)
    print("PATTERN FILTERING TEST")
    print("=" * 80)

    # Load existing patterns
    print("\nLoading patterns from data/patterns.json...")
    with open('data/patterns.json', 'r') as f:
        patterns = json.load(f)

    print(f"Total patterns loaded: {len(patterns)}")

    # Show examples of duplicates mentioned by user
    print("\n" + "=" * 80)
    print("EXAMPLE DUPLICATES (patterns 166, 4531, 4532):")
    print("=" * 80)
    for i in [166, 4531, 4532]:
        if i < len(patterns):
            p = patterns[i]
            print(f"\nPattern {i}:")
            print(f"  Conditions: {p.get('pattern', {}).get('conditions', {})}")
            print(f"  Training Success Rate: {p.get('training_success_rate', 'N/A')}")
            print(f"  Validation Success Rate: {p.get('validation_success_rate', 'N/A')}")
            print(f"  Direction: {p.get('direction', 'N/A')}")

    # Create filter and apply
    print("\n" + "=" * 80)
    print("APPLYING PATTERN FILTER")
    print("=" * 80)

    # Convert patterns to the format expected by the filter
    # The patterns.json has nested structure with 'pattern' key
    filter_patterns = []
    for i, p in enumerate(patterns):
        pattern_data = p.get('pattern', {})
        filter_pattern = {
            'conditions': pattern_data.get('conditions', {}),
            'label_col': pattern_data.get('label_col', 'unknown'),
            'success_rate': p.get('training_success_rate', 0),
            'training_success_rate': p.get('training_success_rate', 0),
            'validation_success_rate': p.get('validation_success_rate', 0),
            'direction': p.get('direction', 'unknown'),
            'occurrences': p.get('occurrences', 0)
        }
        filter_patterns.append(filter_pattern)

    # Apply filter
    pattern_filter = PatternFilter()
    filtered_patterns = pattern_filter.filter_patterns(
        filter_patterns,
        remove_exact_duplicates=True,
        remove_condition_duplicates=True,
        remove_subsets=True
    )

    # Get stats
    stats = pattern_filter.get_filter_stats()

    print("\n" + "=" * 80)
    print("FILTERING RESULTS")
    print("=" * 80)
    print(f"Input patterns: {stats['total_input']}")
    print(f"Exact duplicates removed: {stats['exact_duplicates_removed']}")
    print(f"Condition duplicates removed: {stats['condition_duplicates_removed']}")
    print(f"Subset patterns removed: {stats['subset_patterns_removed']}")
    print(f"Output patterns: {stats['total_output']}")
    print(f"Reduction: {(1 - stats['total_output'] / stats['total_input']) * 100:.1f}%")

    # Show what happened to the example duplicates
    print("\n" + "=" * 80)
    print("WHAT HAPPENED TO EXAMPLE DUPLICATES:")
    print("=" * 80)

    # Track which patterns are still in filtered set by their full key
    example_indices = [166, 4531, 4532]
    example_keys = []
    
    for idx in example_indices:
        p = patterns[idx]
        pattern_data = p.get('pattern', {})
        full_key = pattern_filter.get_full_key({
            'conditions': pattern_data.get('conditions', {}),
            'label_col': pattern_data.get('label_col', 'unknown'),
            'direction': p.get('direction', 'unknown')
        })
        example_keys.append((idx, full_key, p.get('training_success_rate', 0), p.get('validation_success_rate', 0)))

    # Build set of filtered keys
    filtered_keys = set()
    for fp in filtered_patterns:
        filtered_keys.add(pattern_filter.get_full_key(fp))

    for idx, key, train_rate, val_rate in example_keys:
        if key in filtered_keys:
            print(f"\nPattern {idx}: KEPT (training: {train_rate:.2f}%, validation: {val_rate:.2f}%)")
        else:
            print(f"\nPattern {idx}: REMOVED (was a duplicate or subset)")

    print("\n" + "=" * 80)
    print("FILTERING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()