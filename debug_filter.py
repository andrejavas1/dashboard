"""Debug script to understand why patterns 166 and 4532 are both being kept."""

import json
import sys
sys.path.insert(0, 'src')
from pattern_filter import PatternFilter

# Load patterns
data = json.load(open('data/patterns.json'))

# Convert to filter format
filter_patterns = []
for i, p in enumerate(data):
    pattern_data = p.get('pattern', {})
    filter_pattern = {
        'conditions': pattern_data.get('conditions', {}),
        'label_col': pattern_data.get('label_col', 'unknown'),
        'success_rate': p.get('training_success_rate', 0),
        'training_success_rate': p.get('training_success_rate', 0),
        'validation_success_rate': p.get('validation_success_rate', 0),
        'direction': p.get('direction', 'unknown'),
        'occurrences': p.get('occurrences', 0),
        '_original_index': i
    }
    filter_patterns.append(filter_pattern)

# Filter
pf = PatternFilter()
filtered = pf.filter_exact_duplicates(filter_patterns)

# Check patterns 166 and 4532
print(f'Total patterns after exact duplicate filter: {len(filtered)}')
print(f'Exact duplicates removed: {pf.filter_stats["exact_duplicates_removed"]}')

# Find which ones were kept
p166_kept = None
p4532_kept = None
for p in filtered:
    if p['_original_index'] == 166:
        p166_kept = p
    elif p['_original_index'] == 4532:
        p4532_kept = p

print(f'\nPattern 166 kept: {p166_kept is not None}')
print(f'Pattern 4532 kept: {p4532_kept is not None}')

if p166_kept:
    print(f'  Pattern 166 validation: {p166_kept["validation_success_rate"]:.2f}%')
if p4532_kept:
    print(f'  Pattern 4532 validation: {p4532_kept["validation_success_rate"]:.2f}%')

# Check their keys
if p166_kept and p4532_kept:
    key166 = pf.get_full_key(p166_kept)
    key4532 = pf.get_full_key(p4532_kept)
    print(f'\nKeys match: {key166 == key4532}')
    print(f'Key 166: {key166}')
    print(f'Key 4532: {key4532}')