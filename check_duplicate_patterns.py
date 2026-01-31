import json

# Load patterns.json
with open('data/patterns.json', 'r') as f:
    patterns = json.load(f)

# Search for patterns with the specific characteristics
matching_patterns = []

for idx, pattern_data in enumerate(patterns):
    pattern = pattern_data.get('pattern', {})
    
    # Check for matching characteristics
    occurrences = pattern.get('occurrences')
    success_rate = pattern.get('success_rate')
    method = pattern.get('method')
    direction = pattern.get('direction')
    conditions = pattern.get('conditions', {})
    
    # Check if it matches the user's description
    if (occurrences == 72 and 
        success_rate == 88.88888888888889 and
        method == 'rule_based' and
        direction == 'long'):
        
        # Check for AD_ROC_20d and Dist_50d_Low conditions
        ad_roc = conditions.get('AD_ROC_20d', {})
        dist_50d = conditions.get('Dist_50d_Low', {})
        
        if (ad_roc.get('operator') == '<=' and 
            abs(ad_roc.get('value', 0) - (-3.71)) < 0.1 and
            dist_50d.get('operator') == '>=' and
            abs(dist_50d.get('value', 0) - 12.27) < 0.1):
            
            matching_patterns.append({
                'index': idx,
                'pattern': pattern,
                'validation': pattern_data.get('validation', {}),
                'statistically_significant': pattern_data.get('statistically_significant'),
                'classification': pattern_data.get('classification')
            })

print(f"Found {len(matching_patterns)} matching patterns:")
print("=" * 80)

for mp in matching_patterns:
    print(f"\nPattern Index: {mp['index']}")
    print(f"Conditions:")
    for cond_name, cond_data in mp['pattern']['conditions'].items():
        print(f"  {cond_name}: {cond_data['operator']} {cond_data['value']}")
    print(f"Occurrences: {mp['pattern']['occurrences']}")
    print(f"Success Rate: {mp['pattern']['success_rate']:.1f}%")
    print(f"Method: {mp['pattern']['method']}")
    print(f"Direction: {mp['pattern']['direction']}")
    
    # Check validation metrics
    validation = mp.get('validation', {})
    if validation:
        print(f"\nValidation Metrics:")
        print(f"  Training Success Rate: {validation.get('training_success_rate', 'N/A'):.1f}%")
        print(f"  Validation Success Rate: {validation.get('validation_success_rate', 'N/A'):.1f}%")
        print(f"  Validation Occurrences: {validation.get('validation_occurrences', 'N/A')}")
        print(f"  Statistically Significant: {mp.get('statistically_significant', 'N/A')}")
        print(f"  Classification: {mp.get('classification', 'N/A')}")
    else:
        print(f"\nStatistically Significant: {mp.get('statistically_significant', 'N/A')}")
        print(f"Classification: {mp.get('classification', 'N/A')}")
    
    print("-" * 80)

# Check if they are duplicates
if len(matching_patterns) >= 2:
    print("\n" + "=" * 80)
    print("DUPLICATE ANALYSIS:")
    print("=" * 80)
    
    # Compare first two patterns
    p1 = matching_patterns[0]
    p2 = matching_patterns[1]
    
    conditions_match = p1['pattern']['conditions'] == p2['pattern']['conditions']
    metrics_match = (p1['pattern']['occurrences'] == p2['pattern']['occurrences'] and
                     p1['pattern']['success_rate'] == p2['pattern']['success_rate'])
    
    if conditions_match and metrics_match:
        print("✓ These patterns appear to be DUPLICATES - identical conditions and metrics")
    else:
        print("✗ These patterns are NOT duplicates - they differ in conditions or metrics")
    
    # Check filter status
    print("\nFILTER STATUS:")
    for mp in matching_patterns:
        index = mp['index']
        sig = mp.get('statistically_significant')
        classification = mp.get('classification')
        
        # Determine if passed filter
        passed = (sig == 'True' or sig is True) and classification == 'ROBUST'
        
        print(f"Pattern #{index}: {'PASSED' if passed else 'FAILED'} filter")
        print(f"  - Statistically Significant: {sig}")
        print(f"  - Classification: {classification}")