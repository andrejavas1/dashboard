"""
Pattern Occurrence Validation Test - Run after pipeline completes.

This script validates that:
1. All occurrences in occurrence files match their pattern conditions
2. Occurrence counts in files match patterns.json
3. Regime-specific occurrence counts are consistent
4. No invalid occurrences exist

Usage: python test_pattern_occurrences.py
"""
import json
import pandas as pd
import os
import sys
from datetime import datetime

def load_patterns():
    """Load all patterns from patterns.json."""
    with open('data/patterns.json', 'r') as f:
        return json.load(f)

def load_occurrences(pattern_id):
    """Load occurrence file for a pattern."""
    filepath = f'data/pattern_{pattern_id}_occurrences.json'
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return []

def load_regime_analysis():
    """Load regime analysis data."""
    try:
        with open('data/pattern_regime_analysis.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def load_features_matrix():
    """Load features matrix."""
    return pd.read_csv('data/features_matrix.csv', index_col='Date', parse_dates=True)

def check_conditions(row, conditions):
    """
    Check if a row matches all pattern conditions.
    Returns (matches: bool, details: dict)
    """
    details = {}
    all_match = True
    
    for feature, condition in conditions.items():
        if feature not in row:
            details[feature] = {'status': 'MISSING', 'value': None, 'expected': condition}
            all_match = False
            continue
        
        # Handle Series vs scalar
        value = row[feature]
        if hasattr(value, '__len__') and not isinstance(value, str):
            try:
                value = float(value.iloc[0]) if hasattr(value, 'iloc') else float(value)
            except:
                details[feature] = {'status': 'ERROR', 'value': str(value), 'expected': condition}
                all_match = False
                continue
        
        if pd.isna(value):
            details[feature] = {'status': 'NAN', 'value': None, 'expected': condition}
            all_match = False
            continue
        
        op = condition.get('operator', '~')
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
        
        status = 'PASS' if matches else 'FAIL'
        details[feature] = {
            'status': status,
            'value': value,
            'operator': op,
            'target': target
        }
        
        if not matches:
            all_match = False
    
    return all_match, details

def test_pattern(pattern_id, pattern, df, regime_data, verbose=False):
    """Test a single pattern's occurrences."""
    conditions = pattern.get('conditions', {})
    direction = pattern.get('direction', 'long')
    pattern_occurrences = pattern.get('occurrences', 0)
    
    # Load actual occurrences
    actual_occurrences = load_occurrences(pattern_id)
    actual_count = len(actual_occurrences)
    
    # Validate each actual occurrence
    invalid_occurrences = []
    for occ in actual_occurrences:
        date = occ['Date']
        if date in df.index:
            row = df.loc[date]
            matches, details = check_conditions(row, conditions)
            if not matches:
                invalid_occurrences.append({
                    'date': date,
                    'details': details
                })
    
    # Check if count matches patterns.json
    count_mismatch = actual_count != pattern_occurrences
    
    # Check regime consistency if data available
    regime_mismatch = False
    regime_info = ""
    if regime_data and str(pattern_id) in regime_data:
        regime_info_data = regime_data[str(pattern_id)]
        if 'regimes' in regime_info_data:
            # Calculate total from regime breakdown
            total_in_regimes = 0
            for regime_name, regime_stats in regime_info_data['regimes'].items():
                if isinstance(regime_stats, dict) and 'total' in regime_stats:
                    total_in_regimes += regime_stats['total']
            
            if total_in_regimes != actual_count:
                regime_mismatch = True
                regime_info = f"regime total ({total_in_regimes}) != file count ({actual_count})"
    
    result = {
        'pattern_id': pattern_id,
        'direction': direction,
        'conditions': conditions,
        'expected_count': pattern_occurrences,
        'actual_count': actual_count,
        'count_mismatch': count_mismatch,
        'regime_mismatch': regime_mismatch,
        'regime_info': regime_info,
        'invalid_count': len(invalid_occurrences),
        'invalid_occurrences': invalid_occurrences
    }
    
    return result

def print_test_result(result, verbose=False):
    """Print test result for a pattern."""
    pid = result['pattern_id']
    is_ok = not result['count_mismatch'] and not result['regime_mismatch'] and result['invalid_count'] == 0
    status = "[OK]" if is_ok else "[FAIL]"
    
    print(f"\n{status} Pattern #{pid} ({result['direction']})")
    print(f"  Expected: {result['expected_count']} occurrences")
    print(f"  Actual: {result['actual_count']} occurrences")
    
    if result['count_mismatch']:
        print(f"  [!] COUNT MISMATCH: patterns.json shows {result['expected_count']}, file has {result['actual_count']}")
    
    if result['regime_mismatch']:
        print(f"  [!] REGIME MISMATCH: {result['regime_info']}")
    
    if result['invalid_count'] > 0:
        print(f"  [!] INVALID: {result['invalid_count']} occurrences fail conditions")
        if verbose:
            for inv in result['invalid_occurrences'][:3]:
                print(f"    Date {inv['date']}:")
                for feat, det in inv['details'].items():
                    if det['status'] != 'PASS':
                        print(f"      {feat}: {det['status']} (value={det.get('value', 'N/A')}, target={det.get('target', 'N/A')})")

def main():
    """Run pattern occurrence validation."""
    print("=" * 80)
    print("PATTERN OCCURRENCE VALIDATION TEST")
    print("=" * 80)
    print(f"Started: {datetime.now()}")
    print()
    
    # Load data
    print("Loading patterns...")
    patterns = load_patterns()
    print(f"  Loaded {len(patterns)} patterns")
    
    print("Loading features matrix...")
    df = load_features_matrix()
    print(f"  Loaded {len(df)} records from {df.index[0]} to {df.index[-1]}")
    
    print("Loading regime analysis...")
    regime_data = load_regime_analysis()
    if regime_data:
        print(f"  Loaded regime data for {len(regime_data)} patterns")
    else:
        print("  No regime analysis data found")
    print()
    
    # Test parameters
    verbose = False  # Set to True for detailed output
    
    # Run tests
    print("Testing patterns...")
    results = []
    patterns_to_test = patterns
    
    for i, pattern in enumerate(patterns_to_test):
        result = test_pattern(i, pattern, df, regime_data, verbose)
        results.append(result)
        print_test_result(result, verbose)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_patterns = len(results)
    perfect_patterns = sum(1 for r in results if not r['count_mismatch'] and not r['regime_mismatch'] and r['invalid_count'] == 0)
    patterns_with_mismatch = sum(1 for r in results if r['count_mismatch'])
    patterns_with_regime_mismatch = sum(1 for r in results if r['regime_mismatch'])
    patterns_with_invalid = sum(1 for r in results if r['invalid_count'] > 0)
    
    total_expected = sum(r['expected_count'] for r in results)
    total_actual = sum(r['actual_count'] for r in results)
    total_invalid = sum(r['invalid_count'] for r in results)
    
    print(f"Patterns tested: {total_patterns}")
    print(f"Perfect patterns: {perfect_patterns} ({100*perfect_patterns/total_patterns:.1f}%)")
    print(f"Patterns with count mismatch: {patterns_with_mismatch}")
    print(f"Patterns with regime mismatch: {patterns_with_regime_mismatch}")
    print(f"Patterns with invalid occurrences: {patterns_with_invalid}")
    print()
    print(f"Total expected occurrences: {total_expected}")
    print(f"Total actual occurrences: {total_actual}")
    print(f"Total invalid: {total_invalid}")
    
    # Return exit code based on results
    has_errors = total_invalid > 0 or patterns_with_mismatch > 0 or patterns_with_regime_mismatch > 0
    exit_code = 0 if not has_errors else 1
    
    if exit_code == 0:
        print("\n[PASS] ALL TESTS PASSED")
    else:
        print("\n[FAIL] SOME TESTS FAILED")
        print("\nThis test verifies that:")
        print("  1. All occurrences in files match pattern conditions")
        print("  2. Occurrence counts match between files and patterns.json")
        print("  3. Regime breakdowns are consistent with occurrence counts")
    
    # Additional check: Verify patterns have IDs for dashboard API
    print("\nChecking pattern IDs for dashboard compatibility...")
    with open('data/patterns.json', 'r') as f:
        patterns_data = json.load(f)
    
    missing_ids = []
    for i, p in enumerate(patterns_data):
        if 'id' not in p or p.get('id') is None:
            missing_ids.append(i)
    
    if missing_ids:
        print(f"  [WARNING] {len(missing_ids)} patterns missing 'id' field")
        print(f"    Patterns without IDs: {missing_ids[:5]}...")
        print("  [INFO] Dashboard server will assign IDs by array index")
    else:
        print("  [OK] All patterns have IDs assigned")
    
    print("\n" + "=" * 80)
    print(f"Completed: {datetime.now()}")
    print("=" * 80)
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
