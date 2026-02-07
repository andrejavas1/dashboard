"""Check pattern data for inconsistencies."""
import json
from pathlib import Path

def check_patterns(ticker=None):
    """Check pattern data for a ticker."""
    if ticker:
        print(f"\n{'='*60}")
        print(f"Checking {ticker} patterns")
        print(f"{'='*60}")
        ticker_dir = Path(f'data/tickers/{ticker}')
        patterns_file = ticker_dir / 'patterns.json'
    else:
        print(f"\n{'='*60}")
        print(f"Checking main data patterns")
        print(f"{'='*60}")
        patterns_file = Path('data/patterns.json')
    
    # Load patterns
    with open(patterns_file) as f:
        patterns = json.load(f)
    
    issues = []
    for i, p in enumerate(patterns):
        occ = p.get('occurrences', 0)
        succ = p.get('success_count', 0)
        
        # Convert to int if string
        if isinstance(occ, str):
            occ = int(occ)
        if isinstance(succ, str):
            succ = int(succ)
        
        if succ > occ:
            issues.append({
                'id': i,
                'occurrences': occ,
                'success_count': succ,
                'success_rate': p.get('success_rate', 0)
            })
    
    if issues:
        print(f"\n[ERROR] Found {len(issues)} patterns with success_count > occurrences:")
        for issue in issues[:10]:
            print(f"  Pattern {issue['id']}: {issue['success_count']} successes / {issue['occurrences']} occurrences ({issue['success_rate']:.1f}%)")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print(f"\n[OK] All {len(patterns)} patterns have valid data")
    
    return issues

if __name__ == '__main__':
    # Check main data patterns (from main.py run)
    check_patterns(None)
