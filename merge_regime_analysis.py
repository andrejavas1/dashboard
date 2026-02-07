"""Merge regime analysis into patterns for each ticker."""
import json
from pathlib import Path

def merge_regime_for_ticker(ticker):
    """Merge regime analysis into patterns for a specific ticker."""
    ticker_dir = Path(f'data/tickers/{ticker}')
    
    # Load patterns
    with open(ticker_dir / 'patterns.json', 'r') as f:
        patterns = json.load(f)
    
    # Load regime analysis (from main data dir - generated before multi-ticker)
    try:
        with open('data/pattern_regime_analysis.json', 'r') as f:
            regime_data = json.load(f)
    except FileNotFoundError:
        print(f"No regime analysis found for {ticker}")
        return
    
    # Merge regime analysis into each pattern
    merged_count = 0
    for i, pattern in enumerate(patterns):
        pattern_id = str(i)
        if pattern_id in regime_data:
            pattern['regime_analysis'] = regime_data[pattern_id]
            merged_count += 1
    
    # Save updated patterns
    with open(ticker_dir / 'patterns.json', 'w') as f:
        json.dump(patterns, f, indent=2)
    
    print(f"Merged regime analysis for {merged_count}/{len(patterns)} patterns in {ticker}")

if __name__ == '__main__':
    for ticker in ['XOM', 'AAPL']:
        merge_regime_for_ticker(ticker)
