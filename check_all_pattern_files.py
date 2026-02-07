#!/usr/bin/env python3
"""Check all pattern files"""
import json
import os

base_dir = os.path.abspath('.')

print("=" * 70)
print("ALL PATTERN FILES CHECK")
print("=" * 70)

for ticker in ['XOM', 'AAPL', 'TSLA']:
    print(f"\nTicker: {ticker}")
    print("-" * 40)
    
    ticker_dir = os.path.join(base_dir, 'data', 'tickers', ticker)
    
    files_to_check = [
        'patterns.json',
        'discovered_patterns.json', 
        'validated_patterns.json',
        'final_portfolio.json'
    ]
    
    for filename in files_to_check:
        filepath = os.path.join(ticker_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        print(f"  {filename}: {len(data)} items (list)")
                    elif isinstance(data, dict):
                        if 'robust' in data:
                            total = len(data.get('robust', [])) + len(data.get('degraded', [])) + len(data.get('failed', []))
                            print(f"  {filename}: robust={len(data.get('robust', []))}, degraded={len(data.get('degraded', []))}, failed={len(data.get('failed', []))}")
                        else:
                            print(f"  {filename}: dict with keys {list(data.keys())[:3]}")
                    else:
                        print(f"  {filename}: {type(data)}")
                except Exception as e:
                    print(f"  {filename}: ERROR - {e}")
        else:
            print(f"  {filename}: NOT FOUND")

print("\n" + "=" * 70)
