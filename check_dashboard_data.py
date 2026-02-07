#!/usr/bin/env python3
"""Check dashboard data files"""
import json
import os

base_dir = os.path.abspath('.')

print("=" * 70)
print("DASHBOARD DATA CHECK")
print("=" * 70)

for ticker in ['XOM', 'AAPL', 'TSLA']:
    print(f"\nTicker: {ticker}")
    print("-" * 40)
    
    # Check patterns.json
    patterns_path = os.path.join(base_dir, 'data', 'tickers', ticker, 'patterns.json')
    print(f"  patterns.json: exists={os.path.exists(patterns_path)}")
    if os.path.exists(patterns_path):
        with open(patterns_path, 'r') as f:
            data = json.load(f)
        print(f"    Type: {type(data).__name__}, Len: {len(data)}")
    
    # Check ohlcv.json
    ohlcv_path = os.path.join(base_dir, 'data', 'tickers', ticker, 'ohlcv.json')
    print(f"  ohlcv.json: exists={os.path.exists(ohlcv_path)}")
    if os.path.exists(ohlcv_path):
        with open(ohlcv_path, 'r') as f:
            data = json.load(f)
        print(f"    Records: {len(data)}")
    
    # Check occurrences directory
    occ_dir = os.path.join(base_dir, 'data', 'tickers', ticker, 'occurrences')
    if os.path.exists(occ_dir):
        occ_files = [f for f in os.listdir(occ_dir) if f.endswith('.json')]
        print(f"  occurrences: {len(occ_files)} files")

print("\n" + "=" * 70)
