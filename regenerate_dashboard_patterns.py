#!/usr/bin/env python3
"""Regenerate dashboard patterns.json for all tickers with correct filters"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.phase7_portfolio_construction import PortfolioConstruction
import json

tickers = ['XOM', 'AAPL', 'TSLA']

print("=" * 70)
print("REGENERATING DASHBOARD PATTERNS")
print("=" * 70)

for ticker in tickers:
    print(f"\nProcessing {ticker}...")
    
    # Create portfolio constructor (will read run_mode from config)
    pc = PortfolioConstruction()
    
    # Load validated patterns from ticker directory
    validated_path = f'data/tickers/{ticker}/validated_patterns.json'
    if os.path.exists(validated_path):
        with open(validated_path, 'r') as f:
            data = json.load(f)
        
        # Combine all patterns
        if isinstance(data, dict):
            all_patterns = []
            for classification in ['robust', 'degraded', 'failed']:
                for p in data.get(classification, []):
                    p['classification'] = classification.upper()
                    all_patterns.append(p)
            pc.patterns = all_patterns
            print(f"  Loaded {len(all_patterns)} patterns from {validated_path}")
        else:
            pc.patterns = data
            print(f"  Loaded {len(data)} patterns")
        
        # Save dashboard patterns with correct filters
        ticker_dir = f'data/tickers/{ticker}'
        pc.save_dashboard_patterns(ticker_dir)
        print(f"  Saved patterns.json to {ticker_dir}")
    else:
        print(f"  ERROR: {validated_path} not found")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
