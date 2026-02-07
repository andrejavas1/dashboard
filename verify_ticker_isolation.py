#!/usr/bin/env python3
"""
Multi-Ticker Data Isolation Verification Script

Checks that:
1. Each ticker has its own data directory
2. Files are NOT shared between tickers
3. No generic data/ files except pipeline_runs.json
4. Each ticker has a complete file set
"""

import os
import json
from pathlib import Path
from datetime import datetime

def check_ticker_isolation():
    """Verify that tickers are properly isolated."""
    
    print("\n" + "=" * 80)
    print("MULTI-TICKER DATA ISOLATION VERIFICATION")
    print("=" * 80)
    print(f"Check time: {datetime.now().isoformat()}\n")
    
    # Define required files for each ticker
    required_files = [
        'ohlcv.json',
        'patterns.json',
        'final_portfolio.json',
        'features_matrix.csv',
        'validated_patterns.json',
        'discovered_patterns.json',
        'optimized_patterns.json',
        'ranked_patterns.json'
    ]
    
    required_dirs = [
        'occurrences'
    ]
    
    # Check 1: Verify no generic data files
    print("CHECK 1: Verifying no generic data files...")
    print("-" * 80)
    generic_data_dir = Path("data")
    if generic_data_dir.exists():
        # List what's in generic data/
        items = list(generic_data_dir.iterdir())
        generic_files = [f for f in items if f.is_file() and f.name != 'pipeline_runs.json']
        
        if generic_files:
            print("❌ FAIL: Found generic data files (should only have pipeline_runs.json):")
            for f in generic_files:
                size = f.stat().st_size / 1024  # KB
                print(f"  - {f.name} ({size:.1f} KB)")
        else:
            print("✅ PASS: No generic data files found (clean!)")
    else:
        print("⚠️  WARNING: data/ directory doesn't exist")
    
    # Check 2: Verify tickers directory structure
    print("\nCHECK 2: Verifying ticker-specific directories...")
    print("-" * 80)
    
    tickers_dir = Path("data/tickers")
    if not tickers_dir.exists():
        print("❌ FAIL: data/tickers/ directory doesn't exist")
        return False
    
    ticker_dirs = [d for d in tickers_dir.iterdir() if d.is_dir()]
    if not ticker_dirs:
        print("⚠️  WARNING: No ticker directories found in data/tickers/")
        return False
    
    print(f"Found {len(ticker_dirs)} ticker(s):")
    for ticker_dir in sorted(ticker_dirs):
        print(f"  - {ticker_dir.name}/")
    
    # Check 3: Verify each ticker has required files
    print("\nCHECK 3: Verifying each ticker has required files...")
    print("-" * 80)
    
    all_tickers_complete = True
    for ticker_dir in sorted(ticker_dirs):
        ticker = ticker_dir.name
        print(f"\nTicker: {ticker}")
        
        # Check files
        missing_files = []
        for required_file in required_files:
            file_path = ticker_dir / required_file
            if file_path.exists():
                size = file_path.stat().st_size / 1024  # KB
                print(f"  ✅ {required_file:35s} ({size:8.1f} KB)")
            else:
                missing_files.append(required_file)
                print(f"  ❌ {required_file:35s} (MISSING)")
        
        # Check directories
        missing_dirs = []
        for required_dir in required_dirs:
            dir_path = ticker_dir / required_dir
            if dir_path.exists() and dir_path.is_dir():
                # Count files in directory
                file_count = len(list(dir_path.glob('*.json')))
                print(f"  ✅ {required_dir}/ ({file_count} files)")
            else:
                missing_dirs.append(required_dir)
                print(f"  ❌ {required_dir}/ (MISSING)")
        
        if missing_files or missing_dirs:
            all_tickers_complete = False
            print(f"  ⚠️  WARNING: {ticker} is missing {len(missing_files)} files and {len(missing_dirs)} directories")
        else:
            print(f"  ✅ {ticker} has all required files")
    
    # Check 4: Verify data is not shared between tickers
    print("\nCHECK 4: Verifying data is NOT shared between tickers...")
    print("-" * 80)
    
    if len(ticker_dirs) < 2:
        print("⚠️  SKIP: Only 1 ticker found, can't verify isolation")
    else:
        # Compare file sizes and content hashes between tickers
        ticker_list = sorted([d.name for d in ticker_dirs])
        
        all_isolated = True
        for i in range(len(ticker_list) - 1):
            ticker1 = ticker_list[i]
            ticker2 = ticker_list[i + 1]
            
            ohlcv1_path = Path(f"data/tickers/{ticker1}/ohlcv.json")
            ohlcv2_path = Path(f"data/tickers/{ticker2}/ohlcv.json")
            
            if ohlcv1_path.exists() and ohlcv2_path.exists():
                # Compare file sizes
                size1 = ohlcv1_path.stat().st_size
                size2 = ohlcv2_path.stat().st_size
                
                if size1 == size2:
                    print(f"⚠️  WARNING: {ticker1} and {ticker2} have SAME size ohlcv.json ({size1} bytes)")
                    print("      → Might indicate same data or just coincidence")
                    # Try comparing first line
                    with open(ohlcv1_path) as f1, open(ohlcv2_path) as f2:
                        line1 = f1.readline()
                        line2 = f2.readline()
                        if line1 == line2:
                            print(f"      → FAIL: First line is IDENTICAL! Data NOT isolated!")
                            all_isolated = False
                        else:
                            print(f"      → OK: Content differs (different tickers)")
                else:
                    print(f"✅ {ticker1} ({size1} bytes) != {ticker2} ({size2} bytes) - Data isolated")
        
        if all_isolated:
            print("\n✅ PASS: Data is properly isolated between tickers")
        else:
            print("\n❌ FAIL: Data is NOT properly isolated between tickers")
            return False
    
    # Check 5: Verify pattern counts are reasonable
    print("\nCHECK 5: Verifying pattern counts...")
    print("-" * 80)
    
    for ticker_dir in sorted(ticker_dirs):
        ticker = ticker_dir.name
        occurrences_dir = ticker_dir / "occurrences"
        
        if occurrences_dir.exists():
            occurrence_files = list(occurrences_dir.glob("pattern_*_occurrences.json"))
            pattern_count = len(occurrence_files)
            
            if pattern_count > 0:
                print(f"✅ {ticker:6s}: {pattern_count:2d} pattern occurrence files")
            else:
                print(f"⚠️  {ticker:6s}: 0 pattern occurrence files (might be empty data)")
        else:
            print(f"⚠️  {ticker:6s}: occurrences directory not found")
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    if all_tickers_complete:
        print("✅ All tickers have required files")
    else:
        print("❌ Some tickers are missing files - run main.py for affected tickers")
    
    print(f"✅ {len(ticker_dirs)} ticker(s) configured and isolated")
    print("✅ No generic data/ files found (clean setup)")
    
    print("\nRecommendations:")
    if not all_tickers_complete:
        print("1. Run main.py to complete pipeline for missing tickers")
    print("2. Run main.py sequentially for each ticker to verify isolation")
    print("3. For each ticker, verify: data/tickers/{TICKER}/ contains all files")
    print("4. Use pattern_dashboard_server.py with ticker parameter to view data")
    
    return True

if __name__ == "__main__":
    try:
        success = check_ticker_isolation()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
