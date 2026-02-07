#!/usr/bin/env python3
"""Test dashboard API endpoints"""
import urllib.request
import json

BASE_URL = 'http://localhost:5001'

def test_endpoint(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {'error': str(e)}

print("="*70)
print("DASHBOARD API TEST")
print("="*70)

# Test tickers
print("\n1. Testing /api/tickers")
data = test_endpoint(f'{BASE_URL}/api/tickers')
print(f"   Tickers: {data.get('tickers', [])}")
print(f"   Default: {data.get('default', 'N/A')}")

# Test patterns for each ticker
for ticker in ['XOM', 'AAPL', 'TSLA']:
    print(f"\n2. Testing /api/patterns?ticker={ticker}")
    data = test_endpoint(f'{BASE_URL}/api/patterns?ticker={ticker}')
    if 'error' in data:
        print(f"   Error: {data['error']}")
    else:
        print(f"   Patterns: {len(data)}")
        if data:
            print(f"   First: {data[0].get('method', 'N/A')} ({data[0].get('direction', 'N/A')})")
            print(f"   Success rate: {data[0].get('success_rate', 0):.1f}%")

# Test occurrences
print("\n3. Testing /api/pattern/0/occurrences?ticker=TSLA")
data = test_endpoint(f'{BASE_URL}/api/pattern/0/occurrences?ticker=TSLA')
if 'error' in data:
    print(f"   Error: {data['error']}")
else:
    print(f"   Total occurrences: {data.get('total_occurrences', 0)}")
    print(f"   Success count: {data.get('success_count', 0)}")
    print(f"   Success rate: {data.get('success_rate', 0):.1f}%")

# Test OHLCV
print("\n4. Testing /api/ohlcv?ticker=TSLA")
data = test_endpoint(f'{BASE_URL}/api/ohlcv?ticker=TSLA')
if 'error' in data:
    print(f"   Error: {data['error']}")
else:
    print(f"   Records: {len(data)}")

print("\n" + "="*70)
print("API TEST COMPLETE")
print("="*70)
