"""
Analyze pattern performance by market regime.
This script calculates success rates for each pattern in different market regimes.
Uses MA Alignment with 105-day minimum duration - same as dashboard.
"""

import json
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np


def get_ticker_from_args():
    """Get ticker from command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='XOM', help='Ticker symbol')
    args = parser.parse_args()
    return args.ticker


def load_regime_history(ticker='XOM') -> Dict[str, str]:
    """Load regime history and create date-to-regime lookup."""
    regime_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'tickers', ticker, 'regime_history.json')
    try:
        with open(regime_path, 'r') as f:
            regime_history = json.load(f)
        
        # Create date-to-regime lookup
        # Sort by date to ensure proper ordering
        sorted_regimes = sorted(regime_history, key=lambda x: x['date'])
        
        # Return list for binary search style lookup
        return sorted_regimes
    except FileNotFoundError:
        return []


def get_regime_for_date(date_str: str, regime_history: List[Dict]) -> str:
    """Find the regime active on a given date."""
    current_regime = 'Unknown'
    for r in regime_history:
        if r['date'] <= date_str:
            current_regime = r['trend_regime']
        else:
            break
    return current_regime


def load_pattern_occurrences(pattern_id: int, ticker='XOM') -> List[Dict]:
    """Load occurrence data for a specific pattern."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'tickers', ticker, 'occurrences', f'pattern_{pattern_id}_occurrences.json')
    try:
        with open(data_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def is_successful_outcome(occ: Dict) -> bool:
    """Determine if an occurrence was successful."""
    outcome = occ.get('outcome', 'UNKNOWN')
    target_reached = occ.get('target_reached', False)
    
    if target_reached:
        return True
    if outcome in ['SUCCESS', 'STRONG_UP', 'UP']:
        return True
    if 'UP' in outcome or 'SUCCESS' in outcome:
        return True
    return False


def analyze_pattern_regime_performance(pattern_id: int, regime_history: List[Dict], occurrences: List[Dict]) -> Dict:
    """Analyze a pattern's performance across different market regimes."""
    if not occurrences:
        return {}
    
    # Group occurrences by regime
    regime_stats = {
        'Strong Bull': {'total': 0, 'success': 0},
        'Weak Bull': {'total': 0, 'success': 0},
        'Sideways': {'total': 0, 'success': 0},
        'Weak Bear': {'total': 0, 'success': 0},
        'Strong Bear': {'total': 0, 'success': 0},
        'Unknown': {'total': 0, 'success': 0}
    }
    
    for occ in occurrences:
        date = occ.get('Date', '')
        
        # Get regime for this date from regime_history
        trend_regime = get_regime_for_date(date, regime_history)
        
        # Update stats
        regime_stats[trend_regime]['total'] += 1
        if is_successful_outcome(occ):
            regime_stats[trend_regime]['success'] += 1
    
    # Calculate success rates
    result = {}
    for regime, stats in regime_stats.items():
        if stats['total'] > 0:
            success_rate = (stats['success'] / stats['total']) * 100
            result[regime] = {
                'total_occurrences': stats['total'],
                'successes': stats['success'],
                'success_rate': round(success_rate, 1)
            }
    
    return result


def load_ohlcv_data(ticker='XOM'):
    """Load OHLCV data for specific ticker."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'tickers', ticker, 'ohlcv.json')
    with open(data_path, 'r') as f:
        return json.load(f)


def calculate_historical_regimes(ohlcv_data: List[Dict]) -> List[Dict]:
    """
    Calculate market regimes using MA Alignment with 105-day minimum duration.
    Same method as dashboard for consistency.
    Returns regime history list.
    """
    regimes = []
    maShort = 50
    maLong = 200
    minDuration = 105
    
    currentRegime = None
    regimeStartIdx = 0
    daysInRegime = 0
    pendingRegime = None
    pendingDays = 0
    
    for i in range(maLong, len(ohlcv_data)):
        # Calculate MAs
        prices = [d['Close'] for d in ohlcv_data[:i+1]]
        shortMA = sum(prices[-maShort:]) / maShort
        longMA = sum(prices[-maLong:]) / maLong
        price = ohlcv_data[i]['Close']
        
        # Normalize date
        full_date = ohlcv_data[i]['Date']
        date = full_date.split(' ')[0] if ' ' in full_date else full_date
        
        # MA Alignment classification
        maDiff = ((shortMA - longMA) / longMA) * 100
        
        if shortMA > longMA and price > shortMA and maDiff > 5:
            rawRegime = 'Strong Bull'
        elif shortMA > longMA and price > shortMA:
            rawRegime = 'Weak Bull'
        elif shortMA < longMA and price < shortMA and maDiff < -5:
            rawRegime = 'Strong Bear'
        elif shortMA < longMA and price < shortMA:
            rawRegime = 'Weak Bear'
        else:
            rawRegime = 'Sideways'
        
        if currentRegime is None:
            currentRegime = rawRegime
            regimeStartIdx = i
            daysInRegime = 1
        elif rawRegime == currentRegime:
            daysInRegime += 1
            pendingRegime = None
            pendingDays = 0
        else:
            if pendingRegime is None:
                pendingRegime = rawRegime
                pendingDays = 1
            elif pendingRegime == rawRegime:
                pendingDays += 1
                if pendingDays >= minDuration:
                    # Commit regime change
                    regimes.append({
                        'date': ohlcv_data[regimeStartIdx]['Date'].split(' ')[0] if ' ' in ohlcv_data[regimeStartIdx]['Date'] else ohlcv_data[regimeStartIdx]['Date'],
                        'trend_regime': currentRegime,
                        'volatility_regime': 'Medium',
                        'duration': daysInRegime
                    })
                    currentRegime = pendingRegime
                    regimeStartIdx = i - minDuration + 1
                    daysInRegime = minDuration
                    pendingRegime = None
                    pendingDays = 0
            else:
                pendingRegime = rawRegime
                pendingDays = 1
    
    # Add final regime
    if currentRegime:
        regimes.append({
            'date': ohlcv_data[regimeStartIdx]['Date'].split(' ')[0] if ' ' in ohlcv_data[regimeStartIdx]['Date'] else ohlcv_data[regimeStartIdx]['Date'],
            'trend_regime': currentRegime,
            'volatility_regime': 'Medium',
            'duration': daysInRegime
        })
    
    return regimes


def analyze_all_patterns():
    """Analyze all patterns and save results."""
    # Get ticker from args
    ticker = get_ticker_from_args()
    print(f"Analyzing patterns for ticker: {ticker}")
    
    print("Loading regime history from features matrix...")
    regime_history = load_regime_history(ticker)
    
    if not regime_history:
        print("ERROR: No regime_history.json found. Please generate it first.")
        return
    
    print(f"Loaded {len(regime_history)} regime periods from features matrix")
    
    # Count regimes
    regime_counts = {}
    for r in regime_history:
        regime = r['trend_regime']
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    print(f"Regime distribution: {regime_counts}")
    
    # Load patterns from ticker-specific directory
    patterns_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'tickers', ticker, 'patterns.json')
    with open(patterns_path, 'r') as f:
        patterns = json.load(f)
    
    print(f"Analyzing {len(patterns)} patterns...")
    
    # Analyze each pattern
    all_results = {}
    
    for i, pattern in enumerate(patterns):
        pattern_id = i
        
        # Get pattern info
        if 'pattern' in pattern:
            p = pattern['pattern']
        else:
            p = pattern
        
        direction = p.get('direction', 'N/A')
        overall_success_rate = p.get('success_rate', 0)
        
        # Load occurrences with ticker
        occurrences = load_pattern_occurrences(pattern_id, ticker)
        
        # Analyze regime performance
        regime_performance = analyze_pattern_regime_performance(pattern_id, regime_history, occurrences)
        
        if regime_performance:
            all_results[str(pattern_id)] = {
                'direction': direction,
                'overall_success_rate': overall_success_rate,
                'regime_performance': regime_performance
            }
        
        if (i + 1) % 10 == 0:
            print(f"  Analyzed {i + 1}/{len(patterns)} patterns...")
    
    # Save results to ticker-specific directory
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'tickers', ticker, 'pattern_regime_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to {output_path}")
    print(f"Analyzed {len(all_results)} patterns with regime data")


if __name__ == '__main__':
    analyze_all_patterns()
