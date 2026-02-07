"""
Analyze pattern performance by market regime.
This script calculates success rates for each pattern in different market regimes.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market_regime_detector import MarketRegimeDetector


def load_ohlcv_data():
    """Load OHLCV data."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'ohlcv.json')
    with open(data_path, 'r') as f:
        return json.load(f)


def load_pattern_occurrences(pattern_id: int) -> List[Dict]:
    """Load occurrence data for a specific pattern."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', f'pattern_{pattern_id}_occurrences.json')
    try:
        with open(data_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def calculate_regime_for_date(ohlcv_df: pd.DataFrame, date_str: str, window: int = 20) -> Dict:
    """Calculate market regime for a specific date."""
    # Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(ohlcv_df['Date']):
        ohlcv_df['Date'] = pd.to_datetime(ohlcv_df['Date'])
    
    # Find the index for the given date
    date = pd.to_datetime(date_str)
    idx = ohlcv_df[ohlcv_df['Date'] <= date].index
    
    if len(idx) == 0:
        return {'trend_regime': 'Unknown', 'volatility_regime': 'Unknown'}
    
    current_idx = idx[-1]
    
    # Need at least window days of data
    if current_idx < window:
        return {'trend_regime': 'Unknown', 'volatility_regime': 'Unknown'}
    
    # Get the window of data up to this point
    window_data = ohlcv_df.iloc[current_idx - window + 1:current_idx + 1].copy()
    
    if len(window_data) < 5:
        return {'trend_regime': 'Unknown', 'volatility_regime': 'Unknown'}
    
    # Calculate ATR%
    window_data['TR'] = np.maximum(
        window_data['High'] - window_data['Low'],
        np.maximum(
            abs(window_data['High'] - window_data['Close'].shift(1)),
            abs(window_data['Low'] - window_data['Close'].shift(1))
        )
    )
    atr_pct = (window_data['TR'].mean() / window_data['Close'].mean()) * 100
    
    # Calculate MA slope (price trend)
    closes = window_data['Close'].values
    if len(closes) >= 5:
        ma_slope = ((closes[-1] - closes[0]) / closes[0]) * 100
    else:
        ma_slope = 0
    
    # Classify trend - widened sideways range for more stable regime detection
    if ma_slope > 0.3:
        trend_regime = 'Strong Bull'
    elif ma_slope > 0.1:
        trend_regime = 'Weak Bull'
    elif ma_slope > -0.1:
        trend_regime = 'Sideways'
    elif ma_slope > -0.3:
        trend_regime = 'Weak Bear'
    else:
        trend_regime = 'Strong Bear'
    
    # Classify volatility
    if atr_pct < 1.5:
        vol_regime = 'Low'
    elif atr_pct < 3.0:
        vol_regime = 'Medium'
    else:
        vol_regime = 'High'
    
    return {
        'trend_regime': trend_regime,
        'volatility_regime': vol_regime,
        'atr_pct': round(atr_pct, 2),
        'ma_slope': round(ma_slope, 2)
    }


def is_successful_outcome(occ: Dict) -> bool:
    """Determine if an occurrence was successful based on various outcome formats."""
    outcome = occ.get('outcome', 'UNKNOWN')
    target_reached = occ.get('target_reached', False)
    
    # Check various success indicators
    if target_reached:
        return True
    if outcome in ['SUCCESS', 'STRONG_UP', 'UP']:
        return True
    if 'UP' in outcome or 'SUCCESS' in outcome:
        return True
    return False


def analyze_pattern_regime_performance(pattern_id: int, ohlcv_df: pd.DataFrame) -> Dict:
    """Analyze a pattern's performance across different market regimes."""
    occurrences = load_pattern_occurrences(pattern_id)
    
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
        
        # Get regime for this date
        regime = calculate_regime_for_date(ohlcv_df, date)
        trend_regime = regime.get('trend_regime', 'Unknown')
        
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


def analyze_all_patterns():
    """Analyze all patterns and save results."""
    print("Loading OHLCV data...")
    ohlcv_data = load_ohlcv_data()
    ohlcv_df = pd.DataFrame(ohlcv_data)
    
    # Convert Date column to datetime once
    ohlcv_df['Date'] = pd.to_datetime(ohlcv_df['Date'])
    
    # Load patterns
    patterns_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'patterns.json')
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
        
        # Analyze regime performance
        regime_performance = analyze_pattern_regime_performance(pattern_id, ohlcv_df)
        
        if regime_performance:
            all_results[pattern_id] = {
                'direction': direction,
                'overall_success_rate': overall_success_rate,
                'regime_performance': regime_performance
            }
        
        if (i + 1) % 10 == 0:
            print(f"  Analyzed {i + 1}/{len(patterns)} patterns...")
    
    # Save results
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'pattern_regime_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to {output_path}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total patterns analyzed: {len(all_results)}")
    
    # Find best performing regimes across all patterns
    regime_totals = {}
    for pattern_id, data in all_results.items():
        for regime, stats in data['regime_performance'].items():
            if regime not in regime_totals:
                regime_totals[regime] = {'total': 0, 'success': 0}
            regime_totals[regime]['total'] += stats['total_occurrences']
            regime_totals[regime]['success'] += stats['successes']
    
    print("\nOverall success rates by regime:")
    for regime, totals in sorted(regime_totals.items(), key=lambda x: x[1]['success'] / x[1]['total'] if x[1]['total'] > 0 else 0, reverse=True):
        if totals['total'] > 0:
            rate = (totals['success'] / totals['total']) * 100
            print(f"  {regime}: {rate:.1f}% ({totals['success']}/{totals['total']})")
    
    return all_results


if __name__ == "__main__":
    analyze_all_patterns()
