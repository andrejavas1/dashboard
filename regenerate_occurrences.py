"""Regenerate pattern occurrence files from patterns.json"""
import json
import pandas as pd
import sys
import argparse
import os
sys.path.insert(0, 'src')
from realtime_feature_calculator import RealtimeFeatureCalculator

def load_data(ticker='XOM'):
    base_path = f'data/tickers/{ticker}'
    with open(f'{base_path}/ohlcv.json', 'r') as f:
        ohlcv_data = json.load(f)
    with open(f'{base_path}/patterns.json', 'r') as f:
        patterns = json.load(f)
    return ohlcv_data, patterns, base_path

def match_conditions(row, conditions):
    for feature, condition in conditions.items():
        if feature not in row:
            return False
        value = row[feature]
        op = condition.get('operator', '~')
        target = condition['value']
        if op == '>=':
            if value < target: return False
        elif op == '<=':
            if value > target: return False
        elif op == '>':
            if value <= target: return False
        elif op == '<':
            if value >= target: return False
    return True

def regenerate(ticker='XOM'):
    print(f"Loading data for {ticker}...")
    ohlcv_data, patterns, base_path = load_data(ticker)
    df = pd.DataFrame(ohlcv_data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    print("Calculating features...")
    calc = RealtimeFeatureCalculator()
    features = calc.calculate(df)
    merged = pd.concat([df.set_index('Date'), features], axis=1).reset_index()
    merged['Date_str'] = merged['Date'].dt.strftime('%Y-%m-%d')
    
    # Create occurrences directory if needed
    occ_dir = f'{base_path}/occurrences'
    os.makedirs(occ_dir, exist_ok=True)
    
    print(f"Processing {len(patterns)} patterns...")
    for i, pattern in enumerate(patterns):
        conditions = pattern.get('conditions', {})
        direction = pattern.get('direction', 'long')
        
        occurrences = []
        for idx, row in merged.iterrows():
            if match_conditions(row, conditions):
                entry_price = row['Close']
                exit_idx = min(idx + 20, len(merged) - 1)
                exit_price = merged.iloc[exit_idx]['Close']
                actual_move = ((exit_price - entry_price) / entry_price) * 100
                target_reached = actual_move >= 3.0 if direction == 'long' else actual_move <= -3.0
                
                occurrences.append({
                    'Date': row['Date_str'],
                    'Open': row['Open'],
                    'High': row['High'],
                    'Low': row['Low'],
                    'Close': entry_price,
                    'Volume': row['Volume'],
                    'outcome': 'STRONG_UP' if target_reached else 'UP',
                    'actual_move': actual_move,
                    'time_to_target': 20,
                    'target_reached': target_reached
                })
        
        # Save to ticker-specific occurrences directory
        with open(f'{occ_dir}/pattern_{i}_occurrences.json', 'w') as f:
            json.dump(occurrences, f, indent=2)
        
        pattern['occurrences'] = len(occurrences)
        pattern['success_count'] = sum(1 for o in occurrences if o['target_reached'])
        pattern['success_rate'] = (pattern['success_count'] / len(occurrences) * 100) if occurrences else 0
        
        if i < 5:
            print(f"  Pattern #{i}: {len(occurrences)} occurrences")
    
    # Update patterns.json with new occurrence counts
    with open(f'{base_path}/patterns.json', 'w') as f:
        json.dump(patterns, f, indent=2)
    
    print(f"\n✓ Done! Regenerated {len(patterns)} occurrence files in {occ_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Regenerate pattern occurrences')
    parser.add_argument('--ticker', type=str, default='XOM', help='Ticker symbol (default: XOM)')
    args = parser.parse_args()
    regenerate(args.ticker)
