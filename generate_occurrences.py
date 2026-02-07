"""Generate pattern occurrence files from patterns.json and features"""
import json
import pandas as pd
import numpy as np
import os
import sys

def load_data(ticker=None):
    if ticker:
        base_path = f'data/tickers/{ticker}'
    else:
        base_path = 'data'
    
    with open(f'{base_path}/ohlcv.json', 'r') as f:
        ohlcv_data = json.load(f)
    with open(f'{base_path}/patterns.json', 'r') as f:
        patterns = json.load(f)
    return ohlcv_data, patterns, base_path

def calculate_features(ohlcv_data):
    """Calculate basic features for pattern matching."""
    df = pd.DataFrame(ohlcv_data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Basic technical indicators
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['RSI_14'] = calculate_rsi(df['Close'], 14)
    df['ATR_14'] = calculate_atr(df, 14)
    df['ATR_14_pct'] = (df['ATR_14'] / df['Close']) * 100
    df['CCI_20'] = calculate_cci(df, 20)
    df['Dist_MA_50'] = ((df['Close'] - df['SMA_50']) / df['SMA_50']) * 100
    df['Dist_MA_200'] = ((df['Close'] - df['SMA_200']) / df['SMA_200']) * 100
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    df['Daily_Return'] = df['Close'].pct_change() * 100
    
    # 52-week high/low
    df['52w_High'] = df['Close'].rolling(window=252).max()
    df['52w_Low'] = df['Close'].rolling(window=252).min()
    df['52w_Range_Pct'] = ((df['52w_High'] - df['52w_Low']) / df['52w_Low']) * 100
    df['Days_Since_52w_Low'] = df['Close'].rolling(window=252).apply(lambda x: len(x) - x.argmin() if len(x) > 0 else 0, raw=True)
    
    return df

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()

def calculate_cci(df, period=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    tp_ma = tp.rolling(window=period).mean()
    tp_std = tp.rolling(window=period).std()
    return (tp - tp_ma) / (0.015 * tp_std)

def match_conditions(row, conditions):
    for feature, condition in conditions.items():
        if feature not in row or pd.isna(row[feature]):
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

def generate(ticker=None):
    print("Loading data...")
    ohlcv_data, patterns, base_path = load_data(ticker)
    
    if ticker:
        print(f"  Using ticker: {ticker}")
    
    print("Calculating features...")
    df = calculate_features(ohlcv_data)
    df = df.reset_index(drop=True)
    df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    print(f"Processing {len(patterns)} patterns...")
    
    for i, pattern in enumerate(patterns):
        conditions = pattern.get('conditions', {})
        direction = pattern.get('direction', 'long')
        label_col = pattern.get('label_col', 'Label_3pct_20d')
        
        # Parse target from label_col
        import re
        match = re.search(r'Label_(\d+(?:\.\d+)?)pct_(\d+)d', label_col)
        if match:
            target_pct = float(match.group(1))
            time_window = int(match.group(2))
        else:
            target_pct = 3.0
            time_window = 20
        
        occurrences = []
        for idx, row in df.iterrows():
            if match_conditions(row, conditions):
                entry_price = row['Close']
                entry_date = row['Date_str']
                
                # Calculate exit
                exit_idx = min(idx + time_window, len(df) - 1)
                exit_price = df.iloc[exit_idx]['Close']
                actual_move = ((exit_price - entry_price) / entry_price) * 100
                
                if direction == 'long':
                    target_reached = actual_move >= target_pct
                else:
                    target_reached = actual_move <= -target_pct
                
                occurrences.append({
                    'Date': entry_date,
                    'Open': row['Open'],
                    'High': row['High'],
                    'Low': row['Low'],
                    'Close': entry_price,
                    'Volume': int(row['Volume']),
                    'outcome': 'STRONG_UP' if target_reached else ('UP' if actual_move > 0 else 'DOWN'),
                    'actual_move': actual_move,
                    'time_to_target': time_window if target_reached else time_window,
                    'target_reached': target_reached
                })
        
        # Save to file
        if ticker:
            output_path = f'{base_path}/occurrences/pattern_{i}_occurrences.json'
        else:
            output_path = f'data/pattern_{i}_occurrences.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(occurrences, f, indent=2, default=lambda x: bool(x) if isinstance(x, (np.bool_, np.bool)) else x)
        
        # Update pattern counts
        pattern['occurrences'] = len(occurrences)
        pattern['success_count'] = sum(1 for o in occurrences if o['target_reached'])
        pattern['success_rate'] = (pattern['success_count'] / len(occurrences) * 100) if occurrences else 0
        
        if i < 10 or i % 10 == 0:
            print(f"  Pattern #{i}: {len(occurrences)} occurrences")
    
    # Save updated patterns to ticker directory
    patterns_output_path = f'{base_path}/patterns.json'
    with open(patterns_output_path, 'w') as f:
        json.dump(patterns, f, indent=2)
    
    print(f"\nGenerated {len(patterns)} occurrence files")
    print(f"Updated {patterns_output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate pattern occurrence files')
    parser.add_argument('--ticker', type=str, help='Ticker symbol (e.g., AAPL, TSLA)')
    args = parser.parse_args()
    generate(args.ticker)
