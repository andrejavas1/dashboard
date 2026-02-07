#!/usr/bin/env python3
"""Generate pattern occurrence files from features matrix"""
import json
import pandas as pd
import numpy as np
import os
import sys

def generate_occurrences(ticker):
    print(f"\nGenerating occurrences for {ticker}...")
    
    base_path = f'data/tickers/{ticker}'
    
    # Load patterns
    patterns_path = f'{base_path}/patterns.json'
    if not os.path.exists(patterns_path):
        print(f"  ERROR: {patterns_path} not found")
        return
    
    with open(patterns_path, 'r') as f:
        patterns = json.load(f)
    print(f"  Loaded {len(patterns)} patterns")
    
    # Load features matrix
    features_path = f'{base_path}/features_matrix.csv'
    if not os.path.exists(features_path):
        print(f"  ERROR: {features_path} not found")
        return
    
    df = pd.read_csv(features_path)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"  Loaded {len(df)} records from features matrix")
    
    # Create occurrences directory
    occ_dir = f'{base_path}/occurrences'
    os.makedirs(occ_dir, exist_ok=True)
    
    # Process each pattern
    generated_count = 0
    for i, pattern in enumerate(patterns):
        pattern_id = pattern.get('id', i)
        conditions = pattern.get('conditions', {})
        direction = pattern.get('direction', 'long')
        label_col = pattern.get('label_col', 'Label_1pct_5d')
        
        if not conditions:
            continue
        
        # Build mask for conditions
        mask = pd.Series(True, index=df.index)
        for feature, condition in conditions.items():
            if feature not in df.columns:
                mask = pd.Series(False, index=df.index)
                break
            
            op = condition.get('operator', '>=')
            value = condition.get('value', 0)
            
            if op == '>=':
                mask &= (df[feature] >= value)
            elif op == '<=':
                mask &= (df[feature] <= value)
            elif op == '>':
                mask &= (df[feature] > value)
            elif op == '<':
                mask &= (df[feature] < value)
        
        # Get occurrences
        occurrences_df = df[mask].copy()
        
        if len(occurrences_df) == 0:
            # Save empty occurrence file
            occ_data = {
                'pattern_id': pattern_id,
                'occurrences': [],
                'total_occurrences': 0,
                'success_count': 0,
                'failed_count': 0,
                'success_rate': 0
            }
        else:
            # Determine success column
            parts = label_col.split('_')
            if len(parts) >= 3:
                window = parts[2].replace('d', '')
            else:
                window = '5'
            
            if direction == 'long':
                success_col = f'Max_Up_{window}d'
            else:
                success_col = f'Max_Down_{window}d'
            
            # Build occurrence records
            occurrences = []
            for _, row in occurrences_df.iterrows():
                occ = {
                    'date': row['Date'].strftime('%Y-%m-%d'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                }
                
                # Add target reached info if available
                if success_col in row and 'Label_' in label_col:
                    label_value = row.get(label_col, '')
                    occ['target_reached'] = (label_value == 'STRONG_UP' if direction == 'long' else label_value == 'STRONG_DOWN')
                
                occurrences.append(occ)
            
            # Count successes
            success_count = sum(1 for o in occurrences if o.get('target_reached', False))
            
            occ_data = {
                'pattern_id': pattern_id,
                'occurrences': occurrences,
                'total_occurrences': len(occurrences),
                'success_count': success_count,
                'failed_count': len(occurrences) - success_count,
                'success_rate': (success_count / len(occurrences) * 100) if occurrences else 0
            }
        
        # Save occurrence file
        occ_path = f'{occ_dir}/pattern_{pattern_id}_occurrences.json'
        with open(occ_path, 'w') as f:
            json.dump(occ_data, f, indent=2)
        
        generated_count += 1
        if i < 5 or i % 5 == 0:
            print(f"    Pattern {pattern_id}: {occ_data['total_occurrences']} occurrences")
    
    print(f"  Generated {generated_count} occurrence files in {occ_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate occurrence files from features matrix')
    parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol')
    args = parser.parse_args()
    
    generate_occurrences(args.ticker)
