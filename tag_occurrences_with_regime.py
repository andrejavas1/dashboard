"""Tag pattern occurrences with market regime information."""
import json
import pandas as pd
import os
import argparse

def tag_occurrences(ticker='XOM'):
    """Add regime information to occurrence files."""
    base_path = f'data/tickers/{ticker}'
    
    # Load features matrix with regime data
    features_path = f'{base_path}/features_matrix.csv'
    if not os.path.exists(features_path):
        print(f'Features matrix not found: {features_path}')
        return
    
    print(f'Loading features matrix for {ticker}...')
    df = pd.read_csv(features_path)
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    
    # Create regime lookup: date -> trend regime (already has full regime names)
    regime_lookup = {}
    for _, row in df.iterrows():
        regime = row.get('Trend_Regime', 'Unknown')
        regime_lookup[row['Date']] = regime
    
    print(f'Loaded {len(regime_lookup)} date regimes')
    
    # Process each occurrence file
    occ_dir = f'{base_path}/occurrences'
    if not os.path.exists(occ_dir):
        print(f'Occurrences directory not found: {occ_dir}')
        return
    
    files = [f for f in os.listdir(occ_dir) if f.endswith('_occurrences.json')]
    print(f'Processing {len(files)} occurrence files...')
    
    total_tagged = 0
    for fname in files:
        fpath = os.path.join(occ_dir, fname)
        with open(fpath, 'r') as f:
            occurrences = json.load(f)
        
        # Tag each occurrence with regime
        tagged_count = 0
        for occ in occurrences:
            date = occ.get('Date', '')
            if date in regime_lookup:
                occ['regime'] = regime_lookup[date]
                tagged_count += 1
            else:
                occ['regime'] = 'Unknown'
        
        # Save updated file
        with open(fpath, 'w') as f:
            json.dump(occurrences, f, indent=2)
        
        total_tagged += tagged_count
        if tagged_count > 0:
            print(f'  {fname}: {tagged_count}/{len(occurrences)} tagged')
    
    print(f'\n✓ Done! Tagged {total_tagged} total occurrences with regime information')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='XOM', help='Ticker symbol')
    args = parser.parse_args()
    tag_occurrences(args.ticker)
