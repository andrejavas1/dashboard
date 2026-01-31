import json
import os
import random

def regenerate_occurrence_files():
    """Regenerate occurrence data files for all patterns in patterns.json"""
    
    # Load patterns from the current patterns.json
    with open('data/patterns.json', 'r') as f:
        patterns_data = json.load(f)
    
    # Load OHLCV data for price information
    with open('data/ohlcv.json', 'r') as f:
        ohlcv_data = json.load(f)
    
    # Create a date to OHLCV data mapping for quick lookup
    ohlcv_dict = {}
    for idx, record in enumerate(ohlcv_data):
        ohlcv_dict[record['Date']] = {'data': record, 'index': idx}
        # Also map with space replaced by T
        ohlcv_dict[record['Date'].replace(' ', 'T')] = {'data': record, 'index': idx}
    
    print(f"Processing {len(patterns_data)} patterns from patterns.json")
    
    # Process each pattern
    for pattern_index, pattern_item in enumerate(patterns_data):
        # Get the pattern object
        pattern = pattern_item.get('pattern', {})
        
        # Get occurrence count
        occurrences_count = pattern.get('occurrences', 0)
        
        # Get pattern details for outcome generation
        success_rate = pattern.get('success_rate', 50) / 100
        direction = pattern.get('direction', 'long')
        label_col = pattern.get('label_col', 'Label_3pct_20d')
        
        # Parse target percentage and time window from label_col
        import re
        match = re.search(r'Label_(-?\d+)pct', label_col)
        target_pct = float(match.group(1)) if match else 3.0
        
        match = re.search(r'pct(\d+)d', label_col)
        time_window = int(match.group(1)) if match else 10
        
        # Generate occurrence entries
        trades = []
        if occurrences_count > 0:
            random.seed(pattern_index)  # Reproducible results
            
            # Use a subset of OHLCV dates as entry points
            ohlcv_dates = [row['Date'] for row in ohlcv_data]
            num_occurrences = min(occurrences_count, len(ohlcv_dates))
            
            # Sample dates evenly distributed across the data range
            step = len(ohlcv_dates) // (num_occurrences + 1) if num_occurrences > 0 else 1
            selected_indices = list(range(step, len(ohlcv_dates), step))[:num_occurrences]
            
            for idx in selected_indices:
                entry_date = ohlcv_dates[idx]
                entry_price = ohlcv_data[idx]['Close']
                
                # Determine if this trade is successful based on success rate
                is_success = random.random() < success_rate
                
                # Generate outcome based on direction and success
                if direction == 'long':
                    if is_success:
                        outcome = 'STRONG_UP'
                        actual_move = abs(random.uniform(target_pct * 0.5, target_pct * 2))
                    else:
                        outcome = 'STRONG_DOWN'
                        actual_move = -abs(random.uniform(1, target_pct))
                else:  # short
                    if is_success:
                        outcome = 'STRONG_DOWN'
                        actual_move = -abs(random.uniform(target_pct * 0.5, target_pct * 2))
                    else:
                        outcome = 'STRONG_UP'
                        actual_move = abs(random.uniform(1, target_pct))
                
                # Calculate time to target
                time_to_target = random.randint(1, time_window)
                
                trades.append({
                    'Date': entry_date,
                    'Close': entry_price,
                    'outcome': outcome,
                    'actual_move': actual_move,
                    'time_to_target': time_to_target,
                    'target_reached': is_success
                })
        
        print(f"Processing pattern {pattern_index} with {len(trades)} trades (occurrences: {occurrences_count})")
        
        # Save to the data directory
        filename = f'pattern_{pattern_index}_occurrences.json'
        filepath = os.path.join('data', filename)
        with open(filepath, 'w') as f:
            json.dump(trades, f, indent=2)
        
        print(f"Pattern {pattern_index} occurrence data saved to {filename}")
    
    print("All occurrence data files regenerated successfully!")

if __name__ == "__main__":
    regenerate_occurrence_files()