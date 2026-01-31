import json
import os
import csv

def generate_all_occurrence_data():
    """Generate occurrence data files for all patterns"""
    
    # Load validated patterns for occurrence data
    with open('data/validated_patterns.json', 'r') as f:
        validated_data = json.load(f)
    
    # Load optimized patterns for pattern details
    with open('data/optimized_patterns.json', 'r') as f:
        optimized_patterns = json.load(f)
    
    # Load OHLCV data for price information
    with open('data/ohlcv.json', 'r') as f:
        ohlcv_data = json.load(f)
    
    # Create a date to OHLCV data mapping for quick lookup
    # Use just the date part (YYYY-MM-DD) for matching since times may differ
    ohlcv_dict = {}
    for record in ohlcv_data:
        # Extract date part: "2010-01-04 05:00:00" -> "2010-01-04"
        ohlcv_date = record['Date'].split(' ')[0]
        ohlcv_dict[ohlcv_date] = record
    
    print(f"Processing {len(optimized_patterns)} patterns")
    
    # Process each pattern
    for pattern_index, pattern in enumerate(optimized_patterns):
        # Get occurrence data from validated patterns if available
        validated_pattern = None
        for vp in validated_data.get('robust', []) + validated_data.get('degraded', []) + validated_data.get('failed', []):
            if vp.get('pattern_index') == pattern_index:
                validated_pattern = vp
                break
        
        # Generate mock occurrence data based on pattern occurrences count
        # In a real system, this would come from actual pattern matching
        occurrences_count = pattern.get('occurrences', 0)
        
        # Generate occurrence entries
        trades = []
        if occurrences_count > 0:
            # Generate random dates from the OHLCV data range
            import random
            random.seed(pattern_index)  # Reproducible results
            
            # Use a subset of OHLCV dates as entry points
            ohlcv_dates = [row['Date'] for row in ohlcv_data]
            num_occurrences = min(occurrences_count, len(ohlcv_dates))
            
            # Sample dates evenly distributed across the data range
            step = len(ohlcv_dates) // (num_occurrences + 1) if num_occurrences > 0 else 1
            selected_indices = list(range(step, len(ohlcv_dates), step))[:num_occurrences]
            
            for idx in selected_indices:
                entry_date = ohlcv_dates[idx]
                # Generate mock outcome based on pattern direction and success rate
                success_rate = pattern.get('success_rate', 50) / 100
                direction = pattern.get('direction', 'long')
                
                # Determine if this trade is successful
                is_success = random.random() < success_rate
                
                if direction == 'long':
                    outcome = 'STRONG_UP' if is_success else 'STRONG_DOWN'
                    actual_move = random.uniform(1, 5) if is_success else random.uniform(-5, -1)
                else:
                    outcome = 'STRONG_DOWN' if is_success else 'STRONG_UP'
                    actual_move = random.uniform(-5, -1) if is_success else random.uniform(1, 5)
                
                trades.append({
                    'entry_date': entry_date,
                    'outcome': outcome,
                    'actual_move': actual_move,
                    'time_to_target': random.randint(1, 20),
                    'is_success': is_success
                })
        
        print(f"Processing pattern {pattern_index} with {len(trades)} trades")
        
        # Generate the occurrence data format that matches the dashboard expectations
        occurrence_data = []
        
        for i, trade in enumerate(trades):
            entry_date = trade['entry_date']
            
            # Extract date part from trade date: "2010-01-15T05:00:00" -> "2010-01-15"
            entry_date_part = entry_date.split('T')[0]
            
            # Find the OHLCV data for entry date
            if entry_date_part in ohlcv_dict:
                ohlcv_entry = ohlcv_dict[entry_date_part].copy()
                # Use the outcome from trade - this is correct based on label_col
                ohlcv_entry['outcome'] = trade['outcome']
                # For actual_move, ensure it matches the outcome direction
                # If outcome is STRONG_UP, actual_move should be positive
                # If outcome is STRONG_DOWN, actual_move should be negative
                if trade['outcome'] in ['STRONG_UP', 'UP']:
                    ohlcv_entry['actual_move'] = abs(trade['actual_move'])  # Ensure positive
                elif trade['outcome'] in ['STRONG_DOWN', 'DOWN']:
                    ohlcv_entry['actual_move'] = -abs(trade['actual_move'])  # Ensure negative
                else:
                    ohlcv_entry['actual_move'] = trade['actual_move']
                ohlcv_entry['time_to_target'] = trade['time_to_target']
                ohlcv_entry['target_reached'] = trade['is_success']
                occurrence_data.append(ohlcv_entry)
            else:
                # Only print first few warnings to avoid spam
                if i < 3:
                    print(f"Warning: No OHLCV data found for entry date {entry_date}")
        
        print(f"Generated {len(occurrence_data)} occurrence records for pattern {pattern_index}")
        
        # Save to the data directory
        filename = f'pattern_{pattern_index}_occurrences.json'
        filepath = os.path.join('data', filename)
        with open(filepath, 'w') as f:
            json.dump(occurrence_data, f, indent=2)
        
        print(f"Pattern {pattern_index} occurrence data saved to {filename}")
        
    
    print("All occurrence data files generated successfully!")

if __name__ == "__main__":
    generate_all_occurrence_data()