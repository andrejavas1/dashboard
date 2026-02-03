#!/usr/bin/env python3
"""
Flask server for pattern dashboard with dynamic pattern loading.
Loads pattern occurrence data on-demand when a pattern is selected.
Enhanced with pipeline run ID tracking and improved cache management.
"""

import json
import os
import uuid
from datetime import datetime
from flask import Flask, jsonify, send_from_directory, request

app = Flask(__name__, static_folder='.')

# Cache for loaded data with modification time tracking
cached_data = {}
cache_timestamps = {}
pipeline_run_id = None
pipeline_run_timestamp = None

# Force cache clear on server restart
cached_data.clear()
cache_timestamps.clear()

def get_or_generate_pipeline_run_id():
    """Get or generate pipeline run ID from patterns file."""
    global pipeline_run_id, pipeline_run_timestamp
    
    patterns_path = get_data_path('patterns.json')
    
    try:
        with open(patterns_path, 'r') as f:
            patterns_data = json.load(f)
        
        # Check if pipeline metadata exists
        if isinstance(patterns_data, dict) and 'pipeline_run_id' in patterns_data:
            pipeline_run_id = patterns_data['pipeline_run_id']
            pipeline_run_timestamp = patterns_data.get('pipeline_run_timestamp')
        elif isinstance(patterns_data, list) and len(patterns_data) > 0:
            # Try to find metadata in first pattern
            if isinstance(patterns_data[0], dict) and 'pipeline_run_id' in patterns_data[0]:
                pipeline_run_id = patterns_data[0]['pipeline_run_id']
                pipeline_run_timestamp = patterns_data[0].get('pipeline_run_timestamp')
        
        # Generate new ID if not found
        if pipeline_run_id is None:
            pipeline_run_id = str(uuid.uuid4())[:8]
            pipeline_run_timestamp = datetime.now().isoformat()
    except Exception as e:
        print(f"Error reading pipeline metadata: {e}")
        pipeline_run_id = str(uuid.uuid4())[:8]
        pipeline_run_timestamp = datetime.now().isoformat()
    
    return pipeline_run_id, pipeline_run_timestamp

def get_data_path(filename):
    """Get the full path to a data file."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', filename)

def get_file_modification_time(filepath):
    """Get the modification time of a file as a datetime object."""
    return datetime.fromtimestamp(os.path.getmtime(filepath))

@app.route('/')
def index():
    """Serve the dashboard HTML."""
    return send_from_directory('.', 'DASHBOARD_DYNAMIC.html')

@app.route('/regime-demo')
def regime_demo():
    """Serve the regime comparison demo HTML."""
    return send_from_directory('.', 'REGIME_COMPARISON_DEMO.html')

@app.route('/api/data-info')
def get_data_info():
    """Get information about the data files including modification times and pipeline run ID."""
    patterns_path = get_data_path('patterns.json')
    ohlcv_path = get_data_path('ohlcv.json')
    
    patterns_mtime = get_file_modification_time(patterns_path)
    ohlcv_mtime = get_file_modification_time(ohlcv_path)
    
    # Get pipeline run ID
    run_id, run_timestamp = get_or_generate_pipeline_run_id()
    
    # Get pattern count from file
    with open(patterns_path, 'r') as f:
        patterns_data = json.load(f)
    
    # Get OHLCV record count
    with open(ohlcv_path, 'r') as f:
        ohlcv_data = json.load(f)
    
    return jsonify({
        'patterns_modified': patterns_mtime.isoformat(),
        'patterns_count': len(patterns_data),
        'ohlcv_modified': ohlcv_mtime.isoformat(),
        'ohlcv_count': len(ohlcv_data),
        'pipeline_run_id': run_id,
        'pipeline_run_timestamp': run_timestamp,
        'cache_size': len(cached_data)
    })

@app.route('/api/patterns')
def get_patterns():
    """Get all pattern data (without occurrences)."""
    patterns_path = get_data_path('patterns.json')
    current_mtime = get_file_modification_time(patterns_path)
    
    # Check if cache needs to be invalidated
    if 'patterns' in cached_data:
        cached_mtime = cache_timestamps.get('patterns')
        if cached_mtime and current_mtime > cached_mtime:
            # File was modified, clear cache
            del cached_data['patterns']
    
    if 'patterns' not in cached_data:
        with open(patterns_path, 'r') as f:
            patterns_data = json.load(f)
        
        # Extract pattern stats without occurrences
        pattern_stats = []
        for i, pattern in enumerate(patterns_data):
            if 'pattern' in pattern:
                # Wrapped pattern format (from validated_patterns.json)
                classification = pattern.get('classification', '').upper()
                if classification in ['FAILED', 'REJECTED', 'LOW CONFIDENCE']:
                    continue
                
                p = pattern['pattern']
                stats = {
                    'id': i,
                    'success_rate': p.get('success_rate', 0),
                    'avg_move': p.get('avg_move', 0),
                    'occurrences': p.get('occurrences', 0),
                    'method': p.get('method', 'N/A'),
                    'direction': p.get('direction', 'N/A'),
                    'classification': classification,
                    'validation_success_rate': pattern.get('validation_success_rate', p.get('success_rate', 0)),
                    'label_col': p.get('label_col', 'Label_3pct_20d'),
                    'conditions': p.get('conditions', {})
                }
                pattern_stats.append(stats)
            else:
                # Flat pattern format (direct from discovered_patterns.json)
                stats = {
                    'id': i,
                    'success_rate': pattern.get('success_rate', 0),
                    'avg_move': pattern.get('avg_move', 0),
                    'occurrences': pattern.get('occurrences', 0),
                    'method': pattern.get('method', 'N/A'),
                    'direction': pattern.get('direction', 'N/A'),
                    'classification': pattern.get('classification', 'MEDIUM CONFIDENCE').upper(),
                    'validation_success_rate': pattern.get('success_rate', 0),
                    'label_col': pattern.get('label_col', 'Label_3pct_20d'),
                    'conditions': pattern.get('conditions', {})
                }
                pattern_stats.append(stats)
        
        cached_data['patterns'] = pattern_stats
        cache_timestamps['patterns'] = current_mtime
    
    return jsonify(cached_data['patterns'])

@app.route('/api/pattern/<int:pattern_id>/occurrences')
def get_pattern_occurrences(pattern_id):
    """Get occurrence data for a specific pattern."""
    cache_key = f'pattern_{pattern_id}_occurrences'
    
    if cache_key not in cached_data:
        try:
            with open(get_data_path(f'pattern_{pattern_id}_occurrences.json'), 'r') as f:
                cached_data[cache_key] = json.load(f)
        except FileNotFoundError:
            cached_data[cache_key] = []
    
    return jsonify(cached_data[cache_key])

@app.route('/api/ohlcv-full')
def get_ohlcv_full():
    """Get full historical OHLCV data from 2010 for regime demo."""
    try:
        ohlcv_path = get_data_path('ohlcv_full_history.json')
        with open(ohlcv_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({'error': 'Full history data not available. Run fetch_historical_data.py'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ohlcv')
def get_ohlcv():
    """Get OHLCV data with cache invalidation."""
    ohlcv_path = get_data_path('ohlcv.json')
    current_mtime = get_file_modification_time(ohlcv_path)
    
    # Check if cache needs to be invalidated
    if 'ohlcv' in cached_data:
        cached_mtime = cache_timestamps.get('ohlcv')
        if cached_mtime and current_mtime > cached_mtime:
            # File was modified, clear cache
            del cached_data['ohlcv']
    
    if 'ohlcv' not in cached_data:
        with open(ohlcv_path, 'r') as f:
            cached_data['ohlcv'] = json.load(f)
        cache_timestamps['ohlcv'] = current_mtime
    
    return jsonify(cached_data['ohlcv'])

@app.route('/api/matches-history')
def get_matches_history():
    """Get pattern matches history from streaming system."""
    try:
        matches_path = get_data_path('matches_history.json')
        current_mtime = get_file_modification_time(matches_path)
        
        # Check if cache needs to be invalidated
        if 'matches_history' in cached_data:
            cached_mtime = cache_timestamps.get('matches_history')
            if cached_mtime and current_mtime > cached_mtime:
                # File was modified, clear cache
                del cached_data['matches_history']
        
        if 'matches_history' not in cached_data:
            with open(matches_path, 'r') as f:
                cached_data['matches_history'] = json.load(f)
            cache_timestamps['matches_history'] = current_mtime
        
        # Return most recent matches first (limited to 100)
        history = cached_data['matches_history']
        return jsonify(history[-100:][::-1])  # Reverse to get newest first
    except FileNotFoundError:
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pattern/<int:pattern_id>/markers')
def get_pattern_markers(pattern_id):
    """Get entry and exit markers for a specific pattern."""
    # Get occurrences
    try:
        with open(get_data_path(f'pattern_{pattern_id}_occurrences.json'), 'r') as f:
            occurrences = json.load(f)
    except FileNotFoundError:
        return jsonify({'entry': [], 'exit': []})
    
    # Get OHLCV data for date lookup
    if 'ohlcv' not in cached_data:
        with open(get_data_path('ohlcv.json'), 'r') as f:
            cached_data['ohlcv'] = json.load(f)
    
    ohlcv_data = cached_data['ohlcv']
    
    # Create date index for quick lookup (normalize dates to YYYY-MM-DD)
    date_to_index = {}
    for idx, row in enumerate(ohlcv_data):
        full_date = row['Date']
        # Normalize to YYYY-MM-DD format
        date_key = full_date.split(' ')[0] if ' ' in full_date else full_date
        date_to_index[date_key] = idx
        date_to_index[full_date] = idx
        date_to_index[full_date.replace(' ', 'T')] = idx
    
    # Get pattern data for target parameters
    if 'patterns' not in cached_data:
        with open(get_data_path('patterns.json'), 'r') as f:
            patterns_data = json.load(f)
        cached_data['patterns'] = patterns_data
    else:
        patterns_data = cached_data['patterns']
    
    # Get pattern info
    pattern_data = patterns_data[pattern_id].get('pattern', {}) if pattern_id < len(patterns_data) else {}
    label_col = pattern_data.get('label_col', 'Label_3pct_20d')
    pattern_direction = pattern_data.get('direction', 'long')
    
    # Parse target percentage and time window
    import re
    match = re.search(r'Label_(-?\d+)pct', label_col)
    target_pct = float(match.group(1)) if match else 3.0
    
    match = re.search(r'pct(\d+)d', label_col)
    time_window = int(match.group(1)) if match else 10
    
    # Generate markers
    entry_markers = []
    exit_markers = []
    
    sorted_occurrences = sorted(occurrences, key=lambda x: x['Date'])
    open_until_index = -1
    occurrence_index = 0
    
    for occ in sorted_occurrences:
        if occ['Date'] not in date_to_index:
            continue
        
        idx = date_to_index[occ['Date']]
        
        if idx < open_until_index:
            continue
        
        occurrence_index += 1
        
        # Entry marker
        entry_markers.append({
            'x': idx,
            'y': occ['Close'],
            'pattern': pattern_id,
            'outcome': occ.get('outcome', 'UNKNOWN'),
            'date': occ['Date'],
            'occurrence': occurrence_index
        })
        
        # Calculate exit
        entry_price = occ['Close']
        
        if pattern_direction == 'long':
            target_price = entry_price * (1 + target_pct / 100)
        else:
            target_price = entry_price * (1 - target_pct / 100)
        
        exit_idx = idx
        exit_price = entry_price
        exit_found = False
        
        max_idx = min(idx + time_window, len(ohlcv_data) - 1)
        for check_idx in range(idx, max_idx + 1):
            candle = ohlcv_data[check_idx]
            
            if target_price <= candle['High'] and target_price >= candle['Low']:
                exit_idx = check_idx
                exit_price = target_price
                exit_found = True
                break
        
        if not exit_found:
            exit_idx = max_idx
            exit_price = ohlcv_data[exit_idx]['Close']
        
        exit_markers.append({
            'x': exit_idx,
            'y': exit_price,
            'pattern': pattern_id,
            'outcome': occ.get('outcome', 'UNKNOWN'),
            'date': ohlcv_data[exit_idx]['Date'],
            'occurrence': occurrence_index
        })
        
        open_until_index = exit_idx + 1
    
    return jsonify({
        'entry': entry_markers,
        'exit': exit_markers
    })

@app.route('/api/regime-history')
def get_regime_history():
    """Get market regime history for trend visualization on chart."""
    try:
        regime_path = get_data_path('regime_history.json')
        with open(regime_path, 'r') as f:
            regime_data = json.load(f)
        return jsonify(regime_data)
    except FileNotFoundError:
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pattern-regime-analysis')
def get_pattern_regime_analysis():
    """Get pattern performance analysis by market regime."""
    try:
        analysis_path = get_data_path('pattern_regime_analysis.json')
        with open(analysis_path, 'r') as f:
            analysis_data = json.load(f)
        return jsonify(analysis_data)
    except FileNotFoundError:
        return jsonify({})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pattern/<int:pattern_id>/indicators')
def get_pattern_indicators(pattern_id):
    """Get indicator data for a specific pattern with threshold values."""
    try:
        # Load patterns to get conditions
        patterns_path = get_data_path('patterns.json')
        with open(patterns_path, 'r') as f:
            patterns_data = json.load(f)
        
        if pattern_id >= len(patterns_data):
            return jsonify({'error': 'Pattern not found'}), 404
        
        pattern = patterns_data[pattern_id]
        conditions = pattern.get('conditions', {})
        
        # Load features matrix
        import csv
        features_path = get_data_path('features_matrix.csv')
        
        indicators_data = {}
        indicator_names = list(conditions.keys())
        
        with open(features_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                date = row['Date']
                for indicator in indicator_names:
                    if indicator in row:
                        if indicator not in indicators_data:
                            indicators_data[indicator] = {
                                'values': [],
                                'threshold': conditions[indicator].get('value'),
                                'operator': conditions[indicator].get('operator'),
                                'min': float('inf'),
                                'max': float('-inf')
                            }
                        val = float(row[indicator])
                        indicators_data[indicator]['values'].append({
                            'date': date,
                            'value': val
                        })
                        indicators_data[indicator]['min'] = min(indicators_data[indicator]['min'], val)
                        indicators_data[indicator]['max'] = max(indicators_data[indicator]['max'], val)
        
        return jsonify({
            'pattern_id': pattern_id,
            'indicators': indicators_data,
            'direction': pattern.get('direction', 'unknown'),
            'label_col': pattern.get('label_col', '')
        })
    except FileNotFoundError as e:
        return jsonify({'error': f'File not found: {e}'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Pattern Dashboard Server")
    print("=" * 60)
    print("Starting Flask server...")
    print("Dashboard will be available at: http://localhost:5001")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5001, debug=False)
