#!/usr/bin/env python3
"""
Flask server for pattern dashboard with dynamic pattern loading.
Loads pattern occurrence data on-demand when a pattern is selected.
Enhanced with pipeline run ID tracking and improved cache management.
Includes automatic pipeline worker for processing queued tickers.
"""

import json
import os
import uuid
import subprocess
import sys
import threading
import time
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

# Pipeline worker process
worker_process = None

def read_worker_output(process):
    """Read worker output to prevent buffer deadlock."""
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[WORKER] {line.rstrip()}")
    except:
        pass

def start_pipeline_worker():
    """Start the pipeline worker if not already running."""
    global worker_process
    
    if worker_process is not None and worker_process.poll() is None:
        print("[INFO] Pipeline worker already running")
        return True
    
    try:
        print("[INFO] Starting pipeline worker...")
        worker_process = subprocess.Popen(
            [sys.executable, 'src/pipeline_worker.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Start thread to read worker output
        output_thread = threading.Thread(target=read_worker_output, args=(worker_process,), daemon=True)
        output_thread.start()
        
        time.sleep(1)  # Give worker time to start
        
        if worker_process.poll() is None:
            print(f"[INFO] Pipeline worker started (PID: {worker_process.pid})")
            return True
        else:
            print("[ERROR] Worker process exited immediately!")
            worker_process = None
            return False
    except Exception as e:
        print(f"[ERROR] Failed to start worker: {e}")
        return False

def stop_pipeline_worker():
    """Stop the pipeline worker."""
    global worker_process
    
    if worker_process is not None:
        print("[INFO] Stopping pipeline worker...")
        worker_process.terminate()
        try:
            worker_process.wait(timeout=5)
            print("[INFO] Pipeline worker stopped")
        except subprocess.TimeoutExpired:
            worker_process.kill()
            worker_process.wait()
            print("[INFO] Pipeline worker killed")
        worker_process = None

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

def get_data_path(filename, ticker=None):
    """Get the full path to a data file.
    
    Args:
        filename: Name of the file
        ticker: Optional ticker symbol for multi-ticker support
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if ticker:
        # Use ticker-specific directory
        return os.path.join(base_dir, 'data', 'tickers', ticker, filename)
    else:
        # Default to main data directory (backward compatibility)
        return os.path.join(base_dir, 'data', filename)

# Default ticker for backward compatibility
default_ticker = 'AAPL'

# Available tickers (will be populated from config)
available_tickers = ['XOM', 'AAPL', 'TSLA']

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
    ticker = request.args.get('ticker', default_ticker)
    patterns_path = get_data_path('patterns.json', ticker)
    ohlcv_path = get_data_path('ohlcv.json', ticker)
    
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

@app.route('/api/tickers')
def get_tickers():
    """Get list of available tickers."""
    return jsonify({
        'tickers': available_tickers,
        'default': default_ticker
    })

@app.route('/api/test-status')
def get_test_status():
    """Get pattern occurrence validation test status."""
    test_status_path = get_data_path('test_status.json')
    
    try:
        with open(test_status_path, 'r') as f:
            test_data = json.load(f)
        return jsonify(test_data)
    except FileNotFoundError:
        return jsonify({
            'status': 'UNKNOWN',
            'timestamp': None,
            'message': 'No test status available. Run pipeline to generate test results.'
        })

@app.route('/api/patterns')
def get_patterns():
    """Get all pattern data (without occurrences)."""
    # Get ticker from query parameter, default to XOM
    ticker = request.args.get('ticker', default_ticker)
    patterns_path = get_data_path('patterns.json', ticker)
    current_mtime = get_file_modification_time(patterns_path)
    
    # Use ticker-specific cache key
    cache_key = f'{ticker}_patterns'
    
    # Check if cache needs to be invalidated
    if cache_key in cached_data:
        cached_mtime = cache_timestamps.get(cache_key)
        if cached_mtime and current_mtime > cached_mtime:
            # File was modified, clear cache
            del cached_data[cache_key]
    
    if cache_key not in cached_data:
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
                    'conditions': p.get('conditions', {}),
                    'regime_analysis': p.get('regime_analysis', {})
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
                    'conditions': pattern.get('conditions', {}),
                    'regime_analysis': pattern.get('regime_analysis', {})
                }
                pattern_stats.append(stats)
        
        cached_data[cache_key] = pattern_stats
        cache_timestamps[cache_key] = current_mtime
    
    return jsonify(cached_data[cache_key])

@app.route('/api/pattern/<int:pattern_id>/occurrences')
def get_pattern_occurrences(pattern_id):
    """Get occurrence data for a specific pattern."""
    ticker = request.args.get('ticker', default_ticker)
    cache_key = f'{ticker}_pattern_{pattern_id}_occurrences'
    
    if cache_key not in cached_data:
        try:
            # Occurrence files are in 'occurrences' subdirectory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            occ_path = os.path.join(base_dir, 'data', 'tickers', ticker, 'occurrences', f'pattern_{pattern_id}_occurrences.json')
            with open(occ_path, 'r') as f:
                cached_data[cache_key] = json.load(f)
        except FileNotFoundError:
            cached_data[cache_key] = []
    
    return jsonify(cached_data[cache_key])

@app.route('/api/ohlcv-full')
def get_ohlcv_full():
    """Get full historical OHLCV data from 2010 for regime demo."""
    ticker = request.args.get('ticker', default_ticker)
    try:
        # Use ticker-specific full history if available
        ohlcv_path = get_data_path('ohlcv_full_history.json', ticker)
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
    ticker = request.args.get('ticker', default_ticker)
    ohlcv_path = get_data_path('ohlcv.json', ticker)
    current_mtime = get_file_modification_time(ohlcv_path)
    
    # Use ticker-specific cache key
    cache_key = f'{ticker}_ohlcv'
    
    # Check if cache needs to be invalidated
    if cache_key in cached_data:
        cached_mtime = cache_timestamps.get(cache_key)
        if cached_mtime and current_mtime > cached_mtime:
            # File was modified, clear cache
            del cached_data[cache_key]
    
    if cache_key not in cached_data:
        with open(ohlcv_path, 'r') as f:
            cached_data[cache_key] = json.load(f)
        cache_timestamps[cache_key] = current_mtime
    
    return jsonify(cached_data[cache_key])

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
    ticker = request.args.get('ticker', default_ticker)
    cache_key = f'{ticker}_ohlcv'
    
    # Get occurrences
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        occ_path = os.path.join(base_dir, 'data', 'tickers', ticker, 'occurrences', f'pattern_{pattern_id}_occurrences.json')
        with open(occ_path, 'r') as f:
            occurrences = json.load(f)
    except FileNotFoundError:
        return jsonify({'entry': [], 'exit': []})
    
    # Get OHLCV data for date lookup
    if cache_key not in cached_data:
        with open(get_data_path('ohlcv.json', ticker), 'r') as f:
            cached_data[cache_key] = json.load(f)
    
    ohlcv_data = cached_data[cache_key]
    
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
    patterns_cache_key = f'{ticker}_patterns'
    if patterns_cache_key not in cached_data:
        with open(get_data_path('patterns.json', ticker), 'r') as f:
            patterns_data = json.load(f)
        cached_data[patterns_cache_key] = patterns_data
    else:
        patterns_data = cached_data[patterns_cache_key]
    
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
        target_reached = occ.get('target_reached', False)
        actual_move = occ.get('actual_move', 0)
        
        # Use actual exit price based on outcome
        if target_reached:
            # Target was hit - calculate exit based on target %
            if pattern_direction == 'long':
                exit_price = entry_price * (1 + target_pct / 100)
            else:
                exit_price = entry_price * (1 - target_pct / 100)
        else:
            # Target NOT reached - use actual move to calculate exit
            exit_price = entry_price * (1 + actual_move / 100)
        
        # Find exit index based on time_to_target or time_window
        time_to_target = occ.get('time_to_target', time_window)
        exit_idx = min(idx + time_to_target, len(ohlcv_data) - 1)
        
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
        ticker = request.args.get('ticker', default_ticker)
        regime_path = get_data_path('regime_history.json', ticker)
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
        ticker = request.args.get('ticker', default_ticker)
        analysis_path = get_data_path('pattern_regime_analysis.json', ticker)
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
        ticker = request.args.get('ticker', default_ticker)
        # Load patterns to get conditions
        patterns_path = get_data_path('patterns.json', ticker)
        with open(patterns_path, 'r') as f:
            patterns_data = json.load(f)
        
        if pattern_id >= len(patterns_data):
            return jsonify({'error': 'Pattern not found'}), 404
        
        pattern = patterns_data[pattern_id]
        conditions = pattern.get('conditions', {})
        
        # Load features matrix
        import csv
        features_path = get_data_path('features_matrix.csv', ticker)
        
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

@app.route('/api/trade-outcomes')
def get_trade_outcomes():
    """Get all trade outcomes from live trading."""
    try:
        outcomes_path = get_data_path('trade_outcomes.json')
        if not os.path.exists(outcomes_path):
            return jsonify([])
        with open(outcomes_path, 'r') as f:
            outcomes = json.load(f)
        return jsonify(outcomes[-100:][::-1])  # Last 100, newest first
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/open-trades')
def get_open_trades():
    """Get currently open trades."""
    try:
        trades_path = get_data_path('open_trades.json')
        if not os.path.exists(trades_path):
            return jsonify([])
        with open(trades_path, 'r') as f:
            trades = json.load(f)
        return jsonify([t for t in trades if t.get('status') == 'OPEN'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pattern/<int:pattern_id>/live-performance')
def get_pattern_live_performance(pattern_id):
    """Get live performance for a specific pattern."""
    try:
        outcomes_path = get_data_path('trade_outcomes.json')
        if not os.path.exists(outcomes_path):
            return jsonify({
                'pattern_id': pattern_id,
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0
            })
        with open(outcomes_path, 'r') as f:
            outcomes = json.load(f)
        
        pattern_outcomes = [o for o in outcomes if o.get('pattern_id') == pattern_id]
        total = len(pattern_outcomes)
        wins = len([o for o in pattern_outcomes if o.get('profit_pct', 0) > 0])
        
        return jsonify({
            'pattern_id': pattern_id,
            'total_trades': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': (wins / total * 100) if total > 0 else 0,
            'avg_profit': sum(o.get('profit_pct', 0) for o in pattern_outcomes) / total if total > 0 else 0,
            'recent_outcomes': pattern_outcomes[-10:][::-1]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/live-performance-summary')
def get_live_performance_summary():
    """Get overall live trading performance summary."""
    try:
        outcomes_path = get_data_path('trade_outcomes.json')
        open_trades_path = get_data_path('open_trades.json')
        
        outcomes = []
        open_trades = []
        
        if os.path.exists(outcomes_path):
            with open(outcomes_path, 'r') as f:
                outcomes = json.load(f)
        
        if os.path.exists(open_trades_path):
            with open(open_trades_path, 'r') as f:
                open_trades = [t for t in json.load(f) if t.get('status') == 'OPEN']
        
        total = len(outcomes)
        wins = len([o for o in outcomes if o.get('profit_pct', 0) > 0])
        total_profit = sum(o.get('profit_pct', 0) for o in outcomes)
        
        return jsonify({
            'total_completed_trades': total,
            'total_wins': wins,
            'total_losses': total - wins,
            'win_rate': (wins / total * 100) if total > 0 else 0,
            'total_profit_pct': total_profit,
            'avg_profit_per_trade': total_profit / total if total > 0 else 0,
            'open_trades_count': len(open_trades),
            'target_hits': len([o for o in outcomes if o.get('outcome') == 'TARGET_HIT']),
            'stop_hits': len([o for o in outcomes if o.get('outcome') == 'STOP_HIT']),
            'timeouts': len([o for o in outcomes if o.get('outcome') == 'TIMEOUT'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# TICKER QUEUE MANAGEMENT ENDPOINTS

@app.route('/api/ticker-queue/status')
def get_queue_status():
    """Get current ticker queue status."""
    try:
        from src.ticker_queue_manager import TickerQueueManager
        manager = TickerQueueManager()
        return jsonify(manager.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ticker-queue/progress')
def get_pipeline_progress():
    """Get current pipeline processing progress."""
    try:
        progress_file = 'ticker_progress.json'
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            return jsonify(progress)
        else:
            # Return default idle state
            return jsonify({
                'ticker': None,
                'current_phase': 0,
                'total_phases': 10,
                'phase_name': 'Idle',
                'progress_pct': 0,
                'status': 'idle',
                'elapsed_seconds': 0,
                'error': None
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ticker-queue/add', methods=['POST'])
def add_ticker():
    """Add a new ticker to the processing queue and auto-start worker."""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').upper().strip()
        
        if not ticker:
            return jsonify({'error': 'Ticker symbol required'}), 400
        
        from src.ticker_queue_manager import TickerQueueManager
        manager = TickerQueueManager()
        
        if manager.add_ticker(ticker):
            # Auto-start pipeline worker if not running
            worker_started = start_pipeline_worker()
            
            return jsonify({
                'success': True,
                'message': f'Ticker {ticker} added to queue',
                'worker_started': worker_started,
                'status': manager.get_status()
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Could not add ticker {ticker}',
                'status': manager.get_status()
            }), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ticker-queue/cancel/<ticker>', methods=['DELETE'])
def cancel_ticker(ticker):
    """Cancel a queued ticker."""
    try:
        from src.ticker_queue_manager import TickerQueueManager
        manager = TickerQueueManager()
        
        if manager.cancel_queued(ticker.upper()):
            return jsonify({
                'success': True,
                'message': f'Cancelled {ticker}',
                'status': manager.get_status()
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Could not cancel {ticker}'
            }), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ticker-queue/retry/<ticker>', methods=['POST'])
def retry_ticker(ticker):
    """Retry a failed ticker."""
    try:
        from src.ticker_queue_manager import TickerQueueManager
        manager = TickerQueueManager()
        
        if manager.retry_failed(ticker.upper()):
            return jsonify({
                'success': True,
                'message': f'Retrying {ticker}',
                'status': manager.get_status()
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Could not retry {ticker}'
            }), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ticker-queue/remove/<ticker>', methods=['DELETE'])
def remove_ticker(ticker):
    """Remove a ticker from portfolio and delete its data."""
    try:
        from src.ticker_queue_manager import TickerQueueManager
        manager = TickerQueueManager()
        
        if manager.remove_ticker(ticker.upper()):
            return jsonify({
                'success': True,
                'message': f'Removed {ticker}',
                'status': manager.get_status()
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Ticker {ticker} not found'
            }), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tickers/portfolio')
def get_portfolio():
    """Get list of available portfolio tickers."""
    try:
        from src.ticker_queue_manager import TickerQueueManager
        manager = TickerQueueManager()
        
        portfolio = manager.get_portfolio()
        default_ticker = manager.default_ticker()
        
        return jsonify({
            'tickers': portfolio,
            'default': default_ticker,
            'count': len(portfolio)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    """Serve the dashboard HTML file."""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'DASHBOARD_DYNAMIC.html')

if __name__ == '__main__':
    print("=" * 60)
    print("Pattern Dashboard Server")
    print("=" * 60)
    print("Starting Flask server...")
    print("Dashboard will be available at: http://localhost:5001")
    print("=" * 60)
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=False)
    finally:
        # Ensure worker is stopped when server shuts down
        stop_pipeline_worker()
