#!/usr/bin/env python3
"""
Generate a standalone HTML dashboard with candlestick chart and pattern markers.
No server required - all data is embedded in the HTML file.
"""

import json
import os
from datetime import datetime

def load_data():
    """Load all required data files"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load OHLCV data
    with open(os.path.join(base_dir, 'data', 'ohlcv.json'), 'r') as f:
        ohlcv_data = json.load(f)
    
    # Load patterns data
    with open(os.path.join(base_dir, 'data', 'patterns.json'), 'r') as f:
        patterns_data = json.load(f)
    
    # Load realistic data (optional)
    try:
        with open(os.path.join(base_dir, 'data', 'realistic.json'), 'r') as f:
            realistic_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        realistic_data = {}
    
    # Load occurrence files
    occurrences = {}
    for i in range(20):
        try:
            with open(os.path.join(base_dir, 'data', f'pattern_{i}_occurrences.json'), 'r') as f:
                occurrences[i] = json.load(f)
        except FileNotFoundError:
            occurrences[i] = []
    
    return ohlcv_data, patterns_data, realistic_data, occurrences

def format_conditions(conditions):
    """Format pattern conditions for display"""
    formatted = []
    for feature, cond in conditions.items():
        op = cond.get('operator', '~')
        val = cond.get('value', 0)
        if op == '~':
            formatted.append(f"{feature} ≈ {val:.2f}")
        elif op == '>=':
            formatted.append(f"{feature} ≥ {val:.2f}")
        elif op == '<=':
            formatted.append(f"{feature} ≤ {val:.2f}")
        elif op == '>':
            formatted.append(f"{feature} > {val:.2f}")
        elif op == '<':
            formatted.append(f"{feature} < {val:.2f}")
    return formatted

def generate_standalone_html():
    """Generate a standalone HTML file with embedded data"""
    ohlcv_data, patterns_data, realistic_data, occurrences = load_data()
    
    # Use all OHLCV data
    ohlcv_sample = ohlcv_data
    
    # Create date index for quick lookup - handle both date formats
    date_to_index = {}
    for idx, row in enumerate(ohlcv_sample):
        # Store both formats for matching
        date_to_index[row['Date']] = idx  # Original format: "2010-01-04 05:00:00"
        date_to_index[row['Date'].replace(' ', 'T')] = idx  # T format: "2010-01-04T05:00:00"
    
    # Prepare candlestick data arrays
    dates = [row['Date'] for row in ohlcv_sample]
    opens = [row['Open'] for row in ohlcv_sample]
    highs = [row['High'] for row in ohlcv_sample]
    lows = [row['Low'] for row in ohlcv_sample]
    closes = [row['Close'] for row in ohlcv_sample]
    
    # Prepare all pattern markers (will be filtered by selection)
    all_entry_markers = {}
    all_exit_markers = {}
    
    # Create pattern data map for quick lookup
    pattern_data_map = {}
    for i, pattern in enumerate(patterns_data):
        if 'pattern' in pattern:
            pattern_data_map[i] = pattern['pattern']
    
    for pattern_id in range(len(occurrences)):
        entry_markers = []
        exit_markers = []
        
        # Sort occurrences by date
        sorted_occurrences = sorted(occurrences[pattern_id], key=lambda x: x['Date'])
        
        # Track open positions to prevent overlapping
        open_until_index = -1  # Index until which position is open
        
        for occ in sorted_occurrences:
            if occ['Date'] in date_to_index:
                idx = date_to_index[occ['Date']]
                
                # Skip if position is still open (no overlapping positions)
                if idx < open_until_index:
                    continue
                
                # Entry marker
                entry_markers.append({
                    'x': idx,
                    'y': occ['Close'],
                    'pattern': pattern_id,
                    'outcome': occ.get('outcome', 'UNKNOWN'),
                    'date': occ['Date']
                })
                
                # Calculate exit based on pattern conditions
                # Exit when price threshold is reached OR at end of time window
                # No minimum holding period - can exit same day (time_to_target can be 0)
                # Only maximum time window applies
                entry_price = occ['Close']
                
                # Get pattern information to determine target percentage and time window
                if pattern_id in pattern_data_map:
                    pattern_data = pattern_data_map[pattern_id]
                else:
                    # Use default values if pattern not found
                    pattern_data = {}
                label_col = pattern_data.get('label_col', 'Label_3pct_20d')
                pattern_direction = pattern_data.get('direction', 'long')
                
                # Parse target percentage from label_col (e.g., "Label_3pct_20d" -> 3%)
                import re
                match = re.search(r'Label_(-?\d+)pct', label_col)
                if match:
                    target_pct = float(match.group(1))
                else:
                    # Default to 3% if parsing fails
                    target_pct = 3.0
                
                # Parse time window from label_col (e.g., "Label_3pct_20d" -> 20 days)
                match = re.search(r'pct(\d+)d', label_col)
                if match:
                    time_window = int(match.group(1))
                else:
                    # Default to 10 days if parsing fails
                    time_window = 10
                
                # Determine exit index and price
                # Check each candle in the time window to see if target was reached
                exit_idx = idx
                exit_price = entry_price
                exit_found = False
                
                # Calculate target price based on pattern direction and target percentage
                if pattern_direction == 'long':
                    target_price = entry_price * (1 + target_pct / 100)
                else:  # short
                    target_price = entry_price * (1 - target_pct / 100)
                
                # Search through time window for target being reached
                max_idx = min(idx + time_window, len(ohlcv_sample) - 1)
                for check_idx in range(idx, max_idx + 1):
                    candle = ohlcv_sample[check_idx]
                    
                    # Check if target price was reached within this candle's range
                    if target_price <= candle['High'] and target_price >= candle['Low']:
                        # Target reached during this candle
                        exit_idx = check_idx
                        exit_price = target_price
                        exit_found = True
                        break
                
                # If target not reached, exit at close of last candle in window
                if not exit_found:
                    exit_idx = max_idx
                    exit_price = ohlcv_sample[exit_idx]['Close']
                
                exit_markers.append({
                    'x': exit_idx,
                    'y': exit_price,
                    'pattern': pattern_id,
                    'outcome': occ.get('outcome', 'UNKNOWN'),
                    'date': ohlcv_sample[exit_idx]['Date']
                })
                
                # Mark position as open until exit
                open_until_index = exit_idx + 1  # Next position can start after this one closes
        
        all_entry_markers[pattern_id] = entry_markers
        all_exit_markers[pattern_id] = exit_markers
    
    # Extract pattern stats correctly (nested structure)
    pattern_stats = []
    for i, pattern in enumerate(patterns_data):
        if 'pattern' in pattern:
            # Skip failed/rejected patterns
            classification = pattern.get('classification', '').upper()
            if classification in ['FAILED', 'REJECTED', 'LOW CONFIDENCE']:
                continue
            
            p = pattern['pattern']
            training_rate = p.get('success_rate', 0)
            validation_rate = pattern.get('validation_success_rate', training_rate)
            
            # Format entry conditions
            conditions = p.get('conditions', {})
            entry_conditions = format_conditions(conditions)
            
            # Format exit conditions based on label_col
            label_col = p.get('label_col', 'Label_3pct_20d')
            parts = label_col.split('_')
            if len(parts) >= 3:
                threshold = parts[1].replace('pct', '')
                window = parts[2].replace('d', '')
                exit_conditions = [
                    f"Exit when price moves {threshold}% {'up' if p.get('direction') == 'long' else 'down'}",
                    f"within {window} days"
                ]
            else:
                exit_conditions = ["Exit conditions not specified"]
            
            stats = {
                'id': i,
                'success_rate': training_rate,
                'avg_move': p.get('avg_move', 0),
                'occurrences': p.get('occurrences', 0),
                'method': p.get('method', 'N/A'),
                'direction': p.get('direction', 'N/A'),
                'classification': classification,
                'validation_success_rate': validation_rate,
                'entry_conditions': entry_conditions,
                'exit_conditions': exit_conditions,
                'label_col': label_col
            }
            pattern_stats.append(stats)
    
    # Generate HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pattern Trading Dashboard - Standalone</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial@0.2.1/dist/chartjs-chart-financial.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon@3.4.4/build/global/luxon.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.3.1/dist/chartjs-adapter-luxon.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 20px;
            color: #4ecca3;
        }}
        .info {{
            background: #16213e;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .info strong {{
            color: #4ecca3;
        }}
        .chart-container {{
            background: #16213e;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            height: 600px;
            position: relative;
        }}
        .zoom-controls {{
            position: absolute;
            top: 30px;
            right: 30px;
            display: flex;
            gap: 10px;
            z-index: 10;
        }}
        .zoom-btn {{
            background: #0f3460;
            color: #4ecca3;
            border: 1px solid #4ecca3;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        .zoom-btn:hover {{
            background: #4ecca3;
            color: #1a1a2e;
        }}
        .pattern-selector {{
            background: #16213e;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .pattern-selector label {{
            color: #4ecca3;
            margin-right: 10px;
            font-weight: bold;
        }}
        .pattern-selector select {{
            background: #0f3460;
            color: #eee;
            border: 1px solid #4ecca3;
            padding: 8px 15px;
            border-radius: 4px;
            font-size: 14px;
            cursor: pointer;
        }}
        .patterns-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 10px;
            margin-top: 20px;
        }}
        .pattern-card {{
            background: #16213e;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #0f3460;
            cursor: pointer;
            transition: all 0.3s;
        }}
        .pattern-card:hover {{
            border-color: #4ecca3;
            transform: translateY(-2px);
        }}
        .pattern-card.selected {{
            border-color: #4ecca3;
            box-shadow: 0 0 10px rgba(78, 204, 163, 0.3);
        }}
        .pattern-card h3 {{
            color: #4ecca3;
            margin-bottom: 5px;
            font-size: 14px;
        }}
        .stat {{
            display: flex;
            justify-content: space-between;
            margin: 2px 0;
            padding: 2px 0;
            border-bottom: 1px solid #0f3460;
            font-size: 12px;
        }}
        .stat-label {{
            color: #aaa;
            font-size: 11px;
        }}
        .stat-value {{
            color: #4ecca3;
            font-weight: bold;
            font-size: 12px;
        }}
        .positive {{ color: #4ecca3; }}
        .negative {{ color: #e94560; }}
        .conditions-section {{
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid #0f3460;
        }}
        .conditions-title {{
            color: #ffd700;
            font-weight: bold;
            margin-bottom: 3px;
            font-size: 11px;
            cursor: pointer;
            user-select: none;
            display: block;
        }}
        .conditions-title:hover {{
            text-decoration: underline;
        }}
        .condition-item {{
            font-size: 10px;
            color: #ccc;
            margin: 1px 0;
            padding-left: 8px;
            display: none;
        }}
        .conditions-section.visible .condition-item {{
            display: block;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            margin-top: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-color {{
            width: 15px;
            height: 15px;
            border-radius: 50%;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Pattern Trading Dashboard</h1>
        
        <div class="info">
            <p><strong>Data Range:</strong> {ohlcv_sample[0]['Date'][:10]} to {ohlcv_sample[-1]['Date'][:10]}</p>
            <p><strong>Total Records:</strong> {len(ohlcv_data):,}</p>
            <p><strong>Patterns Found:</strong> {len(patterns_data)}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Zoom:</strong> Use mouse wheel to zoom, drag to pan. Click "Reset Zoom" to reset.</p>
            <p><strong>Pattern Selection:</strong> Click on a pattern card below to view its markers on the chart.</p>
        </div>
        
        <div class="pattern-selector">
            <label for="patternSelect">Select Pattern:</label>
            <select id="patternSelect" onchange="selectPattern(this.value)">
                <option value="all">All Patterns</option>
                <option value="none">None</option>
                {''.join([f'<option value="{i}">Pattern #{i} - {p.get("direction", "N/A")} ({p.get("success_rate", 0):.1f}%)</option>' for i, p in enumerate(patterns_data) if 'pattern' in p])}
            </select>
        </div>
        
        <div class="chart-container">
            <div class="zoom-controls">
                <button class="zoom-btn" onclick="resetZoom()">Reset Zoom</button>
            </div>
            <canvas id="candlestickChart"></canvas>
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #4ecca3;"></div>
                <span>Bullish Candle</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #e94560;"></div>
                <span>Bearish Candle</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #ffd700;"></div>
                <span>Pattern Entry</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #ff6b6b;"></div>
                <span>Pattern Exit</span>
            </div>
        </div>
        
        <h2>Pattern Performance (Click to Select)</h2>
        <div class="patterns-grid" id="patternsGrid">
            <!-- Patterns will be inserted here -->
        </div>
    </div>
    
    <script>
        // Register zoom plugin
        Chart.register(ChartZoom);
        
        // Custom plugin to draw candle wicks
        const candleWickPlugin = {{
            id: 'candleWickPlugin',
            afterDatasetsDraw: function(chart) {{
                const ctx = chart.ctx;
                const yAxis = chart.scales.y;
                const xAxis = chart.scales.x;
                
                // Get the candlestick dataset (first dataset)
                const dataset = chart.data.datasets[0];
                if (!dataset) return;
                
                ctx.save();
                ctx.lineWidth = 3;
                
                dataset.data.forEach((d, i) => {{
                    const x = xAxis.getPixelForValue(d.x);
                    const high = yAxis.getPixelForValue(d.h);
                    const low = yAxis.getPixelForValue(d.l);
                    
                    // Set wick color based on candle color
                    const isBullish = d.c >= d.o;
                    ctx.strokeStyle = isBullish ? '#4ecca3' : '#e94560';
                    
                    // Draw wick (vertical line from low to high)
                    ctx.beginPath();
                    ctx.moveTo(x, low);
                    ctx.lineTo(x, high);
                    ctx.stroke();
                    
                    // Draw horizontal lines at High and Low
                    ctx.beginPath();
                    ctx.moveTo(x - 3, high);
                    ctx.lineTo(x + 3, high);
                    ctx.stroke();
                    
                    ctx.beginPath();
                    ctx.moveTo(x - 3, low);
                    ctx.lineTo(x + 3, low);
                    ctx.stroke();
                }});
                
                ctx.restore();
            }}
        }};
        
        // Register custom plugin
        Chart.register(candleWickPlugin);
        
        // Embedded OHLCV Data
        const dates = {json.dumps(dates)};
        const opens = {json.dumps(opens)};
        const highs = {json.dumps(highs)};
        const lows = {json.dumps(lows)};
        const closes = {json.dumps(closes)};
        
        // Embedded All Pattern Markers
        const allEntryMarkers = {json.dumps(all_entry_markers)};
        const allExitMarkers = {json.dumps(all_exit_markers)};
        
        // Embedded Pattern Stats
        const patternStats = {json.dumps(pattern_stats)};
        
        // Currently selected pattern
        let selectedPattern = 'all';
        
        // Reset zoom function
        function resetZoom() {{
            chart.resetZoom();
        }}
        
        // Select pattern function
        function selectPattern(patternId) {{
            selectedPattern = patternId;
            updateChartMarkers();
            
            // Update card selection
            document.querySelectorAll('.pattern-card').forEach(card => {{
                card.classList.remove('selected');
            }});
            if (patternId !== 'all' && patternId !== 'none') {{
                const card = document.getElementById('pattern-card-' + patternId);
                if (card) {{
                    card.classList.add('selected');
                }}
            }}
        }}
        
        // Update chart markers based on selection
        function updateChartMarkers() {{
            let entryData = [];
            let exitData = [];
            
            if (selectedPattern === 'all') {{
                // Show all patterns
                for (let pid in allEntryMarkers) {{
                    entryData = entryData.concat(allEntryMarkers[pid]);
                    exitData = exitData.concat(allExitMarkers[pid]);
                }}
            }} else if (selectedPattern !== 'none') {{
                // Show selected pattern only
                entryData = allEntryMarkers[selectedPattern] || [];
                exitData = allExitMarkers[selectedPattern] || [];
            }}
            
            chart.data.datasets[1].data = entryData.map(m => ({{x: m.x, y: m.y}}));
            chart.data.datasets[2].data = exitData.map(m => ({{x: m.x, y: m.y}}));
            chart.update();
        }}
        
        // Candlestick Chart
        const ctx = document.getElementById('candlestickChart').getContext('2d');
        
        // Create candlestick data with proper OHLC format for financial chart
        const candlestickData = [];
        for (let i = 0; i < dates.length; i++) {{
            candlestickData.push({{
                x: i,
                o: opens[i],
                h: highs[i],
                l: lows[i],
                c: closes[i]
            }});
        }}
        
        const chart = new Chart(ctx, {{
            type: 'candlestick',
            data: {{
                datasets: [
                    {{
                        label: 'Candlesticks',
                        data: candlestickData,
                        color: {{
                            up: '#4ecca3',
                            down: '#e94560',
                            unchanged: '#888'
                        }},
                        borderColor: {{
                            up: '#4ecca3',
                            down: '#e94560',
                            unchanged: '#888'
                        }},
                        order: 3
                    }},
                    {{
                        label: 'Pattern Entries',
                        data: [],
                        backgroundColor: '#ffd700',
                        borderColor: '#ffaa00',
                        borderWidth: 2,
                        pointRadius: 8,
                        pointHoverRadius: 10,
                        type: 'scatter',
                        order: 1
                    }},
                    {{
                        label: 'Pattern Exits',
                        data: [],
                        backgroundColor: '#ff6b6b',
                        borderColor: '#ff4444',
                        borderWidth: 2,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        type: 'scatter',
                        order: 2
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    zoom: {{
                        zoom: {{
                            wheel: {{
                                enabled: true
                            }},
                            pinch: {{
                                enabled: true
                            }},
                            mode: 'xy'
                        }},
                        pan: {{
                            enabled: true,
                            mode: 'xy'
                        }}
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                if (context.dataset.type === 'scatter') {{
                                    if (context.datasetIndex === 1) {{
                                        const markers = selectedPattern === 'all' 
                                            ? Object.values(allEntryMarkers).flat()
                                            : (allEntryMarkers[selectedPattern] || []);
                                        const marker = markers[context.dataIndex];
                                        if (marker) {{
                                            return [
                                                'Pattern Entry #' + marker.pattern,
                                                'Date: ' + marker.date,
                                                'Price: ' + marker.y.toFixed(2),
                                                'Outcome: ' + marker.outcome
                                            ];
                                        }}
                                    }} else {{
                                        const markers = selectedPattern === 'all'
                                            ? Object.values(allExitMarkers).flat()
                                            : (allExitMarkers[selectedPattern] || []);
                                        const marker = markers[context.dataIndex];
                                        if (marker) {{
                                            return [
                                                'Pattern Exit #' + marker.pattern,
                                                'Date: ' + marker.date,
                                                'Price: ' + marker.y.toFixed(2),
                                                'Outcome: ' + marker.outcome
                                            ];
                                        }}
                                    }}
                                }} else {{
                                    const d = context.raw;
                                    return [
                                        'Date: ' + dates[context.dataIndex],
                                        'Open: ' + d.o.toFixed(2),
                                        'High: ' + d.h.toFixed(2),
                                        'Low: ' + d.l.toFixed(2),
                                        'Close: ' + d.c.toFixed(2)
                                    ];
                                }}
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        type: 'linear',
                        ticks: {{
                            color: '#aaa',
                            callback: function(value) {{
                                if (value >= 0 && value < dates.length && Math.floor(value) === value) {{
                                    return dates[value].substring(0, 10);
                                }}
                                return '';
                            }},
                            maxTicksLimit: 15
                        }},
                        grid: {{
                            color: '#0f3460'
                        }},
                        title: {{
                            display: true,
                            text: 'Date',
                            color: '#aaa'
                        }}
                    }},
                    y: {{
                        ticks: {{
                            color: '#aaa'
                        }},
                        grid: {{
                            color: '#0f3460'
                        }},
                        title: {{
                            display: true,
                            text: 'Price',
                            color: '#aaa'
                        }}
                    }}
                }}
            }}
        }});
        
        // Display Pattern Cards
        const patternsGrid = document.getElementById('patternsGrid');
        
        patternStats.forEach((stats, index) => {{
            const card = document.createElement('div');
            card.className = 'pattern-card';
            card.id = 'pattern-card-' + stats.id;
            
            const successRate = stats.success_rate || 0;
            const avgMove = stats.avg_move || 0;
            const occCount = stats.occurrences || 0;
            const validationRate = stats.validation_success_rate || 0;
            
            // Format entry conditions HTML
            const entryConditionsHtml = stats.entry_conditions
                .map(c => `<div class="condition-item">• ${{c}}</div>`)
                .join('');
            
            // Format exit conditions HTML
            const exitConditionsHtml = stats.exit_conditions
                .map(c => `<div class="condition-item">• ${{c}}</div>`)
                .join('');
            
            card.innerHTML = `
                <h3>Pattern #` + stats.id + ` <small>(${{stats.direction}})</small></h3>
                <div class="stat">
                    <span class="stat-label">Training Success:</span>
                    <span class="stat-value ` + (successRate >= 50 ? 'positive' : 'negative') + `">` + successRate.toFixed(1) + `%</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Validation Success:</span>
                    <span class="stat-value ` + (validationRate >= 50 ? 'positive' : 'negative') + `">` + validationRate.toFixed(1) + `%</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Avg Move:</span>
                    <span class="stat-value ` + (avgMove >= 0 ? 'positive' : 'negative') + `">` + avgMove.toFixed(2) + `%</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Occurrences:</span>
                    <span class="stat-value">` + occCount + `</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Method:</span>
                    <span class="stat-value">` + stats.method + `</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Classification:</span>
                    <span class="stat-value">` + stats.classification + `</span>
                </div>
                <div class="conditions-section" id="conditions-` + stats.id + `">
                    <div class="conditions-title" onclick="toggleConditions(` + stats.id + `); event.stopPropagation();">📥 Entry Conditions (click to show)</div>
                    ` + entryConditionsHtml + `
                </div>
                <div class="conditions-section" id="exit-conditions-` + stats.id + `">
                    <div class="conditions-title" onclick="toggleConditions(` + stats.id + `); event.stopPropagation();">📤 Exit Conditions (click to show)</div>
                    ` + exitConditionsHtml + `
                </div>
            `;
            
            // Add click handler for selecting pattern
            card.onclick = function() {{
                document.getElementById('patternSelect').value = stats.id;
                selectPattern(stats.id);
            }};
            
            patternsGrid.appendChild(card);
        }});
        
        // Toggle conditions visibility
        function toggleConditions(patternId) {{
            const entrySection = document.getElementById('conditions-' + patternId);
            const exitSection = document.getElementById('exit-conditions-' + patternId);
            
            if (entrySection.classList.contains('visible')) {{
                entrySection.classList.remove('visible');
                exitSection.classList.remove('visible');
            }} else {{
                entrySection.classList.add('visible');
                exitSection.classList.add('visible');
            }}
        }}
        
        // Initialize with all patterns
        updateChartMarkers();
    </script>
</body>
</html>'''
    
    # Save the HTML file
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PATTERN_DASHBOARD.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Standalone dashboard generated: {output_path}")
    print(f"   Open this file directly in your browser - no server needed!")
    print(f"   Data range: {ohlcv_sample[0]['Date'][:10]} to {ohlcv_sample[-1]['Date'][:10]}")
    print(f"   Total OHLCV records: {len(ohlcv_data):,}")
    
    total_entries = sum(len(m) for m in all_entry_markers.values())
    total_exits = sum(len(m) for m in all_exit_markers.values())
    print(f"   Total entry markers: {total_entries}")
    print(f"   Total exit markers: {total_exits}")
    print(f"   Zoom enabled: Yes - use mouse wheel to zoom, drag to pan")
    print(f"   Pattern selection: Click on pattern cards to view specific pattern markers")

if __name__ == '__main__':
    generate_standalone_html()