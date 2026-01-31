#!/usr/bin/env python3
"""
Add filters to the PATTERN_DASHBOARD.html file.
Filters:
- Training success rate
- Validation success rate
- Direction (long/short)
- Average move
"""

# Read the dashboard HTML
with open('PATTERN_DASHBOARD.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Define the filter section HTML to add after the pattern-selector div
filter_section = '''
        <div class="filter-section">
            <h3>🔍 Pattern Filters</h3>
            <div class="filter-row">
                <div class="filter-group">
                    <label for="trainingSuccessFilter">Training Success ≥:</label>
                    <input type="number" id="trainingSuccessFilter" min="0" max="100" value="0" step="1" onchange="applyFilters()">
                    <span>%</span>
                </div>
                <div class="filter-group">
                    <label for="validationSuccessFilter">Validation Success ≥:</label>
                    <input type="number" id="validationSuccessFilter" min="0" max="100" value="0" step="1" onchange="applyFilters()">
                    <span>%</span>
                </div>
                <div class="filter-group">
                    <label for="directionFilter">Direction:</label>
                    <select id="directionFilter" onchange="applyFilters()">
                        <option value="all">All</option>
                        <option value="long">Long Only</option>
                        <option value="short">Short Only</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="avgMoveFilter">Avg Move ≥:</label>
                    <input type="number" id="avgMoveFilter" min="-100" max="100" value="-100" step="0.1" onchange="applyFilters()">
                    <span>%</span>
                </div>
                <button class="reset-btn" onclick="resetFilters()">Reset Filters</button>
            </div>
            <div class="filter-info" id="filterInfo">Showing all 5141 patterns</div>
        </div>
'''

# Add CSS styles for filters
filter_styles = '''
        .filter-section {
            background: #16213e;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .filter-section h3 {
            color: #4ecca3;
            margin-bottom: 15px;
            font-size: 16px;
        }
        .filter-row {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: flex-end;
        }
        .filter-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .filter-group label {
            color: #aaa;
            font-size: 12px;
            font-weight: bold;
        }
        .filter-group input,
        .filter-group select {
            background: #0f3460;
            color: #eee;
            border: 1px solid #4ecca3;
            padding: 8px 10px;
            border-radius: 4px;
            font-size: 14px;
            cursor: pointer;
        }
        .filter-group input:focus,
        .filter-group select:focus {
            outline: none;
            border-color: #ffd700;
        }
        .filter-group span {
            color: #aaa;
            font-size: 12px;
        }
        .reset-btn {
            background: #e94560;
            color: #eee;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            margin-left: auto;
            transition: all 0.3s;
        }
        .reset-btn:hover {
            background: #ff6b6b;
            transform: translateY(-2px);
        }
        .filter-info {
            color: #4ecca3;
            font-size: 13px;
            margin-top: 10px;
            font-weight: bold;
        }
        .pattern-card.hidden {
            display: none;
        }
'''

# Insert CSS styles before </style>
content = content.replace('</style>', filter_styles + '\n    </style>')

# Insert filter section after the pattern-selector div (after line 204)
# Find the pattern-selector closing div and insert after it
pattern_selector_end = '        </div>\n        \n        <div class="chart-container">'
content = content.replace(pattern_selector_end, '        </div>\n' + filter_section + '\n        \n        <div class="chart-container">')

# Add JavaScript filter functions before the closing </script>
filter_js = '''
        // Filter state
        let filteredPatternIds = new Set();
        
        // Apply filters to pattern cards
        function applyFilters() {
            const trainingSuccess = parseFloat(document.getElementById('trainingSuccessFilter').value) || 0;
            const validationSuccess = parseFloat(document.getElementById('validationSuccessFilter').value) || 0;
            const direction = document.getElementById('directionFilter').value;
            const avgMove = parseFloat(document.getElementById('avgMoveFilter').value) || -100;
            
            filteredPatternIds.clear();
            let visibleCount = 0;
            
            patternStats.forEach((stats, index) => {
                const card = document.getElementById('pattern-card-' + stats.id);
                if (!card) return;
                
                let matches = true;
                
                // Check training success
                if (stats.success_rate < trainingSuccess) {
                    matches = false;
                }
                
                // Check validation success
                if (stats.validation_success_rate < validationSuccess) {
                    matches = false;
                }
                
                // Check direction
                if (direction !== 'all' && stats.direction !== direction) {
                    matches = false;
                }
                
                // Check average move
                if (stats.avg_move < avgMove) {
                    matches = false;
                }
                
                if (matches) {
                    card.classList.remove('hidden');
                    filteredPatternIds.add(stats.id.toString());
                    visibleCount++;
                } else {
                    card.classList.add('hidden');
                }
            });
            
            // Update filter info
            document.getElementById('filterInfo').textContent = 'Showing ' + visibleCount + ' of ' + patternStats.length + ' patterns';
            
            // Update chart to show only filtered patterns
            updateChartMarkers();
        }
        
        // Reset all filters
        function resetFilters() {
            document.getElementById('trainingSuccessFilter').value = 0;
            document.getElementById('validationSuccessFilter').value = 0;
            document.getElementById('directionFilter').value = 'all';
            document.getElementById('avgMoveFilter').value = -100;
            applyFilters();
        }
'''

# Insert filter JS before the closing </script>
content = content.replace('        // Initialize with all patterns\n        updateChartMarkers();\n    </script>', 
    filter_js + '\n        // Initialize with all patterns\n        updateChartMarkers();\n    </script>')

# Update the updateChartMarkers function to respect filtered patterns
old_update_function = '''        // Update chart markers based on selection
        function updateChartMarkers() {
            let entryData = [];
            let exitData = [];
            
            if (selectedPattern === 'all') {
                // Show all patterns
                for (let pid in allEntryMarkers) {
                    entryData = entryData.concat(allEntryMarkers[pid]);
                    exitData = exitData.concat(allExitMarkers[pid]);
                }
            } else if (selectedPattern !== 'none') {
                // Show selected pattern only
                entryData = allEntryMarkers[selectedPattern] || [];
                exitData = allExitMarkers[selectedPattern] || [];
            }
            
            chart.data.datasets[1].data = entryData.map(m => ({x: m.x, y: m.y}));
            chart.data.datasets[2].data = exitData.map(m => ({x: m.x, y: m.y}));
            chart.update();
        }'''

new_update_function = '''        // Update chart markers based on selection
        function updateChartMarkers() {
            let entryData = [];
            let exitData = [];
            
            if (selectedPattern === 'all') {
                // Show all patterns (respecting filters)
                for (let pid in allEntryMarkers) {
                    // Only include if pattern is not filtered out
                    if (filteredPatternIds.size === 0 || filteredPatternIds.has(pid)) {
                        entryData = entryData.concat(allEntryMarkers[pid]);
                        exitData = exitData.concat(allExitMarkers[pid]);
                    }
                }
            } else if (selectedPattern !== 'none') {
                // Show selected pattern only (if not filtered out)
                if (filteredPatternIds.size === 0 || filteredPatternIds.has(selectedPattern)) {
                    entryData = allEntryMarkers[selectedPattern] || [];
                    exitData = allExitMarkers[selectedPattern] || [];
                }
            }
            
            chart.data.datasets[1].data = entryData.map(m => ({x: m.x, y: m.y}));
            chart.data.datasets[2].data = exitData.map(m => ({x: m.x, y: m.y}));
            chart.update();
        }'''

content = content.replace(old_update_function, new_update_function)

# Write the modified content back
with open('PATTERN_DASHBOARD.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Filters added successfully to PATTERN_DASHBOARD.html")
print("The dashboard now includes:")
print("  - Training Success Rate filter")
print("  - Validation Success Rate filter")
print("  - Direction (Long/Short) filter")
print("  - Average Move filter")
print("  - Reset Filters button")