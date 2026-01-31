#!/usr/bin/env python3
"""
Add occurrences filter and sorting functionality to PATTERN_DASHBOARD.html
"""

# Read the dashboard HTML
with open('PATTERN_DASHBOARD.html', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add occurrences filter to the filter section
# Find the avgMoveFilter div and add occurrences filter after it
old_avg_move_filter = '''                <div class="filter-group">
                    <label for="avgMoveFilter">Avg Move ≥:</label>
                    <input type="number" id="avgMoveFilter" min="-100" max="100" value="-100" step="0.1" onchange="applyFilters()">
                    <span>%</span>
                </div>
                <button class="reset-btn" onclick="resetFilters()">Reset Filters</button>'''

new_avg_move_filter = '''                <div class="filter-group">
                    <label for="avgMoveFilter">Avg Move ≥:</label>
                    <input type="number" id="avgMoveFilter" min="-100" max="100" value="-100" step="0.1" onchange="applyFilters()">
                    <span>%</span>
                </div>
                <div class="filter-group">
                    <label for="occurrencesFilter">Occurrences ≥:</label>
                    <input type="number" id="occurrencesFilter" min="0" max="1000" value="0" step="1" onchange="applyFilters()">
                </div>
                <button class="reset-btn" onclick="resetFilters()">Reset Filters</button>'''

content = content.replace(old_avg_move_filter, new_avg_move_filter)

# 2. Add sorting controls to the filter section
# Add sorting dropdown after filter info
old_filter_info = '''            <div class="filter-info" id="filterInfo">Showing all 5141 patterns</div>
        </div>'''

new_filter_info = '''            <div class="filter-info" id="filterInfo">Showing all 5141 patterns</div>
            <div class="sort-controls">
                <label for="sortBy">Sort by:</label>
                <select id="sortBy" onchange="sortPatterns()">
                    <option value="id">Pattern ID</option>
                    <option value="success_rate">Training Success</option>
                    <option value="validation_success_rate">Validation Success</option>
                    <option value="avg_move">Avg Move</option>
                    <option value="occurrences">Occurrences</option>
                </select>
                <select id="sortOrder" onchange="sortPatterns()">
                    <option value="desc">Descending</option>
                    <option value="asc">Ascending</option>
                </select>
            </div>
        </div>'''

content = content.replace(old_filter_info, new_filter_info)

# 3. Add CSS for sort controls
old_filter_info_css = '''        .filter-info {
            color: #4ecca3;
            font-size: 13px;
            margin-top: 10px;
            font-weight: bold;
        }
        .pattern-card.hidden {
            display: none;
        }'''

new_filter_info_css = '''        .filter-info {
            color: #4ecca3;
            font-size: 13px;
            margin-top: 10px;
            font-weight: bold;
        }
        .sort-controls {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #0f3460;
        }
        .sort-controls label {
            color: #aaa;
            font-size: 12px;
            font-weight: bold;
        }
        .sort-controls select {
            background: #0f3460;
            color: #eee;
            border: 1px solid #4ecca3;
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 13px;
            cursor: pointer;
        }
        .sort-controls select:focus {
            outline: none;
            border-color: #ffd700;
        }
        .pattern-card.hidden {
            display: none;
        }'''

content = content.replace(old_filter_info_css, new_filter_info_css)

# 4. Update applyFilters function to include occurrences filter
old_apply_filters = '''        // Apply filters to pattern cards
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
        }'''

new_apply_filters = '''        // Apply filters to pattern cards
        function applyFilters() {
            const trainingSuccess = parseFloat(document.getElementById('trainingSuccessFilter').value) || 0;
            const validationSuccess = parseFloat(document.getElementById('validationSuccessFilter').value) || 0;
            const direction = document.getElementById('directionFilter').value;
            const avgMove = parseFloat(document.getElementById('avgMoveFilter').value) || -100;
            const occurrences = parseFloat(document.getElementById('occurrencesFilter').value) || 0;
            
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
                
                // Check occurrences
                if (stats.occurrences < occurrences) {
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
            
            // Re-sort patterns after filtering
            sortPatterns();
        }'''

content = content.replace(old_apply_filters, new_apply_filters)

# 5. Update resetFilters function to reset occurrences filter
old_reset_filters = '''        // Reset all filters
        function resetFilters() {
            document.getElementById('trainingSuccessFilter').value = 0;
            document.getElementById('validationSuccessFilter').value = 0;
            document.getElementById('directionFilter').value = 'all';
            document.getElementById('avgMoveFilter').value = -100;
            applyFilters();
        }'''

new_reset_filters = '''        // Reset all filters
        function resetFilters() {
            document.getElementById('trainingSuccessFilter').value = 0;
            document.getElementById('validationSuccessFilter').value = 0;
            document.getElementById('directionFilter').value = 'all';
            document.getElementById('avgMoveFilter').value = -100;
            document.getElementById('occurrencesFilter').value = 0;
            document.getElementById('sortBy').value = 'id';
            document.getElementById('sortOrder').value = 'desc';
            applyFilters();
        }'''

content = content.replace(old_reset_filters, new_reset_filters)

# 6. Add sortPatterns function before applyFilters
sort_function = '''
        // Sort pattern cards
        function sortPatterns() {
            const sortBy = document.getElementById('sortBy').value;
            const sortOrder = document.getElementById('sortOrder').value;
            const patternsGrid = document.getElementById('patternsGrid');
            
            // Get all pattern cards
            const cards = Array.from(patternsGrid.children);
            
            // Sort cards based on the selected criteria
            cards.sort((a, b) => {
                const idA = parseInt(a.id.replace('pattern-card-', ''));
                const idB = parseInt(b.id.replace('pattern-card-', ''));
                
                const statsA = patternStats.find(s => s.id === idA);
                const statsB = patternStats.find(s => s.id === idB);
                
                if (!statsA || !statsB) return 0;
                
                let valueA, valueB;
                
                switch(sortBy) {
                    case 'id':
                        valueA = statsA.id;
                        valueB = statsB.id;
                        break;
                    case 'success_rate':
                        valueA = statsA.success_rate;
                        valueB = statsB.success_rate;
                        break;
                    case 'validation_success_rate':
                        valueA = statsA.validation_success_rate;
                        valueB = statsB.validation_success_rate;
                        break;
                    case 'avg_move':
                        valueA = statsA.avg_move;
                        valueB = statsB.avg_move;
                        break;
                    case 'occurrences':
                        valueA = statsA.occurrences;
                        valueB = statsB.occurrences;
                        break;
                    default:
                        return 0;
                }
                
                if (sortOrder === 'desc') {
                    return valueB - valueA;
                } else {
                    return valueA - valueB;
                }
            });
            
            // Re-append cards in sorted order
            cards.forEach(card => patternsGrid.appendChild(card));
        }
'''

# Insert sort function before applyFilters
content = content.replace('        // Apply filters to pattern cards', sort_function + '        // Apply filters to pattern cards')

# Write the modified content back
with open('PATTERN_DASHBOARD.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Occurrences filter and sorting functionality added successfully!")
print("New features:")
print("  - Occurrences filter (minimum number of occurrences)")
print("  - Sort by: Pattern ID, Training Success, Validation Success, Avg Move, Occurrences")
print("  - Sort order: Ascending or Descending")
print("  - Sorting works with filters applied")