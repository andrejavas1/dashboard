#!/usr/bin/env python3
"""
Fix chart markers to only show visible patterns
"""

# Read the dashboard HTML
with open('PATTERN_DASHBOARD.html', 'r', encoding='utf-8') as f:
    content = f.read()

# The issue is in the updateChartMarkers function - it's not properly respecting the filteredPatternIds
# Let's fix the updateChartMarkers function

old_update_function = '''        // Update chart markers based on selection
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

new_update_function = '''        // Update chart markers based on selection
        function updateChartMarkers() {
            let entryData = [];
            let exitData = [];
            
            if (selectedPattern === 'all') {
                // Show all patterns (respecting filters)
                for (let pid in allEntryMarkers) {
                    // Only include if pattern is visible (not hidden by filters)
                    const card = document.getElementById('pattern-card-' + pid);
                    if (card && !card.classList.contains('hidden')) {
                        entryData = entryData.concat(allEntryMarkers[pid]);
                        exitData = exitData.concat(allExitMarkers[pid]);
                    }
                }
            } else if (selectedPattern !== 'none') {
                // Show selected pattern only (if visible)
                const card = document.getElementById('pattern-card-' + selectedPattern);
                if (card && !card.classList.contains('hidden')) {
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

print("Chart markers fixed!")
print("Now the chart will only show markers from visible (non-hidden) pattern cards")