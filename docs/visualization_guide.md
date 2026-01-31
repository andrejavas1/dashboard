# Pattern Visualization Library - User Guide

## Overview

The Pattern Visualization Library (`src/pattern_visualization.py`) provides comprehensive tools for displaying trading patterns, their conditions, performance results, and historical backtesting visualizations. This library is designed for analyzing and monitoring pattern performance with interactive features.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Classes](#core-classes)
4. [API Reference](#api-reference)
5. [Examples](#examples)
6. [Visualization Types](#visualization-types)
7. [Backtesting](#backtesting)
8. [Advanced Usage](#advanced-usage)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Installation

### Requirements

```bash
pip install pandas numpy matplotlib seaborn
```

### Import

```python
from src.pattern_visualization import (
    PatternVisualization,
    PatternMetrics,
    BacktestResult,
    VisualizationType,
    visualize_pattern,
    compare_patterns
)
```

---

## Quick Start

### Basic Usage

```python
import pandas as pd
from src.pattern_visualization import PatternVisualization

# Load historical data
data = pd.read_csv('data/features_matrix.csv', index_col='Date', parse_dates=True)

# Load patterns
import json
with open('data/final_portfolio.json', 'r') as f:
    patterns = json.load(f)

# Create visualizer
viz = PatternVisualization(data=data)

# Create all visualizations
paths = viz.create_all_visualizations(patterns)

# Export summary
viz.export_visualization_summary(paths)
```

### Quick Single Pattern Visualization

```python
from src.pattern_visualization import visualize_pattern

paths = visualize_pattern(
    pattern=patterns[0],
    pattern_id=0,
    data=data,
    output_dir='charts'
)
```

### Quick Pattern Comparison

```python
from src.pattern_visualization import compare_patterns

paths = compare_patterns(
    patterns=patterns[:5],
    data=data,
    output_dir='charts'
)
```

---

## Core Classes

### PatternVisualization

Main class for creating pattern visualizations.

#### Constructor

```python
PatternVisualization(data: Optional[pd.DataFrame] = None, 
                    config: Optional[Dict] = None)
```

**Parameters:**
- `data`: Historical price and feature data (DataFrame with Date index)
- `config`: Configuration dictionary with options:
  - `charts_dir`: Output directory for charts (default: 'charts')
  - `reports_dir`: Output directory for reports (default: 'reports')
  - `dpi`: Image resolution (default: 150)
  - `figsize`: Figure size tuple (default: (14, 8))
  - `style`: Matplotlib style (default: 'seaborn-v0_8-darkgrid')

#### Methods

| Method | Description |
|--------|-------------|
| `load_data(data_path)` | Load historical data from CSV file |
| `load_patterns(patterns_path)` | Load patterns from JSON file |
| `add_pattern(pattern)` | Add a pattern to the library |
| `extract_pattern_metrics(pattern, pattern_id)` | Extract metrics from pattern |
| `find_pattern_occurrences(pattern)` | Find historical occurrences |
| `create_overview_chart(pattern, pattern_id, save)` | Create overview chart |
| `create_occurrences_chart(pattern, pattern_id, save)` | Create occurrences chart |
| `create_statistics_chart(pattern, pattern_id, save)` | Create statistics chart |
| `create_equity_curve_chart(pattern, pattern_id, save)` | Create equity curve |
| `create_comparison_chart(patterns, save)` | Create comparison chart |
| `create_performance_heatmap(patterns, save)` | Create performance heatmap |
| `create_all_visualizations(patterns)` | Create all visualizations |
| `export_visualization_summary(paths)` | Export summary to JSON |
| `run_backtest(pattern, pattern_id, initial_capital, position_size)` | Run backtest |

### PatternMetrics

Data class for pattern performance metrics.

#### Attributes

```python
@dataclass
class PatternMetrics:
    pattern_id: int              # Pattern identifier
    name: str                    # Pattern name
    direction: str               # 'long' or 'short'
    success_rate: float          # Validation success rate (%)
    occurrences: int             # Number of occurrences
    avg_move: float              # Average move (%)
    avg_time: float              # Average time to target (days)
    false_positive_rate: float   # False positive rate (%)
    composite_score: float       # Overall quality score
    classification: str          # 'ROBUST', 'MEDIUM', or 'FAILED'
    validation_ratio: float      # Validation/training ratio
    training_success_rate: float # Training success rate (%)
```

### BacktestResult

Data class for backtesting results.

#### Attributes

```python
@dataclass
class BacktestResult:
    pattern_id: int              # Pattern identifier
    initial_capital: float       # Starting capital
    final_capital: float         # Ending capital
    total_return: float           # Total return (%)
    total_trades: int            # Number of trades
    winning_trades: int          # Number of winning trades
    losing_trades: int           # Number of losing trades
    win_rate: float              # Win rate (%)
    max_drawdown: float          # Maximum drawdown (%)
    sharpe_ratio: float          # Sharpe ratio
    equity_curve: pd.Series      # Equity curve over time
    trade_history: pd.DataFrame  # Detailed trade history
```

### VisualizationType

Enumeration of available visualization types.

```python
class VisualizationType(Enum):
    OVERVIEW = "overview"
    OCCURRENCES = "occurrences"
    STATISTICS = "statistics"
    EQUITY_CURVE = "equity_curve"
    BACKTEST = "backtest"
    COMPARISON = "comparison"
    HEATMAP = "heatmap"
    SCATTER = "scatter"
    DISTRIBUTION = "distribution"
    PERFORMANCE = "performance"
```

---

## API Reference

### PatternVisualization.load_data()

Load historical data from a CSV file.

```python
def load_data(self, data_path: str) -> pd.DataFrame:
    """Load historical data from file.
    
    Args:
        data_path: Path to CSV file with Date index
        
    Returns:
        DataFrame with historical data
    """
```

**Example:**
```python
viz = PatternVisualization()
data = viz.load_data('data/features_matrix.csv')
```

### PatternVisualization.load_patterns()

Load patterns from a JSON file.

```python
def load_patterns(self, patterns_path: str) -> List[Dict]:
    """Load patterns from JSON file.
    
    Args:
        patterns_path: Path to patterns JSON file
        
    Returns:
        List of pattern dictionaries
    """
```

**Example:**
```python
patterns = viz.load_patterns('data/final_portfolio.json')
```

### PatternVisualization.find_pattern_occurrences()

Find all historical occurrences of a pattern.

```python
def find_pattern_occurrences(self, pattern: Dict) -> pd.DataFrame:
    """Find all historical occurrences of a pattern.
    
    Args:
        pattern: Pattern dictionary
        
    Returns:
        DataFrame with occurrence dates and outcomes
    """
```

**Returns:**
DataFrame with columns:
- `outcome`: Pattern outcome
- `actual_move`: Actual price movement (%)
- `target_reached`: Whether target was reached (boolean)
- `time_to_target`: Days to reach target

**Example:**
```python
occurrences = viz.find_pattern_occurrences(patterns[0])
print(f"Found {len(occurrences)} occurrences")
print(f"Success rate: {occurrences['target_reached'].mean() * 100:.1f}%")
```

### PatternVisualization.create_overview_chart()

Create overview chart for a pattern.

```python
def create_overview_chart(self, pattern: Dict, pattern_id: int, 
                         save: bool = True) -> Optional[str]:
    """Create overview chart for a pattern.
    
    Args:
        pattern: Pattern dictionary
        pattern_id: Pattern identifier
        save: Whether to save the chart
        
    Returns:
        Path to saved chart or None
    """
```

**Chart Contents:**
- Success rate comparison (training vs validation)
- Total occurrences display
- Pattern conditions list
- Expected performance metrics

**Example:**
```python
path = viz.create_overview_chart(patterns[0], pattern_id=0)
print(f"Overview chart saved to: {path}")
```

### PatternVisualization.create_occurrences_chart()

Create chart showing all historical occurrences.

```python
def create_occurrences_chart(self, pattern: Dict, pattern_id: int,
                             save: bool = True) -> Optional[str]:
    """Create chart showing all historical occurrences.
    
    Args:
        pattern: Pattern dictionary
        pattern_id: Pattern identifier
        save: Whether to save the chart
        
    Returns:
        Path to saved chart or None
    """
```

**Chart Contents:**
- Price chart with occurrence markers
- Successful trades (green circles)
- Failed trades (red X marks)
- Volume chart below price

**Example:**
```python
path = viz.create_occurrences_chart(patterns[0], pattern_id=0)
```

### PatternVisualization.create_statistics_chart()

Create statistical dashboard for a pattern.

```python
def create_statistics_chart(self, pattern: Dict, pattern_id: int,
                            save: bool = True) -> Optional[str]:
    """Create statistical dashboard for a pattern.
    
    Args:
        pattern: Pattern dictionary
        pattern_id: Pattern identifier
        save: Whether to save the chart
        
    Returns:
        Path to saved chart or None
    """
```

**Chart Contents:**
- Outcome distribution pie chart
- Move distribution histogram
- Time to target distribution
- Monthly success rate bar chart

**Example:**
```python
path = viz.create_statistics_chart(patterns[0], pattern_id=0)
```

### PatternVisualization.run_backtest()

Run historical backtest for a pattern.

```python
def run_backtest(self, pattern: Dict, pattern_id: int,
                 initial_capital: float = 10000.0,
                 position_size: float = 0.1) -> BacktestResult:
    """Run historical backtest for a pattern.
    
    Args:
        pattern: Pattern dictionary
        pattern_id: Pattern identifier
        initial_capital: Starting capital
        position_size: Percentage of capital per trade (0-1)
        
    Returns:
        BacktestResult object with detailed results
    """
```

**Example:**
```python
result = viz.run_backtest(
    pattern=patterns[0],
    pattern_id=0,
    initial_capital=10000,
    position_size=0.1
)

print(f"Total Return: {result.total_return:.2f}%")
print(f"Win Rate: {result.win_rate:.1f}%")
print(f"Max Drawdown: {result.max_drawdown:.2f}%")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
```

### PatternVisualization.create_equity_curve_chart()

Create equity curve visualization.

```python
def create_equity_curve_chart(self, pattern: Dict, pattern_id: int,
                               save: bool = True) -> Optional[str]:
    """Create equity curve visualization for a pattern.
    
    Args:
        pattern: Pattern dictionary
        pattern_id: Pattern identifier
        save: Whether to save the chart
        
    Returns:
        Path to saved chart or None
    """
```

**Chart Contents:**
- Pattern strategy equity curve
- Buy-and-hold benchmark
- Key metrics display (total return, win rate, max drawdown, Sharpe ratio)

**Example:**
```python
path = viz.create_equity_curve_chart(patterns[0], pattern_id=0)
```

### PatternVisualization.create_comparison_chart()

Create comparison chart for multiple patterns.

```python
def create_comparison_chart(self, patterns: List[Dict], 
                            save: bool = True) -> Optional[str]:
    """Create comparison chart for multiple patterns.
    
    Args:
        patterns: List of pattern dictionaries
        save: Whether to save the chart
        
    Returns:
        Path to saved chart or None
    """
```

**Chart Contents:**
- Success rate comparison bar chart
- Frequency comparison bar chart
- Quality comparison bar chart
- Success rate vs frequency scatter plot

**Example:**
```python
path = viz.create_comparison_chart(patterns[:10])
```

### PatternVisualization.create_performance_heatmap()

Create performance heatmap for multiple patterns.

```python
def create_performance_heatmap(self, patterns: List[Dict],
                               save: bool = True) -> Optional[str]:
    """Create performance heatmap for multiple patterns.
    
    Args:
        patterns: List of pattern dictionaries
        save: Whether to save the chart
        
    Returns:
        Path to saved chart or None
    """
```

**Chart Contents:**
- Heatmap showing normalized performance metrics
- Metrics: Success Rate, Occurrences, Avg Move, Win Rate, Composite Score, Validation Ratio

**Example:**
```python
path = viz.create_performance_heatmap(patterns[:10])
```

### PatternVisualization.create_all_visualizations()

Create all visualizations for loaded patterns.

```python
def create_all_visualizations(self, patterns: Optional[List[Dict]] = None) -> Dict[str, List[str]]:
    """Create all visualizations for loaded patterns.
    
    Args:
        patterns: Optional list of patterns (uses loaded patterns if None)
        
    Returns:
        Dictionary mapping pattern IDs to visualization paths
    """
```

**Example:**
```python
paths = viz.create_all_visualizations(patterns)

# Access paths for specific pattern
pattern_0_paths = paths.get('pattern_0', [])
print(f"Pattern 0 visualizations: {pattern_0_paths}")

# Access comparison charts
comparison_paths = paths.get('comparison', [])
print(f"Comparison charts: {comparison_paths}")
```

### PatternVisualization.export_visualization_summary()

Export summary of all visualizations to JSON.

```python
def export_visualization_summary(self, paths: Dict[str, List[str]]) -> str:
    """Export summary of all visualizations to JSON.
    
    Args:
        paths: Dictionary of visualization paths
        
    Returns:
        Path to saved summary file
    """
```

**Example:**
```python
summary_path = viz.export_visualization_summary(paths)
print(f"Summary saved to: {summary_path}")
```

---

## Examples

### Example 1: Complete Workflow

```python
import pandas as pd
import json
from src.pattern_visualization import PatternVisualization

# Step 1: Load data and patterns
data = pd.read_csv('data/features_matrix.csv', index_col='Date', parse_dates=True)
with open('data/final_portfolio.json', 'r') as f:
    patterns = json.load(f)

# Step 2: Initialize visualizer
viz = PatternVisualization(data=data)

# Step 3: Create all visualizations
paths = viz.create_all_visualizations(patterns)

# Step 4: Export summary
summary_path = viz.export_visualization_summary(paths)

# Step 5: Print results
print(f"Created visualizations for {len([k for k in paths.keys() if k.startswith('pattern_')])} patterns")
print(f"Total visualizations: {sum(len(v) for v in paths.values())}")
```

### Example 2: Analyze Specific Pattern

```python
from src.pattern_visualization import PatternVisualization

# Load data
viz = PatternVisualization()
data = viz.load_data('data/features_matrix.csv')

# Load patterns
patterns = viz.load_patterns('data/final_portfolio.json')

# Analyze first pattern
pattern = patterns[0]
pattern_id = 0

# Extract metrics
metrics = viz.extract_pattern_metrics(pattern, pattern_id)
print(f"Pattern: {metrics.name}")
print(f"Success Rate: {metrics.success_rate:.1f}%")
print(f"Occurrences: {metrics.occurrences}")
print(f"Classification: {metrics.classification}")

# Find occurrences
occurrences = viz.find_pattern_occurrences(pattern)
print(f"\nHistorical Occurrences: {len(occurrences)}")
print(f"Success Rate: {occurrences['target_reached'].mean() * 100:.1f}%")

# Run backtest
backtest = viz.run_backtest(pattern, pattern_id)
print(f"\nBacktest Results:")
print(f"Total Return: {backtest.total_return:.2f}%")
print(f"Win Rate: {backtest.win_rate:.1f}%")
print(f"Max Drawdown: {backtest.max_drawdown:.2f}%")

# Create visualizations
viz.create_overview_chart(pattern, pattern_id)
viz.create_occurrences_chart(pattern, pattern_id)
viz.create_statistics_chart(pattern, pattern_id)
viz.create_equity_curve_chart(pattern, pattern_id)
```

### Example 3: Compare Top Patterns

```python
from src.pattern_visualization import PatternVisualization

# Load data and patterns
viz = PatternVisualization()
data = viz.load_data('data/features_matrix.csv')
patterns = viz.load_patterns('data/final_portfolio.json')

# Sort patterns by composite score
sorted_patterns = sorted(patterns, 
                        key=lambda p: p.get('composite_score', 0), 
                        reverse=True)

# Take top 10 patterns
top_patterns = sorted_patterns[:10]

# Create comparison visualizations
viz.create_comparison_chart(top_patterns)
viz.create_performance_heatmap(top_patterns)

# Print comparison
print("Top 10 Patterns by Composite Score:")
for i, pattern in enumerate(top_patterns):
    metrics = viz.extract_pattern_metrics(pattern, i)
    print(f"{i+1}. {metrics.name}: Score={metrics.composite_score:.2f}, "
          f"Success={metrics.success_rate:.1f}%, "
          f"Freq={metrics.occurrences}")
```

### Example 4: Custom Configuration

```python
from src.pattern_visualization import PatternVisualization

# Custom configuration
config = {
    'charts_dir': 'my_charts',
    'reports_dir': 'my_reports',
    'dpi': 200,
    'figsize': (16, 10),
    'style': 'seaborn-v0_8-whitegrid'
}

# Initialize with custom config
viz = PatternVisualization(data=data, config=config)

# Create visualizations with custom settings
paths = viz.create_all_visualizations(patterns)
```

### Example 5: Interactive Pattern Exploration

```python
from src.pattern_visualization import PatternVisualization

viz = PatternVisualization()
data = viz.load_data('data/features_matrix.csv')
patterns = viz.load_patterns('data/final_portfolio.json')

# Explore patterns interactively
for i, pattern in enumerate(patterns):
    metrics = viz.extract_pattern_metrics(pattern, i)
    
    # Only visualize robust patterns
    if metrics.classification == 'ROBUST' and metrics.success_rate >= 80:
        print(f"\nVisualizing Pattern {i}: {metrics.name}")
        print(f"  Success Rate: {metrics.success_rate:.1f}%")
        print(f"  Occurrences: {metrics.occurrences}")
        
        # Create visualizations
        viz.create_overview_chart(pattern, i)
        viz.create_equity_curve_chart(pattern, i)
```

---

## Visualization Types

### Overview Chart

Shows a comprehensive summary of pattern performance.

**Components:**
- Success rate comparison (training vs validation)
- Total occurrences count
- Pattern conditions list
- Expected performance metrics

**Use Case:** Quick assessment of pattern quality and characteristics

### Occurrences Chart

Displays all historical occurrences on a price chart.

**Components:**
- Price line chart
- Successful trade markers (green circles)
- Failed trade markers (red X)
- Volume chart

**Use Case:** Visualizing when patterns trigger and their outcomes

### Statistics Chart

Provides statistical analysis of pattern performance.

**Components:**
- Outcome distribution pie chart
- Move distribution histogram
- Time to target distribution
- Monthly success rate

**Use Case:** Understanding pattern behavior across different conditions

### Equity Curve Chart

Shows simulated trading performance over time.

**Components:**
- Pattern strategy equity curve
- Buy-and-hold benchmark
- Key metrics (total return, win rate, max drawdown, Sharpe ratio)

**Use Case:** Evaluating historical trading performance

### Comparison Chart

Compares multiple patterns side-by-side.

**Components:**
- Success rate comparison
- Frequency comparison
- Quality comparison
- Success rate vs frequency scatter plot

**Use Case:** Selecting the best patterns from a portfolio

### Performance Heatmap

Shows normalized performance metrics across patterns.

**Components:**
- Heatmap with color-coded performance
- Metrics: Success Rate, Occurrences, Avg Move, Win Rate, Composite Score, Validation Ratio

**Use Case:** Identifying patterns with balanced performance

---

## Backtesting

### Running a Backtest

```python
result = viz.run_backtest(
    pattern=patterns[0],
    pattern_id=0,
    initial_capital=10000,
    position_size=0.1
)
```

### Backtest Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `pattern` | Pattern dictionary | Required |
| `pattern_id` | Pattern identifier | Required |
| `initial_capital` | Starting capital | 10000.0 |
| `position_size` | Position size (0-1) | 0.1 |

### Backtest Results

```python
# Access results
print(f"Initial Capital: ${result.initial_capital:,.2f}")
print(f"Final Capital: ${result.final_capital:,.2f}")
print(f"Total Return: {result.total_return:.2f}%")
print(f"Total Trades: {result.total_trades}")
print(f"Winning Trades: {result.winning_trades}")
print(f"Losing Trades: {result.losing_trades}")
print(f"Win Rate: {result.win_rate:.1f}%")
print(f"Max Drawdown: {result.max_drawdown:.2f}%")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")

# Access equity curve
result.equity_curve.plot()

# Access trade history
print(result.trade_history.head())
```

### Custom Backtest Strategy

```python
# Run backtests with different position sizes
for pos_size in [0.05, 0.1, 0.2, 0.3]:
    result = viz.run_backtest(
        pattern=patterns[0],
        pattern_id=0,
        initial_capital=10000,
        position_size=pos_size
    )
    print(f"Position Size {pos_size*100}%: "
          f"Return={result.total_return:.2f}%, "
          f"Drawdown={result.max_drawdown:.2f}%")
```

---

## Advanced Usage

### Filtering Patterns

```python
# Filter by classification
robust_patterns = [p for p in patterns 
                   if p.get('classification') == 'ROBUST']

# Filter by success rate
high_success = [p for p in patterns 
                if p.get('validation_success_rate', 0) >= 80]

# Filter by occurrences
frequent = [p for p in patterns 
            if p.get('validation_occurrences', 0) >= 50]
```

### Custom Metrics Calculation

```python
# Calculate custom metrics
for i, pattern in enumerate(patterns):
    metrics = viz.extract_pattern_metrics(pattern, i)
    
    # Custom score: success rate weighted by frequency
    custom_score = metrics.success_rate * (1 + np.log(metrics.occurrences + 1) / 10)
    
    print(f"Pattern {i}: Custom Score = {custom_score:.2f}")
```

### Batch Processing

```python
# Process patterns in batches
batch_size = 10
for i in range(0, len(patterns), batch_size):
    batch = patterns[i:i+batch_size]
    paths = viz.create_all_visualizations(batch)
    print(f"Processed batch {i//batch_size + 1}")
```

### Export to Different Formats

```python
import matplotlib.pyplot as plt

# Create chart without saving
viz.create_overview_chart(patterns[0], 0, save=False)
plt.savefig('custom_output.pdf', format='pdf')
plt.savefig('custom_output.svg', format='svg')
plt.close()
```

### Interactive Exploration with Pandas

```python
# Create DataFrame of all pattern metrics
metrics_data = []
for i, pattern in enumerate(patterns):
    metrics = viz.extract_pattern_metrics(pattern, i)
    metrics_data.append({
        'id': metrics.pattern_id,
        'name': metrics.name,
        'success_rate': metrics.success_rate,
        'occurrences': metrics.occurrences,
        'composite_score': metrics.composite_score,
        'classification': metrics.classification
    })

df = pd.DataFrame(metrics_data)

# Analyze
print(df.describe())
print(df.groupby('classification').mean())

# Filter and sort
top_patterns = df[df['success_rate'] >= 80].sort_values(
    'composite_score', ascending=False
)
print(top_patterns.head(10))
```

---

## Best Practices

### 1. Data Preparation

- Ensure data has a Date index
- Include all required features for pattern conditions
- Validate data quality before visualization

```python
# Check data
print(data.info())
print(data.describe())
print(data.isnull().sum())
```

### 2. Pattern Validation

- Validate pattern structure before visualization
- Handle nested pattern structures correctly
- Check for required fields

```python
def validate_pattern(pattern):
    """Validate pattern structure."""
    if 'pattern' in pattern:
        pattern_data = pattern['pattern']
    else:
        pattern_data = pattern
    
    required = ['conditions', 'direction', 'label_col']
    for field in required:
        if field not in pattern_data:
            raise ValueError(f"Missing required field: {field}")
    
    return True
```

### 3. Memory Management

- Process patterns in batches for large portfolios
- Clear cache when not needed
- Use appropriate figure sizes

```python
# Clear occurrences cache
viz.occurrences_cache.clear()

# Close all figures
plt.close('all')
```

### 4. Performance Optimization

- Cache occurrences when processing multiple visualizations
- Use vectorized operations for data manipulation
- Limit the number of patterns in comparison charts

```python
# Cache occurrences
occurrences = viz.find_pattern_occurrences(pattern)
viz.occurrences_cache[pattern_id] = occurrences
```

### 5. Error Handling

- Handle missing data gracefully
- Provide meaningful error messages
- Log warnings for non-critical issues

```python
try:
    path = viz.create_equity_curve_chart(pattern, pattern_id)
except ValueError as e:
    logger.warning(f"Could not create equity curve: {e}")
```

---

## Troubleshooting

### Issue: "Data not loaded"

**Problem:** Attempting to visualize patterns without loading data first.

**Solution:**
```python
viz = PatternVisualization()
viz.load_data('data/features_matrix.csv')
```

### Issue: "No occurrences found"

**Problem:** Pattern conditions don't match any historical data.

**Solutions:**
1. Check if features exist in data
2. Verify condition values are reasonable
3. Review pattern conditions

```python
# Check features
print(data.columns.tolist())

# Check conditions
pattern_data = pattern.get('pattern', pattern)
print(pattern_data['conditions'])
```

### Issue: Charts not saving

**Problem:** Output directory doesn't exist or no write permissions.

**Solution:**
```python
# Create directories
import os
os.makedirs('charts', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Check permissions
import stat
os.chmod('charts', stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
```

### Issue: Memory errors with large datasets

**Problem:** Processing too many patterns at once.

**Solution:**
```python
# Process in batches
batch_size = 5
for i in range(0, len(patterns), batch_size):
    batch = patterns[i:i+batch_size]
    viz.create_all_visualizations(batch)
    plt.close('all')  # Clear figures
```

### Issue: Slow visualization generation

**Problem:** Creating many visualizations takes too long.

**Solutions:**
1. Reduce DPI setting
2. Process fewer patterns
3. Use caching

```python
config = {
    'dpi': 100,  # Lower resolution
    'figsize': (12, 6)  # Smaller figures
}
viz = PatternVisualization(data=data, config=config)
```

---

## API Reference Summary

### Quick Reference

```python
# Initialization
viz = PatternVisualization(data=data, config=config)

# Data Loading
viz.load_data('data/features_matrix.csv')
viz.load_patterns('data/final_portfolio.json')

# Visualization Creation
viz.create_overview_chart(pattern, pattern_id)
viz.create_occurrences_chart(pattern, pattern_id)
viz.create_statistics_chart(pattern, pattern_id)
viz.create_equity_curve_chart(pattern, pattern_id)
viz.create_comparison_chart(patterns)
viz.create_performance_heatmap(patterns)

# Batch Processing
paths = viz.create_all_visualizations(patterns)

# Backtesting
result = viz.run_backtest(pattern, pattern_id, initial_capital, position_size)

# Export
viz.export_visualization_summary(paths)
```

### Convenience Functions

```python
# Quick single pattern visualization
paths = visualize_pattern(pattern, pattern_id, data, output_dir)

# Quick pattern comparison
paths = compare_patterns(patterns, data, output_dir)
```

---

## Version History

- **v1.0**: Initial release with comprehensive visualization tools
  - Overview, occurrences, statistics, and equity curve charts
  - Pattern comparison and performance heatmap
  - Historical backtesting with detailed metrics
  - Interactive exploration features
  - Comprehensive API and documentation

---

*Last Updated: 2026-01-23*