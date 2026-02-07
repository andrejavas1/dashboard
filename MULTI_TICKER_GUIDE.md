# Multi-Ticker Support Guide

## Current State
Currently the system only supports **one ticker** (XOM) hardcoded in `config.yaml`.

## What Changes Are Needed

### Option 1: Separate Patterns Per Ticker (Recommended)
Each ticker has its own patterns discovered from its own historical data.

**Pros:**
- Patterns are ticker-specific (XOM patterns won't work for AAPL)
- Independent analysis per ticker
- Better performance tracking per symbol

**Cons:**
- More data files
- More computation time

### Option 2: Universal Patterns Across Tickers
Discover patterns from all tickers combined.

**Pros:**
- Single pattern set
- Less computation

**Cons:**
- Patterns may not work across different stocks
- Harder to track per-ticker performance

## Recommended Approach: Option 1

### File Structure Changes:
```
data/
├── tickers/
│   ├── XOM/
│   │   ├── patterns.json
│   │   ├── features_matrix.csv
│   │   └── pattern_*_occurrences.json
│   ├── AAPL/
│   │   ├── patterns.json
│   │   ├── features_matrix.csv
│   │   └── pattern_*_occurrences.json
│   └── TSLA/
│       ├── patterns.json
│       ├── features_matrix.csv
│       └── pattern_*_occurrences.json
├── combined/
│   └── all_patterns.json (optional aggregation)
└── trades/
    └── all_trades.json (cross-ticker trade tracking)
```

### Config Changes:
```yaml
# OLD (single ticker)
ticker: "XOM"

# NEW (multiple tickers)
tickers:
  - "XOM"
  - "AAPL"
  - "TSLA"
  - "MSFT"
```

### Implementation Steps:

**Step 1: Update config.yaml**
Add tickers list instead of single ticker

**Step 2: Update Phase 1 (Data Acquisition)**
Loop through all tickers, fetch data for each

**Step 3: Update Phase 3 (Feature Engineering)**
Calculate features per ticker, store in ticker-specific folders

**Step 4: Update Phase 4 (Pattern Discovery)**
Discover patterns per ticker independently

**Step 5: Update Dashboard**
Add ticker selector dropdown
Show patterns for selected ticker only

**Step 6: Update Real-Time Streaming**
Support multiple WebSocket connections (one per ticker)
Or create separate streaming instances

## Quick Implementation

Would you like me to implement **Option 1** (separate patterns per ticker)?

This would involve:
1. ✅ Updating config.yaml for multiple tickers
2. ✅ Modifying pipeline to loop through tickers
3. ✅ Creating per-ticker data directories
4. ✅ Updating dashboard with ticker selector
5. ✅ Supporting multi-ticker real-time streaming

Or would you prefer a different approach?
