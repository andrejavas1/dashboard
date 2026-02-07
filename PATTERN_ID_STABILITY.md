# Pattern ID Stability - Critical Fix

## The Problem

When pipeline re-runs weekly, patterns used to get NEW IDs, causing:
1. ❌ Open trades linked to wrong patterns
2. ❌ Trade history disconnected from patterns  
3. ❌ Live stats lost or attributed to wrong patterns
4. ❌ Confusion about which pattern triggered which trade

**Example of what would happen WITHOUT this fix:**
```
Week 1:
  Pattern #5 triggers → Trade opened (pattern_id=5)
  Pattern #5 live stats: 3 trades, 2 wins

Week 2 (pipeline re-runs):
  Pattern #5 becomes Pattern #12 (new ID assigned)
  Open trade still has pattern_id=5 → Links to NOTHING
  Pattern #12 live stats: 0 trades (lost history!)
  Pattern #5 shows in outcomes but not in active patterns
```

## The Solution: Pattern ID Manager

**File:** `src/pattern_id_manager.py`

### How It Works

**1. Pattern Fingerprinting**
```python
# Each pattern gets unique fingerprint from its conditions
fingerprint = hash(direction + label_col + conditions)

Example:
  Pattern: Long, RSI<30, Close>SMA20, Label_2pct_10d
  Fingerprint: "a3f7b2d9e1c5..." (always same for this pattern)
```

**2. Registry System**
```
data/pattern_registry.json:
{
  "patterns": {
    "5": {
      "fingerprint": "a3f7b2d9e1c5...",
      "first_seen": "2026-01-01",
      "last_seen": "2026-02-05",
      "live_stats": {...}
    }
  },
  "next_id": 52
}
```

**3. Sync Process (runs after pipeline)**
```
Pipeline discovers pattern:
  ↓
Generate fingerprint
  ↓
Check registry for existing fingerprint
  ↓
IF found → Use existing ID (e.g., #5)
IF new → Assign new ID (e.g., #52)
IF old pattern missing → Archive it
```

### What This Fixes

**✅ Open Trades Stay Linked**
```
Week 1: Pattern #5 triggers → Open trade (pattern_id=5)
Week 2: Pattern still #5 → Trade still linked correctly
```

**✅ Trade History Preserved**
```
Week 1: Pattern #5 has 10 trades in outcomes
Week 2: Pattern #5 still has those 10 trades
```

**✅ Live Stats Continue**
```
Week 1: Pattern #5 live win rate: 70%
Week 2: Pattern #5 live win rate: 68% (includes new results)
```

**✅ Removed Patterns Archived**
```
Pattern #3 not rediscovered → Moved to archive
Trade outcomes for #3 still exist → Referenced in archive
```

## Weekly Pipeline Flow (Updated)

```
1. Sunday 2 AM → Pipeline scheduler starts
2. Backup current data
3. Run pipeline → Discover new patterns
4. ⚡ SYNC PATTERN IDs ⚡
   - Match fingerprints to existing IDs
   - Assign new IDs to truly new patterns
   - Archive patterns that weren't rediscovered
5. Save patterns.json with stable IDs
6. Restart server → All links intact
```

## Data Files

**Active Registry:** `data/pattern_registry.json`
- Maps pattern IDs to fingerprints
- Tracks first_seen, last_seen, discovery_count
- Preserves live_stats across runs

**Archive:** `data/pattern_archive.json`
- Patterns no longer active
- Keeps history for reference
- Shows when archived, final stats

**Sync Report:** `data/pattern_sync_report.json`
- What happened during sync
- How many preserved, new, archived

## Testing

**Test the sync:**
```bash
python src/pattern_id_manager.py
```

**Expected output:**
```
Pattern ID Sync Complete:
  Active patterns: 52
  Preserved IDs: 48
  New patterns: 4
  Archived: 2
```

## Integration

The Pattern ID Manager is automatically called by:
- `weekly_pipeline_scheduler.py` after pipeline completes
- Can be run manually after any pipeline run

This ensures pattern identity persists across the entire system lifecycle.
