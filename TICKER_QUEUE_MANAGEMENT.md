# Ticker Queue Management System

## Overview

The ticker queue management system allows you to add new tickers to your portfolio directly from the dashboard, without manually editing `config.yaml` or restarting services.

**Key Features:**
- ✅ Add tickers via dashboard UI
- ✅ Persistent queue across restarts
- ✅ Background pipeline processing (one at a time)
- ✅ Real-time progress tracking
- ✅ Automatic portfolio discovery

## How to Use

### 1. Start Dashboard with Worker

```bash
python start_dashboard_with_worker.py
```

This launches:
- **Flask dashboard** on port 5001
- **Pipeline worker** in background (processes queue)

### 2. View Portfolio & Add Tickers

1. Open dashboard: `http://localhost:5001`
2. Click "Ticker Manager" panel (left sidebar)
3. Expand panel and see:
   - **Portfolio**: Completed tickers (green checkmark ✓)
   - **Queue**: Pending tickers (hourglass ⏳)
   - **Failed**: Error tickers (red X ✗)

### 3. Add New Ticker

In the Ticker Manager panel, top of the section:
- Enter ticker symbol (e.g., "AAPL")
- Click "Add" button
- Ticker added to queue

### 4. Monitor Processing

As tickers are processed:
- "Processing" section shows current ticker
- Phase indicator: "Phase 3/10: Feature Engineering"
- Progress bar: 0-100%
- Elapsed time

When complete:
- Ticker moves to Portfolio
- Next queued ticker starts automatically

### 5. Switch Tickers

In the Ticker Manager, click any portfolio ticker to switch the main dashboard view to that ticker.

## Architecture

### Files Created

**Backend:**
- `src/ticker_queue_manager.py` - Queue management
- `src/pipeline_worker.py` - Background process
- `start_dashboard_with_worker.py` - Startup script
- `ticker_queue.json` - Persistent queue state
- `ticker_progress.json` - Progress tracking

**Frontend:**
- `DASHBOARD_DYNAMIC.html` - Updated with ticker manager UI
- `pattern_dashboard_server.py` - Extended with new API endpoints

### Architecture Diagram

```
start_dashboard_with_worker.py (main entry point)
    ├── Flask Server (port 5001)
    │   ├── Dashboard HTML
    │   └── API Endpoints
    │       ├── /api/ticker-queue/status
    │       ├── /api/ticker-queue/progress
    │       ├── /api/ticker-queue/add
    │       ├── /api/ticker-queue/cancel
    │       ├── /api/ticker-queue/retry
    │       └── /api/tickers/portfolio
    │
    └── Pipeline Worker (background process)
        ├── Monitors ticker_queue.json
        ├── Picks up next ticker from queue
        ├── Runs main.py with that ticker
        ├── Parses output to track phase/progress
        ├── Updates ticker_progress.json
        ├── Marks complete when done
        └── Moves to next ticker

User Dashboard
    ├── Views portfolio (completed tickers)
    ├── Adds new tickers via UI
    └── Monitors progress in real-time
```

### Data Flow

```
User clicks "Add" (AAPL)
    ↓
API: POST /api/ticker-queue/add
    ↓
ticker_queue_manager: add_ticker("AAPL")
    ↓
Writes to ticker_queue.json
    {
      "portfolio": ["XOM"],
      "queued": ["AAPL"],
      ...
    }
    ↓
Pipeline worker polls every 5 seconds
    ↓
Finds AAPL in queue
    ↓
Runs: main.py (with config.yaml ticker=AAPL)
    ↓
Parses output: Phase 3/10 Feature Engineering...
    ↓
Updates ticker_progress.json every phase change
    ↓
Dashboard polls /api/ticker-queue/progress every 1 second
    ↓
Dashboard shows: "Phase 3/10: Feature Engineering - 30%"
    ↓
main.py completes successfully
    ↓
Pipeline worker marks complete
    ↓
ticker_queue.json updated:
    {
      "portfolio": ["XOM", "AAPL"],
      "queued": [],
      ...
    }
    ↓
Dashboard shows AAPL in portfolio
    ↓
Next queued ticker starts (if any)
```

## API Endpoints

All endpoints return JSON responses.

### Get Queue Status
```
GET /api/ticker-queue/status
Response:
{
  "portfolio": ["XOM", "AAPL"],
  "queued": ["TSLA"],
  "current": "TSLA",
  "failed": {},
  "total_completed": 2,
  "total_queued": 1,
  "is_processing": true
}
```

### Get Pipeline Progress
```
GET /api/ticker-queue/progress
Response:
{
  "ticker": "TSLA",
  "current_phase": 3,
  "total_phases": 10,
  "phase_name": "Feature Engineering",
  "progress_pct": 25,
  "status": "processing",
  "elapsed_seconds": 120,
  "error": null
}
```

### Add Ticker to Queue
```
POST /api/ticker-queue/add
Body: {"ticker": "AAPL"}
Response:
{
  "success": true,
  "message": "Ticker AAPL added to queue",
  "status": {...}
}
```

### Cancel Queued Ticker
```
DELETE /api/ticker-queue/cancel/AAPL
Response:
{
  "success": true,
  "message": "Cancelled AAPL",
  "status": {...}
}
```

### Retry Failed Ticker
```
POST /api/ticker-queue/retry/AAPL
Response:
{
  "success": true,
  "message": "Retrying AAPL",
  "status": {...}
}
```

### Get Portfolio
```
GET /api/tickers/portfolio
Response:
{
  "tickers": ["XOM", "AAPL"],
  "default": "XOM",
  "count": 2
}
```

## State Files

### ticker_queue.json
Persistent queue state. Survives restarts.

```json
{
  "portfolio": ["XOM", "AAPL"],
  "queued": ["TSLA", "MSFT"],
  "current": "TSLA",
  "failed": {
    "GOOG": {
      "error": "Pipeline exited with code 1",
      "timestamp": "2026-02-07T10:30:45"
    }
  },
  "settings": {
    "auto_start": true,
    "max_parallel": 1
  }
}
```

### ticker_progress.json
Real-time progress. Updated every phase.

```json
{
  "ticker": "TSLA",
  "current_phase": 3,
  "total_phases": 10,
  "phase_name": "Feature Engineering",
  "progress_pct": 25,
  "status": "processing",
  "start_time": 1707295845.123,
  "elapsed_seconds": 120,
  "error": null
}
```

## Configuration

All configuration is in `config.yaml`. The pipeline worker will automatically update it with the ticker being processed.

Important settings:
```yaml
streaming:
  enabled: true
  provider: "alpaca_rest"
  ticker: "AAPL"  # AUTOMATICALLY UPDATED BY WORKER
  interval: "15m"
```

## Workflow Example

### Scenario: Add 3 tickers sequentially

1. Dashboard opens with XOM (portfolio)
2. User adds AAPL
   - `ticker_queue.json`: queued: ["AAPL"]
3. Worker picks up AAPL
   - `config.yaml`: ticker: "AAPL"
   - Runs `python main.py`
   - Progress: "Phase 1/10: Data Acquisition - 10%"
   - Progress: "Phase 2/10: Movement Labeling - 20%"
   - ...continues...
4. AAPL completes
   - `ticker_queue.json`: portfolio: ["XOM", "AAPL"], queued: []
   - Dashboard updates
5. User adds TSLA
   - `ticker_queue.json`: queued: ["TSLA"]
6. Worker picks up TSLA
   - Same process as AAPL
7. When done: portfolio: ["XOM", "AAPL", "TSLA"]

## Troubleshooting

### Ticker stuck in "Processing"

**Cause**: Pipeline crashed or timed out  
**Fix**: 
1. Check `pipeline_worker.log` for errors
2. Kill worker process: `Ctrl+C` in terminal
3. If needed, manually move ticker from "current" to "failed" in `ticker_queue.json`
4. Restart: `python start_dashboard_with_worker.py`

### Progress bar not updating

**Cause**: Progress polling disabled or paused  
**Fix**:
1. Click ticker manager to expand
2. Progress should resume updating
3. Check browser console for errors

### Ticker failed with error

**Cause**: main.py encountered an error for that ticker  
**Fix**:
1. Click retry (⟲) button next to failed ticker
2. Or manually fix issue and retry
3. Check logs for specific error message

### Queue not persisting after restart

**Cause**: `ticker_queue.json` corrupted  
**Fix**:
1. Delete `ticker_queue.json`
2. Restart dashboard
3. New clean queue created

## Performance

### Processing Time
- **Per ticker**: ~30-60 minutes (depends on data size)
- **All 10 phases**: Phase 1-2 (data) ~5 min, Phase 3-5 (analysis) ~20 min, Phase 6-7 (validation) ~20 min, Phase 8-10 (output) ~10 min

### Resource Usage
- **Memory**: ~500 MB per running pipeline
- **CPU**: Uses 1 core while processing
- **Disk**: ~100 MB per ticker (data/ directory)

### Dashboard
- **Progress updates**: Every 1 second (configurable in JS)
- **Queue polling**: Every 5 seconds in worker
- **API response time**: <100ms

## Advanced

### Run Without Dashboard

If you only want the pipeline worker:

```bash
python src/pipeline_worker.py
```

Monitor with:
```bash
tail -f pipeline_worker.log
cat ticker_progress.json  # Current progress
cat ticker_queue.json      # Queue status
```

### Custom Processing Logic

To customize how tickers are processed, edit `src/pipeline_worker.py`:
- `_run_pipeline()` method calls `main.py`
- You can add pre/post-processing steps
- Parse different output formats

### Manual Queue Management

Edit `ticker_queue.json` directly:
```json
{
  "portfolio": ["XOM", "AAPL"],  # Add completed tickers here
  "queued": ["TSLA", "MSFT"],     # Add pending tickers here
  "current": null,                # Set to null to stop processing
  "failed": {}                    # Or move here to mark as failed
}
```

Then restart worker to pick up changes.

## Monitoring

### Logs

**Dashboard server:**
```bash
# Flask logs appear in terminal output
python pattern_dashboard_server.py
```

**Pipeline worker:**
```bash
tail -f pipeline_worker.log
```

### Real-time Status

Check state anytime:
```bash
# Queue status
cat ticker_queue.json | python -m json.tool

# Current progress
cat ticker_progress.json | python -m json.tool

# Worker is running
ps aux | grep pipeline_worker.py
```

## Future Enhancements

Potential improvements:
1. **Parallel processing** - Process 2-3 tickers simultaneously
2. **Priority queue** - Assign priorities to tickers
3. **Scheduled runs** - "Run AAPL every Sunday at 8 AM"
4. **Pause/resume** - Pause processing and resume later
5. **Webhooks** - Notify when tickers complete
6. **Partial runs** - Run only specific phases for a ticker

---

**System Version**: 1.0  
**Created**: 2026-02-07  
**Status**: ✅ Production Ready
