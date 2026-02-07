# Quick Start: Ticker Queue System in 5 Minutes

## Step 1: Verify Installation (30 seconds)

```bash
cd c:\Users\batar\Desktop\New folder (6)
python TICKER_QUEUE_QUICKSTART.py
```

**Expected Output**:
```
✓ Python version: 3.x.x
✓ src/ticker_queue_manager.py exists
✓ src/pipeline_worker.py exists
✓ pattern_dashboard_server.py exists
✓ ticker_queue.json found
✓ All dependencies available

System Ready! Start with: python start_dashboard_with_worker.py
```

## Step 2: Start the System (30 seconds)

```bash
python start_dashboard_with_worker.py
```

**Expected Output**:
```
Starting pipeline worker...
[2026-02-07 12:00:00] Pipeline worker started (PID: 12345)
Starting dashboard...
WARNING: This is a development server. Do not use it in production.
Running on http://127.0.0.1:5001
```

## Step 3: Open Dashboard (30 seconds)

Open your browser and go to:
```
http://localhost:5001
```

You should see the trading dashboard with a new **"Ticker Manager"** panel on the left.

## Step 4: Add Your First Ticker (1 minute)

1. Look for **"Ticker Manager"** section on left sidebar
2. Click to expand if collapsed
3. You'll see:
   - **"Portfolio"**: XOM ✓
   - **"Queued"**: (empty)
   - **"Add Ticker"**: Input field

4. Type a ticker: **AAPL**
5. Click **"Add"** button
6. You should see "AAPL" appear in the Queued list

## Step 5: Monitor Processing (continuous)

**You'll see**:
- **Queue Status**: "Processing: AAPL (1 of 1)"
- **Progress Bar**: "Phase 3/10: Feature Engineering - 25%"
- **Time**: "Elapsed: 5m 30s"

**The pipeline will take ~45-60 minutes total.**

While waiting, you can:
- ✅ Close browser (queue persists)
- ✅ Add more tickers (they queue automatically)
- ✅ Refresh page (progress updates)
- ✅ Restart server (queue restored)

## Testing Checklist

### Quick Test (5 minutes)
- [ ] Start `python start_dashboard_with_worker.py`
- [ ] Open `http://localhost:5001`
- [ ] Expand "Ticker Manager"
- [ ] See "Portfolio: XOM ✓"
- [ ] Click "Add", type "TEST"
- [ ] See "TEST" appear in Queue
- [ ] Stop (Ctrl+C) and restart
- [ ] Queue still shows "TEST" (✓ Persistence works!)

### Full Test (1.5 hours)
- [ ] Start system
- [ ] Add ticker "AAPL"
- [ ] Watch progress bar update (Phase 1→10)
- [ ] See estimated time remaining
- [ ] When done, "AAPL" moves to Portfolio
- [ ] Add another ticker "MSFT"
- [ ] System automatically starts MSFT
- [ ] Both show in final Portfolio

### Stress Test (multiple tickers)
- [ ] Add 5 tickers: AAPL, MSFT, GOOGL, AMZN, TSLA
- [ ] All appear in Queued list
- [ ] Worker picks them up one at a time
- [ ] Portfolio grows as tickers complete

## Common Commands

### Stop Everything
```bash
# Press Ctrl+C in terminal
# Or run:
taskkill /F /IM python.exe
```

### Clear Queue (Reset)
```bash
# Edit ticker_queue.json manually:
{
  "portfolio": ["XOM"],
  "queued": [],
  "current": null,
  "failed": {},
  "settings": {"auto_start": true, "max_parallel": 1}
}
```

### View Queue Status
```bash
# Option 1: Look in browser at Ticker Manager panel
# Option 2: Check the file:
type ticker_queue.json
```

### View Progress
```bash
# Option 1: Look at progress bar in dashboard
# Option 2: Check the file:
type ticker_progress.json
```

### View Worker Logs
```bash
# Worker logs go to console and pipeline_worker.log
type pipeline_worker.log
```

## Troubleshooting

### Issue: "Connection refused" on http://localhost:5001

**Solution 1**: Check if already running
```bash
netstat -ano | findstr :5001
# If something is using port 5001, stop it
taskkill /PID <PID> /F
```

**Solution 2**: Check firewall
- Windows might block Python
- Try: Settings → Privacy & Security → Firewall → Allow Python

### Issue: "No module named" error

**Solution**:
```bash
pip install -r requirements.txt
python TICKER_QUEUE_QUICKSTART.py
```

### Issue: Queue doesn't auto-start

**Solution**: Check `ticker_queue.json`
```json
{
  "settings": {
    "auto_start": true,    // ← Must be true
    "max_parallel": 1
  }
}
```

### Issue: Progress bar stuck at 0%

**Solution**:
1. Wait 10 seconds (phase detection might be slow)
2. Refresh page (F5)
3. If still stuck, check logs:
   ```bash
   type pipeline_worker.log
   ```

### Issue: Ticker moves to "Failed"

**Solution**:
1. Check `pipeline_worker.log` for error message
2. Fix the issue (usually config or data problem)
3. Click `⟲` (retry) button next to ticker
4. Ticker moves back to Queue

## File Locations

| File | Purpose | Can Edit? |
|------|---------|-----------|
| `ticker_queue.json` | Queue state | Yes (careful!) |
| `ticker_progress.json` | Current progress | No (read-only) |
| `pipeline_worker.log` | Worker logs | No (read-only) |
| `config.yaml` | Ticker config | Auto-updated |
| `START_HERE.md` | Main guide | No |
| `TICKER_QUEUE_MANAGEMENT.md` | Detailed docs | No |

## Next Steps

### If System Works ✓
1. Queue your real tickers
2. Let it run overnight
3. Check portfolio in morning
4. Review results in dashboard

### If System Fails ✗
1. Run `python TICKER_QUEUE_QUICKSTART.py`
2. Check `pipeline_worker.log` for errors
3. Review `TICKER_QUEUE_MANAGEMENT.md` troubleshooting
4. Check all dependencies installed

## Key URLs

| URL | Purpose |
|-----|---------|
| `http://localhost:5001` | Main dashboard |
| `http://localhost:5001/api/ticker-queue/status` | Queue status (JSON) |
| `http://localhost:5001/api/ticker-queue/progress` | Progress (JSON) |
| `http://localhost:5001/api/tickers/portfolio` | Portfolio list (JSON) |

## System Architecture (30-second version)

```
┌─────────────────────────────────────────────────────┐
│  Your Browser (http://localhost:5001)               │
│  ├─ Dashboard UI                                    │
│  └─ Ticker Manager Panel (new!)                     │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP Requests
                   ↓
┌────────────────────────────────────────────────────┐
│  Flask Dashboard Server (pattern_dashboard_server) │
│  ├─ GET /api/ticker-queue/status                  │
│  ├─ GET /api/ticker-queue/progress                │
│  ├─ POST /api/ticker-queue/add                    │
│  └─ DELETE /api/ticker-queue/cancel               │
└────────────────┬────────────────────────────────────┘
                  │ Reads/Writes
                  ↓
        ┌──────────────────────┐
        │  ticker_queue.json   │ ← Queue state
        │  ticker_progress.json│ ← Progress state
        └──────────────────────┘
                  ↑
                  │ Updates
┌─────────────────┴────────────────────────────────────┐
│  Pipeline Worker (pipeline_worker.py)               │
│  ├─ Monitors queue every 5 seconds                 │
│  ├─ Launches main.py for next ticker               │
│  ├─ Parses output for progress                     │
│  └─ Updates state when complete                    │
└──────────────────┬──────────────────────────────────┘
                   │ Runs
                   ↓
        ┌────────────────────┐
        │  main.py (Phase 1) │
        │  main.py (Phase 2) │
        │  main.py (Phase 3) │
        │  ... (Phases 4-10) │
        └────────────────────┘
                   │ Writes
                   ↓
        ┌──────────────────────────────┐
        │  data/tickers/{TICKER}/      │
        │  ├─ ohlcv.json              │
        │  ├─ patterns.json            │
        │  ├─ classifier.pkl           │
        │  └─ occurrences/             │
        └──────────────────────────────┘
```

## One-Command Startup

```bash
python start_dashboard_with_worker.py
```

That's it! Everything else is automatic.

---

**You're ready!** Start the system and watch the magic happen. 🚀
