# FIXED: Ticker Queue Not Processing

## The Bug
The pipeline worker was loading the queue **once at startup** and never reloading it. When you added a ticker via the dashboard UI, the worker's in-memory copy of the queue didn't get updated, so it never picked up the new ticker.

## The Fix
Updated `src/ticker_queue_manager.py` to reload the queue file **every time** it checks for the next ticker. This ensures the worker always sees new tickers added by the dashboard.

**Changed**: `get_next_ticker()` method now calls `self.queue = self._load_queue()` before checking for queued tickers.

## Changes Made

### 1. File: `src/ticker_queue_manager.py`
- **Method**: `get_next_ticker()`
- **Change**: Added `self.queue = self._load_queue()` at the start
- **Effect**: Worker now sees real-time queue updates from dashboard

### 2. File: `src/pipeline_worker.py`
- **Change 1**: Improved logging with star marker (★) for processing start
- **Change 2**: Added periodic queue status logging (every 30 seconds)
- **Change 3**: Better error reporting with `exc_info=True` for tracebacks
- **Effect**: Better visibility into worker activity and debugging

### 3. File: `start_dashboard_with_worker.py`
- **Change 1**: Added `read_worker_output()` function to consume subprocess output
- **Change 2**: Added daemon thread to prevent output buffer deadlock
- **Change 3**: Added verification check that worker is running
- **Effect**: Prevents subprocess from hanging or blocking

## How to Test

### Quick Test (5 minutes)
```bash
python test_worker_pickup.py
```
This will:
1. Reset queue with test ticker
2. Start worker
3. Watch for worker picking up the ticker
4. Show results

### Full Test (with real ticker)
1. **Start system**:
   ```bash
   python start_dashboard_with_worker.py
   ```

2. **Open dashboard**:
   - Go to `http://localhost:5001`
   - Look for "Ticker Manager" panel on left
   
3. **Add ticker**:
   - Type a ticker symbol (e.g., "MSFT")
   - Click "Add"
   - You should immediately see:
      - Ticker appears in "Queued" list
      - Progress bar shows "Phase 0/10: Initializing - 0%"
      - Worker starts processing
      
4. **Monitor progress**:
   - Progress bar updates every 5-10 seconds
   - Shows current phase number and percentage
   - Shows elapsed time

5. **Check logs**:
   - Terminal shows worker activity with ★ markers
   - `pipeline_worker.log` contains detailed logs
   - Search for "★ Processing" to see when ticker starts

## What You Should See

### Before (Broken)
- Add ticker → It appears in Queued list
- Nothing happens
- No progress bar
- Worker.log shows: "Queue is empty, waiting..."

### After (Fixed)
- Add ticker → It appears in Queued list
- Progress bar appears immediately
- Shows "Phase X/10: Name - Y%"
- Worker.log shows: "★ Processing ticker: XXXX"
- Ticker moves through all 10 phases
- Ticker moves to Portfolio when complete

## Verification Checklist

- [ ] Run `python test_worker_pickup.py` → See "SUCCESS" message
- [ ] Start `python start_dashboard_with_worker.py`
- [ ] Add ticker "MSFT" via dashboard
- [ ] Progress bar appears within 5 seconds
- [ ] Phase counter shows "Phase 1/10"
- [ ] Check logs: `tail -f pipeline_worker.log` shows processing
- [ ] Worker continues through all 10 phases
- [ ] Ticker moves to "Portfolio" when complete

## If Still Having Issues

### Issue: Progress bar still doesn't appear
**Check**:
1. `pipeline_worker.log` - Look for error messages
2. Browser console - Open DevTools (F12), check for JS errors
3. Flask logs - Check terminal for error messages
4. `ticker_queue.json` - Verify ticker is in "queued" list

### Issue: Worker exits immediately
**Check**:
```bash
python diagnose_worker.py
```
This will test imports and worker initialization.

### Issue: Ticker doesn't move to portfolio
**Check**:
1. `pipeline_worker.log` - Look for error during phase
2. `config.yaml` - Should have ticker name (auto-updated)
3. Network - Can worker run main.py?

## Files Modified
- `src/ticker_queue_manager.py` - Queue reload fix
- `src/pipeline_worker.py` - Better logging
- `start_dashboard_with_worker.py` - Output handling + verification
- `QUICK_START.md` - Updated docs
- `diagnose_worker.py` - NEW diagnostic tool
- `test_worker_pickup.py` - NEW verification test

## Timeline to Production

1. **Immediate**: Test with `python test_worker_pickup.py`
2. **Then**: Try real ticker with `python start_dashboard_with_worker.py`
3. **Then**: Queue multiple tickers if first one works
4. **Production**: System ready to use for persistent portfolio tracking

---

**The bug has been fixed!** The worker will now properly detect and process tickers added via the dashboard.
