#!/usr/bin/env python3
"""
Diagnostic script to test worker and queue functionality
"""

import sys
import os
import json
import subprocess
import time
from pathlib import Path

print("=" * 80)
print("DIAGNOSTIC CHECK: Pipeline Worker & Queue System")
print("=" * 80)

# Check 1: Verify files exist
print("\n1. Checking required files...")
required_files = [
    'src/ticker_queue_manager.py',
    'src/pipeline_worker.py',
    'pattern_dashboard_server.py',
    'ticker_queue.json'
]

all_exist = True
for f in required_files:
    exists = Path(f).exists()
    status = "✓" if exists else "✗"
    print(f"   {status} {f}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n✗ Missing files! Check workspace structure.")
    sys.exit(1)

# Check 2: Verify ticker_queue.json is valid
print("\n2. Checking ticker_queue.json...")
try:
    with open('ticker_queue.json', 'r') as f:
        queue = json.load(f)
    print(f"   ✓ Valid JSON")
    print(f"   Portfolio: {queue.get('portfolio', [])}")
    print(f"   Queued: {queue.get('queued', [])}")
    print(f"   Current: {queue.get('current')}")
    print(f"   Failed: {queue.get('failed', {})}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Check 3: Test ticker_queue_manager in isolation
print("\n3. Testing TickerQueueManager...")
try:
    sys.path.insert(0, 'src')
    from ticker_queue_manager import TickerQueueManager
    
    manager = TickerQueueManager()
    print(f"   ✓ Manager initialized")
    
    # Test getting next ticker
    ticker = manager.get_next_ticker()
    print(f"   Next ticker from queue: {ticker}")
    
    if ticker:
        print(f"   ✓ Manager found queued ticker: {ticker}")
    else:
        print(f"   ℹ Queue is empty (this is OK if no tickers are queued)")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 4: Test if worker can be imported
print("\n4. Testing PipelineWorker import...")
try:
    from pipeline_worker import PipelineWorker
    print(f"   ✓ Worker class imports successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 5: Test if main.py exists
print("\n5. Checking main.py...")
if Path('main.py').exists():
    print(f"   ✓ main.py exists")
else:
    print(f"   ✗ main.py not found!")

# Check 6: Test config.yaml
print("\n6. Checking config.yaml...")
if Path('config.yaml').exists():
    print(f"   ✓ config.yaml exists")
else:
    print(f"   ✗ config.yaml not found!")

# Check 7: Test if pipeline_worker.py can run
print("\n7. Testing worker subprocess...")
try:
    proc = subprocess.Popen(
        [sys.executable, 'src/pipeline_worker.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=5
    )
    
    # Wait briefly
    time.sleep(2)
    
    # Check if still running
    poll_result = proc.poll()
    
    if poll_result is None:
        print(f"   ✓ Worker process is running (PID: {proc.pid})")
        proc.terminate()
        proc.wait(timeout=2)
        print(f"   ✓ Worker terminated cleanly")
    else:
        print(f"   ⚠ Worker exited with code: {poll_result}")
        stdout, stderr = proc.communicate()
        if stderr:
            print(f"   Error output:\n{stderr}")
        
except subprocess.TimeoutExpired:
    print(f"   ✓ Worker is running (didn't exit within timeout)")
    proc.kill()
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Check 8: Check if log file exists
print("\n8. Checking worker log...")
if Path('pipeline_worker.log').exists():
    print(f"   ✓ pipeline_worker.log exists")
    # Show last 10 lines
    with open('pipeline_worker.log', 'r') as f:
        lines = f.readlines()[-10:]
    if lines:
        print(f"   Last entries:")
        for line in lines:
            print(f"      {line.rstrip()}")
else:
    print(f"   ℹ pipeline_worker.log not found (will be created on first run)")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

print("\nTo fix issues:")
print("1. If queue manager has errors → Check imports and paths")
print("2. If worker subprocess fails → Check pipeline_worker.py for runtime errors")
print("3. If no log file → Worker hasn't run yet (normal)")
print("4. If worker exits → Check pipeline_worker.log for errors after running system")

print("\nNext step: Run: python start_dashboard_with_worker.py")
print("Then wait 10 seconds and check pipeline_worker.log for worker activity")
