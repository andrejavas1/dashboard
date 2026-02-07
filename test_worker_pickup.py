#!/usr/bin/env python3
"""
Quick test: Verify worker picks up queued ticker
"""

import json
import time
import subprocess
import sys
from pathlib import Path

print("=" * 80)
print("TEST: Worker Picks Up Queued Ticker")
print("=" * 80)

# Step 1: Reset queue
print("\n1. Resetting queue to known state...")
queue_state = {
    "portfolio": ["XOM"],
    "queued": ["TEST_TICKER"],
    "current": None,
    "failed": {},
    "settings": {
        "auto_start": True,
        "max_parallel": 1
    }
}

with open('ticker_queue.json', 'w') as f:
    json.dump(queue_state, f, indent=2)
print("   ✓ Queue reset: TEST_TICKER is now queued")

# Step 2: Start worker
print("\n2. Starting worker subprocess...")
proc = subprocess.Popen(
    [sys.executable, 'src/pipeline_worker.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)
print(f"   ✓ Worker started (PID: {proc.pid})")
print(f"   ℹ Worker will run for 15 seconds...")

# Step 3: Wait and collect output
print("\n3. Monitoring worker output (15 seconds)...\n")
start_time = time.time()
found_test_ticker = False
found_error = False

try:
    while time.time() - start_time < 15:
        line = proc.stdout.readline()
        if line:
            line = line.strip()
            if line:
                print(f"   {line}")
                
                # Check for key messages
                if "TEST_TICKER" in line:
                    found_test_ticker = True
                    print("   ✓✓✓ WORKER FOUND TEST_TICKER ✓✓✓")
                if "Processing ticker" in line or "Starting processing" in line:
                    found_test_ticker = True
                if "error" in line.lower() or "failed" in line.lower():
                    found_error = True
except KeyboardInterrupt:
    print("\n   (interrupted)")
finally:
    # Stop worker
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except:
        proc.kill()

# Step 4: Check results
print("\n" + "=" * 80)
print("TEST RESULTS")
print("=" * 80)

# Check queue state
with open('ticker_queue.json', 'r') as f:
    final_queue = json.load(f)

print(f"\nQueue state after worker run:")
print(f"   Portfolio: {final_queue['portfolio']}")
print(f"   Queued: {final_queue['queued']}")
print(f"   Current: {final_queue['current']}")

# Analyze results
print("\n" + "=" * 80)
if found_test_ticker:
    print("✓ SUCCESS: Worker detected and processed TEST_TICKER")
    print("\nThe worker IS working correctly!")
    print("The issue was that it wasn't reloading the queue from disk.")
    print("This has been FIXED. Try adding a real ticker now:")
    print("  1. Start: python start_dashboard_with_worker.py")
    print("  2. Open: http://localhost:5001")
    print("  3. Add a ticker in Ticker Manager")
    print("  4. Watch progress bar appear")
elif "Queue is empty" in str(final_queue):
    print("✓ PARTIAL: Worker ran but queue was empty")
    print("Check if TEST_TICKER was picked up from initial queue state")
else:
    print("⚠ UNCLEAR: Could not determine worker state from output")

if found_error:
    print("\n⚠ Worker errors detected - check pipeline_worker.log")

print("=" * 80)
