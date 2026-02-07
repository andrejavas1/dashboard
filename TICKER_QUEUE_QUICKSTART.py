#!/usr/bin/env python3
"""
Quick Start Guide for Ticker Queue Management System

Run this to verify everything is set up correctly.
"""

import os
import sys
from pathlib import Path

def check_setup():
    """Check that all required files exist."""
    print("\n" + "=" * 60)
    print("TICKER QUEUE SYSTEM - SETUP CHECK")
    print("=" * 60 + "\n")
    
    required_files = [
        "src/ticker_queue_manager.py",
        "src/pipeline_worker.py",
        "start_dashboard_with_worker.py",
        "ticker_queue.json",
        "pattern_dashboard_server.py",
        "DASHBOARD_DYNAMIC.html",
    ]
    
    all_exist = True
    for file in required_files:
        exists = Path(file).exists()
        status = "✅" if exists else "❌"
        print(f"{status} {file}")
        if not exists:
            all_exist = False
    
    print("\n" + "=" * 60)
    if all_exist:
        print("✅ ALL FILES PRESENT - Ready to start!")
    else:
        print("❌ MISSING FILES - Please check above")
        return False
    
    print("=" * 60 + "\n")
    return True

def show_quick_start():
    """Show quick start instructions."""
    print("\n" + "=" * 60)
    print("QUICK START")
    print("=" * 60 + "\n")
    
    print("1. START THE SYSTEM:")
    print("   python start_dashboard_with_worker.py\n")
    
    print("2. OPEN DASHBOARD:")
    print("   http://localhost:5001\n")
    
    print("3. USE TICKER MANAGER:")
    print("   - Click 'Ticker Manager' in left sidebar")
    print("   - See existing portfolio tickers")
    print("   - Enter new ticker (e.g., AAPL)")
    print("   - Click 'Add' button")
    print("   - Watch progress as it processes\n")
    
    print("4. MONITOR PROGRESS:")
    print("   - Phase counter: 1/10, 2/10, etc.")
    print("   - Progress bar: 0-100%")
    print("   - Status: Processing, Complete\n")
    
    print("5. SWITCH TICKERS:")
    print("   - Click any ticker name in Portfolio list")
    print("   - Dashboard switches to that ticker's data\n")
    
    print("=" * 60 + "\n")

def show_architecture():
    """Show system architecture."""
    print("\n" + "=" * 60)
    print("SYSTEM ARCHITECTURE")
    print("=" * 60 + "\n")
    
    print("""
Start Script
└── start_dashboard_with_worker.py
    ├── Flask Server (Dashboard UI)
    │   ├── Port: 5001
    │   ├── URL: http://localhost:5001
    │   └── API: /api/ticker-queue/*
    │
    └── Pipeline Worker (Background)
        ├── Monitors: ticker_queue.json
        ├── Runs: main.py for each ticker
        ├── Updates: ticker_progress.json
        └── One ticker at a time

Dashboard UI (in browser)
├── Ticker Manager Panel
│   ├── Add Ticker Form
│   ├── Processing Status Bar
│   ├── Portfolio Tickers List
│   ├── Queued Tickers List
│   └── Failed Tickers List
│
└── Main Dashboard
    ├── Switch ticker via dropdown
    ├── View patterns
    └── See real-time updates

Data Files
├── ticker_queue.json (persistent)
│   └── Tracks queue state across restarts
├── ticker_progress.json (real-time)
│   └── Updated every phase
└── data/tickers/{TICKER}/ (outputs)
    └── Completed ticker data
""")
    
    print("=" * 60 + "\n")

def show_troubleshooting():
    """Show common troubleshooting."""
    print("\n" + "=" * 60)
    print("TROUBLESHOOTING")
    print("=" * 60 + "\n")
    
    print("Issue: Dashboard not loading")
    print("  → Check: Is Flask server running?")
    print("  → Try: http://localhost:5001")
    print("  → Fix: Restart with: python start_dashboard_with_worker.py\n")
    
    print("Issue: Ticker stuck in 'Processing'")
    print("  → Check: pipeline_worker.log for errors")
    print("  → Try: Press Ctrl+C and restart worker")
    print("  → Fix: Manually move to 'failed' in ticker_queue.json\n")
    
    print("Issue: Progress not updating")
    print("  → Check: Is ticker manager panel expanded?")
    print("  → Try: Click ticker manager button to expand")
    print("  → Fix: Refresh browser page\n")
    
    print("Issue: New ticker not appearing")
    print("  → Check: Did you click 'Add' button?")
    print("  → Try: Refresh dashboard")
    print("  → Fix: Check browser console for errors (F12)\n")
    
    print("=" * 60 + "\n")

if __name__ == "__main__":
    # Check setup
    if not check_setup():
        sys.exit(1)
    
    # Show instructions
    show_quick_start()
    show_architecture()
    show_troubleshooting()
    
    print("Ready? Start with:")
    print("  python start_dashboard_with_worker.py")
    print()
