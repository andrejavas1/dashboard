"""
Start the complete real-time detection system
"""
import subprocess
import sys
import time
import os

def start_servers():
    """Start both the dashboard server and streaming orchestrator."""
    
    print("=" * 60)
    print("Starting Real-Time Pattern Detection System")
    print("=" * 60)
    
    # Start Pattern Dashboard Server (Flask)
    print("\n1. Starting Pattern Dashboard Server...")
    print("   URL: http://localhost:5000")
    dashboard = subprocess.Popen(
        [sys.executable, "pattern_dashboard_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    time.sleep(2)
    
    # Start Streaming Orchestrator (WebSocket API on port 5010)
    print("\n2. Starting Streaming Orchestrator...")
    print("   WebSocket: ws://localhost:5010")
    print("   This connects to Alpaca for real-time data")
    streaming = subprocess.Popen(
        [sys.executable, "run_streaming.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    time.sleep(2)
    
    print("\n" + "=" * 60)
    print("SYSTEM READY!")
    print("=" * 60)
    print("\nAccess Points:")
    print("  • Dashboard:     http://localhost:5000")
    print("  • WebSocket API: ws://localhost:5010")
    print("  • Data Info:     http://localhost:5000/api/data-info")
    print("\nReal-time features:")
    print("  • Pattern matches appear in dashboard automatically")
    print("  • Alerts shown when patterns trigger (≥90% confidence)")
    print("  • Market regime changes broadcast to all connected clients")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping servers...")
        dashboard.terminate()
        streaming.terminate()
        print("Done!")

if __name__ == "__main__":
    start_servers()
