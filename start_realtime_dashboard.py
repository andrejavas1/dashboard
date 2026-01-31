"""
Launch Real-Time Pattern Detection System
Starts both the Flask dashboard server AND the WebSocket streaming server.
"""

import subprocess
import sys
import time
import webbrowser
import os

def main():
    print("=" * 70)
    print("  Real-Time Pattern Detection System")
    print("=" * 70)
    print()
    print("Architecture:")
    print("  • Flask Server: Serves dashboard at http://localhost:5000")
    print("  • WebSocket Server: Real-time data at ws://localhost:5010")
    print("  • Fetches 15m prices → Updates daily bars → Matches patterns")
    print()
    
    # Start Flask dashboard server
    print("🚀 Starting Flask Dashboard Server...")
    flask_process = subprocess.Popen(
        [sys.executable, "pattern_dashboard_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    time.sleep(2)
    
    # Start WebSocket streaming server
    print("🚀 Starting WebSocket Streaming Server...")
    ws_process = subprocess.Popen(
        [sys.executable, "-m", "src.realtime_streaming_system"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    time.sleep(2)
    
    # Open dashboard
    print("\n🌐 Opening Dashboard...")
    webbrowser.open("http://localhost:5000")
    
    print("\n" + "=" * 70)
    print("  SYSTEM READY!")
    print("=" * 70)
    print("\nInstructions:")
    print("  1. Dashboard loads with historical pattern data")
    print("  2. Click 'Connect Real-Time' to enable live updates")
    print("  3. Watch real-time panel for:")
    print("     • Today's updating daily bar (O/H/L/C)")
    print("     • Market regime (Trend/Volatility)")
    print("     • Active pattern count")
    print("     • Triggered pattern alerts")
    print("\nPress Ctrl+C to stop both servers\n")
    
    try:
        # Monitor both processes
        while True:
            # Check Flask output
            flask_line = flask_process.stdout.readline()
            if flask_line:
                print(f"[Flask] {flask_line}", end='')
            
            # Check WebSocket output
            ws_line = ws_process.stdout.readline()
            if ws_line:
                print(f"[WS] {ws_line}", end='')
                
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping servers...")
        flask_process.terminate()
        ws_process.terminate()
        flask_process.wait()
        ws_process.wait()
        print("✅ Servers stopped")

if __name__ == "__main__":
    main()
