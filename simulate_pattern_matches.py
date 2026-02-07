"""
Simulate Active Pattern Matches for Dashboard Testing
Sends fake pattern match data via WebSocket to test the realtime display.
"""

import asyncio
import json
import websockets
import random
from datetime import datetime, timezone

async def simulate_pattern_matches():
    """Send simulated pattern match data to the dashboard."""
    uri = "ws://localhost:5012"
    
    # Sample patterns to simulate - matching dashboard expected format
    sample_patterns = [
        {"pattern_id": 1, "name": "Pattern #1", "conditions": {"RSI_14": {"min": 30, "max": 50}}, "direction": "long"},
        {"pattern_id": 5, "name": "Pattern #5", "conditions": {"MACD": {"min": -0.5, "max": 0.5}}, "direction": "long"},
        {"pattern_id": 12, "name": "Pattern #12", "conditions": {"Vol_Change": {"min": 1.2, "max": 2.0}}, "direction": "short"},
        {"pattern_id": 23, "name": "Pattern #23", "conditions": {"BB_Width": {"min": 0.02, "max": 0.08}}, "direction": "long"},
        {"pattern_id": 34, "name": "Pattern #34", "conditions": {"ATR_14": {"min": 2.0, "max": 5.0}}, "direction": "short"},
    ]
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"[OK] Connected to {uri}")
            print("Sending simulated pattern matches every 5 seconds...")
            print("Press Ctrl+C to stop\n")
            
            while True:
                # Simulate 1-3 random pattern matches
                num_matches = random.randint(1, 3)
                matches = random.sample(sample_patterns, num_matches)
                
                # Add match details - using field names dashboard expects
                for match in matches:
                    match['confidence'] = round(random.uniform(85, 98), 1)
                    match['probability'] = round(random.uniform(75, 92), 1)
                    match['timestamp'] = datetime.now(timezone.utc).isoformat()
                
                # Create update messages matching the streaming system format
                # 1. Daily bar update
                await websocket.send(json.dumps({
                    "type": "daily_bar_update",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "daily_bar": {
                        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        "open": round(138.50 + random.uniform(-2, 2), 2),
                        "high": round(142.00 + random.uniform(-1, 3), 2),
                        "low": round(137.00 + random.uniform(-2, 1), 2),
                        "close": round(141.00 + random.uniform(-3, 3), 2),
                        "volume": int(25000000 + random.uniform(-5000000, 5000000)),
                        "updates": random.randint(10, 50)
                    }
                }))
                
                # 2. Regime update
                await websocket.send(json.dumps({
                    "type": "regime_update",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "regime": {
                        "volatility": random.choice(["Low", "Medium", "High"]),
                        "trend": random.choice(["Strong Bull", "Bull", "Sideways", "Bear", "Strong Bear"])
                    }
                }))
                
                # 3. Matches update (this is what shows in Active Pattern Matches)
                await websocket.send(json.dumps({
                    "type": "matches_update",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "matches": matches,
                    "triggered_count": len([m for m in matches if m.get('match_score', 0) > 90])
                }))
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Sent {num_matches} pattern matches: {[m['name'] for m in matches]}")
                
                await asyncio.sleep(5)
                
    except OSError as e:
        print(f"[ERROR] Could not connect to {uri}")
        print("Make sure the realtime streaming system is running:")
        print("  python -m src.realtime_streaming_system")
    except KeyboardInterrupt:
        print("\n[OK] Simulation stopped")
    except Exception as e:
        print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Pattern Match Simulator for Dashboard Testing")
    print("=" * 60)
    print()
    asyncio.run(simulate_pattern_matches())
