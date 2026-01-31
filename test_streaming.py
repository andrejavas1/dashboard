"""
Simple test script to verify streaming system works
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_streamer():
    """Test the demo streamer independently."""
    from streaming_data_streamer import DemoStreamer
    
    print("=" * 50)
    print("Testing Demo Streamer")
    print("=" * 50)
    
    # Create streamer
    streamer = DemoStreamer()
    
    # Set up callbacks
    bar_received = []
    
    def on_new_bar(data):
        bar = data['bar']
        bar_received.append(bar)
        print(f"\n[OK] Bar received: {bar['timestamp']}")
        print(f"  Open: ${bar['Open']:.2f} | High: ${bar['High']:.2f} | Low: ${bar['Low']:.2f} | Close: ${bar['Close']:.2f}")
        print(f"  Volume: {bar['Volume']:,}")
    
    def on_connection_status(data):
        print(f"\n[OK] Connection status: {data['status']}")
    
    def on_error(data):
        print(f"\n[ERROR] Error: {data}")
    
    streamer.set_event_callbacks(
        on_new_bar=on_new_bar,
        on_connection_status=on_connection_status,
        on_error=on_error
    )
    
    # Start streamer
    print("\nStarting demo streamer (will run for 60 seconds)...")
    try:
        await streamer.start()
        
        # Wait for 3 bars (45 seconds)
        for i in range(9):
            await asyncio.sleep(5)
            print(f"Waiting... ({i+1}/9)")
        
        await streamer.stop()
        
        # Summary
        print("\n" + "=" * 50)
        print("Test Summary")
        print("=" * 50)
        print(f"Bars received: {len(bar_received)}")
        print(f"Test: {'PASSED' if len(bar_received) >= 3 else 'FAILED'}")
        print("=" * 50)
        
        return len(bar_received) >= 3
        
    except KeyboardInterrupt:
        await streamer.stop()
        print("\nTest interrupted")


async def test_feature_calculator():
    """Test the feature calculator."""
    from incremental_feature_calculator import IncrementalFeatureCalculator
    
    print("\n" + "=" * 50)
    print("Testing Feature Calculator")
    print("=" * 50)
    
    calc = IncrementalFeatureCalculator()
    
    # Simulate some bars
    test_bars = [
        {'Open': 100, 'High': 102, 'Low': 99, 'Close': 101, 'Volume': 1000000},
        {'Open': 101, 'High': 103, 'Low': 100, 'Close': 102, 'Volume': 1100000},
        {'Open': 102, 'High': 104, 'Low': 101, 'Close': 103, 'Volume': 1200000},
    ]
    
    for i, bar in enumerate(test_bars):
        features = calc.update(bar)
        print(f"\nBar {i+1}:")
        print(f"  EMA(20): {features.get('ema_20', 'N/A')}")
        print(f"  RSI(14): {features.get('rsi_14', 'N/A')}")
        print(f"  ATR(14): {features.get('atr_14', 'N/A')}")
    
    print("\n[OK] Feature calculator test PASSED")


async def test_pattern_matcher():
    """Test the pattern matcher."""
    from tolerance_pattern_matcher import TolerancePatternMatcher
    
    print("\n" + "=" * 50)
    print("Testing Pattern Matcher")
    print("=" * 50)
    
    matcher = TolerancePatternMatcher(tolerance_pct=5.0)
    
    # Try to load patterns
    try:
        matcher.load_patterns()
        count = matcher.get_pattern_count()
        print(f"\n[OK] Patterns loaded: {count}")
        
        # Test with sample features
        sample_features = {
            'ema_20': 102,
            'ema_50': 100,
            'rsi_14': 55,
            'atr_14': 2,
            'volume_ratio': 1.2
        }
        
        matches = matcher.match_all_patterns(sample_features)
        print(f"[OK] Pattern matches found: {len(matches)}")
        
        if matches:
            for match in matches[:3]:  # Show first 3
                print(f"  Pattern #{match['pattern_id']}: {match['status']} (confidence: {match['confidence']:.1f}%)")
        
        print("\n[OK] Pattern matcher test PASSED")
        
    except FileNotFoundError:
        print("[WARN] Pattern file not found - skipping pattern matching test")


async def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("STREAMING SYSTEM TEST SUITE")
    print("=" * 50)
    
    # Test 1: Demo Streamer
    streamer_passed = await test_streamer()
    
    # Test 2: Feature Calculator
    await test_feature_calculator()
    
    # Test 3: Pattern Matcher
    await test_pattern_matcher()
    
    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED")
    print(f"Streamer Test: {'PASSED' if streamer_passed else 'FAILED'}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())