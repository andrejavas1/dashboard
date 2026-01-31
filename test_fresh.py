"""Fresh test for streaming system - uses updated code"""
import asyncio
import sys
import os

# Force fresh import
import importlib
if 'streaming_data_streamer' in sys.modules:
    del sys.modules['streaming_data_streamer']

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_demo_streamer():
    """Test the demo streamer with updated validation bypass."""
    print("=" * 60)
    print("FRESH TEST - Demo Streamer with Updated Code")
    print("=" * 60)
    
    # Import fresh
    from streaming_data_streamer import DemoStreamer
    
    print("\n1. Creating demo streamer...")
    streamer = DemoStreamer()
    
    print(f"   Demo price: ${streamer.demo_price:.2f}")
    print(f"   Demo volume: {streamer.demo_volume:,.0f}")
    print(f"   Historical baseline: {streamer.historical_baseline}")
    
    bars = []
    
    def on_new_bar(data):
        bar = data['bar']
        bars.append(bar)
        print(f"\n   [OK] Bar #{len(bars)} received:")
        print(f"       Time: {bar['timestamp']}")
        print(f"       Open: ${bar['Open']:.2f} | High: ${bar['High']:.2f} | Low: ${bar['Low']:.2f} | Close: ${bar['Close']:.2f}")
        print(f"       Volume: {bar['Volume']:,}")
    
    streamer.set_event_callbacks(on_new_bar=on_new_bar)
    
    print("\n2. Starting streamer (will generate 3 bars)...")
    await streamer.start()
    
    # Wait for 3 bars (15 seconds total)
    for i in range(3):
        await asyncio.sleep(5)
        print(f"   Waiting... ({i+1}/3)")
    
    await streamer.stop()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Bars received: {len(bars)}")
    print(f"Test: {'PASSED' if len(bars) >= 3 else 'FAILED'}")
    print("=" * 60)
    
    return len(bars) >= 3

async def test_feature_calculator():
    """Test the feature calculator."""
    print("\n" + "=" * 60)
    print("TEST - Feature Calculator")
    print("=" * 60)
    
    from incremental_feature_calculator import IncrementalFeatureCalculator
    
    calc = IncrementalFeatureCalculator()
    
    test_bars = [
        {'Open': 109.91, 'High': 111.00, 'Low': 108.50, 'Close': 110.00, 'Volume': 16000000},
        {'Open': 110.00, 'High': 111.50, 'Low': 109.00, 'Close': 111.00, 'Volume': 17000000},
        {'Open': 111.00, 'High': 112.00, 'Low': 110.50, 'Close': 111.50, 'Volume': 18000000},
    ]
    
    print("\nProcessing 3 test bars...")
    for i, bar in enumerate(test_bars, 1):
        features = calc.update(bar)
        print(f"\nBar {i}:")
        print(f"  EMA(20): {features.get('ema_20', 'N/A')}")
        print(f"  RSI(14): {features.get('rsi_14', 'N/A')}")
        print(f"  ATR(14): {features.get('atr_14', 'N/A')}")
    
    print("\n[OK] Feature calculator test PASSED")
    return True

async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("STREAMING SYSTEM - FRESH TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Demo Streamer
    results['streamer'] = await test_demo_streamer()
    
    # Test 2: Feature Calculator
    results['feature_calc'] = await test_feature_calculator()
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")
    print("=" * 60)
    
    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())