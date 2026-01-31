"""Quick test for demo streamer"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test():
    from streaming_data_streamer import DemoStreamer
    
    print("Creating demo streamer...")
    streamer = DemoStreamer()
    
    print(f"Demo price: {streamer.demo_price}")
    print(f"Demo volume: {streamer.demo_volume}")
    print(f"Price deviation threshold: {streamer.max_price_deviation_pct}%")
    
    bars = []
    
    def on_new_bar(data):
        bars.append(data['bar'])
        print(f"[OK] Bar #{len(bars)}: ${data['bar']['Close']:.2f}")
    
    streamer.set_event_callbacks(on_new_bar=on_new_bar)
    
    print("\nStarting streamer for 15 seconds...")
    await streamer.start()
    
    for i in range(3):
        await asyncio.sleep(5)
        print(f"Waiting... ({i+1}/3)")
    
    await streamer.stop()
    
    print(f"\nTotal bars received: {len(bars)}")
    print(f"Test: {'PASSED' if len(bars) >= 3 else 'FAILED'}")

if __name__ == "__main__":
    asyncio.run(test())