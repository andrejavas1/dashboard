"""
Run streaming orchestrator with Alpaca REST API
"""
import asyncio
import sys
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from streaming_orchestrator import run_orchestrator

if __name__ == "__main__":
    print("Starting streaming orchestrator...")
    print(f"Provider: alpaca_rest (from config.yaml)")
    print(f"API Key: {os.getenv('ALPACA_API_KEY', 'NOT SET')[:10]}...")
    asyncio.run(run_orchestrator())
