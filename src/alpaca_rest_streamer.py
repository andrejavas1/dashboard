"""
Alpaca REST API Streamer Module
Polls Alpaca Data API for real-time bars (free tier compatible)
"""

import os
import logging
import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Callable
from collections import deque
import pandas as pd
import numpy as np
import yaml
import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlpacaRESTStreamer:
    """
    Real-time data streamer using Alpaca REST API polling.
    Works with Alpaca free tier by polling every 15 seconds.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the Alpaca REST streamer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.streaming_config = self.config.get('streaming', {})
        
        # Connection settings
        self.ticker = self.streaming_config.get('ticker', 'XOM')
        self.interval = self._convert_timeframe(self.streaming_config.get('interval', '15m'))
        self.buffer_size = self.streaming_config.get('buffer_size', 252)
        # Poll interval should match the bar timeframe (e.g., 15m bars = check every 15m)
        self.poll_interval = self._get_poll_interval(self.streaming_config.get('interval', '15m'))
        
        # Alpaca API settings
        self.base_url = "https://data.alpaca.markets/v2"
        self.api_key = os.getenv('ALPACA_API_KEY', '')
        self.api_secret = os.getenv('ALPACA_API_SECRET', '')
        
        # Validation settings
        validation_config = self.streaming_config.get('validation', {})
        self.max_price_deviation_pct = validation_config.get('max_price_deviation_pct', 5)
        self.max_volume_deviation_pct = validation_config.get('max_volume_deviation_pct', 10)
        
        # Data storage
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.historical_baseline = None
        self.last_valid_bar = None
        self.last_bar_timestamp = None
        
        # Connection state
        self.is_connected = False
        self.is_running = False
        self.session = None
        
        # Timeframe mapping
        self.timeframe_map = {
            '1m': '1Min',
            '5m': '5Min',
            '15m': '15Min',
            '30m': '30Min',
            '1h': '1Hour',
            '1d': '1Day'
        }
        
        # Event callbacks
        self.on_new_bar: Optional[Callable] = None
        self.on_connection_status: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        logger.info(f"Alpaca REST Streamer initialized for {self.ticker}")
        logger.info(f"API Key: {self.api_key[:10]}..." if self.api_key else "API Key: Not set")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return {}
    
    def set_event_callbacks(self, on_new_bar: Callable = None, 
                           on_connection_status: Callable = None,
                           on_error: Callable = None):
        """Set event callback functions."""
        if on_new_bar:
            self.on_new_bar = on_new_bar
        if on_connection_status:
            self.on_connection_status = on_connection_status
        if on_error:
            self.on_error = on_error
    
    def _convert_timeframe(self, interval: str) -> str:
        """Convert config interval to Alpaca timeframe format."""
        return self.timeframe_map.get(interval, interval)
    
    def _get_poll_interval(self, interval: str) -> int:
        """Get appropriate poll interval in seconds based on bar timeframe."""
        # Poll at the bar interval (e.g., 15m bars = check every 15 minutes)
        # For shorter timeframes, we poll slightly more frequently to catch new bars
        interval_map = {
            '1m': 60,      # 1 minute bars: check every 1 minute
            '5m': 300,     # 5 minute bars: check every 5 minutes
            '15m': 900,    # 15 minute bars: check every 15 minutes
            '30m': 1800,   # 30 minute bars: check every 30 minutes
            '1h': 3600,    # 1 hour bars: check every hour
            '1d': 3600,    # Daily bars: check every hour (market open check)
        }
        return interval_map.get(interval, 900)  # Default to 15 minutes
    
    def load_historical_baseline(self, data_path: str = "data/ohlcv.json"):
        """Load historical OHLCV data for validation baseline."""
        try:
            with open(data_path, 'r') as f:
                historical_data = json.load(f)
            
            df = pd.DataFrame(historical_data)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            self.historical_baseline = {
                'avg_close': df['Close'].mean(),
                'avg_volume': df['Volume'].mean(),
                'std_close': df['Close'].std(),
                'std_volume': df['Volume'].std(),
                'min_close': df['Close'].min(),
                'max_close': df['Close'].max(),
                'min_volume': df['Volume'].min(),
                'max_volume': df['Volume'].max()
            }
            
            logger.info(f"Historical baseline loaded: Avg Close=${self.historical_baseline['avg_close']:.2f}")
            
        except FileNotFoundError:
            logger.warning(f"Historical data not found at {data_path}")
            self.historical_baseline = None
        except Exception as e:
            logger.error(f"Error loading historical baseline: {e}")
            self.historical_baseline = None
    
    def validate_bar(self, bar: Dict) -> tuple[bool, str]:
        """Validate incoming bar against historical baseline."""
        if self.historical_baseline is None:
            return True, "No baseline available"
        
        close = bar.get('Close', 0)
        volume = bar.get('Volume', 0)
        
        if close == 0 or volume == 0:
            return False, "Missing price or volume data"
        
        # Check price deviation
        avg_close = self.historical_baseline['avg_close']
        price_deviation = abs(close - avg_close) / avg_close * 100
        
        if price_deviation > self.max_price_deviation_pct:
            return False, f"Price deviation {price_deviation:.2f}% exceeds threshold"
        
        return True, "Valid"
    
    def _emit(self, event_type: str, data: Dict):
        """Emit event to registered callbacks."""
        if event_type == 'new_bar' and self.on_new_bar:
            if asyncio.iscoroutinefunction(self.on_new_bar):
                asyncio.create_task(self.on_new_bar(data))
            else:
                self.on_new_bar(data)
        elif event_type == 'connection_status' and self.on_connection_status:
            if asyncio.iscoroutinefunction(self.on_connection_status):
                asyncio.create_task(self.on_connection_status(data))
            else:
                self.on_connection_status(data)
        elif event_type == 'error' and self.on_error:
            if asyncio.iscoroutinefunction(self.on_error):
                asyncio.create_task(self.on_error(data))
            else:
                self.on_error(data)
    
    async def _fetch_latest_bars(self) -> Optional[Dict]:
        """Fetch latest bars from Alpaca REST API."""
        if not self.session:
            return None
        
        try:
            # Calculate time range (last 5 minutes)
            end = datetime.now(timezone.utc)
            start = end - timedelta(minutes=5)
            
            url = f"{self.base_url}/stocks/{self.ticker}/bars"
            params = {
                'start': start.isoformat(),
                'end': end.isoformat(),
                'timeframe': self.interval,
                'feed': 'iex',  # Use IEX feed (free tier)
                'limit': 10
            }
            
            headers = {
                'APCA-API-KEY-ID': self.api_key,
                'APCA-API-SECRET-KEY': self.api_secret
            }
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    bars = data.get('bars', [])
                    
                    if bars:
                        # Get the most recent bar
                        latest = bars[-1]
                        return {
                            'timestamp': datetime.fromisoformat(latest['t'].replace('Z', '+00:00')),
                            'Open': float(latest['o']),
                            'High': float(latest['h']),
                            'Low': float(latest['l']),
                            'Close': float(latest['c']),
                            'Volume': int(latest['v'])
                        }
                    else:
                        logger.debug("No new bars available")
                        return None
                else:
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching bars: {e}")
            return None
    
    async def _poll_loop(self):
        """Main polling loop."""
        poll_minutes = self.poll_interval // 60
        logger.info(f"Starting Alpaca REST polling (every {poll_minutes}m for {self.interval} bars)")
        
        while self.is_running:
            try:
                bar = await self._fetch_latest_bars()
                
                if bar:
                    # Check if this is a new bar
                    if self.last_bar_timestamp is None or bar['timestamp'] > self.last_bar_timestamp:
                        self.last_bar_timestamp = bar['timestamp']
                        
                        # Validate bar
                        is_valid, reason = self.validate_bar(bar)
                        
                        if is_valid:
                            self.last_valid_bar = bar
                            self.data_buffer.append(bar)
                            
                            # Emit new bar event
                            self._emit('new_bar', {
                                'bar': bar,
                                'timestamp': bar['timestamp'].isoformat(),
                                'buffer_size': len(self.data_buffer)
                            })
                            
                            logger.info(f"New bar: {bar['timestamp']} | Close: ${bar['Close']:.2f} | Vol: {bar['Volume']:,}")
                        else:
                            logger.warning(f"Invalid bar rejected: {reason}")
                    else:
                        logger.debug("Duplicate bar, skipping")
                
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error(f"Error in poll loop: {e}")
                await asyncio.sleep(self.poll_interval)
    
    async def start(self):
        """Start the Alpaca REST streamer."""
        if self.is_running:
            logger.warning("Streamer already running")
            return
        
        if not self.api_key or not self.api_secret:
            logger.error("Alpaca API keys not configured")
            self._emit('error', {'type': 'config_error', 'message': 'API keys not set'})
            return
        
        self.is_running = True
        
        # Load historical baseline
        if self.historical_baseline is None:
            self.load_historical_baseline()
        
        # Create HTTP session
        self.session = aiohttp.ClientSession()
        self.is_connected = True
        
        self._emit('connection_status', {
            'status': 'connected',
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info("Alpaca REST streamer started")
        
        # Start polling
        await self._poll_loop()
    
    async def stop(self):
        """Stop the Alpaca REST streamer."""
        logger.info("Stopping Alpaca REST streamer...")
        self.is_running = False
        
        if self.session:
            await self.session.close()
            self.session = None
        
        self.is_connected = False
        
        self._emit('connection_status', {
            'status': 'stopped',
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info("Alpaca REST streamer stopped")
    
    def get_latest_bar(self) -> Optional[Dict]:
        """Get the most recent valid bar."""
        return self.last_valid_bar
    
    def get_buffer(self) -> List[Dict]:
        """Get the current data buffer."""
        return list(self.data_buffer)


if __name__ == "__main__":
    # Demo usage
    async def on_new_bar(data):
        print(f"New bar: {data['timestamp']} | Close: ${data['bar']['Close']:.2f}")
    
    async def on_connection_status(data):
        print(f"Status: {data['status']}")
    
    async def on_error(data):
        print(f"Error: {data}")
    
    async def main():
        streamer = AlpacaRESTStreamer()
        streamer.set_event_callbacks(
            on_new_bar=on_new_bar,
            on_connection_status=on_connection_status,
            on_error=on_error
        )
        
        try:
            await streamer.start()
        except KeyboardInterrupt:
            await streamer.stop()
    
    asyncio.run(main())
