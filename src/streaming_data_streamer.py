"""
WebSocket Data Streamer Module
Establishes real-time connection to fetch 15-minute OHLCV bars
"""

import os
import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from collections import deque
import pandas as pd
import numpy as np
import yaml
from websockets.client import connect
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WebSocketStreamer:
    """
    Real-time data streamer using WebSocket connection.
    Fetches 15-minute OHLCV bars and validates against historical baseline.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the WebSocket streamer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.streaming_config = self.config.get('streaming', {})
        
        # Connection settings
        self.ticker = self.streaming_config.get('ticker', 'XOM')
        self.interval = self.streaming_config.get('interval', '15m')
        self.buffer_size = self.streaming_config.get('buffer_size', 252)
        self.ws_url = self.streaming_config.get('ws_url', 'wss://stream.alpaca.markets/v2/iex')
        self.reconnect_interval = self.streaming_config.get('reconnect_interval', 5)
        self.max_reconnect_attempts = self.streaming_config.get('max_reconnect_attempts', 10)
        
        # Validation settings
        validation_config = self.streaming_config.get('validation', {})
        self.confidence_threshold = validation_config.get('confidence_threshold', 98)
        self.max_price_deviation_pct = validation_config.get('max_price_deviation_pct', 5)
        self.max_volume_deviation_pct = validation_config.get('max_volume_deviation_pct', 10)
        
        # Data storage
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.historical_baseline = None
        self.last_valid_bar = None
        
        # Connection state
        self.websocket = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.is_running = False
        
        # Event callbacks
        self.on_new_bar: Optional[Callable] = None
        self.on_connection_status: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # API keys (from .env or config)
        self.api_key = os.getenv('ALPACA_API_KEY', 'demo')
        self.api_secret = os.getenv('ALPACA_API_SECRET', 'demo')
        
        logger.info(f"WebSocket Streamer initialized for {self.ticker} at {self.interval} interval")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
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
    
    def load_historical_baseline(self, data_path: str = "data/ohlcv.json"):
        """
        Load historical OHLCV data for validation baseline.
        
        Args:
            data_path: Path to historical data file
        """
        try:
            with open(data_path, 'r') as f:
                historical_data = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Calculate baseline statistics
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
            
            logger.info(f"Historical baseline loaded from {data_path}")
            logger.info(f"  Avg Close: ${self.historical_baseline['avg_close']:.2f}")
            logger.info(f"  Avg Volume: {self.historical_baseline['avg_volume']:,.0f}")
            
        except FileNotFoundError:
            logger.warning(f"Historical data not found at {data_path}, validation disabled")
            self.historical_baseline = None
        except Exception as e:
            logger.error(f"Error loading historical baseline: {e}")
            self.historical_baseline = None
    
    def validate_bar(self, bar: Dict) -> tuple[bool, str]:
        """
        Validate incoming bar against historical baseline.
        
        Args:
            bar: OHLCV bar dictionary
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Demo mode: always accept bars
        if isinstance(self, DemoStreamer):
            return True, "Demo mode"
        
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
            return False, f"Price deviation {price_deviation:.2f}% exceeds threshold {self.max_price_deviation_pct}%"
        
        # Check volume deviation
        avg_volume = self.historical_baseline['avg_volume']
        volume_deviation = abs(volume - avg_volume) / avg_volume * 100
        
        if volume_deviation > self.max_volume_deviation_pct:
            return False, f"Volume deviation {volume_deviation:.2f}% exceeds threshold {self.max_volume_deviation_pct}%"
        
        # Check for reasonable OHLC relationships
        high = bar.get('High', close)
        low = bar.get('Low', close)
        open_price = bar.get('Open', close)
        
        if not (low <= open_price <= high):
            return False, "Invalid OHLC: Open not within High-Low range"
        
        if not (low <= close <= high):
            return False, "Invalid OHLC: Close not within High-Low range"
        
        return True, "Valid"
    
    def _emit(self, event_type: str, data: Dict):
        """Emit event to registered callbacks."""
        if event_type == 'new_bar' and self.on_new_bar:
            # Check if callback is async
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
    
    def _parse_bar_message(self, message: str) -> Optional[Dict]:
        """
        Parse WebSocket message into OHLCV bar format.
        
        Args:
            message: WebSocket message string
            
        Returns:
            Parsed bar dictionary or None
        """
        try:
            data = json.loads(message)
            
            # Handle different message formats
            if 'T' in data:  # Alpaca format
                if data['T'] == 'q':  # Quote
                    # Convert quote to bar (simplified)
                    return {
                        'timestamp': datetime.fromtimestamp(data['t'] / 1000),
                        'Open': data.get('bp', 0),
                        'High': data.get('ap', 0),
                        'Low': data.get('bp', 0),
                        'Close': data.get('ap', 0),
                        'Volume': data.get('bs', 0) + data.get('as', 0)
                    }
                elif data['T'] == 'b':  # Bar
                    return {
                        'timestamp': datetime.fromtimestamp(data['t'] / 1000),
                        'Open': data.get('o', 0),
                        'High': data.get('h', 0),
                        'Low': data.get('l', 0),
                        'Close': data.get('c', 0),
                        'Volume': data.get('v', 0)
                    }
            elif 'symbol' in data:  # Generic format
                return {
                    'timestamp': datetime.now(),
                    'Open': data.get('open', 0),
                    'High': data.get('high', 0),
                    'Low': data.get('low', 0),
                    'Close': data.get('close', 0),
                    'Volume': data.get('volume', 0)
                }
            
            return None
        except json.JSONDecodeError:
            logger.error(f"Failed to parse message: {message}")
            return None
        except Exception as e:
            logger.error(f"Error parsing bar message: {e}")
            return None
    
    async def _connect(self):
        """Establish WebSocket connection."""
        try:
            logger.info(f"Connecting to {self.ws_url}...")
            self.websocket = await connect(self.ws_url)
            
            # Authenticate with Alpaca
            if self.api_key and self.api_secret and self.api_key != 'demo':
                auth_message = {
                    "action": "auth",
                    "key": self.api_key,
                    "secret": self.api_secret
                }
                await self.websocket.send(json.dumps(auth_message))
                logger.info("Authentication sent")
                
                # Wait for auth response
                auth_response = await self.websocket.recv()
                logger.info(f"Auth response: {auth_response}")
            
            self.is_connected = True
            self.reconnect_attempts = 0
            
            self._emit('connection_status', {
                'status': 'connected',
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info("WebSocket connected successfully")
            
        except OSError as e:
            logger.error(f"Connection error: {e}")
            await self._handle_disconnect()
        except Exception as e:
            logger.error(f"Unexpected error connecting: {e}")
            await self._handle_disconnect()
    
    async def _handle_disconnect(self):
        """Handle WebSocket disconnection."""
        self.is_connected = False
        self.websocket = None
        
        self._emit('connection_status', {
            'status': 'disconnected',
            'timestamp': datetime.now().isoformat()
        })
        
        logger.warning("WebSocket disconnected")
        
        # Attempt reconnection
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            logger.info(f"Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} in {self.reconnect_interval}s...")
            await asyncio.sleep(self.reconnect_interval)
            await self._connect()
        else:
            logger.error("Max reconnection attempts reached. Stopping.")
            self.is_running = False
    
    async def _subscribe(self):
        """Subscribe to ticker data."""
        if not self.is_connected or self.websocket is None:
            logger.error("Cannot subscribe: not connected")
            return
        
        try:
            # Alpaca subscription format
            subscribe_message = {
                "action": "subscribe",
                "trades": [self.ticker],
                "quotes": [self.ticker],
                "bars": [self.ticker]
            }
            
            await self.websocket.send(json.dumps(subscribe_message))
            logger.info(f"Subscribed to {self.ticker}")
            
        except Exception as e:
            logger.error(f"Error subscribing: {e}")
    
    async def _listen(self):
        """Listen for incoming messages."""
        if not self.is_connected or self.websocket is None:
            return
        
        try:
            async for message in self.websocket:
                bar = self._parse_bar_message(message)
                
                if bar:
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
                        
                        logger.debug(f"New bar: {bar['timestamp']} | Close: ${bar['Close']:.2f} | Vol: {bar['Volume']:,}")
                    else:
                        logger.warning(f"Invalid bar rejected: {reason}")
                        self._emit('error', {
                            'type': 'validation_error',
                            'reason': reason,
                            'bar': bar
                        })
        
        except ConnectionClosed:
            logger.warning("Connection closed by server")
            await self._handle_disconnect()
        except Exception as e:
            logger.error(f"Error listening for messages: {e}")
            await self._handle_disconnect()
    
    async def start(self):
        """Start the WebSocket streamer."""
        if self.is_running:
            logger.warning("Streamer already running")
            return
        
        self.is_running = True
        
        # Load historical baseline
        if self.historical_baseline is None:
            self.load_historical_baseline()
        
        # Connect and start listening
        await self._connect()
        
        if self.is_connected:
            await self._subscribe()
            await self._listen()
    
    async def stop(self):
        """Stop the WebSocket streamer."""
        logger.info("Stopping WebSocket streamer...")
        self.is_running = False
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        self.is_connected = False
        
        self._emit('connection_status', {
            'status': 'stopped',
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info("WebSocket streamer stopped")
    
    def get_latest_bar(self) -> Optional[Dict]:
        """Get the most recent valid bar."""
        return self.last_valid_bar
    
    def get_buffer(self) -> List[Dict]:
        """Get the current data buffer."""
        return list(self.data_buffer)
    
    def get_buffer_as_dataframe(self) -> pd.DataFrame:
        """Convert buffer to pandas DataFrame."""
        if not self.data_buffer:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.data_buffer)
        df.set_index('timestamp', inplace=True)
        return df


# Demo mode for testing without real WebSocket
class DemoStreamer(WebSocketStreamer):
    """
    Demo streamer that generates synthetic data for testing.
    Useful when WebSocket API is not available.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        super().__init__(config_path)
        self.demo_interval = 5  # seconds between bars (demo speed)
        # Start near historical baseline for validation
        if self.historical_baseline:
            self.demo_price = self.historical_baseline['avg_close']
            self.demo_volume = self.historical_baseline['avg_volume']
        else:
            self.demo_price = 100.0
            self.demo_volume = 1000000
        
        # Disable validation for demo mode
        self.historical_baseline = None
    
    async def _connect(self):
        """Override to skip real WebSocket connection."""
        logger.info("Demo mode: Starting synthetic data generation")
        self.is_connected = True
        self.reconnect_attempts = 0
        
        self._emit('connection_status', {
            'status': 'connected',
            'timestamp': datetime.now().isoformat()
        })
    
    async def _generate_demo_bar(self) -> Dict:
        """Generate a synthetic OHLCV bar."""
        # Random price movement
        change = np.random.normal(0, 0.5)  # Mean 0, std 0.5
        open_price = self.demo_price
        close_price = open_price + change
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.2))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.2))
        
        # Random volume
        volume = int(self.demo_volume * np.random.uniform(0.8, 1.2))
        
        bar = {
            'timestamp': datetime.now(),
            'Open': round(open_price, 2),
            'High': round(high_price, 2),
            'Low': round(low_price, 2),
            'Close': round(close_price, 2),
            'Volume': volume
        }
        
        self.demo_price = close_price
        return bar
    
    async def _listen(self):
        """Generate demo bars at regular intervals."""
        logger.info(f"Starting demo mode - generating bars every {self.demo_interval}s")
        
        while self.is_running:
            bar = await self._generate_demo_bar()
            
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
                
                logger.info(f"Demo bar: {bar['timestamp']} | Close: ${bar['Close']:.2f} | Vol: {bar['Volume']:,}")
            else:
                logger.warning(f"Invalid demo bar rejected: {reason}")
            
            await asyncio.sleep(self.demo_interval)


if __name__ == "__main__":
    # Demo usage
    async def on_new_bar(data):
        print(f"New bar received: {data['timestamp']} | Close: ${data['bar']['Close']:.2f}")
    
    async def on_connection_status(data):
        print(f"Connection status: {data['status']}")
    
    async def on_error(data):
        print(f"Error: {data}")
    
    async def main():
        # Create demo streamer
        streamer = DemoStreamer()
        streamer.set_event_callbacks(
            on_new_bar=on_new_bar,
            on_connection_status=on_connection_status,
            on_error=on_error
        )
        
        # Run for 60 seconds
        try:
            await streamer.start()
            await asyncio.sleep(60)
            await streamer.stop()
        except KeyboardInterrupt:
            await streamer.stop()
    
    asyncio.run(main())