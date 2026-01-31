"""
Real-Time Streaming System - Integrated Daily Bar Updates with Dashboard
Fetches 15m prices, reconstructs daily bars, calculates features, matches patterns.
Broadcasts updates via WebSocket to connected dashboard clients.
"""

import os
import sys
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set
import websockets
from websockets.server import serve
import aiohttp
import yaml
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from realtime_daily_reconstructor import RealtimeDailyReconstructor
from realtime_feature_calculator import RealtimeFeatureCalculator
from tolerance_pattern_matcher import TolerancePatternMatcher
from market_regime_detector import MarketRegimeDetector
from probability_estimator import ProbabilityEstimator

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealtimeStreamingSystem:
    """
    Complete real-time streaming system that:
    1. Fetches 15m prices from Alpaca
    2. Reconstructs updating daily bars
    3. Recalculates all 140+ features on daily bars
    4. Matches patterns against daily bars
    5. Broadcasts updates to WebSocket clients
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the real-time streaming system."""
        self.config = self._load_config(config_path)
        self.streaming_config = self.config.get('streaming', {})
        
        # Component settings
        self.ticker = self.streaming_config.get('ticker', 'XOM')
        self.interval = '15Min'  # We fetch 15m bars
        self.poll_interval = 900  # Poll every 15 minutes
        
        # WebSocket server settings
        self.ws_port = self.streaming_config.get('dashboard', {}).get('ws_port', 5010)
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.ws_server = None
        
        # Initialize components
        self.daily_reconstructor = RealtimeDailyReconstructor()
        self.feature_calculator = RealtimeFeatureCalculator()
        self.pattern_matcher = TolerancePatternMatcher(
            tolerance_pct=self.streaming_config.get('pattern_matching', {}).get('tolerance_pct', 5.0)
        )
        self.pattern_matcher.load_patterns()
        self.regime_detector = MarketRegimeDetector()
        self.probability_estimator = ProbabilityEstimator()
        
        # State
        self.is_running = False
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_update_time: Optional[datetime] = None
        
        # Current day tracking
        self.current_regime: Optional[Dict] = None
        self.current_matches: List[Dict] = []
        self.triggered_patterns: List[Dict] = []
        
        logger.info(f"Realtime Streaming System initialized for {self.ticker}")
        logger.info(f"Loaded {self.pattern_matcher.get_pattern_count()} patterns for matching")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Config not found at {config_path}, using defaults")
            return {}
    
    # ========== WebSocket Server Methods ==========
    
    async def _handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle a new WebSocket client connection."""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send initial state
        await self._send_to_client(websocket, {
            'type': 'connected',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'ticker': self.ticker,
            'patterns_loaded': self.pattern_matcher.get_pattern_count()
        })
        
        # Send current regime if available
        if self.current_regime:
            await self._send_to_client(websocket, {
                'type': 'regime_update',
                'data': self.current_regime
            })
        
        # Send current daily bar
        current_bar = self.daily_reconstructor.get_current_daily_bar()
        if current_bar:
            await self._send_to_client(websocket, {
                'type': 'daily_bar_update',
                'data': current_bar
            })
        
        try:
            async for message in websocket:
                await self._handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        finally:
            self.clients.discard(websocket)
    
    async def _handle_client_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """Handle incoming message from client."""
        try:
            data = json.loads(message)
            action = data.get('action')
            
            if action == 'ping':
                await self._send_to_client(websocket, {
                    'type': 'pong',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            elif action == 'get_patterns':
                await self._send_to_client(websocket, {
                    'type': 'pattern_list',
                    'patterns': self.pattern_matcher.get_all_patterns()
                })
            elif action == 'get_matches':
                await self._send_to_client(websocket, {
                    'type': 'current_matches',
                    'matches': self.current_matches
                })
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from client: {message}")
    
    async def _send_to_client(self, websocket: websockets.WebSocketServerProtocol, data: Dict):
        """Send data to a specific client."""
        try:
            await websocket.send(json.dumps(data))
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
            self.clients.discard(websocket)
    
    async def _broadcast(self, data: Dict):
        """Broadcast data to all connected clients."""
        if not self.clients:
            return
        
        message = json.dumps(data)
        disconnected = []
        
        for client in self.clients:
            try:
                await client.send(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(client)
        
        # Remove disconnected clients
        for client in disconnected:
            self.clients.discard(client)
    
    # ========== Data Fetching Methods ==========
    
    async def _fetch_latest_15m_bars(self) -> List[Dict]:
        """Fetch latest 15-minute bars from Alpaca."""
        if not self.session:
            return []
        
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(minutes=20)  # Get last 20 minutes
            
            url = f"https://data.alpaca.markets/v2/stocks/{self.ticker}/bars"
            params = {
                'start': start.isoformat(),
                'end': end.isoformat(),
                'timeframe': self.interval,
                'feed': 'iex',
                'limit': 5
            }
            headers = {
                'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY', ''),
                'APCA-API-SECRET-KEY': os.getenv('ALPACA_API_SECRET', '')
            }
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    bars = data.get('bars', [])
                    
                    # Convert to our format
                    result = []
                    for bar in bars:
                        result.append({
                            'timestamp': datetime.fromisoformat(bar['t'].replace('Z', '+00:00')),
                            'Open': float(bar['o']),
                            'High': float(bar['h']),
                            'Low': float(bar['l']),
                            'Close': float(bar['c']),
                            'Volume': int(bar['v'])
                        })
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching bars: {e}")
            return []
    
    # ========== Core Processing Methods ==========
    
    async def _process_new_bars(self, bars_15m: List[Dict]):
        """Process new 15m bars and update everything."""
        for bar in bars_15m:
            # Skip if we've already processed this bar
            if self.last_update_time and bar['timestamp'] <= self.last_update_time:
                continue
            
            self.last_update_time = bar['timestamp']
            
            # 1. Update daily bar reconstruction
            daily_bar, is_new_day = self.daily_reconstructor.process_15m_bar(bar)
            
            if is_new_day:
                logger.info(f"New trading day: {daily_bar['Date']}")
            
            # 2. Get full daily data for feature calculation
            daily_df = self.daily_reconstructor.get_full_daily_data(include_current=True)
            
            if len(daily_df) < 50:
                logger.warning(f"Need 50 days of data, have {len(daily_df)}")
                continue
            
            # 3. Calculate all features on daily bars
            features = self.feature_calculator.calculate_all_features(daily_df)
            
            # 4. Detect market regime
            new_regime = self.regime_detector.update(features)
            
            # Broadcast regime change if occurred
            if self.current_regime and self._regime_changed(self.current_regime, new_regime):
                await self._broadcast({
                    'type': 'regime_change',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'from': self.current_regime,
                    'to': new_regime
                })
                logger.info(f"Regime changed: {self.current_regime.get('trend_regime')} → {new_regime.get('trend_regime')}")
            
            self.current_regime = new_regime
            
            # 5. Match patterns
            self.current_matches = self.pattern_matcher.match_all_patterns(features)
            
            # 6. Calculate probabilities and check for triggers
            self.triggered_patterns = []
            for match in self.current_matches:
                if match.get('confidence_score', 0) >= 0.9:  # 90% threshold
                    probability = self.probability_estimator.estimate_probability(
                        match, new_regime, features['Close']
                    )
                    
                    trigger_data = {
                        'pattern_id': match['pattern_id'],
                        'confidence': match['confidence_score'],
                        'probability': probability.get('final_probability', 0),
                        'signal': probability.get('actionable_signals', {}).get('entry_signal', 'HOLD'),
                        'entry_price': probability.get('actionable_signals', {}).get('entry_price'),
                        'target_price': probability.get('price_targets', [{}])[0].get('target_price'),
                        'stop_loss': probability.get('actionable_signals', {}).get('stop_loss')
                    }
                    
                    self.triggered_patterns.append(trigger_data)
                    
                    # Broadcast high-confidence trigger
                    await self._broadcast({
                        'type': 'pattern_triggered',
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'data': trigger_data
                    })
            
            # 7. Broadcast updates to all clients
            await self._broadcast({
                'type': 'daily_bar_update',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'data': daily_bar,
                'is_new_day': is_new_day
            })
            
            await self._broadcast({
                'type': 'feature_update',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'features': {
                    'Close': features.get('Close'),
                    'RSI_14': features.get('RSI_14'),
                    'SMA_50': features.get('SMA_50'),
                    'ATR_14_pct': features.get('ATR_14_pct'),
                    'Volume_Ratio': features.get('Volume_Ratio')
                }
            })
            
            await self._broadcast({
                'type': 'matches_update',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'matches': self.current_matches,
                'triggered_count': len(self.triggered_patterns)
            })
            
            # Log summary
            logger.info(f"Updated daily bar {daily_bar['Date']}: "
                       f"C:${daily_bar['Close']:.2f} | "
                       f"RSI:{features.get('RSI_14', 0):.1f} | "
                       f"Matches:{len(self.current_matches)} | "
                       f"Triggered:{len(self.triggered_patterns)}")
    
    def _regime_changed(self, prev: Dict, curr: Dict) -> bool:
        """Check if regime has changed."""
        return (
            prev.get('volatility_regime') != curr.get('volatility_regime') or
            prev.get('trend_regime') != curr.get('trend_regime')
        )
    
    # ========== Main Loop ==========
    
    async def _data_fetching_loop(self):
        """Main loop for fetching data and processing updates."""
        logger.info(f"Starting data fetching loop (every {self.poll_interval}s)")
        
        while self.is_running:
            try:
                bars = await self._fetch_latest_15m_bars()
                
                if bars:
                    await self._process_new_bars(bars)
                else:
                    logger.debug("No new bars available")
                
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error(f"Error in data fetching loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def start(self):
        """Start the real-time streaming system."""
        if self.is_running:
            logger.warning("System already running")
            return
        
        self.is_running = True
        
        # Create HTTP session for Alpaca API
        self.session = aiohttp.ClientSession()
        
        # Start WebSocket server
        self.ws_server = await serve(
            self._handle_client,
            "0.0.0.0",
            self.ws_port,
            ping_interval=30,
            ping_timeout=20
        )
        
        logger.info(f"WebSocket server started on ws://0.0.0.0:{self.ws_port}")
        logger.info("Dashboard can connect to receive real-time updates")
        
        # Start data fetching loop
        await self._data_fetching_loop()
    
    async def stop(self):
        """Stop the real-time streaming system."""
        logger.info("Stopping Realtime Streaming System...")
        self.is_running = False
        
        if self.ws_server:
            self.ws_server.close()
            await self.ws_server.wait_closed()
        
        if self.session:
            await self.session.close()
        
        # Close all client connections
        for client in list(self.clients):
            try:
                await client.close()
            except:
                pass
        self.clients.clear()
        
        logger.info("Realtime Streaming System stopped")


async def main():
    """Run the real-time streaming system."""
    system = RealtimeStreamingSystem()
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await system.stop()


if __name__ == "__main__":
    asyncio.run(main())
