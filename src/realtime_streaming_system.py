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
from pattern_outcome_tracker import PatternOutcomeTracker

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
        self.poll_interval = 900  # Poll every 15 minutes (900 seconds)
        self.retry_interval = 60  # Retry every 1 minute on error
        self.consecutive_errors = 0
        self.max_retries_before_backoff = 5
        
        # WebSocket server settings
        self.ws_port = self.streaming_config.get('dashboard', {}).get('ws_port', 5012)
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.ws_server = None
        
        # Initialize components with ticker support
        self.daily_reconstructor = RealtimeDailyReconstructor(ticker=self.ticker)
        self.feature_calculator = RealtimeFeatureCalculator()
        self.pattern_matcher = TolerancePatternMatcher(
            tolerance_pct=self.streaming_config.get('pattern_matching', {}).get('tolerance_pct', 5.0)
        )
        # CRITICAL: Pass ticker to load patterns from ticker-specific directory
        self.pattern_matcher.load_patterns(ticker=self.ticker)
        self.regime_detector = MarketRegimeDetector()
        self.outcome_tracker = PatternOutcomeTracker()
        self.probability_estimator = ProbabilityEstimator()
        
        # State
        self.is_running = False
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_update_time: Optional[datetime] = None
        
        # Current day tracking
        self.current_regime: Optional[Dict] = None
        self.current_matches: List[Dict] = []
        self.triggered_patterns: List[Dict] = []
        
        # Pattern matches history (timestamped log of all matches)
        self.matches_history: List[Dict] = []
        self.max_history_size = 1000  # Keep last 1000 match events
        
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
        
        # Send current daily bar with regime info
        current_bar = self.daily_reconstructor.get_current_daily_bar()
        if current_bar:
            logger.info(f"Sending initial daily bar to new client: {current_bar}")
            await self._send_to_client(websocket, {
                'type': 'daily_bar_update',
                'data': current_bar,  # FIX: Use 'data' key to match broadcast format
                'regime': self.current_regime or {},
                'intraday_stats': {'updates': current_bar.get('IntradayUpdates', 0)}
            })
        else:
            logger.warning("No current daily bar available to send to new client")
        
        # Send current regime if available
        if self.current_regime:
            await self._send_to_client(websocket, {
                'type': 'regime_update',
                'data': self.current_regime
            })
        
        # Send current matches
        if self.current_matches:
            await self._send_to_client(websocket, {
                'type': 'matches_update',
                'matches': self.current_matches,
                'triggered_count': len(self.triggered_patterns)
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
                await self._send_to_client(websocket, {'type': 'pong'})
            elif action == 'get_regime_history':
                # Send regime history
                history = self.regime_detector.get_regime_history()
                await self._send_to_client(websocket, {
                    'type': 'regime_history',
                    'data': history
                })
            else:
                logger.warning(f"Unknown action: {action}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received: {message}")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
    
    async def _send_to_client(self, websocket: websockets.WebSocketServerProtocol, data: Dict):
        """Send data to a specific client."""
        try:
            await websocket.send(json.dumps(data))
        except websockets.exceptions.ConnectionClosed:
            logger.debug("Client connection closed during send")
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
    
    async def _broadcast(self, data: Dict):
        """Broadcast data to all connected clients."""
        if not self.clients:
            return
        
        message = json.dumps(data)
        disconnected = set()
        
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        for client in disconnected:
            self.clients.discard(client)
    
    # ========== Data Fetching Methods ==========
    
    async def _fetch_latest_15m_bars(self) -> List[Dict]:
        """Fetch latest 15-minute bars from Alpaca."""
        try:
            api_key = os.getenv('ALPACA_API_KEY')
            api_secret = os.getenv('ALPACA_API_SECRET')
            
            if not api_key or not api_secret:
                logger.error("Alpaca API credentials not found")
                return []
            
            # Calculate time range (fetch last 24 hours to ensure we have data)
            end = datetime.now(timezone.utc)
            start = end - timedelta(hours=24)
            
            url = f"https://data.alpaca.markets/v2/stocks/{self.ticker}/bars"
            
            # Use IEX feed for free tier (SIP requires paid subscription)
            # Free tier: 200 requests/minute, IEX feed only
            feed = self.streaming_config.get('alpaca', {}).get('feed', 'iex')
            
            params = {
                'timeframe': self.interval,
                'start': start.isoformat(),
                'end': end.isoformat(),
                'feed': feed,  # 'iex' for free tier, 'sip' for paid
                'sort': 'asc'
            }
            
            logger.debug(f"Fetching bars with {feed} feed...")
            headers = {
                'APCA-API-KEY-ID': api_key,
                'APCA-API-SECRET-KEY': api_secret
            }
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    bars = data.get('bars', [])
                    
                    # Convert to our format
                    result = []
                    for bar in bars:
                        result.append({
                            'timestamp': bar['t'],
                            'open': bar['o'],
                            'high': bar['h'],
                            'low': bar['l'],
                            'close': bar['c'],
                            'volume': bar['v']
                        })
                    
                    return result
                else:
                    text = await response.text()
                    logger.error(f"Error fetching bars: {response.status} - {text}")
                    return []
                    
        except Exception as e:
            logger.error(f"Exception fetching bars: {e}")
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
            features_df = self.feature_calculator.calculate(daily_df)
            # Convert last row to dict for regime detector and pattern matching
            if features_df is not None and not features_df.empty:
                features = features_df.iloc[-1].to_dict()
            else:
                features = {}
            
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
            
            # Save regime history to file for dashboard
            await self._save_regime_history()
            
            # 5. Match patterns
            self.current_matches = self.pattern_matcher.match_all_patterns(features)
            
            # 5.5 Store matches in history with timestamp
            if self.current_matches:
                match_entry = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'date': daily_bar['Date'],
                    'close': daily_bar['Close'],
                    'regime': self.current_regime.copy() if self.current_regime else {},
                    'matches_count': len(self.current_matches),
                    'triggered_count': 0,
                    'matches': []
                }
                
                for match in self.current_matches:
                    match_info = {
                        'pattern_id': match.get('pattern_id'),
                        'confidence_score': match.get('confidence_score', 0),
                        'direction': match.get('direction', 'unknown'),
                        'triggered': match.get('confidence_score', 0) >= 0.9
                    }
                    if match_info['triggered']:
                        match_entry['triggered_count'] += 1
                    match_entry['matches'].append(match_info)
                
                self.matches_history.append(match_entry)
                
                # Trim history if too large
                if len(self.matches_history) > self.max_history_size:
                    self.matches_history = self.matches_history[-self.max_history_size:]
                
                # Save to file
                await self._save_matches_history()
            
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
                    
                    # Record trade in outcome tracker
                    entry_price = trigger_data['entry_price'] or daily_bar['Close']
                    target_price = trigger_data['target_price'] or (entry_price * 1.02)
                    stop_price = trigger_data['stop_loss'] or (entry_price * 0.99)
                    
                    self.outcome_tracker.record_pattern_trigger(
                        pattern_id=match['pattern_id'],
                        trigger_date=daily_bar['Date'],
                        entry_price=entry_price,
                        direction=match.get('direction', 'long'),
                        target_price=target_price,
                        stop_price=stop_price,
                        label_col=match.get('label_col', 'Label_2.0pct_10d')
                    )
                    
                    # Broadcast high-confidence trigger
                    await self._broadcast({
                        'type': 'pattern_triggered',
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'data': trigger_data
                    })
            
            # 6.5 Check open trades for outcomes (only on new days)
            if is_new_day:
                closed_trades = self.outcome_tracker.check_open_trades(
                    daily_df, daily_bar['Date']
                )
                for trade in closed_trades:
                    logger.info(f"Trade outcome: Pattern #{trade['pattern_id']} "
                              f"{trade['outcome']} ({trade['profit_pct']:.2f}%)")
                    
                    # Broadcast trade outcome
                    await self._broadcast({
                        'type': 'trade_outcome',
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'data': trade
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
    
    async def _save_regime_history(self):
        """Save regime history to file for dashboard access."""
        try:
            history = self.regime_detector.get_regime_history()
            if history:
                data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
                os.makedirs(data_dir, exist_ok=True)
                filepath = os.path.join(data_dir, 'regime_history.json')
                
                with open(filepath, 'w') as f:
                    json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving regime history: {e}")
    
    async def _save_matches_history(self):
        """Save pattern matches history to file for dashboard access."""
        try:
            if self.matches_history:
                data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
                os.makedirs(data_dir, exist_ok=True)
                filepath = os.path.join(data_dir, 'matches_history.json')
                
                with open(filepath, 'w') as f:
                    json.dump(self.matches_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving matches history: {e}")
    
    def get_matches_history(self, limit: int = 100) -> List[Dict]:
        """Get pattern matches history (most recent first)."""
        return self.matches_history[-limit:][::-1]  # Reverse to get newest first
    
    # ========== Main Loop ==========
    
    async def _data_fetching_loop(self):
        """Main loop that fetches data periodically with smart retry logic."""
        logger.info(f"Starting data fetching loop - polling every {self.poll_interval}s (15 min)")
        logger.info(f"Alpaca free tier: 200 requests/min, we use ~4 requests/hour")
        
        while self.is_running:
            try:
                if not self.session:
                    self.session = aiohttp.ClientSession()
                
                # Fetch latest bars
                logger.debug("Fetching latest 15m bars from Alpaca...")
                bars = await self._fetch_latest_15m_bars()
                
                if bars:
                    await self._process_new_bars(bars)
                    self.consecutive_errors = 0  # Reset error counter on success
                    logger.info(f"Next poll in {self.poll_interval} seconds ({self.poll_interval/60:.0f} minutes)")
                    await asyncio.sleep(self.poll_interval)
                else:
                    # No bars received (API error or no data)
                    self.consecutive_errors += 1
                    
                    # Use exponential backoff for retries
                    if self.consecutive_errors <= self.max_retries_before_backoff:
                        retry_delay = self.retry_interval
                        logger.warning(f"No bars received. Retrying in {retry_delay}s... (error #{self.consecutive_errors})")
                    else:
                        # After 5 errors, increase delay to avoid rate limiting
                        retry_delay = min(self.poll_interval, self.retry_interval * (self.consecutive_errors - self.max_retries_before_backoff + 1))
                        logger.warning(f"Multiple errors detected. Backing off: retry in {retry_delay}s...")
                    
                    await asyncio.sleep(retry_delay)
                
            except Exception as e:
                self.consecutive_errors += 1
                logger.error(f"Error in data fetching loop: {e}")
                logger.info(f"Retrying in {self.retry_interval}s... (error #{self.consecutive_errors})")
                await asyncio.sleep(self.retry_interval)
    
    async def start(self):
        """Start the streaming system."""
        self.is_running = True
        
        # Initialize HTTP session
        self.session = aiohttp.ClientSession()
        
        # Initialize with historical data
        await self._initialize_with_historical_data()
        
        # Start WebSocket server
        self.ws_server = await serve(
            self._handle_client,
            'localhost',
            self.ws_port,
            ping_interval=20,
            ping_timeout=10
        )
        
        logger.info(f"WebSocket server started on ws://localhost:{self.ws_port}")
        logger.info(f"Dashboard URL: http://localhost:5001")
        
        # Start data fetching loop
        await self._data_fetching_loop()
    
    async def _initialize_with_historical_data(self):
        """Initialize system with last day's data when market is closed."""
        try:
            logger.info("Initializing with historical data...")
            
            # Use the daily_reconstructor's historical data (already loaded from data/ohlcv.json)
            daily_history = self.daily_reconstructor.daily_history
            
            if daily_history and len(daily_history) > 0:
                # Get the last historical bar as current daily bar
                last_bar = daily_history[-1]
                
                # Set it as the current updating daily bar
                from datetime import datetime, timezone
                self.daily_reconstructor.current_daily_bar = {
                    'Date': last_bar['Date'],
                    'Open': last_bar['Open'],
                    'High': last_bar['High'],
                    'Low': last_bar['Low'],
                    'Close': last_bar['Close'],
                    'Volume': last_bar['Volume'],
                    'Timestamp': datetime.now(timezone.utc).isoformat(),
                    'IntradayUpdates': 1,
                    'IsRealtime': False  # This is historical data
                }
                # Handle date format that may include time
                date_str = last_bar['Date']
                if ' ' in date_str:
                    date_str = date_str.split(' ')[0]  # Extract just the date part
                self.daily_reconstructor.current_date = datetime.strptime(date_str, '%Y-%m-%d')
                
                logger.info(f"Set current daily bar from historical data: {last_bar['Date']} "
                           f"O:{last_bar['Open']:.2f} H:{last_bar['High']:.2f} "
                           f"L:{last_bar['Low']:.2f} C:{last_bar['Close']:.2f}")
                
                # Calculate initial features and regime
                daily_df = self.daily_reconstructor.get_full_daily_data(include_current=True)
                if len(daily_df) >= 50:
                    features_df = self.feature_calculator.calculate(daily_df)
                    if features_df is not None and not features_df.empty:
                        features = features_df.iloc[-1].to_dict()
                        self.current_regime = self.regime_detector.update(features)
                        logger.info(f"Initial regime: {self.current_regime}")
                        
                        # Calculate initial pattern matches
                        self.current_matches = self.pattern_matcher.match_all_patterns(features)
                        logger.info(f"Initial pattern matches: {len(self.current_matches)}")
            else:
                logger.warning("No historical data available in daily_reconstructor")
                
        except Exception as e:
            logger.error(f"Error initializing with historical data: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def stop(self):
        """Stop the streaming system."""
        self.is_running = False
        
        if self.ws_server:
            self.ws_server.close()
            await self.ws_server.wait_closed()
        
        if self.session:
            await self.session.close()
        
        logger.info("Streaming system stopped")


async def main():
    """Run the real-time streaming system."""
    system = RealtimeStreamingSystem()
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        await system.stop()


if __name__ == "__main__":
    asyncio.run(main())
