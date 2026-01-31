"""
Real-Time Dashboard API Module
WebSocket API for real-time dashboard updates
"""

import os
import logging
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Set
import websockets
from websockets.server import serve

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StreamingDashboardAPI:
    """
    WebSocket API for real-time dashboard updates.
    Broadcasts pattern matches, regime changes, and alerts to connected clients.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the streaming dashboard API.
        
        Args:
            config_path: Path to configuration file
        """
        import yaml
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.streaming_config = config.get('streaming', {})
            dashboard_config = self.streaming_config.get('dashboard', {})
        except FileNotFoundError:
            self.streaming_config = {}
            dashboard_config = {}
        
        # Server settings
        self.ws_port = dashboard_config.get('ws_port', 5001)
        self.http_port = dashboard_config.get('http_port', 5002)
        self.update_interval = dashboard_config.get('update_interval', 1)
        
        # Connected clients
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # Subscriptions per client
        self.subscriptions: Dict[websockets.WebSocketServerProtocol, Set[str]] = {}
        
        # Server state
        self.server = None
        self.is_running = False
        
        # Latest data for new connections
        self.latest_matches = []
        self.latest_regime = {}
        self.latest_alerts = []
        self.latest_daily_bar = None
        self.latest_intraday_stats = {}
        
        logger.info(f"Streaming Dashboard API initialized on port {self.ws_port}")
    
    async def register_client(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new client connection."""
        self.clients.add(websocket)
        self.subscriptions[websocket] = set()
        
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send initial data to new client
        await self.send_initial_data(websocket)
    
    async def unregister_client(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a client connection."""
        if websocket in self.clients:
            self.clients.remove(websocket)
        
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
        
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def send_initial_data(self, websocket: websockets.WebSocketServerProtocol):
        """Send initial data to a newly connected client."""
        # Send latest daily bar (most important - shows current/historical data)
        if self.latest_daily_bar:
            await self._send_to_client(websocket, {
                'type': 'daily_bar_update',
                'timestamp': datetime.now().isoformat(),
                'daily_bar': self.latest_daily_bar,
                'is_new_day': False,
                'intraday_stats': self.latest_intraday_stats,
                'regime': self.latest_regime,
                'pattern_matches': len(self.latest_matches)
            })
        
        # Send current regime
        if self.latest_regime:
            await self._send_to_client(websocket, {
                'type': 'regime_update',
                'timestamp': datetime.now().isoformat(),
                'data': self.latest_regime
            })
        
        # Send latest matches
        if self.latest_matches:
            await self._send_to_client(websocket, {
                'type': 'pattern_matches',
                'timestamp': datetime.now().isoformat(),
                'matches': self.latest_matches
            })
        
        # Send latest alerts
        if self.latest_alerts:
            for alert in self.latest_alerts:
                await self._send_to_client(websocket, alert)
    
    async def handle_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """Handle incoming message from client."""
        try:
            data = json.loads(message)
            action = data.get('action')
            
            if action == 'subscribe':
                channels = data.get('channels', [])
                self.subscriptions[websocket].update(channels)
                logger.info(f"Client subscribed to channels: {channels}")
                
                await self._send_to_client(websocket, {
                    'type': 'subscription_confirmed',
                    'channels': list(self.subscriptions[websocket]),
                    'timestamp': datetime.now().isoformat()
                })
            
            elif action == 'unsubscribe':
                channels = data.get('channels', [])
                for channel in channels:
                    self.subscriptions[websocket].discard(channel)
                logger.info(f"Client unsubscribed from channels: {channels}")
            
            elif action == 'ping':
                await self._send_to_client(websocket, {
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                })
        
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _send_to_client(self, websocket: websockets.WebSocketServerProtocol, data: Dict):
        """Send data to a specific client."""
        try:
            await websocket.send(json.dumps(data))
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
            await self.unregister_client(websocket)
    
    async def broadcast(self, data: Dict, channel: str = None):
        """
        Broadcast data to all subscribed clients.
        
        Args:
            data: Data to broadcast
            channel: Channel name (optional, for filtering)
        """
        if not self.clients:
            return
        
        message = json.dumps(data)
        
        # Create list of clients to avoid modification during iteration
        clients_to_send = list(self.clients)
        
        for client in clients_to_send:
            # Check if client is subscribed to this channel
            if channel and client in self.subscriptions:
                if channel not in self.subscriptions[client]:
                    continue
            
            try:
                await client.send(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                await self.unregister_client(client)
    
    async def broadcast_pattern_match(self, match: Dict, probability: Dict):
        """Broadcast a pattern match to clients."""
        data = {
            'type': 'pattern_match',
            'timestamp': datetime.now().isoformat(),
            'pattern_id': match.get('pattern_id'),
            'confidence': match.get('confidence_score'),
            'status': match.get('status'),
            'probability': probability.get('final_probability'),
            'direction': probability.get('direction'),
            'entry_signal': probability.get('actionable_signals', {}).get('entry_signal'),
            'entry_price': probability.get('actionable_signals', {}).get('entry_price'),
            'target_price': probability.get('price_targets', [{}])[0].get('target_price'),
            'stop_loss': probability.get('actionable_signals', {}).get('stop_loss'),
            'risk_reward_ratio': probability.get('actionable_signals', {}).get('risk_reward_ratio')
        }
        
        await self.broadcast(data, 'pattern_matches')
    
    async def broadcast_regime_change(self, from_regime: Dict, to_regime: Dict):
        """Broadcast a regime change to clients."""
        data = {
            'type': 'regime_change',
            'timestamp': datetime.now().isoformat(),
            'from': {
                'volatility': from_regime.get('volatility_regime'),
                'trend': from_regime.get('trend_regime')
            },
            'to': {
                'volatility': to_regime.get('volatility_regime'),
                'trend': to_regime.get('trend_regime')
            },
            'current': to_regime
        }
        
        self.latest_regime = to_regime
        await self.broadcast(data, 'regime_changes')
    
    async def broadcast_alert(self, alert: Dict):
        """Broadcast an alert to clients."""
        data = {
            'type': 'alert',
            'timestamp': datetime.now().isoformat(),
            **alert
        }
        
        self.latest_alerts.append(data)
        # Keep only last 50 alerts
        self.latest_alerts = self.latest_alerts[-50:]
        
        await self.broadcast(data, 'alerts')
    
    async def broadcast_matches_update(self, matches: List[Dict]):
        """Broadcast all current matches to clients."""
        data = {
            'type': 'matches_update',
            'timestamp': datetime.now().isoformat(),
            'matches': matches,
            'count': len(matches)
        }
        
        self.latest_matches = matches
        await self.broadcast(data, 'pattern_matches')
    
    async def broadcast_daily_bar_update(self, update_data: Dict):
        """Broadcast daily bar update to clients.
        
        Args:
            update_data: Dictionary containing:
                - daily_bar: Current updating daily OHLCV bar
                - is_new_day: Whether this is a new trading day
                - intraday_stats: Statistics about today's progress
                - current_features: Current calculated features
                - regime: Current market regime
                - pattern_matches: Number of pattern matches
        """
        # Store latest data for new connections
        self.latest_daily_bar = update_data.get('daily_bar')
        self.latest_intraday_stats = update_data.get('intraday_stats', {})
        if update_data.get('regime'):
            self.latest_regime = update_data.get('regime')
        
        data = {
            'type': 'daily_bar_update',
            'timestamp': datetime.now().isoformat(),
            'daily_bar': update_data.get('daily_bar'),
            'is_new_day': update_data.get('is_new_day', False),
            'intraday_stats': update_data.get('intraday_stats', {}),
            'regime': update_data.get('regime', {}),
            'pattern_matches': update_data.get('pattern_matches', 0)
        }
        
        await self.broadcast(data, 'price_updates')
    
    async def broadcast_connection_status(self, status: str):
        """Broadcast connection status to clients."""
        data = {
            'type': 'connection_status',
            'timestamp': datetime.now().isoformat(),
            'status': status
        }
        
        await self.broadcast(data, 'connection')
    
    async def broadcast_error(self, error_type: str, reason: str):
        """Broadcast error to clients."""
        data = {
            'type': 'error',
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'reason': reason
        }
        
        await self.broadcast(data, 'errors')
    
    def create_alert(self, pattern_id: int, probability: Dict, level: str = 'HIGH') -> Dict:
        """
        Create an alert from a pattern match.
        
        Args:
            pattern_id: Pattern ID
            probability: Probability estimation
            level: Alert level (HIGH, MEDIUM, LOW)
            
        Returns:
            Alert dictionary
        """
        actionable = probability.get('actionable_signals', {})
        targets = probability.get('price_targets', [])
        
        alert = {
            'level': level,
            'message': f"Pattern #{pattern_id} triggered - {actionable.get('entry_signal', 'WATCH')} signal",
            'pattern_id': pattern_id,
            'pattern_name': probability.get('pattern_name', f'Pattern {pattern_id}'),
            'entry_price': actionable.get('entry_price'),
            'target_price': targets[0].get('target_price') if targets else None,
            'stop_loss': actionable.get('stop_loss'),
            'probability': probability.get('final_probability'),
            'direction': probability.get('direction'),
            'risk_reward_ratio': actionable.get('risk_reward_ratio'),
            'position_size_pct': actionable.get('position_size_pct')
        }
        
        return alert
    
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle a client connection."""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            await self.unregister_client(websocket)
    
    async def start(self):
        """Start the WebSocket server."""
        if self.is_running:
            logger.warning("Dashboard API already running")
            return
        
        self.is_running = True
        
        logger.info(f"Starting WebSocket server on port {self.ws_port}...")
        
        self.server = await serve(
            self.handle_client,
            "0.0.0.0",
            self.ws_port,
            ping_interval=30,
            ping_timeout=20
        )
        
        logger.info(f"WebSocket server started on ws://0.0.0.0:{self.ws_port}")
    
    async def stop(self):
        """Stop the WebSocket server."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("WebSocket server stopped")
    
    def get_client_count(self) -> int:
        """Get the number of connected clients."""
        return len(self.clients)


# Singleton instance for global access
_dashboard_api = None


def get_dashboard_api(config_path: str = "config.yaml") -> StreamingDashboardAPI:
    """Get or create the dashboard API singleton."""
    global _dashboard_api
    if _dashboard_api is None:
        _dashboard_api = StreamingDashboardAPI(config_path)
    return _dashboard_api


if __name__ == "__main__":
    # Demo usage
    async def main():
        api = StreamingDashboardAPI()
        
        # Start server
        await api.start()
        
        print(f"WebSocket server running on ws://0.0.0.0:{api.ws_port}")
        print("Press Ctrl+C to stop...")
        
        # Simulate some broadcasts
        await asyncio.sleep(2)
        
        # Simulate regime change
        from_regime = {'volatility_regime': 'Medium', 'trend_regime': 'Sideways'}
        to_regime = {'volatility_regime': 'High', 'trend_regime': 'Strong Bull'}
        await api.broadcast_regime_change(from_regime, to_regime)
        print("Broadcasted regime change")
        
        await asyncio.sleep(2)
        
        # Simulate pattern match
        match = {
            'pattern_id': 0,
            'confidence_score': 92.5,
            'status': 'TRIGGERED'
        }
        probability = {
            'final_probability': 87.5,
            'direction': 'long',
            'actionable_signals': {
                'entry_signal': 'STRONG BUY',
                'entry_price': 106.00,
                'target_price': 108.12,
                'stop_loss': 103.70,
                'risk_reward_ratio': 2.5
            },
            'price_targets': [{'target_price': 108.12}]
        }
        await api.broadcast_pattern_match(match, probability)
        print("Broadcasted pattern match")
        
        await asyncio.sleep(2)
        
        # Simulate alert
        alert = api.create_alert(0, probability, 'HIGH')
        await api.broadcast_alert(alert)
        print("Broadcasted alert")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await api.stop()
            print("\nServer stopped")
    
    asyncio.run(main())