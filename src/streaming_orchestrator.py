"""
Streaming Orchestrator Module - Daily Bar Based
Main coordinator for real-time streaming pattern analysis system.

Key Design: Patterns were discovered on daily bars, so we must:
1. Fetch 15-minute price updates
2. Reconstruct the current day's daily OHLCV bar
3. Recalculate all 140+ indicators on daily bars with full history
4. Match patterns against daily bar features
5. Display updating daily candlestick in dashboard
"""

import os
import sys
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StreamingOrchestrator:
    """
    Main orchestrator for the real-time streaming system.
    Coordinates data streaming, daily bar reconstruction, feature calculation,
    pattern matching, regime detection, probability estimation, and dashboard updates.
    
    CRITICAL: All pattern matching is done against DAILY bars, not 15m bars,
    because that's how patterns were discovered and validated (85-95% success rates).
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the streaming orchestrator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.streaming_config = self.config.get('streaming', {})
        
        # Check if streaming is enabled
        self.enabled = self.streaming_config.get('enabled', True)
        
        if not self.enabled:
            logger.info("Streaming is disabled in config")
            return
        
        # Initialize components
        self.streamer = None
        self.daily_reconstructor = None
        self.feature_calculator = None
        self.pattern_matcher = None
        self.regime_detector = None
        self.probability_estimator = None
        self.dashboard_api = None
        
        # State
        self.is_running = False
        self.previous_regime = None
        self.bar_count = 0
        self.last_daily_bar = None
        
        # Alert settings
        alert_config = self.streaming_config.get('alerts', {})
        self.alerts_enabled = alert_config.get('enabled', True)
        self.alert_levels = alert_config.get('levels', {'HIGH': 90, 'MEDIUM': 70, 'LOW': 50})
        
        logger.info("Streaming Orchestrator initialized (Daily Bar Mode)")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        import yaml
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
            return {}
    
    async def initialize(self):
        """Initialize all streaming components."""
        if not self.enabled:
            return
        
        logger.info("Initializing streaming components...")
        
        # Import components
        import streaming_data_streamer
        import realtime_daily_reconstructor
        import realtime_feature_calculator
        import tolerance_pattern_matcher
        import market_regime_detector
        import probability_estimator
        import streaming_dashboard_api
        
        DemoStreamer = streaming_data_streamer.DemoStreamer
        WebSocketStreamer = streaming_data_streamer.WebSocketStreamer
        RealtimeDailyReconstructor = realtime_daily_reconstructor.RealtimeDailyReconstructor
        RealtimeFeatureCalculator = realtime_feature_calculator.RealtimeFeatureCalculator
        TolerancePatternMatcher = tolerance_pattern_matcher.TolerancePatternMatcher
        MarketRegimeDetector = market_regime_detector.MarketRegimeDetector
        ProbabilityEstimator = probability_estimator.ProbabilityEstimator
        StreamingDashboardAPI = streaming_dashboard_api.StreamingDashboardAPI
        
        # Initialize daily bar reconstructor
        self.daily_reconstructor = RealtimeDailyReconstructor()
        logger.info(f"Daily reconstructor initialized with {len(self.daily_reconstructor.daily_history)} historical days")
        
        # Initialize feature calculator (works on daily bars)
        self.feature_calculator = RealtimeFeatureCalculator(self.config_path)
        logger.info("Feature calculator initialized (daily bar mode)")
        
        # Initialize pattern matcher
        self.pattern_matcher = TolerancePatternMatcher(
            tolerance_pct=self.streaming_config.get('pattern_matching', {}).get('tolerance_pct', 5.0),
            config_path=self.config_path
        )
        self.pattern_matcher.load_patterns()
        logger.info(f"Pattern matcher initialized with {self.pattern_matcher.get_pattern_count()} patterns")
        
        # Initialize regime detector
        self.regime_detector = MarketRegimeDetector(self.config_path)
        logger.info("Regime detector initialized")
        
        # Initialize probability estimator
        self.probability_estimator = ProbabilityEstimator(self.config_path)
        logger.info("Probability estimator initialized")
        
        # Initialize dashboard API
        self.dashboard_api = StreamingDashboardAPI(self.config_path)
        await self.dashboard_api.start()
        logger.info("Dashboard API started")
        
        # Check if market is open - if not, send last historical data
        await self._send_last_historical_data()
        
        # Initialize data streamer based on provider config
        provider = self.streaming_config.get('provider', 'demo')
        if provider == 'alpaca':
            self.streamer = WebSocketStreamer(self.config_path)
            logger.info(f"Using Alpaca WebSocket streamer")
        elif provider == 'alpaca_rest':
            import alpaca_rest_streamer
            self.streamer = alpaca_rest_streamer.AlpacaRESTStreamer(self.config_path)
            logger.info(f"Using Alpaca REST streamer")
        else:
            self.streamer = DemoStreamer(self.config_path)
            logger.info(f"Using Demo streamer (provider: {provider})")
        
        self.streamer.load_historical_baseline()
        
        # Set up event callbacks
        self.streamer.set_event_callbacks(
            on_new_bar=self._on_new_15m_bar,
            on_connection_status=self._on_connection_status,
            on_error=self._on_error
        )
        
        logger.info("All streaming components initialized")
    
    async def _on_new_15m_bar(self, data: Dict):
        """
        Handle new 15-minute bar event from streamer.
        
        This is the core processing pipeline:
        1. Convert 15m bar to updating daily bar
        2. Recalculate all features on daily bars
        3. Detect regime
        4. Match patterns
        5. Send updates to dashboard
        """
        bar_15m = data['bar']
        self.bar_count += 1
        
        logger.debug(f"Processing 15m bar {self.bar_count}: {bar_15m['timestamp']}")
        
        # Step 1: Process 15m bar and update daily reconstruction
        daily_bar, is_new_day = self.daily_reconstructor.process_15m_bar(bar_15m)
        
        if is_new_day:
            logger.info(f"New trading day: {daily_bar['Date']} Open: ${daily_bar['Open']:.2f}")
        
        # Get full daily data (historical + current updating bar)
        full_daily_df = self.daily_reconstructor.get_full_daily_data(include_current=True)
        
        # Check if daily bar actually changed (new Close/High/Low)
        daily_changed = self._daily_bar_changed(daily_bar)
        
        if daily_changed:
            # Step 2: Calculate features on daily bars
            features = self.feature_calculator.calculate(full_daily_df)
            current_features = features.iloc[-1].to_dict() if features is not None else {}
            
            # Step 3: Update regime detection
            current_regime = self.regime_detector.update(current_features)
            
            # Check for regime change
            if self.previous_regime and self._regime_changed(self.previous_regime, current_regime):
                logger.info(f"Regime changed: {self.previous_regime} -> {current_regime}")
                await self.dashboard_api.broadcast_regime_change(self.previous_regime, current_regime)
            
            self.previous_regime = current_regime
            
            # Step 4: Match patterns against daily features
            matches = self.pattern_matcher.match_all_patterns(current_features)
            
            if matches:
                logger.info(f"Found {len(matches)} pattern matches on daily bar")
                
                # Step 5: Estimate probabilities for matches
                current_price = daily_bar['Close']
                estimations = self.probability_estimator.batch_estimate(matches, current_regime, current_price)
                
                # Broadcast matches
                await self.dashboard_api.broadcast_matches_update(estimations)
                
                # Process triggered patterns
                for match, estimation in zip(matches, estimations):
                    if match['status'] == 'TRIGGERED' and self.alerts_enabled:
                        level = self._get_alert_level(estimation['final_probability'])
                        alert = self.dashboard_api.create_alert(
                            match['pattern_id'],
                            estimation,
                            level
                        )
                        await self.dashboard_api.broadcast_alert(alert)
                        logger.info(f"Alert: Pattern #{match['pattern_id']} triggered - {estimation['actionable_signals']['entry_signal']}")
                    
                    await self.dashboard_api.broadcast_pattern_match(match, estimation)
            
            # Step 6: Broadcast daily bar update to dashboard
            intraday_stats = self.daily_reconstructor.get_intraday_stats()
            await self.dashboard_api.broadcast_daily_bar_update({
                'daily_bar': daily_bar,
                'is_new_day': is_new_day,
                'intraday_stats': intraday_stats,
                'current_features': current_features,
                'regime': current_regime,
                'pattern_matches': len(matches) if matches else 0
            })
            
            self.last_daily_bar = daily_bar.copy()
    
    def _daily_bar_changed(self, daily_bar: Dict) -> bool:
        """Check if daily bar has meaningfully changed since last update."""
        if self.last_daily_bar is None:
            return True
        
        return (
            daily_bar['Close'] != self.last_daily_bar.get('Close') or
            daily_bar['High'] != self.last_daily_bar.get('High') or
            daily_bar['Low'] != self.last_daily_bar.get('Low') or
            daily_bar['Volume'] != self.last_daily_bar.get('Volume')
        )
    
    async def _on_connection_status(self, data: Dict):
        """Handle connection status event."""
        status = data['status']
        logger.info(f"Connection status: {status}")
        await self.dashboard_api.broadcast_connection_status(status)
    
    async def _on_error(self, data: Dict):
        """Handle error event."""
        error_type = data.get('type', 'unknown')
        reason = data.get('reason', 'unknown')
        logger.error(f"Error ({error_type}): {reason}")
        await self.dashboard_api.broadcast_error(error_type, reason)
    
    def _regime_changed(self, prev: Dict, curr: Dict) -> bool:
        """Check if regime has changed."""
        return (
            prev.get('volatility_regime') != curr.get('volatility_regime') or
            prev.get('trend_regime') != curr.get('trend_regime')
        )
    
    def _get_alert_level(self, probability: float) -> str:
        """Get alert level based on probability."""
        if probability >= self.alert_levels.get('HIGH', 90):
            return 'HIGH'
        elif probability >= self.alert_levels.get('MEDIUM', 70):
            return 'MEDIUM'
        else:
            return 'LOW'
    
    async def _send_last_historical_data(self):
        """
        Send the last historical day's data to the dashboard.
        This is useful when market is closed - shows last trading day's info.
        """
        try:
            import pandas as pd
            
            # Get historical daily data
            full_daily_df = self.daily_reconstructor.get_full_daily_data(include_current=False)
            
            if full_daily_df is None or len(full_daily_df) == 0:
                logger.warning("No historical data available for last day fallback")
                return
            
            # Get last day's bar
            last_bar = full_daily_df.iloc[-1]
            
            # Convert to dict format
            daily_bar = {
                'Date': last_bar['Date'].strftime('%Y-%m-%d') if hasattr(last_bar['Date'], 'strftime') else str(last_bar['Date']),
                'Open': float(last_bar['Open']),
                'High': float(last_bar['High']),
                'Low': float(last_bar['Low']),
                'Close': float(last_bar['Close']),
                'Volume': float(last_bar['Volume']),
                'IsHistorical': True
            }
            
            logger.info(f"Sending last historical day: {daily_bar['Date']} Close: ${daily_bar['Close']:.2f}")
            
            # Calculate features on historical data
            features = self.feature_calculator.calculate(full_daily_df)
            current_features = features.iloc[-1].to_dict() if features is not None else {}
            
            # Get regime from last day
            current_regime = self.regime_detector.update(current_features)
            self.previous_regime = current_regime
            
            # Match patterns
            matches = self.pattern_matcher.match_all_patterns(current_features)
            
            if matches:
                logger.info(f"Found {len(matches)} pattern matches on last historical day")
                current_price = daily_bar['Close']
                estimations = self.probability_estimator.batch_estimate(matches, current_regime, current_price)
                await self.dashboard_api.broadcast_matches_update(estimations)
            
            # Send update to dashboard
            await self.dashboard_api.broadcast_daily_bar_update({
                'daily_bar': daily_bar,
                'is_new_day': False,
                'intraday_stats': {
                    'date': daily_bar['Date'],
                    'bars_received': 0,
                    'market_hours_pct': 0,
                    'note': 'Market closed - showing last trading day'
                },
                'current_features': current_features,
                'regime': current_regime,
                'pattern_matches': len(matches) if matches else 0
            })
            
            self.last_daily_bar = daily_bar.copy()
            
        except Exception as e:
            logger.error(f"Error sending last historical data: {e}")
    
    async def start(self):
        """Start the streaming orchestrator."""
        if not self.enabled:
            logger.warning("Streaming is disabled, not starting")
            return
        
        logger.info("Starting streaming orchestrator...")
        self.is_running = True
        
        try:
            await self.streamer.start()
        except Exception as e:
            logger.error(f"Error starting streamer: {e}")
            self.is_running = False
            raise
    
    async def stop(self):
        """Stop the streaming orchestrator."""
        logger.info("Stopping streaming orchestrator...")
        self.is_running = False
        
        if self.streamer:
            await self.streamer.stop()
        
        if self.dashboard_api:
            await self.dashboard_api.stop()
        
        logger.info("Streaming orchestrator stopped")


# Convenience function for running the orchestrator
async def run_orchestrator(config_path: str = "config.yaml"):
    """Run the streaming orchestrator."""
    orchestrator = StreamingOrchestrator(config_path)
    await orchestrator.initialize()
    
    try:
        await orchestrator.start()
        
        # Keep running until interrupted
        while orchestrator.is_running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(run_orchestrator())
