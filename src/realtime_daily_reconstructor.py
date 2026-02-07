"""
Real-Time Daily Bar Reconstructor
Converts 15-minute price updates into updating daily OHLCV bars.
Maintains full historical daily context for pattern matching.
"""

import os
import logging
import json
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from collections import deque
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealtimeDailyReconstructor:
    """
    Reconstructs daily OHLCV bars from 15-minute intraday updates.
    
    Key insight: Patterns were discovered on daily bars (50-day MA, 14-period RSI, etc.)
    So we must match against daily bars that update throughout the trading day,
    NOT against 15-minute bars directly.
    """
    
    def __init__(self, historical_data_path: str = None, ticker: str = None):
        """
        Initialize the daily bar reconstructor.
        
        Args:
            historical_data_path: Path to historical daily OHLCV data. If None, constructs from ticker
            ticker: Ticker symbol for path construction
        """
        # If path not provided, construct from ticker
        if historical_data_path is None:
            if ticker:
                historical_data_path = f"data/tickers/{ticker}/ohlcv.json"
            else:
                historical_data_path = "data/ohlcv.json"
        
        self.historical_data_path = historical_data_path
        self.daily_history: List[Dict] = []
        self.current_daily_bar: Optional[Dict] = None
        self.current_date: Optional[datetime] = None
        
        # Track intraday 15m bars for current day
        self.intraday_bars: List[Dict] = []
        
        # Market hours (EST)
        self.market_open = 9  # 9:30 AM
        self.market_close = 16  # 4:00 PM
        
        # Load historical daily data
        self._load_historical_data()
        
        logger.info(f"Daily Reconstructor initialized with {len(self.daily_history)} historical days")
        if self.daily_history:
            logger.info(f"Last historical date: {self.daily_history[-1]['Date']}")
    
    def _load_historical_data(self):
        """Load historical daily OHLCV data."""
        try:
            with open(self.historical_data_path, 'r') as f:
                data = json.load(f)
            
            # Convert to list of dicts if needed
            if isinstance(data, dict):
                # Assume it's a single record
                data = [data]
            
            self.daily_history = data
            logger.info(f"Loaded {len(self.daily_history)} historical daily bars")
            
        except FileNotFoundError:
            logger.warning(f"Historical data not found at {self.historical_data_path}")
            self.daily_history = []
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            self.daily_history = []
    
    def _is_same_trading_day(self, dt1: datetime, dt2: datetime) -> bool:
        """Check if two datetimes are in the same trading day."""
        # Normalize to date (accounting for overnight sessions if needed)
        return dt1.date() == dt2.date()
    
    def _market_is_open(self, dt: datetime) -> bool:
        """Check if market is open for the given datetime (EST)."""
        # Simple check - in production would handle holidays, early close, etc.
        hour = dt.hour
        return self.market_open <= hour < self.market_close
    
    def _create_new_daily_bar(self, timestamp: datetime, bar_15m: Dict) -> Dict:
        """Create a new daily bar from the first 15m bar of the day."""
        date_str = timestamp.strftime('%Y-%m-%d')
        
        return {
            'Date': date_str,
            'Open': bar_15m['Open'],
            'High': bar_15m['High'],
            'Low': bar_15m['Low'],
            'Close': bar_15m['Close'],
            'Volume': bar_15m['Volume'],
            'Timestamp': timestamp.isoformat(),
            'IntradayUpdates': 1,
            'IsRealtime': True
        }
    
    def _update_daily_bar(self, bar_15m: Dict) -> Dict:
        """Update the current daily bar with new 15m data."""
        if self.current_daily_bar is None:
            return None
        
        # Update High if new high
        if bar_15m['High'] > self.current_daily_bar['High']:
            self.current_daily_bar['High'] = bar_15m['High']
        
        # Update Low if new low
        if bar_15m['Low'] < self.current_daily_bar['Low']:
            self.current_daily_bar['Low'] = bar_15m['Low']
        
        # Always update Close to latest price
        self.current_daily_bar['Close'] = bar_15m['Close']
        
        # Accumulate Volume
        self.current_daily_bar['Volume'] += bar_15m['Volume']
        
        # Update metadata
        self.current_daily_bar['IntradayUpdates'] += 1
        self.current_daily_bar['Timestamp'] = bar_15m.get('timestamp', datetime.now(timezone.utc)).isoformat()
        
        return self.current_daily_bar
    
    def process_15m_bar(self, bar_15m: Dict) -> Tuple[Optional[Dict], bool]:
        """
        Process a new 15-minute bar and update daily reconstruction.
        
        Args:
            bar_15m: 15-minute bar with timestamp, OHLCV
        
        Returns:
            Tuple of (updated_daily_bar, is_new_day)
            - updated_daily_bar: The current day's daily OHLCV (updating throughout day)
            - is_new_day: True if this started a new trading day
        """
        timestamp = bar_15m.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        is_new_day = False
        
        # Check if this is a new trading day
        if (self.current_date is None or 
            not self._is_same_trading_day(timestamp, self.current_date)):
            
            # Save previous day's bar to history if exists
            if self.current_daily_bar is not None:
                self._finalize_daily_bar()
            
            # Start new daily bar
            self.current_date = timestamp
            self.current_daily_bar = self._create_new_daily_bar(timestamp, bar_15m)
            self.intraday_bars = [bar_15m]
            is_new_day = True
            
            logger.info(f"New trading day started: {self.current_daily_bar['Date']} "
                       f"Open: ${self.current_daily_bar['Open']:.2f}")
        else:
            # Update existing daily bar
            self._update_daily_bar(bar_15m)
            self.intraday_bars.append(bar_15m)
        
        return self.current_daily_bar.copy(), is_new_day
    
    def _finalize_daily_bar(self):
        """Finalize the current day's bar and add to history."""
        if self.current_daily_bar:
            # Remove realtime metadata before adding to history
            bar_to_save = {
                'Date': self.current_daily_bar['Date'],
                'Open': self.current_daily_bar['Open'],
                'High': self.current_daily_bar['High'],
                'Low': self.current_daily_bar['Low'],
                'Close': self.current_daily_bar['Close'],
                'Volume': self.current_daily_bar['Volume']
            }
            
            # Replace last bar in history if it's the same date, otherwise append
            if self.daily_history:
                last_hist_date = self.daily_history[-1].get('Date', '')
                if last_hist_date == bar_to_save['Date']:
                    self.daily_history[-1] = bar_to_save
                else:
                    self.daily_history.append(bar_to_save)
            else:
                self.daily_history.append(bar_to_save)
            
            logger.info(f"Finalized daily bar for {bar_to_save['Date']}: "
                       f"O:{bar_to_save['Open']:.2f} H:{bar_to_save['High']:.2f} "
                       f"L:{bar_to_save['Low']:.2f} C:{bar_to_save['Close']:.2f} "
                       f"V:{bar_to_save['Volume']:,}")
    
    def get_full_daily_data(self, include_current: bool = True) -> pd.DataFrame:
        """
        Get complete daily OHLCV data including historical and current updating bar.
        
        Args:
            include_current: Whether to include the current updating daily bar
        
        Returns:
            DataFrame with Date, Open, High, Low, Close, Volume columns
        """
        # Start with historical data
        data = self.daily_history.copy()
        
        # Add current updating bar if requested and exists
        if include_current and self.current_daily_bar is not None:
            current = {
                'Date': self.current_daily_bar['Date'],
                'Open': self.current_daily_bar['Open'],
                'High': self.current_daily_bar['High'],
                'Low': self.current_daily_bar['Low'],
                'Close': self.current_daily_bar['Close'],
                'Volume': self.current_daily_bar['Volume']
            }
            
            # Replace or append
            if data and data[-1]['Date'] == current['Date']:
                data[-1] = current
            else:
                data.append(current)
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
        
        return df
    
    def get_current_daily_bar(self) -> Optional[Dict]:
        """Get the current day's updating daily bar."""
        return self.current_daily_bar.copy() if self.current_daily_bar else None
    
    def get_intraday_stats(self) -> Dict:
        """Get statistics about today's intraday progress."""
        if not self.intraday_bars:
            return {}
        
        return {
            'date': self.current_date.strftime('%Y-%m-%d') if self.current_date else None,
            'bars_received': len(self.intraday_bars),
            'market_hours_pct': min(100, (len(self.intraday_bars) / 26) * 100),  # 26 bars = full day
            'current_daily_open': self.current_daily_bar['Open'] if self.current_daily_bar else None,
            'current_daily_close': self.current_daily_bar['Close'] if self.current_daily_bar else None,
            'daily_range_pct': ((self.current_daily_bar['High'] - self.current_daily_bar['Low']) / 
                               self.current_daily_bar['Open'] * 100) if self.current_daily_bar else 0
        }
    
    def is_ready_for_feature_calc(self) -> bool:
        """Check if we have enough data for feature calculation (need at least 50 days)."""
        return len(self.daily_history) >= 50


# Demo/test
if __name__ == "__main__":
    reconstructor = RealtimeDailyReconstructor()
    
    # Simulate 15m bars throughout a trading day
    base_date = datetime(2026, 1, 30, 9, 30, tzinfo=timezone.utc)
    
    test_bars = [
        {'Open': 100.0, 'High': 100.5, 'Low': 99.8, 'Close': 100.2, 'Volume': 100000},
        {'Open': 100.2, 'High': 101.0, 'Low': 100.1, 'Close': 100.8, 'Volume': 150000},
        {'Open': 100.8, 'High': 101.5, 'Low': 100.5, 'Close': 101.2, 'Volume': 200000},
        {'Open': 101.2, 'High': 101.8, 'Low': 100.9, 'Close': 101.0, 'Volume': 180000},
    ]
    
    for i, bar in enumerate(test_bars):
        timestamp = base_date + timedelta(minutes=15*i)
        bar['timestamp'] = timestamp
        
        daily_bar, is_new_day = reconstructor.process_15m_bar(bar)
        
        print(f"\n15m bar {i+1} @ {timestamp.strftime('%H:%M')}:")
        print(f"  O:{bar['Open']:.2f} H:{bar['High']:.2f} L:{bar['Low']:.2f} C:{bar['Close']:.2f}")
        print(f"  Daily bar: O:{daily_bar['Open']:.2f} H:{daily_bar['High']:.2f} "
              f"L:{daily_bar['Low']:.2f} C:{daily_bar['Close']:.2f} "
              f"V:{daily_bar['Volume']:,}")
        print(f"  Is new day: {is_new_day}")
    
    print(f"\nFinal intraday stats: {reconstructor.get_intraday_stats()}")
    
    full_data = reconstructor.get_full_daily_data()
    print(f"\nFull daily data shape: {full_data.shape}")
    print(f"Last 3 rows:\n{full_data.tail(3)}")
