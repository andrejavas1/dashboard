"""
Stable Market Regime Detector
Uses long-term trends (50/200 MA) with minimum 63-day (3-month) regime duration
to prevent excessive regime changes.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class RegimePeriod:
    start_date: str
    end_date: str
    trend_regime: str
    volatility_regime: str
    duration: int


class StableRegimeDetector:
    """
    Detects stable market regimes using:
    - 50-day vs 200-day MA alignment for trend
    - 20-day ATR percentile for volatility (ticker-specific)
    - Minimum 63-day (3-month) regime duration
    """
    
    def __init__(self, min_duration: int = 63):
        """
        Initialize detector.
        
        Args:
            min_duration: Minimum days in a regime before allowing change (default 63 = 3 months)
        """
        self.min_duration = min_duration
        self.ma_short = 50
        self.ma_long = 200
        self.vol_window = 20
        
    def detect_regimes(self, ohlcv_data: List[Dict]) -> List[Dict]:
        """
        Detect stable regimes from OHLCV data.
        
        Args:
            ohlcv_data: List of OHLCV bars with 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'
            
        Returns:
            List of regime periods
        """
        if len(ohlcv_data) < self.ma_long:
            return []
        
        # Calculate MAs and ATR
        df = self._prepare_data(ohlcv_data)
        
        # Calculate trend and volatility for each day
        df = self._calculate_trend(df)
        df = self._calculate_volatility(df)
        
        # Apply minimum duration smoothing
        regimes = self._apply_min_duration(df)
        
        return regimes
    
    def _prepare_data(self, ohlcv_data: List[Dict]) -> pd.DataFrame:
        """Convert OHLCV to DataFrame."""
        df = pd.DataFrame(ohlcv_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        return df
    
    def _calculate_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend using MA alignment."""
        df['MA_50'] = df['Close'].rolling(window=self.ma_short).mean()
        df['MA_200'] = df['Close'].rolling(window=self.ma_long).mean()
        
        # Calculate MA difference percentage
        df['MA_Diff_Pct'] = ((df['MA_50'] - df['MA_200']) / df['MA_200']) * 100
        
        # Price vs MA50 position
        df['Price_vs_MA50'] = ((df['Close'] - df['MA_50']) / df['MA_50']) * 100
        
        # Classify trend
        def classify_trend(row):
            if pd.isna(row['MA_Diff_Pct']) or pd.isna(row['Price_vs_MA50']):
                return 'Unknown'
            
            ma_diff = row['MA_Diff_Pct']
            price_pos = row['Price_vs_MA50']
            
            # Strong Bull: MA50 > MA200 by >5%, Price > MA50
            if ma_diff > 5 and price_pos > 0:
                return 'Strong Bull'
            # Weak Bull: MA50 > MA200 by 0-5%, Price > MA50
            elif ma_diff > 0 and price_pos > 0:
                return 'Weak Bull'
            # Strong Bear: MA50 < MA200 by <-5%, Price < MA50
            elif ma_diff < -5 and price_pos < 0:
                return 'Strong Bear'
            # Weak Bear: MA50 < MA200 by 0 to -5%, Price < MA50
            elif ma_diff < 0 and price_pos < 0:
                return 'Weak Bear'
            # Sideways: Price near MA50 or mixed signals
            else:
                return 'Sideways'
        
        df['Raw_Trend'] = df.apply(classify_trend, axis=1)
        return df
    
    def _calculate_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility using ATR percentile (ticker-specific)."""
        # Calculate True Range
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = abs(df['High'] - df['Close'].shift(1))
        df['Low_Close'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        
        # Calculate ATR as percentage of price
        df['ATR'] = df['TR'].rolling(window=self.vol_window).mean()
        df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
        
        # Calculate rolling percentiles for this ticker (ticker-specific)
        df['ATR_Pctl'] = df['ATR_Pct'].rolling(window=252, min_periods=63).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )
        
        # Classify volatility based on percentiles
        def classify_volatility(row):
            if pd.isna(row['ATR_Pctl']):
                return 'Medium'
            
            pctl = row['ATR_Pctl']
            if pctl > 70:
                return 'High'
            elif pctl < 30:
                return 'Low'
            else:
                return 'Medium'
        
        df['Volatility'] = df.apply(classify_volatility, axis=1)
        return df
    
    def _apply_min_duration(self, df: pd.DataFrame) -> List[Dict]:
        """Apply minimum duration to prevent excessive regime changes."""
        regimes = []
        current_regime = None
        regime_start_idx = 0
        days_in_regime = 0
        pending_regime = None
        pending_days = 0
        
        for idx in range(self.ma_long, len(df)):
            row = df.iloc[idx]
            date = row['Date'].strftime('%Y-%m-%d')
            
            raw_trend = row['Raw_Trend']
            volatility = row['Volatility']
            
            if current_regime is None:
                current_regime = {
                    'trend': raw_trend,
                    'volatility': volatility,
                    'start_date': date,
                    'start_idx': idx
                }
                regime_start_idx = idx
                days_in_regime = 1
            elif raw_trend == current_regime['trend']:
                # Continue current regime
                days_in_regime += 1
                pending_regime = None
                pending_days = 0
            else:
                # Potential regime change
                if pending_regime is None:
                    pending_regime = raw_trend
                    pending_days = 1
                elif pending_regime == raw_trend:
                    pending_days += 1
                    if pending_days >= self.min_duration:
                        # Commit regime change
                        regimes.append({
                            'date': current_regime['start_date'],
                            'trend_regime': current_regime['trend'],
                            'volatility_regime': current_regime['volatility'],
                            'duration': days_in_regime
                        })
                        
                        current_regime = {
                            'trend': pending_regime,
                            'volatility': volatility,
                            'start_date': df.iloc[idx - self.min_duration + 1]['Date'].strftime('%Y-%m-%d'),
                            'start_idx': idx - self.min_duration + 1
                        }
                        regime_start_idx = idx - self.min_duration + 1
                        days_in_regime = self.min_duration
                        pending_regime = None
                        pending_days = 0
                else:
                    # Different pending regime, reset
                    pending_regime = raw_trend
                    pending_days = 1
        
        # Add final regime
        if current_regime is not None and len(df) > 0:
            regimes.append({
                'date': current_regime['start_date'],
                'trend_regime': current_regime['trend'],
                'volatility_regime': current_regime['volatility'],
                'duration': len(df) - regime_start_idx
            })
        
        return regimes


def generate_stable_regimes(ticker: str, min_duration: int = 63):
    """
    Generate stable regime history for a ticker.
    
    Args:
        ticker: Ticker symbol
        min_duration: Minimum days in regime (default 63 = 3 months)
    """
    import os
    
    # Load OHLCV data
    data_path = f'data/tickers/{ticker}/ohlcv.json'
    with open(data_path, 'r') as f:
        ohlcv_data = json.load(f)
    
    print(f"\nGenerating stable regimes for {ticker}...")
    print(f"Data: {len(ohlcv_data)} days")
    print(f"Min regime duration: {min_duration} days (3 months)")
    
    # Detect regimes
    detector = StableRegimeDetector(min_duration=min_duration)
    regimes = detector.detect_regimes(ohlcv_data)
    
    print(f"Detected {len(regimes)} stable regimes:")
    for r in regimes:
        print(f"  {r['date']}: {r['trend_regime']} / {r['volatility_regime']} ({r['duration']} days)")
    
    # Save regime history
    output_path = f'data/tickers/{ticker}/regime_history.json'
    with open(output_path, 'w') as f:
        json.dump(regimes, f, indent=2)
    
    print(f"\nSaved to {output_path}")
    
    return regimes


if __name__ == '__main__':
    import sys
    
    tickers = ['AAPL', 'XOM']
    
    for ticker in tickers:
        try:
            generate_stable_regimes(ticker, min_duration=63)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
