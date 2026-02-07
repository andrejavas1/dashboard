"""Pattern Outcome Tracker - Tracks whether triggered patterns succeed or fail.

This module monitors open trades from pattern triggers and records outcomes
when targets are hit, stops are hit, or holding periods expire.
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class PatternOutcomeTracker:
    """Tracks outcomes of pattern-triggered trades."""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # File paths
        self.open_trades_file = self.data_dir / 'open_trades.json'
        self.trade_outcomes_file = self.data_dir / 'trade_outcomes.json'
        
        # Load existing data
        self.open_trades: List[Dict] = self._load_json(self.open_trades_file, [])
        self.trade_outcomes: List[Dict] = self._load_json(self.trade_outcomes_file, [])
        
        logger.info(f"OutcomeTracker loaded: {len(self.open_trades)} open trades, {len(self.trade_outcomes)} completed")
    
    def _load_json(self, filepath: Path, default):
        """Load JSON file or return default."""
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
        return default
    
    def _save_json(self, filepath: Path, data):
        """Save data to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving {filepath}: {e}")
    
    def record_pattern_trigger(self, 
                               pattern_id: int,
                               trigger_date: str,
                               entry_price: float,
                               direction: str,
                               target_price: float,
                               stop_price: float,
                               label_col: str = 'Label_2.0pct_10d') -> Dict:
        """Record when a pattern triggers an entry signal.
        
        Args:
            pattern_id: The pattern that triggered
            trigger_date: Date of trigger (YYYY-MM-DD)
            entry_price: Entry price
            direction: 'long' or 'short'
            target_price: Profit target price
            stop_price: Stop loss price
            label_col: Label column for holding period (e.g., 'Label_2.0pct_10d')
        
        Returns:
            Trade record dict
        """
        # Parse holding period from label_col
        holding_days = self._parse_holding_period(label_col)
        
        trade = {
            'trade_id': f"{pattern_id}_{trigger_date}_{datetime.now().strftime('%H%M%S')}",
            'pattern_id': pattern_id,
            'trigger_date': trigger_date,
            'entry_price': entry_price,
            'direction': direction,
            'target_price': target_price,
            'stop_price': stop_price,
            'holding_days': holding_days,
            'max_holding_date': (datetime.strptime(trigger_date, '%Y-%m-%d') + 
                                timedelta(days=holding_days)).strftime('%Y-%m-%d'),
            'status': 'OPEN',
            'created_at': datetime.now().isoformat()
        }
        
        self.open_trades.append(trade)
        self._save_json(self.open_trades_file, self.open_trades)
        
        logger.info(f"Recorded open trade: Pattern #{pattern_id} {direction} @ {entry_price:.2f}")
        return trade
    
    def _parse_holding_period(self, label_col: str) -> int:
        """Parse holding period from label column name.
        
        Examples:
            'Label_2.0pct_10d' -> 10 days
            'Label_3.0pct_20d' -> 20 days
        """
        try:
            # Split by underscore and get last part
            parts = label_col.split('_')
            if len(parts) >= 3:
                days_part = parts[-1]  # e.g., '10d'
                return int(days_part.replace('d', ''))
        except Exception as e:
            logger.warning(f"Could not parse holding period from {label_col}: {e}")
        
        return 10  # Default 10 days
    
    def check_open_trades(self, price_data: pd.DataFrame, current_date: str) -> List[Dict]:
        """Check all open trades against current price data.
        
        Args:
            price_data: DataFrame with OHLCV data, indexed by date
            current_date: Current date being checked (YYYY-MM-DD)
        
        Returns:
            List of trades that were closed
        """
        closed_trades = []
        still_open = []
        
        for trade in self.open_trades:
            if trade['status'] != 'OPEN':
                continue
            
            outcome = self._check_trade_outcome(trade, price_data, current_date)
            
            if outcome:
                # Trade closed
                trade.update(outcome)
                trade['status'] = 'CLOSED'
                trade['closed_at'] = current_date
                self.trade_outcomes.append(trade)
                closed_trades.append(trade)
                logger.info(f"Trade closed: Pattern #{trade['pattern_id']} - {outcome['outcome']} "
                          f"({outcome['profit_pct']:.2f}%)")
            else:
                # Still open
                still_open.append(trade)
        
        # Update files
        self.open_trades = still_open
        self._save_json(self.open_trades_file, self.open_trades)
        self._save_json(self.trade_outcomes_file, self.trade_outcomes)
        
        return closed_trades
    
    def _check_trade_outcome(self, trade: Dict, price_data: pd.DataFrame, 
                            current_date: str) -> Optional[Dict]:
        """Check if a trade has hit target, stop, or max holding period.
        
        Returns:
            Outcome dict if trade closed, None if still open
        """
        trigger_date = trade['trigger_date']
        direction = trade['direction']
        target = trade['target_price']
        stop = trade['stop_price']
        max_date = trade['max_holding_date']
        
        # Get price data from trigger date to current date
        mask = (price_data.index >= trigger_date) & (price_data.index <= current_date)
        trade_data = price_data.loc[mask]
        
        if trade_data.empty:
            return None
        
        # Check if target or stop was hit
        for date, row in trade_data.iterrows():
            date_str = date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date)
            
            if direction == 'long':
                # For long: check if high hit target or low hit stop
                if row['High'] >= target:
                    return {
                        'outcome': 'TARGET_HIT',
                        'exit_price': target,
                        'exit_date': date_str,
                        'profit_pct': ((target - trade['entry_price']) / trade['entry_price']) * 100,
                        'days_held': (datetime.strptime(date_str, '%Y-%m-%d') - 
                                     datetime.strptime(trigger_date, '%Y-%m-%d')).days
                    }
                elif row['Low'] <= stop:
                    return {
                        'outcome': 'STOP_HIT',
                        'exit_price': stop,
                        'exit_date': date_str,
                        'profit_pct': ((stop - trade['entry_price']) / trade['entry_price']) * 100,
                        'days_held': (datetime.strptime(date_str, '%Y-%m-%d') - 
                                     datetime.strptime(trigger_date, '%Y-%m-%d')).days
                    }
            else:  # short
                # For short: check if low hit target or high hit stop
                if row['Low'] <= target:
                    return {
                        'outcome': 'TARGET_HIT',
                        'exit_price': target,
                        'exit_date': date_str,
                        'profit_pct': ((trade['entry_price'] - target) / trade['entry_price']) * 100,
                        'days_held': (datetime.strptime(date_str, '%Y-%m-%d') - 
                                     datetime.strptime(trigger_date, '%Y-%m-%d')).days
                    }
                elif row['High'] >= stop:
                    return {
                        'outcome': 'STOP_HIT',
                        'exit_price': stop,
                        'exit_date': date_str,
                        'profit_pct': ((trade['entry_price'] - stop) / trade['entry_price']) * 100,
                        'days_held': (datetime.strptime(date_str, '%Y-%m-%d') - 
                                     datetime.strptime(trigger_date, '%Y-%m-%d')).days
                    }
        
        # Check max holding period
        if current_date >= max_date:
            # Close at current price
            last_price = trade_data.iloc[-1]['Close']
            if direction == 'long':
                profit_pct = ((last_price - trade['entry_price']) / trade['entry_price']) * 100
            else:
                profit_pct = ((trade['entry_price'] - last_price) / trade['entry_price']) * 100
            
            return {
                'outcome': 'TIMEOUT',
                'exit_price': last_price,
                'exit_date': current_date,
                'profit_pct': profit_pct,
                'days_held': (datetime.strptime(current_date, '%Y-%m-%d') - 
                             datetime.strptime(trigger_date, '%Y-%m-%d')).days
            }
        
        # Still open
        return None
    
    def get_pattern_performance(self, pattern_id: Optional[int] = None) -> Dict:
        """Get performance statistics for patterns.
        
        Args:
            pattern_id: Specific pattern, or None for all patterns
        
        Returns:
            Performance statistics dict
        """
        outcomes = self.trade_outcomes
        if pattern_id is not None:
            outcomes = [o for o in outcomes if o['pattern_id'] == pattern_id]
        
        if not outcomes:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'total_profit_pct': 0
            }
        
        total = len(outcomes)
        wins = len([o for o in outcomes if o.get('profit_pct', 0) > 0])
        losses = total - wins
        profits = [o.get('profit_pct', 0) for o in outcomes]
        
        return {
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': (wins / total * 100) if total > 0 else 0,
            'avg_profit': sum(profits) / len(profits) if profits else 0,
            'total_profit_pct': sum(profits),
            'target_hits': len([o for o in outcomes if o.get('outcome') == 'TARGET_HIT']),
            'stop_hits': len([o for o in outcomes if o.get('outcome') == 'STOP_HIT']),
            'timeouts': len([o for o in outcomes if o.get('outcome') == 'TIMEOUT'])
        }
    
    def get_open_trades(self) -> List[Dict]:
        """Get list of currently open trades."""
        return self.open_trades
    
    def get_trade_outcomes(self, pattern_id: Optional[int] = None) -> List[Dict]:
        """Get list of completed trade outcomes."""
        if pattern_id is not None:
            return [o for o in self.trade_outcomes if o['pattern_id'] == pattern_id]
        return self.trade_outcomes
