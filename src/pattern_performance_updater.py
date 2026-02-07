"""Pattern Performance Updater - Nightly job to update pattern stats with live results.

This module updates patterns.json with live trading performance,
disables underperforming patterns, and generates alerts.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class PatternPerformanceUpdater:
    """Updates pattern performance based on live trading results."""
    
    # Thresholds for auto-disable
    MIN_TRADES_FOR_EVALUATION = 5  # Minimum trades before evaluating
    MIN_WIN_RATE = 50.0  # Disable if win rate below 50%
    MAX_CONSECUTIVE_LOSSES = 3  # Disable after 3 consecutive losses
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.patterns_file = self.data_dir / 'patterns.json'
        self.trade_outcomes_file = self.data_dir / 'trade_outcomes.json'
        self.performance_log_file = self.data_dir / 'performance_updates.json'
        
        # Load data
        self.patterns = self._load_json(self.patterns_file, [])
        self.trade_outcomes = self._load_json(self.trade_outcomes_file, [])
        
        logger.info(f"PerformanceUpdater: {len(self.patterns)} patterns, {len(self.trade_outcomes)} outcomes")
    
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
    
    def update_all_patterns(self) -> Dict:
        """Update all patterns with live performance data.
        
        Returns:
            Update summary dict
        """
        updates = {
            'timestamp': datetime.now().isoformat(),
            'patterns_updated': 0,
            'patterns_disabled': 0,
            'patterns_enabled': 0,
            'alerts': []
        }
        
        # Group outcomes by pattern
        outcomes_by_pattern = {}
        for outcome in self.trade_outcomes:
            pid = outcome.get('pattern_id')
            if pid not in outcomes_by_pattern:
                outcomes_by_pattern[pid] = []
            outcomes_by_pattern[pid].append(outcome)
        
        # Update each pattern
        for pattern in self.patterns:
            pattern_id = self._get_pattern_id(pattern)
            outcomes = outcomes_by_pattern.get(pattern_id, [])
            
            if len(outcomes) >= self.MIN_TRADES_FOR_EVALUATION:
                result = self._update_pattern(pattern, outcomes)
                updates['patterns_updated'] += 1
                
                if result.get('disabled'):
                    updates['patterns_disabled'] += 1
                    updates['alerts'].append({
                        'type': 'PATTERN_DISABLED',
                        'pattern_id': pattern_id,
                        'reason': result.get('disable_reason'),
                        'live_win_rate': result.get('live_win_rate')
                    })
                elif result.get('enabled'):
                    updates['patterns_enabled'] += 1
        
        # Save updated patterns
        self._save_json(self.patterns_file, self.patterns)
        
        # Log update
        self._log_update(updates)
        
        logger.info(f"Performance update complete: {updates['patterns_updated']} updated, "
                   f"{updates['patterns_disabled']} disabled")
        
        return updates
    
    def _get_pattern_id(self, pattern: Dict) -> int:
        """Extract pattern ID from pattern dict."""
        # Try different possible locations for ID
        if 'id' in pattern:
            return pattern['id']
        if 'pattern' in pattern and 'id' in pattern['pattern']:
            return pattern['pattern']['id']
        # Return index if no ID found
        return self.patterns.index(pattern)
    
    def _update_pattern(self, pattern: Dict, outcomes: List[Dict]) -> Dict:
        """Update a single pattern with live performance.
        
        Returns:
            Result dict with update info
        """
        result = {
            'disabled': False,
            'enabled': False,
            'live_win_rate': 0,
            'disable_reason': None
        }
        
        # Calculate live stats
        total = len(outcomes)
        wins = len([o for o in outcomes if o.get('profit_pct', 0) > 0])
        win_rate = (wins / total * 100) if total > 0 else 0
        
        # Check for consecutive losses
        consecutive_losses = 0
        max_consecutive = 0
        for outcome in sorted(outcomes, key=lambda x: x.get('trigger_date', '')):
            if outcome.get('profit_pct', 0) <= 0:
                consecutive_losses += 1
                max_consecutive = max(max_consecutive, consecutive_losses)
            else:
                consecutive_losses = 0
        
        # Update pattern with live stats
        live_stats = {
            'live_trades': total,
            'live_wins': wins,
            'live_losses': total - wins,
            'live_win_rate': round(win_rate, 2),
            'live_avg_profit': round(sum(o.get('profit_pct', 0) for o in outcomes) / total, 2) if total > 0 else 0,
            'consecutive_losses': consecutive_losses,
            'max_consecutive_losses': max_consecutive,
            'last_updated': datetime.now().isoformat()
        }
        
        # Store live stats in pattern
        if 'pattern' in pattern:
            pattern['pattern']['live_stats'] = live_stats
        else:
            pattern['live_stats'] = live_stats
        
        result['live_win_rate'] = round(win_rate, 2)
        
        # Check if pattern should be disabled
        current_status = pattern.get('status', 'ACTIVE')
        
        if total >= self.MIN_TRADES_FOR_EVALUATION:
            if win_rate < self.MIN_WIN_RATE:
                if current_status != 'DISABLED':
                    pattern['status'] = 'DISABLED'
                    pattern['disabled_reason'] = f'Win rate {win_rate:.1f}% below {self.MIN_WIN_RATE}%'
                    pattern['disabled_at'] = datetime.now().isoformat()
                    result['disabled'] = True
                    result['disable_reason'] = 'Low win rate'
            
            elif max_consecutive >= self.MAX_CONSECUTIVE_LOSSES:
                if current_status != 'DISABLED':
                    pattern['status'] = 'DISABLED'
                    pattern['disabled_reason'] = f'{max_consecutive} consecutive losses'
                    pattern['disabled_at'] = datetime.now().isoformat()
                    result['disabled'] = True
                    result['disable_reason'] = 'Consecutive losses'
            
            elif current_status == 'DISABLED' and win_rate >= self.MIN_WIN_RATE + 10:
                # Re-enable if performance improves significantly
                pattern['status'] = 'ACTIVE'
                pattern['enabled_at'] = datetime.now().isoformat()
                result['enabled'] = True
        
        return result
    
    def _log_update(self, updates: Dict):
        """Log performance update to file."""
        try:
            history = []
            if self.performance_log_file.exists():
                with open(self.performance_log_file, 'r') as f:
                    history = json.load(f)
            
            history.append(updates)
            
            # Keep last 100 updates
            history = history[-100:]
            
            with open(self.performance_log_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Error logging update: {e}")
    
    def get_disabled_patterns(self) -> List[Dict]:
        """Get list of disabled patterns."""
        disabled = []
        for pattern in self.patterns:
            if pattern.get('status') == 'DISABLED':
                disabled.append({
                    'pattern_id': self._get_pattern_id(pattern),
                    'disabled_reason': pattern.get('disabled_reason'),
                    'disabled_at': pattern.get('disabled_at'),
                    'live_stats': pattern.get('live_stats', {})
                })
        return disabled
    
    def get_performance_summary(self) -> Dict:
        """Get overall performance summary."""
        active = [p for p in self.patterns if p.get('status') != 'DISABLED']
        disabled = [p for p in self.patterns if p.get('status') == 'DISABLED']
        
        total_trades = sum(p.get('live_stats', {}).get('live_trades', 0) for p in self.patterns)
        total_wins = sum(p.get('live_stats', {}).get('live_wins', 0) for p in self.patterns)
        
        return {
            'total_patterns': len(self.patterns),
            'active_patterns': len(active),
            'disabled_patterns': len(disabled),
            'total_live_trades': total_trades,
            'total_live_wins': total_wins,
            'overall_live_win_rate': (total_wins / total_trades * 100) if total_trades > 0 else 0
        }


def main():
    """Run performance updater as standalone script."""
    logging.basicConfig(level=logging.INFO)
    
    updater = PatternPerformanceUpdater()
    updates = updater.update_all_patterns()
    
    print("\n" + "="*60)
    print("PATTERN PERFORMANCE UPDATE")
    print("="*60)
    print(f"Patterns updated: {updates['patterns_updated']}")
    print(f"Patterns disabled: {updates['patterns_disabled']}")
    print(f"Patterns enabled: {updates['patterns_enabled']}")
    
    if updates['alerts']:
        print("\nAlerts:")
        for alert in updates['alerts']:
            print(f"  [{alert['type']}] Pattern #{alert['pattern_id']}: {alert['reason']}")
    
    summary = updater.get_performance_summary()
    print(f"\nOverall Performance:")
    print(f"  Active patterns: {summary['active_patterns']}")
    print(f"  Total live trades: {summary['total_live_trades']}")
    print(f"  Overall win rate: {summary['overall_live_win_rate']:.1f}%")


if __name__ == '__main__':
    main()
