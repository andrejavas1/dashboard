"""
Phase 9: Real-Time Pattern Detection System Module
Creates automated monitoring system for detecting patterns in real-time.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yaml
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealTimePatternDetection:
    """
    Real-time pattern detection and alerting system.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the real-time detection system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Data storage
        self.data = None
        self.portfolio = None
        self.detected_patterns = []
        self.alerts = []
        
        # Alert thresholds
        self.almost_triggered_threshold = 0.9  # 90% of conditions met
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'data_sources': {'ticker': 'XOM'},
            'output': {'reports_dir': 'reports', 'data_dir': 'data'}
        }
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load features matrix from file.
        
        Args:
            data_path: Path to features matrix CSV file
            
        Returns:
            DataFrame with features and labels
        """
        logger.info(f"Loading data from {data_path}")
        self.data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
        logger.info(f"Loaded {len(self.data)} records")
        return self.data
    
    def load_portfolio(self, portfolio_path: str) -> List[Dict]:
        """
        Load portfolio from file.
        
        Args:
            portfolio_path: Path to portfolio JSON file
            
        Returns:
            List of patterns
        """
        logger.info(f"Loading portfolio from {portfolio_path}")
        with open(portfolio_path, 'r') as f:
            self.portfolio = json.load(f)
        logger.info(f"Loaded {len(self.portfolio)} patterns")
        return self.portfolio
    
    def check_pattern_conditions(self, pattern: Dict, current_data: pd.Series) -> Tuple[bool, float, Dict]:
        """
        Check if pattern conditions are met for current data.
        
        Args:
            pattern: Pattern dictionary (may be nested with 'pattern' key)
            current_data: Current market data as Series
            
        Returns:
            Tuple of (is_triggered, confidence_score, condition_status)
        """
        # Handle nested pattern structure
        if 'pattern' in pattern:
            pattern_data = pattern['pattern']
        else:
            pattern_data = pattern
        
        conditions = pattern_data['conditions']
        conditions_met = 0
        total_conditions = len(conditions)
        condition_status = {}
        
        for feature, condition in conditions.items():
            if feature not in current_data.index:
                condition_status[feature] = {'met': False, 'value': None, 'required': condition['value']}
                continue
            
            current_value = current_data[feature]
            required_value = condition['value']
            operator = condition['operator']
            
            is_met = False
            if operator == '>=':
                is_met = current_value >= required_value
            elif operator == '<=':
                is_met = current_value <= required_value
            elif operator == '>':
                is_met = current_value > required_value
            elif operator == '<':
                is_met = current_value < required_value
            
            condition_status[feature] = {
                'met': is_met,
                'current_value': current_value,
                'required_value': required_value,
                'operator': operator
            }
            
            if is_met:
                conditions_met += 1
        
        # Calculate confidence score
        confidence_score = conditions_met / total_conditions if total_conditions > 0 else 0
        is_triggered = confidence_score >= 1.0
        
        return is_triggered, confidence_score, condition_status
    
    def scan_latest_data(self) -> List[Dict]:
        """
        Scan the latest data for pattern triggers.
        
        Returns:
            List of detected patterns
        """
        if self.data is None or self.portfolio is None:
            logger.error("Data or portfolio not loaded")
            return []
        
        # Get latest data point
        latest_data = self.data.iloc[-1]
        latest_date = self.data.index[-1]
        
        logger.info(f"Scanning data for {latest_date}...")
        
        detected = []
        almost_triggered = []
        
        for pattern in self.portfolio:
            # Check pattern conditions
            is_triggered, confidence, condition_status = self.check_pattern_conditions(
                pattern, latest_data
            )
            
            if is_triggered:
                # Pattern triggered
                detection = {
                    'pattern_id': self.portfolio.index(pattern),
                    'pattern': pattern,
                    'date': str(latest_date),
                    'confidence_score': confidence,
                    'condition_status': condition_status,
                    'status': 'TRIGGERED'
                }
                detected.append(detection)
            elif confidence >= self.almost_triggered_threshold:
                # Pattern almost triggered
                warning = {
                    'pattern_id': self.portfolio.index(pattern),
                    'pattern': pattern,
                    'date': str(latest_date),
                    'confidence_score': confidence,
                    'condition_status': condition_status,
                    'status': 'ALMOST_TRIGGERED'
                }
                almost_triggered.append(warning)
        
        self.detected_patterns = detected
        
        logger.info(f"  Triggered: {len(detected)} patterns")
        logger.info(f"  Almost Triggered: {len(almost_triggered)} patterns")
        
        return detected + almost_triggered
    
    def generate_alert(self, detection: Dict) -> Dict:
        """
        Generate alert for a detected pattern.
        
        Args:
            detection: Detection dictionary
            
        Returns:
            Alert dictionary
        """
        pattern = detection['pattern']
        # Handle nested pattern structure
        if 'pattern' in pattern:
            pattern_data = pattern['pattern']
            wrapper = pattern
        else:
            pattern_data = pattern
            wrapper = pattern
        
        pattern_id = detection['pattern_id']
        confidence = detection['confidence_score']
        condition_status = detection['condition_status']
        
        # Extract pattern information
        label_col = pattern_data.get('label_col', '')
        parts = label_col.split('_')
        threshold = float(parts[1].replace('pct', '')) if len(parts) > 1 else 5
        window = int(parts[2].replace('d', '')) if len(parts) > 2 else 10
        
        # Get validation metrics
        validation_rate = wrapper.get('validation_success_rate', 0)
        avg_move = wrapper.get('validation_avg_move', 0)
        avg_time = wrapper.get('validation_avg_time', 0)
        
        # Get current price
        current_price = self.data.iloc[-1]['Close']
        target_price = current_price * (1 + threshold / 100)
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'pattern_id': pattern_id,
            'pattern_name': f"Pattern {pattern_id}: {label_col}",
            'status': detection['status'],
            'confidence_score': confidence,
            'current_price': current_price,
            'target_price': target_price,
            'expected_move': f"+{threshold}%",
            'time_window': f"{window} days",
            'historical_success_rate': validation_rate,
            'average_move': f"{avg_move:.2f}%",
            'average_time_to_target': f"{avg_time:.1f} days",
            'conditions': condition_status,
            'actionable_info': {
                'entry_price': current_price,
                'target_price': target_price,
                'stop_loss': current_price * 0.98,  # 2% stop loss
                'time_horizon': f"{window} trading days",
                'probability': f"{validation_rate:.0f}%"
            }
        }
        
        return alert
    
    def generate_daily_report(self) -> Dict:
        """
        Generate daily report of detected patterns.
        
        Returns:
            Daily report dictionary
        """
        # Scan for patterns
        detections = self.scan_latest_data()
        
        # Generate alerts
        alerts = []
        for detection in detections:
            if detection['status'] in ['TRIGGERED', 'ALMOST_TRIGGERED']:
                alert = self.generate_alert(detection)
                alerts.append(alert)
        
        self.alerts = alerts
        
        # Create report
        report = {
            'report_date': datetime.now().isoformat(),
            'data_date': str(self.data.index[-1]),
            'ticker': self.config['data_sources']['ticker'],
            'summary': {
                'total_alerts': len(alerts),
                'triggered_patterns': len([a for a in alerts if a['status'] == 'TRIGGERED']),
                'almost_triggered': len([a for a in alerts if a['status'] == 'ALMOST_TRIGGERED'])
            },
            'alerts': alerts,
            'market_state': self._get_market_state()
        }
        
        return report
    
    def _get_market_state(self) -> Dict:
        """
        Get current market state information.
        
        Returns:
            Market state dictionary
        """
        latest = self.data.iloc[-1]
        
        return {
            'date': str(self.data.index[-1]),
            'close': latest['Close'],
            'volume': latest['Volume'],
            'rsi': latest.get('RSI_14', None),
            'atr_pct': latest.get('ATR_20_Pct', None),
            'trend_regime': latest.get('Trend_Regime', 'Unknown'),
            'volatility_regime': latest.get('Vol_Regime', 'Unknown'),
            'dist_from_ma20': latest.get('Dist_MA_20', None),
            'dist_from_ma200': latest.get('Dist_MA_50', None)  # Use MA_50 for quick mode
        }
    
    def save_alerts(self, report: Dict, output_dir: str = None):
        """
        Save alerts to file.
        
        Args:
            report: Daily report dictionary
            output_dir: Directory to save alerts
        """
        if output_dir is None:
            output_dir = self.config['output']['reports_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save daily report
        report_path = os.path.join(output_dir, f"daily_report_{datetime.now().strftime('%Y%m%d')}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Daily report saved to {report_path}")
        
        # Save latest alerts for easy access
        latest_path = os.path.join(output_dir, "latest_alerts.json")
        with open(latest_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Latest alerts saved to {latest_path}")
    
    def create_pattern_monitoring_dashboard(self) -> Dict:
        """
        Create monitoring dashboard data.
        
        Returns:
            Dashboard data dictionary
        """
        if self.data is None or self.portfolio is None:
            return {}
        
        latest = self.data.iloc[-1]
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'market_state': self._get_market_state(),
            'pattern_status': []
        }
        
        for i, pattern in enumerate(self.portfolio):
            is_triggered, confidence, condition_status = self.check_pattern_conditions(
                pattern, latest
            )
            
            # Handle nested pattern structure
            if 'pattern' in pattern:
                pattern_data = pattern['pattern']
                wrapper = pattern
            else:
                pattern_data = pattern
                wrapper = pattern
            
            pattern_status = {
                'pattern_id': i,
                'label': pattern_data.get('label_col', ''),
                'is_triggered': is_triggered,
                'confidence_score': confidence,
                'status': 'TRIGGERED' if is_triggered else
                          'ALMOST_TRIGGERED' if confidence >= self.almost_triggered_threshold else
                          'NOT_TRIGGERED',
                'conditions_met': sum(1 for c in condition_status.values() if c['met']),
                'total_conditions': len(condition_status),
                'validation_success_rate': wrapper.get('validation_success_rate', 0)
            }
            
            dashboard['pattern_status'].append(pattern_status)
        
        return dashboard
    
    def run_phase9(self, data_path: str = None, portfolio_path: str = None) -> Dict:
        """
        Run complete Phase 9: Real-Time Pattern Detection.
        
        Args:
            data_path: Path to features matrix CSV file
            portfolio_path: Path to portfolio JSON file
            
        Returns:
            Daily report dictionary
        """
        logger.info("\n" + "=" * 60)
        logger.info("STARTING PHASE 9: REAL-TIME PATTERN DETECTION")
        logger.info("=" * 60)
        
        # Load data
        if data_path:
            self.load_data(data_path)
        elif self.data is None:
            default_path = os.path.join("data", "features_matrix.csv")
            if os.path.exists(default_path):
                self.load_data(default_path)
            else:
                logger.error("No data path provided and default file not found")
                return {}
        
        # Load portfolio
        if portfolio_path:
            self.load_portfolio(portfolio_path)
        elif self.portfolio is None:
            default_path = os.path.join("data", "final_portfolio.json")
            if os.path.exists(default_path):
                self.load_portfolio(default_path)
            else:
                logger.error("No portfolio path provided and default file not found")
                return {}
        
        # Generate daily report
        report = self.generate_daily_report()
        
        # Save alerts
        self.save_alerts(report)
        
        # Create dashboard
        dashboard = self.create_pattern_monitoring_dashboard()
        dashboard_path = os.path.join(self.config['output']['reports_dir'], "dashboard.json")
        with open(dashboard_path, 'w') as f:
            json.dump(dashboard, f, indent=2, default=str)
        logger.info(f"Dashboard data saved to {dashboard_path}")
        
        # Log summary
        logger.info(f"\nDaily Report Summary:")
        logger.info(f"  Total Alerts: {report['summary']['total_alerts']}")
        logger.info(f"  Triggered Patterns: {report['summary']['triggered_patterns']}")
        logger.info(f"  Almost Triggered: {report['summary']['almost_triggered']}")
        
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 9 COMPLETE")
        logger.info("=" * 60)
        
        return report


if __name__ == "__main__":
    # Run Phase 9
    rtd = RealTimePatternDetection()
    report = rtd.run_phase9()
    
    print(f"\nFinal Results:")
    print(f"  Total Alerts: {report['summary']['total_alerts']}")
    print(f"  Triggered: {report['summary']['triggered_patterns']}")
    print(f"  Almost Triggered: {report['summary']['almost_triggered']}")