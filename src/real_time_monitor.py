"""
Real-Time Pattern Monitoring System

Comprehensive real-time monitoring system for tracking pattern occurrences,
generating alerts for high-probability patterns, and monitoring system performance.

Author: Agent_Visualization
Task: Task 5.3 - Real-Time Monitoring System
"""

import os
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Alert types."""
    PATTERN_TRIGGERED = "pattern_triggered"
    PATTERN_ALMOST_TRIGGERED = "pattern_almost_triggered"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SYSTEM_ERROR = "system_error"
    DATA_ANOMALY = "data_anomaly"
    THRESHOLD_EXCEEDED = "threshold_exceeded"


class NotificationChannel(Enum):
    """Notification channels."""
    LOG = "log"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    DATABASE = "database"


@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    timestamp: str
    alert_type: AlertType
    severity: AlertSeverity
    pattern_id: Optional[int]
    pattern_name: Optional[str]
    message: str
    details: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False
    resolution_notes: Optional[str] = None


@dataclass
class PatternTrigger:
    """Pattern trigger event."""
    pattern_id: int
    pattern_name: str
    timestamp: str
    confidence_score: float
    conditions_met: int
    total_conditions: int
    condition_status: Dict[str, Dict]
    current_price: float
    target_price: float
    expected_move: str
    time_window: str
    historical_success_rate: float
    average_move: float
    average_time_to_target: float


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: str
    cpu_usage: float
    memory_usage: float
    scan_duration: float
    patterns_scanned: int
    patterns_triggered: int
    alerts_generated: int
    data_points_processed: int
    last_data_update: str


@dataclass
class MonitoringConfig:
    """Monitoring system configuration."""
    # Data sources
    data_path: str = "data/features_matrix.csv"
    portfolio_path: str = "data/final_portfolio.json"
    
    # Monitoring settings
    scan_interval: int = 60  # seconds
    almost_triggered_threshold: float = 0.9
    high_probability_threshold: float = 0.85
    critical_threshold: float = 0.95
    max_patterns_display: int = 20  # Maximum patterns to display in dashboard
    
    # Alert settings
    enable_alerts: bool = True
    alert_cooldown: int = 300  # seconds between same alert
    max_alerts_per_hour: int = 100
    
    # Notification channels
    notification_channels: List[str] = field(default_factory=lambda: ["log", "database"])
    
    # Storage
    alerts_dir: str = "reports/alerts"
    dashboard_dir: str = "reports/dashboard"
    metrics_dir: str = "reports/metrics"
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    metrics_retention_hours: int = 24
    
    # Visualization integration
    enable_visualization: bool = True
    charts_dir: str = "charts/monitoring"


class AlertManager:
    """Manages alert generation, routing, and lifecycle."""
    
    def __init__(self, config: MonitoringConfig):
        """
        Initialize alert manager.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config
        self.alerts: List[Alert] = []
        self.alert_history: deque = deque(maxlen=1000)
        self.last_alert_times: Dict[str, datetime] = {}
        self.alert_count_hourly: deque = deque()
        
        # Create alerts directory
        os.makedirs(self.config.alerts_dir, exist_ok=True)
    
    def generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        return f"ALT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.alerts):04d}"
    
    def check_alert_cooldown(self, alert_key: str) -> bool:
        """
        Check if alert is in cooldown period.
        
        Args:
            alert_key: Unique key for alert type
            
        Returns:
            True if alert can be sent, False if in cooldown
        """
        if alert_key in self.last_alert_times:
            elapsed = (datetime.now() - self.last_alert_times[alert_key]).total_seconds()
            if elapsed < self.config.alert_cooldown:
                return False
        return True
    
    def check_hourly_limit(self) -> bool:
        """
        Check if hourly alert limit has been reached.
        
        Returns:
            True if limit not exceeded, False otherwise
        """
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Remove old entries
        while self.alert_count_hourly and self.alert_count_hourly[0] < hour_ago:
            self.alert_count_hourly.popleft()
        
        return len(self.alert_count_hourly) < self.config.max_alerts_per_hour
    
    def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        pattern_id: Optional[int],
        pattern_name: Optional[str],
        message: str,
        details: Dict[str, Any]
    ) -> Optional[Alert]:
        """
        Create a new alert.
        
        Args:
            alert_type: Type of alert
            severity: Alert severity
            pattern_id: Associated pattern ID
            pattern_name: Associated pattern name
            message: Alert message
            details: Additional details
            
        Returns:
            Alert object or None if limits exceeded
        """
        if not self.config.enable_alerts:
            return None
        
        # Check cooldown
        alert_key = f"{alert_type.value}_{pattern_id}"
        if not self.check_alert_cooldown(alert_key):
            logger.debug(f"Alert in cooldown: {alert_key}")
            return None
        
        # Check hourly limit
        if not self.check_hourly_limit():
            logger.warning("Hourly alert limit exceeded")
            return None
        
        alert = Alert(
            alert_id=self.generate_alert_id(),
            timestamp=datetime.now().isoformat(),
            alert_type=alert_type,
            severity=severity,
            pattern_id=pattern_id,
            pattern_name=pattern_name,
            message=message,
            details=details
        )
        
        self.alerts.append(alert)
        self.alert_history.append(alert)
        self.last_alert_times[alert_key] = datetime.now()
        self.alert_count_hourly.append(datetime.now())
        
        logger.info(f"Alert created: {alert.alert_id} - {message}")
        
        return alert
    
    def send_alert(self, alert: Alert):
        """
        Send alert through configured channels.
        
        Args:
            alert: Alert to send
        """
        for channel in self.config.notification_channels:
            try:
                if channel == "log":
                    self._send_log_alert(alert)
                elif channel == "database":
                    self._save_to_database(alert)
                elif channel == "email":
                    self._send_email_alert(alert)
                elif channel == "webhook":
                    self._send_webhook_alert(alert)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {e}")
    
    def _send_log_alert(self, alert: Alert):
        """Send alert to log."""
        severity_map = {
            AlertSeverity.LOW: logging.INFO,
            AlertSeverity.MEDIUM: logging.WARNING,
            AlertSeverity.HIGH: logging.WARNING,
            AlertSeverity.CRITICAL: logging.ERROR
        }
        log_level = severity_map.get(alert.severity, logging.INFO)
        
        logger.log(
            log_level,
            f"[{alert.alert_id}] {alert.severity.value.upper()}: {alert.message}"
        )
    
    def _save_to_database(self, alert: Alert):
        """Save alert to database (JSON file)."""
        alert_file = os.path.join(
            self.config.alerts_dir,
            f"alert_{datetime.now().strftime('%Y%m%d')}.json"
        )
        
        # Load existing alerts
        alerts = []
        if os.path.exists(alert_file):
            with open(alert_file, 'r') as f:
                alerts = json.load(f)
        
        # Add new alert
        alert_dict = asdict(alert)
        alert_dict['alert_type'] = alert.alert_type.value
        alert_dict['severity'] = alert.severity.value
        
        alerts.append(alert_dict)
        
        # Save
        with open(alert_file, 'w') as f:
            json.dump(alerts, f, indent=2)
    
    def _send_email_alert(self, alert: Alert):
        """Send alert via email (placeholder)."""
        logger.info(f"Email alert would be sent: {alert.alert_id}")
        # TODO: Implement email notification
    
    def _send_webhook_alert(self, alert: Alert):
        """Send alert via webhook (placeholder)."""
        logger.info(f"Webhook alert would be sent: {alert.alert_id}")
        # TODO: Implement webhook notification
    
    def acknowledge_alert(self, alert_id: str, notes: Optional[str] = None):
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID to acknowledge
            notes: Optional notes
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.resolution_notes = notes
                logger.info(f"Alert acknowledged: {alert_id}")
                return
    
    def resolve_alert(self, alert_id: str, notes: Optional[str] = None):
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert ID to resolve
            notes: Optional notes
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolution_notes = notes
                logger.info(f"Alert resolved: {alert_id}")
                return
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [a for a in self.alerts if not a.resolved]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity."""
        return [a for a in self.alerts if a.severity == severity and not a.resolved]


class PerformanceMonitor:
    """Monitors system performance metrics."""
    
    def __init__(self, config: MonitoringConfig):
        """
        Initialize performance monitor.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config
        self.metrics_history: deque = deque(maxlen=1000)
        self.start_time = datetime.now()
        
        # Create metrics directory
        os.makedirs(self.config.metrics_dir, exist_ok=True)
    
    def record_metrics(
        self,
        scan_duration: float,
        patterns_scanned: int,
        patterns_triggered: int,
        alerts_generated: int,
        data_points_processed: int
    ) -> SystemMetrics:
        """
        Record system metrics.
        
        Args:
            scan_duration: Time taken for scan
            patterns_scanned: Number of patterns scanned
            patterns_triggered: Number of patterns triggered
            alerts_generated: Number of alerts generated
            data_points_processed: Number of data points
            
        Returns:
            SystemMetrics object
        """
        metrics = SystemMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_usage=self._get_cpu_usage(),
            memory_usage=self._get_memory_usage(),
            scan_duration=scan_duration,
            patterns_scanned=patterns_scanned,
            patterns_triggered=patterns_triggered,
            alerts_generated=alerts_generated,
            data_points_processed=data_points_processed,
            last_data_update=datetime.now().isoformat()
        )
        
        self.metrics_history.append(metrics)
        self._save_metrics(metrics)
        
        return metrics
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage (placeholder)."""
        # In production, use psutil or similar
        return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage (placeholder)."""
        # In production, use psutil or similar
        return 0.0
    
    def _save_metrics(self, metrics: SystemMetrics):
        """Save metrics to file."""
        metrics_file = os.path.join(
            self.config.metrics_dir,
            f"metrics_{datetime.now().strftime('%Y%m%d')}.json"
        )
        
        # Load existing metrics
        metrics_list = []
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics_list = json.load(f)
        
        # Add new metrics
        metrics_dict = asdict(metrics)
        metrics_list.append(metrics_dict)
        
        # Save
        with open(metrics_file, 'w') as f:
            json.dump(metrics_list, f, indent=2)
    
    def get_average_scan_duration(self) -> float:
        """Get average scan duration."""
        if not self.metrics_history:
            return 0.0
        return np.mean([m.scan_duration for m in self.metrics_history])
    
    def get_uptime(self) -> timedelta:
        """Get system uptime."""
        return datetime.now() - self.start_time
    
    def get_recent_metrics(self, hours: int = 1) -> List[SystemMetrics]:
        """
        Get metrics from last N hours.
        
        Args:
            hours: Number of hours
            
        Returns:
            List of metrics
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            m for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) > cutoff
        ]


class RealTimeMonitor:
    """
    Main real-time monitoring system.
    
    Integrates pattern detection, alerting, and performance monitoring.
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None, config_path: str = "config.yaml"):
        """
        Initialize real-time monitor.
        
        Args:
            config: Monitoring configuration (uses defaults if None)
            config_path: Path to config.yaml file
        """
        # Load settings from config.yaml if available
        if config is None:
            config = self._load_config_from_yaml(config_path)
        
        self.config = config or MonitoringConfig()
        
        # Core components
        self.alert_manager = AlertManager(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)
        
        # Data storage
        self.data: Optional[pd.DataFrame] = None
        self.portfolio: Optional[List[Dict]] = None
        self.pattern_triggers: deque = deque(maxlen=self.config.max_triggers_history)
        
        # Monitoring state
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Create directories
        os.makedirs(self.config.dashboard_dir, exist_ok=True)
        os.makedirs(self.config.charts_dir, exist_ok=True)
        
        logger.info(f"Real-Time Monitor initialized (max_patterns_display: {self.config.max_patterns_display})")
    
    def _load_config_from_yaml(self, config_path: str) -> Optional[MonitoringConfig]:
        """
        Load monitoring configuration from YAML file.
        
        Args:
            config_path: Path to config.yaml file
            
        Returns:
            MonitoringConfig object or None
        """
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            monitoring_config = yaml_config.get('monitoring', {})
            
            return MonitoringConfig(
                max_patterns_display=monitoring_config.get('max_patterns_display', 50),
                scan_interval=monitoring_config.get('scan_interval', 60),
                almost_triggered_threshold=monitoring_config.get('almost_triggered_threshold', 0.9),
                high_probability_threshold=monitoring_config.get('high_probability_threshold', 0.85),
                critical_threshold=monitoring_config.get('critical_threshold', 0.95),
                max_triggers_history=monitoring_config.get('max_triggers_history', 500)
            )
        except Exception as e:
            logger.warning(f"Could not load monitoring config from {config_path}: {e}")
            return None
    
    def load_config(self, config_path: str):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
        """
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Update config
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            logger.info(f"Configuration loaded from {config_path}")
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    def load_data(self, data_path: Optional[str] = None):
        """
        Load market data.
        
        Args:
            data_path: Path to data file (uses config default if None)
        """
        path = data_path or self.config.data_path
        
        try:
            logger.info(f"Loading data from {path}")
            self.data = pd.read_csv(path, index_col='Date', parse_dates=True)
            logger.info(f"Loaded {len(self.data)} records")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def load_portfolio(self, portfolio_path: Optional[str] = None):
        """
        Load pattern portfolio.
        
        Args:
            portfolio_path: Path to portfolio file (uses config default if None)
        """
        path = portfolio_path or self.config.portfolio_path
        
        try:
            logger.info(f"Loading portfolio from {path}")
            with open(path, 'r') as f:
                self.portfolio = json.load(f)
            logger.info(f"Loaded {len(self.portfolio)} patterns")
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
            raise
    
    def check_pattern_conditions(
        self,
        pattern: Dict,
        current_data: pd.Series
    ) -> Tuple[bool, float, Dict]:
        """
        Check if pattern conditions are met.
        
        Args:
            pattern: Pattern dictionary
            current_data: Current market data
            
        Returns:
            Tuple of (is_triggered, confidence_score, condition_status)
        """
        # Handle nested pattern structure
        if 'pattern' in pattern:
            pattern_data = pattern['pattern']
        else:
            pattern_data = pattern
        
        conditions = pattern_data.get('conditions', {})
        conditions_met = 0
        total_conditions = len(conditions)
        condition_status = {}
        
        for feature, condition in conditions.items():
            if feature not in current_data.index:
                condition_status[feature] = {
                    'met': False,
                    'value': None,
                    'required': condition.get('value', 0)
                }
                continue
            
            current_value = current_data[feature]
            required_value = condition.get('value', 0)
            operator = condition.get('operator', '>=')
            
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
        
        confidence_score = conditions_met / total_conditions if total_conditions > 0 else 0
        is_triggered = confidence_score >= 1.0
        
        return is_triggered, confidence_score, condition_status
    
    def scan_patterns(self) -> List[PatternTrigger]:
        """
        Scan all patterns for triggers.
        
        Returns:
            List of pattern triggers
        """
        if self.data is None or self.portfolio is None:
            logger.error("Data or portfolio not loaded")
            return []
        
        start_time = time.time()
        latest_data = self.data.iloc[-1]
        latest_date = self.data.index[-1]
        
        logger.info(f"Scanning patterns for {latest_date}...")
        
        triggers = []
        for i, pattern in enumerate(self.portfolio):
            is_triggered, confidence, condition_status = self.check_pattern_conditions(
                pattern, latest_data
            )
            
            # Handle nested pattern structure
            if 'pattern' in pattern:
                pattern_data = pattern['pattern']
                wrapper = pattern
            else:
                pattern_data = pattern
                wrapper = pattern
            
            # Determine if we should create a trigger
            if is_triggered or confidence >= self.config.almost_triggered_threshold:
                # Extract pattern info
                label_col = pattern_data.get('label_col', '')
                parts = label_col.split('_')
                threshold = int(parts[1].replace('pct', '')) if len(parts) > 1 else 5
                window = int(parts[2].replace('d', '')) if len(parts) > 2 else 10
                
                current_price = latest_data['Close']
                target_price = current_price * (1 + threshold / 100)
                
                trigger = PatternTrigger(
                    pattern_id=i,
                    pattern_name=label_col or f"Pattern {i}",
                    timestamp=str(latest_date),
                    confidence_score=confidence,
                    conditions_met=sum(1 for c in condition_status.values() if c['met']),
                    total_conditions=len(condition_status),
                    condition_status=condition_status,
                    current_price=current_price,
                    target_price=target_price,
                    expected_move=f"+{threshold}%",
                    time_window=f"{window} days",
                    historical_success_rate=wrapper.get('validation_success_rate', 0),
                    average_move=wrapper.get('validation_avg_move', 0),
                    average_time_to_target=wrapper.get('validation_avg_time', 0)
                )
                
                triggers.append(trigger)
                
                # Generate alert
                self._generate_pattern_alert(trigger, is_triggered)
        
        scan_duration = time.time() - start_time
        
        # Record performance metrics
        self.performance_monitor.record_metrics(
            scan_duration=scan_duration,
            patterns_scanned=len(self.portfolio),
            patterns_triggered=len([t for t in triggers if t.confidence_score >= 1.0]),
            alerts_generated=len(triggers),
            data_points_processed=len(latest_data)
        )
        
        self.pattern_triggers.extend(triggers)
        
        logger.info(f"Scan complete: {len(triggers)} triggers found in {scan_duration:.2f}s")
        
        return triggers
    
    def _generate_pattern_alert(self, trigger: PatternTrigger, is_triggered: bool):
        """
        Generate alert for pattern trigger.
        
        Args:
            trigger: Pattern trigger
            is_triggered: Whether pattern is fully triggered
        """
        # Determine alert type and severity
        if is_triggered:
            alert_type = AlertType.PATTERN_TRIGGERED
            
            # Determine severity based on historical success rate
            if trigger.historical_success_rate >= self.config.critical_threshold:
                severity = AlertSeverity.CRITICAL
            elif trigger.historical_success_rate >= self.config.high_probability_threshold:
                severity = AlertSeverity.HIGH
            else:
                severity = AlertSeverity.MEDIUM
        else:
            alert_type = AlertType.PATTERN_ALMOST_TRIGGERED
            severity = AlertSeverity.LOW
        
        # Create alert message
        message = (
            f"Pattern {trigger.pattern_id} ({trigger.pattern_name}) "
            f"{'TRIGGERED' if is_triggered else 'ALMOST TRIGGERED'} - "
            f"Confidence: {trigger.confidence_score:.1%}"
        )
        
        # Create alert details
        details = {
            'confidence_score': trigger.confidence_score,
            'conditions_met': trigger.conditions_met,
            'total_conditions': trigger.total_conditions,
            'current_price': trigger.current_price,
            'target_price': trigger.target_price,
            'expected_move': trigger.expected_move,
            'time_window': trigger.time_window,
            'historical_success_rate': trigger.historical_success_rate,
            'average_move': trigger.average_move,
            'average_time_to_target': trigger.average_time_to_target,
            'condition_status': trigger.condition_status
        }
        
        # Create and send alert
        alert = self.alert_manager.create_alert(
            alert_type=alert_type,
            severity=severity,
            pattern_id=trigger.pattern_id,
            pattern_name=trigger.pattern_name,
            message=message,
            details=details
        )
        
        if alert:
            self.alert_manager.send_alert(alert)
    
    def create_dashboard_data(self) -> Dict:
        """
        Create dashboard data for visualization.
        
        Returns:
            Dashboard data dictionary
        """
        if self.data is None or self.portfolio is None:
            return {}
        
        latest = self.data.iloc[-1]
        recent_metrics = self.performance_monitor.get_recent_metrics(hours=1)
        
        # Pattern status
        pattern_status = []
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
            
            status = {
                'pattern_id': i,
                'label': pattern_data.get('label_col', ''),
                'is_triggered': is_triggered,
                'confidence_score': confidence,
                'status': (
                    'TRIGGERED' if is_triggered else
                    'ALMOST_TRIGGERED' if confidence >= self.config.almost_triggered_threshold else
                    'NOT_TRIGGERED'
                ),
                'conditions_met': sum(1 for c in condition_status.values() if c['met']),
                'total_conditions': len(condition_status),
                'validation_success_rate': wrapper.get('validation_success_rate', 0),
                'classification': wrapper.get('classification', 'UNKNOWN')
            }
            pattern_status.append(status)
        
        # Recent triggers
        recent_triggers = [
            {
                'pattern_id': t.pattern_id,
                'pattern_name': t.pattern_name,
                'timestamp': t.timestamp,
                'confidence_score': t.confidence_score,
                'current_price': t.current_price,
                'target_price': t.target_price
            }
            for t in self.pattern_triggers[-20:]  # Last 20 triggers
        ]
        
        # Active alerts
        active_alerts = [
            {
                'alert_id': a.alert_id,
                'alert_type': a.alert_type.value,
                'severity': a.severity.value,
                'pattern_id': a.pattern_id,
                'pattern_name': a.pattern_name,
                'message': a.message,
                'timestamp': a.timestamp
            }
            for a in self.alert_manager.get_active_alerts()
        ]
        
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'system_status': {
                'is_running': self.is_running,
                'uptime': str(self.performance_monitor.get_uptime()),
                'last_scan': datetime.now().isoformat(),
                'scan_interval': self.config.scan_interval
            },
            'market_state': {
                'date': str(self.data.index[-1]),
                'close': float(latest['Close']),
                'volume': float(latest['Volume']),
                'rsi': float(latest.get('RSI_14', 0)),
                'atr_pct': float(latest.get('ATR_20_Pct', 0)),
                'trend_regime': latest.get('Trend_Regime', 'Unknown'),
                'volatility_regime': latest.get('Vol_Regime', 'Unknown')
            },
            'pattern_status': pattern_status,
            'recent_triggers': recent_triggers,
            'active_alerts': active_alerts,
            'performance_metrics': {
                'avg_scan_duration': self.performance_monitor.get_average_scan_duration(),
                'total_scans': len(self.performance_monitor.metrics_history),
                'total_alerts': len(self.alert_manager.alerts),
                'active_alerts_count': len(self.alert_manager.get_active_alerts())
            },
            'summary': {
                'total_patterns': len(self.portfolio),
                'triggered_patterns': len([p for p in pattern_status if p['is_triggered']]),
                'almost_triggered': len([p for p in pattern_status if p['status'] == 'ALMOST_TRIGGERED']),
                'not_triggered': len([p for p in pattern_status if p['status'] == 'NOT_TRIGGERED'])
            }
        }
        
        return dashboard
    
    def save_dashboard(self, dashboard: Dict):
        """
        Save dashboard data to file.
        
        Args:
            dashboard: Dashboard data
        """
        dashboard_path = os.path.join(self.config.dashboard_dir, "monitoring_dashboard.json")
        with open(dashboard_path, 'w') as f:
            json.dump(dashboard, f, indent=2, default=str)
        logger.info(f"Dashboard saved to {dashboard_path}")
    
    def run_single_scan(self):
        """Run a single pattern scan."""
        try:
            triggers = self.scan_patterns()
            dashboard = self.create_dashboard_data()
            self.save_dashboard(dashboard)
            return triggers
        except Exception as e:
            logger.error(f"Error during scan: {e}")
            # Generate system error alert
            alert = self.alert_manager.create_alert(
                alert_type=AlertType.SYSTEM_ERROR,
                severity=AlertSeverity.HIGH,
                pattern_id=None,
                pattern_name=None,
                message=f"Scan error: {str(e)}",
                details={'error': str(e)}
            )
            if alert:
                self.alert_manager.send_alert(alert)
            return []
    
    def start_monitoring(self):
        """Start continuous monitoring in background thread."""
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Monitoring started (interval: {self.config.scan_interval}s)")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                self.run_single_scan()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Wait for next scan or stop event
            self.stop_event.wait(self.config.scan_interval)
        
        logger.info("Monitoring loop stopped")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        if not self.is_running:
            logger.warning("Monitoring not running")
            return
        
        self.is_running = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Monitoring stopped")
    
    def get_status(self) -> Dict:
        """
        Get current monitoring status.
        
        Returns:
            Status dictionary
        """
        return {
            'is_running': self.is_running,
            'uptime': str(self.performance_monitor.get_uptime()),
            'total_scans': len(self.performance_monitor.metrics_history),
            'total_alerts': len(self.alert_manager.alerts),
            'active_alerts': len(self.alert_manager.get_active_alerts()),
            'last_scan': datetime.now().isoformat(),
            'config': {
                'scan_interval': self.config.scan_interval,
                'almost_triggered_threshold': self.config.almost_triggered_threshold,
                'high_probability_threshold': self.config.high_probability_threshold,
                'critical_threshold': self.config.critical_threshold
            }
        }


def create_monitoring_dashboard_html(dashboard_data: Dict, output_path: str):
    """
    Create HTML monitoring dashboard.
    
    Args:
        dashboard_data: Dashboard data dictionary
        output_path: Path to save HTML file
    """
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-Time Pattern Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            padding: 20px;
        }
        .dashboard {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
        }
        .header h1 {
            color: #00d4ff;
            margin-bottom: 10px;
        }
        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }
        .status-running {
            background: #00c853;
            color: white;
        }
        .status-stopped {
            background: #ff5252;
            color: white;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .card h3 {
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 18px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .metric:last-child {
            border-bottom: none;
        }
        .metric-label {
            color: #a0a0a0;
        }
        .metric-value {
            font-weight: bold;
            color: #00d4ff;
        }
        .alert-item {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }
        .alert-critical {
            background: rgba(255, 82, 82, 0.2);
            border-left-color: #ff5252;
        }
        .alert-high {
            background: rgba(255, 152, 0, 0.2);
            border-left-color: #ff9800;
        }
        .alert-medium {
            background: rgba(255, 193, 7, 0.2);
            border-left-color: #ffc107;
        }
        .alert-low {
            background: rgba(0, 200, 83, 0.2);
            border-left-color: #00c853;
        }
        .pattern-item {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            background: rgba(255, 255, 255, 0.05);
        }
        .pattern-triggered {
            border-left: 4px solid #00c853;
        }
        .pattern-almost {
            border-left: 4px solid #ffc107;
        }
        .pattern-not-triggered {
            border-left: 4px solid #757575;
        }
        .confidence-bar {
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            margin-top: 5px;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        .confidence-high {
            background: linear-gradient(90deg, #00c853, #00e676);
        }
        .confidence-medium {
            background: linear-gradient(90deg, #ffc107, #ffeb3b);
        }
        .confidence-low {
            background: linear-gradient(90deg, #757575, #9e9e9e);
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            color: #a0a0a0;
            font-size: 12px;
        }
        .refresh-btn {
            background: #00d4ff;
            color: #1a1a2e;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            margin-left: 10px;
        }
        .refresh-btn:hover {
            background: #00b8d4;
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>Real-Time Pattern Monitoring Dashboard</h1>
            <p>
                Last Updated: <span id="last-update">{{timestamp}}</span>
                <button class="refresh-btn" onclick="location.reload()">Refresh</button>
            </p>
            <p>
                Status: <span class="status-badge {{status_class}}">{{status}}</span>
                | Uptime: <span class="metric-value">{{uptime}}</span>
            </p>
        </div>
        
        <div class="grid">
            <!-- Summary Card -->
            <div class="card">
                <h3>Summary</h3>
                <div class="metric">
                    <span class="metric-label">Total Patterns</span>
                    <span class="metric-value">{{total_patterns}}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Triggered</span>
                    <span class="metric-value" style="color: #00c853">{{triggered}}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Almost Triggered</span>
                    <span class="metric-value" style="color: #ffc107">{{almost_triggered}}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Not Triggered</span>
                    <span class="metric-value" style="color: #757575">{{not_triggered}}</span>
                </div>
            </div>
            
            <!-- Market State Card -->
            <div class="card">
                <h3>Market State</h3>
                <div class="metric">
                    <span class="metric-label">Date</span>
                    <span class="metric-value">{{market_date}}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Close Price</span>
                    <span class="metric-value">${{close_price}}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Volume</span>
                    <span class="metric-value">{{volume}}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">RSI (14)</span>
                    <span class="metric-value">{{rsi}}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Trend Regime</span>
                    <span class="metric-value">{{trend_regime}}</span>
                </div>
            </div>
            
            <!-- Performance Card -->
            <div class="card">
                <h3>Performance Metrics</h3>
                <div class="metric">
                    <span class="metric-label">Avg Scan Duration</span>
                    <span class="metric-value">{{avg_scan_duration}}s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Scans</span>
                    <span class="metric-value">{{total_scans}}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Alerts</span>
                    <span class="metric-value">{{total_alerts}}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Active Alerts</span>
                    <span class="metric-value" style="color: #ff5252">{{active_alerts}}</span>
                </div>
            </div>
        </div>
        
        <div class="grid">
            <!-- Active Alerts Card -->
            <div class="card" style="grid-column: span 2;">
                <h3>Active Alerts ({{active_alerts_count}})</h3>
                <div id="alerts-list">
                    {{alerts_html}}
                </div>
            </div>
            
            <!-- Recent Triggers Card -->
            <div class="card" style="grid-column: span 1;">
                <h3>Recent Triggers</h3>
                <div id="triggers-list">
                    {{triggers_html}}
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>Pattern Status</h3>
            <div id="patterns-list">
                {{patterns_html}}
            </div>
        </div>
        
        <div class="footer">
            <p>Real-Time Pattern Monitoring System | Agent_Visualization | Task 5.3</p>
        </div>
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function() {
            location.reload();
        }, 30000);
    </script>
</body>
</html>
"""
    
    # Prepare template variables
    status_class = "status-running" if dashboard_data.get('system_status', {}).get('is_running') else "status-stopped"
    status = "RUNNING" if dashboard_data.get('system_status', {}).get('is_running') else "STOPPED"
    
    # Generate alerts HTML
    alerts = dashboard_data.get('active_alerts', [])
    alerts_html = ""
    if alerts:
        for alert in alerts[:10]:  # Show last 10 alerts
            alerts_html += f"""
            <div class="alert-item alert-{alert['severity']}">
                <strong>{alert['alert_type'].replace('_', ' ').title()}</strong> - {alert['message']}
                <br><small>{alert['timestamp']}</small>
            </div>
            """
    else:
        alerts_html = "<p style='color: #00c853'>No active alerts</p>"
    
    # Generate triggers HTML
    triggers = dashboard_data.get('recent_triggers', [])
    triggers_html = ""
    if triggers:
        for trigger in triggers[:10]:
            triggers_html += f"""
            <div class="pattern-item">
                <strong>Pattern {trigger['pattern_id']}</strong>: {trigger['pattern_name']}
                <br>Confidence: {trigger['confidence_score']:.1%}
                <br>Price: ${trigger['current_price']:.2f} → ${trigger['target_price']:.2f}
            </div>
            """
    else:
        triggers_html = "<p>No recent triggers</p>"
    
    # Generate patterns HTML
    patterns = dashboard_data.get('pattern_status', [])
    max_patterns = self.config.max_patterns_display if hasattr(self.config, 'max_patterns_display') else 50
    patterns_html = ""
    for pattern in patterns[:max_patterns]:  # Show limited patterns
        status_class = (
            "pattern-triggered" if pattern['is_triggered'] else
            "pattern-almost" if pattern['status'] == 'ALMOST_TRIGGERED' else
            "pattern-not-triggered"
        )
        confidence_class = (
            "confidence-high" if pattern['confidence_score'] >= 0.9 else
            "confidence-medium" if pattern['confidence_score'] >= 0.7 else
            "confidence-low"
        )
        
        patterns_html += f"""
        <div class="pattern-item {status_class}">
            <strong>Pattern {pattern['pattern_id']}</strong>: {pattern['label']}
            <br>Status: {pattern['status']}
            <br>Conditions: {pattern['conditions_met']}/{pattern['total_conditions']}
            <br>Success Rate: {pattern['validation_success_rate']:.1f}%
            <div class="confidence-bar">
                <div class="confidence-fill {confidence_class}" style="width: {pattern['confidence_score'] * 100}%"></div>
            </div>
        </div>
        """
    
    # Fill template
    html = html_template.replace('{{timestamp}}', dashboard_data.get('timestamp', 'N/A'))
    html = html.replace('{{status_class}}', status_class)
    html = html.replace('{{status}}', status)
    html = html.replace('{{uptime}}', dashboard_data.get('system_status', {}).get('uptime', 'N/A'))
    html = html.replace('{{total_patterns}}', str(dashboard_data.get('summary', {}).get('total_patterns', 0)))
    html = html.replace('{{triggered}}', str(dashboard_data.get('summary', {}).get('triggered_patterns', 0)))
    html = html.replace('{{almost_triggered}}', str(dashboard_data.get('summary', {}).get('almost_triggered', 0)))
    html = html.replace('{{not_triggered}}', str(dashboard_data.get('summary', {}).get('not_triggered', 0)))
    html = html.replace('{{market_date}}', dashboard_data.get('market_state', {}).get('date', 'N/A'))
    html = html.replace('{{close_price}}', f"{dashboard_data.get('market_state', {}).get('close', 0):.2f}")
    html = html.replace('{{volume}}', f"{dashboard_data.get('market_state', {}).get('volume', 0):,.0f}")
    html = html.replace('{{rsi}}', f"{dashboard_data.get('market_state', {}).get('rsi', 0):.2f}")
    html = html.replace('{{trend_regime}}', dashboard_data.get('market_state', {}).get('trend_regime', 'N/A'))
    html = html.replace('{{avg_scan_duration}}', f"{dashboard_data.get('performance_metrics', {}).get('avg_scan_duration', 0):.3f}")
    html = html.replace('{{total_scans}}', str(dashboard_data.get('performance_metrics', {}).get('total_scans', 0)))
    html = html.replace('{{total_alerts}}', str(dashboard_data.get('performance_metrics', {}).get('total_alerts', 0)))
    html = html.replace('{{active_alerts}}', str(dashboard_data.get('performance_metrics', {}).get('active_alerts_count', 0)))
    html = html.replace('{{active_alerts_count}}', str(len(alerts)))
    html = html.replace('{{alerts_html}}', alerts_html)
    html = html.replace('{{triggers_html}}', triggers_html)
    html = html.replace('{{patterns_html}}', patterns_html)
    
    # Save HTML
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"HTML dashboard saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Real-Time Pattern Monitoring System")
    print("=" * 60)
    print("\nThis system provides:")
    print("  - Real-time pattern scanning and detection")
    print("  - Alert generation for high-probability patterns")
    print("  - Performance monitoring and metrics")
    print("  - Interactive monitoring dashboard")
    print("\nSee docs/monitoring_guide.md for detailed usage examples.")