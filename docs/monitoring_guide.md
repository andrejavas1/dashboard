# Real-Time Pattern Monitoring System Guide

**Author:** Agent_Visualization  
**Task:** Task 5.3 - Real-Time Monitoring System  
**Version:** 1.0.0

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Core Components](#core-components)
6. [Alert System](#alert-system)
7. [Monitoring Dashboard](#monitoring-dashboard)
8. [API Reference](#api-reference)
9. [Advanced Usage](#advanced-usage)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

---

## Overview

The Real-Time Pattern Monitoring System provides comprehensive monitoring capabilities for pattern-based trading strategies. It continuously scans market data for pattern triggers, generates alerts for high-probability opportunities, and tracks system performance metrics.

### Key Features

- **Real-Time Pattern Scanning**: Continuous monitoring of pattern conditions
- **Intelligent Alerting**: Multi-severity alerts with cooldown and rate limiting
- **Performance Monitoring**: Track system metrics and scan performance
- **Interactive Dashboard**: Web-based visualization of monitoring data
- **Multi-Channel Notifications**: Log, database, email, SMS, and webhook support
- **Configurable Thresholds**: Customize trigger and alert thresholds

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Real-Time Monitor                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Data Loader │  │  Pattern     │  │  Alert       │      │
│  │              │  │  Scanner     │  │  Manager     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                 │                 │               │
│         ▼                 ▼                 ▼               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Performance │  │  Dashboard   │  │  Notification│      │
│  │  Monitor     │  │  Generator   │  │  Channels    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Required packages (see `requirements.txt`)

### Installation Steps

1. **Clone or navigate to the project directory:**
   ```bash
   cd /path/to/project
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "from src.real_time_monitor import RealTimeMonitor; print('OK')"
   ```

### Required Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
pyyaml>=5.4.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

---

## Quick Start

### Basic Usage

```python
from src.real_time_monitor import RealTimeMonitor

# Initialize monitor
monitor = RealTimeMonitor()

# Load data and portfolio
monitor.load_data("data/features_matrix.csv")
monitor.load_portfolio("data/final_portfolio.json")

# Run a single scan
triggers = monitor.run_single_scan()
print(f"Found {len(triggers)} pattern triggers")

# Get dashboard data
dashboard = monitor.create_dashboard_data()
print(f"Triggered patterns: {dashboard['summary']['triggered_patterns']}")
```

### Continuous Monitoring

```python
from src.real_time_monitor import RealTimeMonitor

# Initialize monitor
monitor = RealTimeMonitor()

# Load data and portfolio
monitor.load_data("data/features_matrix.csv")
monitor.load_portfolio("data/final_portfolio.json")

# Start continuous monitoring
monitor.start_monitoring()

# Monitor runs in background...
# Check status
status = monitor.get_status()
print(f"Status: {status}")

# Stop monitoring when done
monitor.stop_monitoring()
```

### Create HTML Dashboard

```python
from src.real_time_monitor import RealTimeMonitor, create_monitoring_dashboard_html

# Initialize and run scan
monitor = RealTimeMonitor()
monitor.load_data("data/features_matrix.csv")
monitor.load_portfolio("data/final_portfolio.json")
monitor.run_single_scan()

# Create dashboard
dashboard = monitor.create_dashboard_data()
create_monitoring_dashboard_html(
    dashboard,
    "dashboard/monitoring_dashboard.html"
)
print("Dashboard created at dashboard/monitoring_dashboard.html")
```

---

## Configuration

### Configuration File

Create a YAML configuration file (`monitoring_config.yaml`):

```yaml
# Data sources
data_path: "data/features_matrix.csv"
portfolio_path: "data/final_portfolio.json"

# Monitoring settings
scan_interval: 60  # seconds between scans
almost_triggered_threshold: 0.9  # 90% of conditions met
high_probability_threshold: 0.85  # 85% historical success rate
critical_threshold: 0.95  # 95% historical success rate

# Alert settings
enable_alerts: true
alert_cooldown: 300  # seconds between same alert
max_alerts_per_hour: 100

# Notification channels
notification_channels:
  - log
  - database
  # - email  # Uncomment to enable
  # - sms    # Uncomment to enable
  # - webhook  # Uncomment to enable

# Storage
alerts_dir: "reports/alerts"
dashboard_dir: "reports/dashboard"
metrics_dir: "reports/metrics"

# Performance monitoring
enable_performance_monitoring: true
metrics_retention_hours: 24

# Visualization integration
enable_visualization: true
charts_dir: "charts/monitoring"
```

### Loading Configuration

```python
from src.real_time_monitor import RealTimeMonitor, MonitoringConfig

# Method 1: Load from YAML file
monitor = RealTimeMonitor()
monitor.load_config("monitoring_config.yaml")

# Method 2: Create config programmatically
config = MonitoringConfig(
    scan_interval=30,
    almost_triggered_threshold=0.85,
    enable_alerts=True
)
monitor = RealTimeMonitor(config=config)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | str | `"data/features_matrix.csv"` | Path to market data file |
| `portfolio_path` | str | `"data/final_portfolio.json"` | Path to pattern portfolio |
| `scan_interval` | int | `60` | Seconds between scans |
| `almost_triggered_threshold` | float | `0.9` | Confidence threshold for "almost triggered" |
| `high_probability_threshold` | float | `0.85` | Success rate for HIGH severity alerts |
| `critical_threshold` | float | `0.95` | Success rate for CRITICAL severity alerts |
| `enable_alerts` | bool | `true` | Enable alert generation |
| `alert_cooldown` | int | `300` | Seconds between identical alerts |
| `max_alerts_per_hour` | int | `100` | Maximum alerts per hour |
| `notification_channels` | list | `["log", "database"]` | Notification channels |
| `alerts_dir` | str | `"reports/alerts"` | Directory for alert storage |
| `dashboard_dir` | str | `"reports/dashboard"` | Directory for dashboard data |
| `metrics_dir` | str | `"reports/metrics"` | Directory for metrics storage |
| `enable_performance_monitoring` | bool | `true` | Enable performance tracking |
| `metrics_retention_hours` | int | `24` | Hours to retain metrics |

---

## Core Components

### RealTimeMonitor

The main monitoring system class.

#### Initialization

```python
from src.real_time_monitor import RealTimeMonitor

# With default configuration
monitor = RealTimeMonitor()

# With custom configuration
from src.real_time_monitor import MonitoringConfig
config = MonitoringConfig(scan_interval=30)
monitor = RealTimeMonitor(config=config)
```

#### Methods

##### `load_data(data_path: Optional[str] = None)`

Load market data from CSV file.

```python
monitor.load_data("data/features_matrix.csv")
```

##### `load_portfolio(portfolio_path: Optional[str] = None)`

Load pattern portfolio from JSON file.

```python
monitor.load_portfolio("data/final_portfolio.json")
```

##### `run_single_scan() -> List[PatternTrigger]`

Run a single pattern scan.

```python
triggers = monitor.run_single_scan()
for trigger in triggers:
    print(f"Pattern {trigger.pattern_id}: {trigger.confidence_score:.1%}")
```

##### `start_monitoring()`

Start continuous monitoring in background thread.

```python
monitor.start_monitoring()
```

##### `stop_monitoring()`

Stop continuous monitoring.

```python
monitor.stop_monitoring()
```

##### `create_dashboard_data() -> Dict`

Create dashboard data for visualization.

```python
dashboard = monitor.create_dashboard_data()
```

##### `get_status() -> Dict`

Get current monitoring status.

```python
status = monitor.get_status()
print(f"Running: {status['is_running']}")
print(f"Uptime: {status['uptime']}")
```

### AlertManager

Manages alert generation and routing.

#### Creating Alerts

```python
from src.real_time_monitor import AlertManager, AlertType, AlertSeverity, MonitoringConfig

config = MonitoringConfig()
alert_manager = AlertManager(config)

# Create an alert
alert = alert_manager.create_alert(
    alert_type=AlertType.PATTERN_TRIGGERED,
    severity=AlertSeverity.HIGH,
    pattern_id=0,
    pattern_name="Pattern 0",
    message="Pattern triggered with high confidence",
    details={"confidence": 0.95}
)

# Send alert
alert_manager.send_alert(alert)
```

#### Managing Alerts

```python
# Get active alerts
active = alert_manager.get_active_alerts()

# Get alerts by severity
high_alerts = alert_manager.get_alerts_by_severity(AlertSeverity.HIGH)

# Acknowledge alert
alert_manager.acknowledge_alert(alert.alert_id, notes="Investigating")

# Resolve alert
alert_manager.resolve_alert(alert.alert_id, notes="Resolved - false positive")
```

### PerformanceMonitor

Tracks system performance metrics.

#### Recording Metrics

```python
from src.real_time_monitor import PerformanceMonitor, MonitoringConfig

config = MonitoringConfig()
perf_monitor = PerformanceMonitor(config)

# Record metrics
metrics = perf_monitor.record_metrics(
    scan_duration=0.5,
    patterns_scanned=50,
    patterns_triggered=3,
    alerts_generated=3,
    data_points_processed=1000
)
```

#### Querying Metrics

```python
# Get average scan duration
avg_duration = perf_monitor.get_average_scan_duration()

# Get system uptime
uptime = perf_monitor.get_uptime()

# Get recent metrics
recent = perf_monitor.get_recent_metrics(hours=1)
```

---

## Alert System

### Alert Types

| Type | Description |
|------|-------------|
| `PATTERN_TRIGGERED` | Pattern conditions fully met |
| `PATTERN_ALMOST_TRIGGERED` | Pattern conditions almost met |
| `PERFORMANCE_DEGRADATION` | System performance issues |
| `SYSTEM_ERROR` | System errors or exceptions |
| `DATA_ANOMALY` | Unusual data patterns |
| `THRESHOLD_EXCEEDED` | Metric threshold exceeded |

### Alert Severity Levels

| Severity | Description | Use Case |
|----------|-------------|----------|
| `LOW` | Informational | Almost triggered patterns |
| `MEDIUM` | Warning | Moderate probability patterns |
| `HIGH` | Important | High probability patterns |
| `CRITICAL` | Urgent | Very high probability patterns |

### Alert Lifecycle

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Created   │───▶│ Acknowledge │───▶│  Resolved   │
└─────────────┘    └─────────────┘    └─────────────┘
       │                                    │
       ▼                                    ▼
┌─────────────┐                    ┌─────────────┐
│    Sent     │                    │   Archived  │
└─────────────┘                    └─────────────┘
```

### Notification Channels

#### Log Channel

Alerts are written to the system log.

```python
config = MonitoringConfig(notification_channels=["log"])
```

#### Database Channel

Alerts are saved to JSON files in the alerts directory.

```python
config = MonitoringConfig(notification_channels=["database"])
```

Alerts are saved to: `reports/alerts/alert_YYYYMMDD.json`

#### Email Channel (Placeholder)

```python
config = MonitoringConfig(notification_channels=["email"])
```

*Note: Email integration requires additional configuration.*

#### Webhook Channel (Placeholder)

```python
config = MonitoringConfig(notification_channels=["webhook"])
```

*Note: Webhook integration requires additional configuration.*

### Alert Cooldown and Rate Limiting

The system implements two protection mechanisms:

1. **Alert Cooldown**: Prevents duplicate alerts within a specified time window
2. **Hourly Rate Limit**: Caps the number of alerts per hour

```python
config = MonitoringConfig(
    alert_cooldown=300,  # 5 minutes
    max_alerts_per_hour=100
)
```

---

## Monitoring Dashboard

### Dashboard Components

The monitoring dashboard provides:

1. **System Status**: Running state, uptime, last scan time
2. **Summary Statistics**: Total patterns, triggered, almost triggered
3. **Market State**: Current price, volume, RSI, trend regime
4. **Performance Metrics**: Scan duration, total scans, alerts
5. **Active Alerts**: List of unresolved alerts
6. **Recent Triggers**: Latest pattern triggers
7. **Pattern Status**: Status of all monitored patterns

### Creating the Dashboard

```python
from src.real_time_monitor import RealTimeMonitor, create_monitoring_dashboard_html

# Initialize and scan
monitor = RealTimeMonitor()
monitor.load_data("data/features_matrix.csv")
monitor.load_portfolio("data/final_portfolio.json")
monitor.run_single_scan()

# Generate dashboard
dashboard = monitor.create_dashboard_data()

# Create HTML file
create_monitoring_dashboard_html(
    dashboard,
    "dashboard/monitoring_dashboard.html"
)
```

### Dashboard Features

- **Auto-refresh**: Dashboard refreshes every 30 seconds
- **Color-coded status**: Visual indicators for pattern states
- **Confidence bars**: Visual representation of pattern confidence
- **Responsive design**: Works on desktop and mobile devices
- **Real-time updates**: Shows latest monitoring data

### Dashboard Data Structure

```python
{
    "timestamp": "2024-01-23T06:00:00",
    "system_status": {
        "is_running": True,
        "uptime": "1:23:45",
        "last_scan": "2024-01-23T06:00:00",
        "scan_interval": 60
    },
    "market_state": {
        "date": "2024-01-23",
        "close": 95.50,
        "volume": 15000000,
        "rsi": 65.5,
        "atr_pct": 1.2,
        "trend_regime": "UPTREND",
        "volatility_regime": "NORMAL"
    },
    "pattern_status": [...],
    "recent_triggers": [...],
    "active_alerts": [...],
    "performance_metrics": {...},
    "summary": {...}
}
```

---

## API Reference

### Data Classes

#### PatternTrigger

```python
@dataclass
class PatternTrigger:
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
```

#### Alert

```python
@dataclass
class Alert:
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
```

#### SystemMetrics

```python
@dataclass
class SystemMetrics:
    timestamp: str
    cpu_usage: float
    memory_usage: float
    scan_duration: float
    patterns_scanned: int
    patterns_triggered: int
    alerts_generated: int
    data_points_processed: int
    last_data_update: str
```

### Enums

#### AlertSeverity

```python
class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

#### AlertType

```python
class AlertType(Enum):
    PATTERN_TRIGGERED = "pattern_triggered"
    PATTERN_ALMOST_TRIGGERED = "pattern_almost_triggered"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SYSTEM_ERROR = "system_error"
    DATA_ANOMALY = "data_anomaly"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
```

#### NotificationChannel

```python
class NotificationChannel(Enum):
    LOG = "log"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    DATABASE = "database"
```

---

## Advanced Usage

### Custom Alert Handlers

```python
from src.real_time_monitor import AlertManager, Alert, AlertSeverity

class CustomAlertManager(AlertManager):
    def _send_webhook_alert(self, alert: Alert):
        """Custom webhook implementation."""
        import requests
        
        webhook_url = "https://your-webhook-url.com/alerts"
        payload = {
            "alert_id": alert.alert_id,
            "severity": alert.severity.value,
            "message": alert.message,
            "details": alert.details
        }
        
        response = requests.post(webhook_url, json=payload)
        logger.info(f"Webhook response: {response.status_code}")
```

### Custom Pattern Scanning

```python
from src.real_time_monitor import RealTimeMonitor

class CustomMonitor(RealTimeMonitor):
    def scan_patterns(self):
        """Custom scanning logic."""
        # Run default scan
        triggers = super().scan_patterns()
        
        # Add custom processing
        for trigger in triggers:
            if trigger.confidence_score > 0.95:
                # Special handling for high-confidence triggers
                self._handle_high_confidence_trigger(trigger)
        
        return triggers
    
    def _handle_high_confidence_trigger(self, trigger):
        """Handle high-confidence triggers specially."""
        logger.info(f"High confidence trigger: {trigger.pattern_id}")
        # Add custom logic here
```

### Integration with Pattern Visualization

```python
from src.real_time_monitor import RealTimeMonitor
from src.pattern_visualization import PatternVisualization

# Initialize both systems
monitor = RealTimeMonitor()
viz = PatternVisualization()

# Load data
data = monitor.load_data("data/features_matrix.csv")
viz.data = data

# Load patterns
portfolio = monitor.load_portfolio("data/final_portfolio.json")
viz.patterns = portfolio

# Run scan
triggers = monitor.run_single_scan()

# Visualize triggered patterns
for trigger in triggers:
    pattern = portfolio[trigger.pattern_id]
    viz.create_overview_chart(pattern, trigger.pattern_id)
    viz.create_occurrences_chart(pattern, trigger.pattern_id)
```

### Scheduled Monitoring

```python
import schedule
import time
from src.real_time_monitor import RealTimeMonitor

monitor = RealTimeMonitor()
monitor.load_data("data/features_matrix.csv")
monitor.load_portfolio("data/final_portfolio.json")

def run_scheduled_scan():
    """Run scheduled scan."""
    print(f"Running scan at {datetime.now()}")
    triggers = monitor.run_single_scan()
    print(f"Found {len(triggers)} triggers")

# Schedule scans every 5 minutes
schedule.every(5).minutes.do(run_scheduled_scan)

# Run scheduler
while True:
    schedule.run_pending()
    time.sleep(1)
```

### Multi-Ticker Monitoring

```python
from src.real_time_monitor import RealTimeMonitor

# Create monitors for different tickers
monitors = {
    "XOM": RealTimeMonitor(config=MonitoringConfig(
        data_path="data/XOM_features.csv",
        portfolio_path="data/XOM_portfolio.json"
    )),
    "AAPL": RealTimeMonitor(config=MonitoringConfig(
        data_path="data/AAPL_features.csv",
        portfolio_path="data/AAPL_portfolio.json"
    ))
}

# Load data for all monitors
for ticker, monitor in monitors.items():
    monitor.load_data()
    monitor.load_portfolio()

# Run scans
for ticker, monitor in monitors.items():
    triggers = monitor.run_single_scan()
    print(f"{ticker}: {len(triggers)} triggers")
```

---

## Troubleshooting

### Common Issues

#### Issue: No patterns triggered

**Symptoms:** Scan runs but no triggers are found.

**Possible Causes:**
1. Pattern conditions are too strict
2. Data doesn't contain required features
3. Thresholds are set too high

**Solutions:**
```python
# Check pattern conditions
pattern = monitor.portfolio[0]
print("Pattern conditions:", pattern['pattern']['conditions'])

# Check data features
print("Available features:", monitor.data.columns.tolist())

# Lower thresholds
config = MonitoringConfig(
    almost_triggered_threshold=0.8,
    high_probability_threshold=0.75
)
monitor = RealTimeMonitor(config=config)
```

#### Issue: Too many alerts

**Symptoms:** Receiving excessive alerts.

**Possible Causes:**
1. Alert cooldown too short
2. Thresholds too low
3. Pattern conditions too loose

**Solutions:**
```python
# Increase cooldown
config = MonitoringConfig(
    alert_cooldown=600,  # 10 minutes
    max_alerts_per_hour=50
)

# Increase thresholds
config.almost_triggered_threshold = 0.95
config.high_probability_threshold = 0.90
```

#### Issue: Slow scan performance

**Symptoms:** Scans take too long to complete.

**Possible Causes:**
1. Too many patterns in portfolio
2. Large dataset
3. Complex pattern conditions

**Solutions:**
```python
# Increase scan interval
config = MonitoringConfig(scan_interval=120)

# Reduce portfolio size
# Filter patterns to only high-quality ones
high_quality_patterns = [
    p for p in portfolio
    if p.get('validation_success_rate', 0) > 80
]
monitor.portfolio = high_quality_patterns

# Monitor performance
status = monitor.get_status()
print(f"Avg scan duration: {status['avg_scan_duration']}s")
```

#### Issue: Data loading errors

**Symptoms:** Cannot load data or portfolio files.

**Possible Causes:**
1. File path incorrect
2. File format incorrect
3. Missing required columns

**Solutions:**
```python
# Check file exists
import os
print("Data file exists:", os.path.exists("data/features_matrix.csv"))
print("Portfolio exists:", os.path.exists("data/final_portfolio.json"))

# Verify data format
import pandas as pd
data = pd.read_csv("data/features_matrix.csv")
print("Data shape:", data.shape)
print("Columns:", data.columns.tolist())

# Verify portfolio format
import json
with open("data/final_portfolio.json") as f:
    portfolio = json.load(f)
print("Portfolio size:", len(portfolio))
```

### Debug Mode

Enable debug logging:

```python
import logging

# Set debug level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Run monitor
monitor = RealTimeMonitor()
monitor.load_data()
monitor.load_portfolio()
triggers = monitor.run_single_scan()
```

---

## Best Practices

### 1. Configuration Management

- Use YAML configuration files for production deployments
- Keep sensitive information (API keys, passwords) in environment variables
- Version control configuration files
- Document custom configuration values

```python
import os
from dotenv import load_dotenv

load_dotenv()

config = MonitoringConfig(
    notification_channels=["log", "webhook"],
    webhook_url=os.getenv("WEBHOOK_URL")
)
```

### 2. Error Handling

- Implement comprehensive error handling
- Log all errors with context
- Gracefully handle missing data
- Implement retry logic for transient failures

```python
try:
    monitor.load_data("data/features_matrix.csv")
except FileNotFoundError:
    logger.error("Data file not found")
    # Fallback to backup data
    monitor.load_data("data/features_backup.csv")
except Exception as e:
    logger.error(f"Error loading data: {e}")
    raise
```

### 3. Performance Optimization

- Monitor scan duration regularly
- Optimize pattern conditions
- Use caching for expensive operations
- Consider parallel processing for large portfolios

```python
import time

start = time.time()
triggers = monitor.run_single_scan()
duration = time.time() - start

if duration > 5.0:
    logger.warning(f"Slow scan: {duration:.2f}s")
```

### 4. Alert Management

- Set appropriate cooldown periods
- Implement alert escalation
- Regularly review and resolve alerts
- Use severity levels appropriately

```python
# Implement alert escalation
if alert.severity == AlertSeverity.CRITICAL:
    # Send to multiple channels
    alert_manager.send_alert(alert)
    # Also send SMS
    send_sms_alert(alert)
```

### 5. Data Quality

- Validate data before processing
- Handle missing values appropriately
- Monitor data freshness
- Implement data quality checks

```python
def validate_data(data: pd.DataFrame) -> bool:
    """Validate data quality."""
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Check required columns
    if not all(col in data.columns for col in required_columns):
        logger.error("Missing required columns")
        return False
    
    # Check for missing values
    if data.isnull().any().any():
        logger.warning("Data contains missing values")
    
    # Check data freshness
    last_date = data.index[-1]
    if (datetime.now() - last_date).days > 1:
        logger.warning("Data is stale")
    
    return True
```

### 6. Monitoring and Maintenance

- Regularly review system logs
- Monitor system resources
- Schedule regular data updates
- Implement health checks

```python
def health_check(monitor: RealTimeMonitor) -> Dict:
    """Perform system health check."""
    status = monitor.get_status()
    
    health = {
        "status": "healthy",
        "checks": {}
    }
    
    # Check if running
    health["checks"]["running"] = status["is_running"]
    
    # Check uptime
    uptime_hours = status["uptime"].total_seconds() / 3600
    health["checks"]["uptime"] = uptime_hours > 0
    
    # Check recent alerts
    if status["active_alerts"] > 10:
        health["status"] = "warning"
    
    return health
```

### 7. Security Considerations

- Secure sensitive configuration
- Implement authentication for dashboard
- Use HTTPS for webhooks
- Validate all external inputs

```python
import hashlib

def hash_sensitive_data(data: str) -> str:
    """Hash sensitive data for logging."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]
```

---

## Appendix

### File Structure

```
project/
├── src/
│   └── real_time_monitor.py      # Main monitoring module
├── data/
│   ├── features_matrix.csv       # Market data
│   └── final_portfolio.json      # Pattern portfolio
├── reports/
│   ├── alerts/                   # Alert storage
│   │   └── alert_YYYYMMDD.json
│   ├── dashboard/                # Dashboard data
│   │   └── monitoring_dashboard.json
│   └── metrics/                  # Metrics storage
│       └── metrics_YYYYMMDD.json
├── dashboard/
│   └── monitoring_dashboard.html # HTML dashboard
├── docs/
│   └── monitoring_guide.md       # This guide
└── monitoring_config.yaml        # Configuration file
```

### Example Configuration File

```yaml
# monitoring_config.yaml
data_path: "data/features_matrix.csv"
portfolio_path: "data/final_portfolio.json"

scan_interval: 60
almost_triggered_threshold: 0.9
high_probability_threshold: 0.85
critical_threshold: 0.95

enable_alerts: true
alert_cooldown: 300
max_alerts_per_hour: 100

notification_channels:
  - log
  - database

alerts_dir: "reports/alerts"
dashboard_dir: "reports/dashboard"
metrics_dir: "reports/metrics"

enable_performance_monitoring: true
metrics_retention_hours: 24

enable_visualization: true
charts_dir: "charts/monitoring"
```

### Glossary

- **Pattern Trigger**: When all pattern conditions are met
- **Almost Triggered**: When most (but not all) conditions are met
- **Confidence Score**: Percentage of conditions met (0.0 to 1.0)
- **Alert Cooldown**: Minimum time between identical alerts
- **Rate Limiting**: Maximum alerts per time period
- **Dashboard**: Web-based visualization interface
- **Metrics**: System performance measurements
- **Notification Channel**: Method for delivering alerts

---

## Support

For issues, questions, or contributions:

1. Review this documentation
2. Check the troubleshooting section
3. Review system logs
4. Contact the development team

---

**Document Version:** 1.0.0  
**Last Updated:** 2024-01-23  
**Author:** Agent_Visualization  
**Task:** Task 5.3 - Real-Time Monitoring System