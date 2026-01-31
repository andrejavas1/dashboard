# Price Movement Probability Discovery System

A comprehensive system for analyzing historical price data of XOM stock to discover specific market conditions that consistently precede significant price movements.

## Features

- **Multi-Source Data Acquisition**: Collects OHLCV data from Yahoo Finance, Alpha Vantage, Tiingo, EODHD, and Quandl
- **Advanced Pattern Discovery**: ML-based and rule-based pattern discovery with >80% success rate target
- **Comprehensive Validation**: Cross-validation, false positive reduction, and performance testing
- **Real-Time Monitoring**: Live pattern detection with alerting system
- **Interactive Visualization**: Dashboards and charts for pattern analysis

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)

The system works with Yahoo Finance (free, no API key required). To enable additional data sources:

1. Copy the `.env.template` file:
```bash
copy .env.template .env
```

2. Edit `.env` and add your API keys:
```
ALPHA_VANTAGE_API_KEY=your_key_here
TIINGO_API_KEY=your_key_here
EODHD_API_KEY=your_key_here
QUANDL_API_KEY=your_key_here
```

### 3. Run the Pipeline

```bash
python main.py
```

To run a specific phase:
```bash
python main.py --phase 1
```

## Project Structure

```
.
├── src/                    # Source code modules
│   ├── data_acquisition.py      # Phase 1: Data acquisition
│   ├── phase2_movement_labeling.py  # Phase 2: Movement labeling
│   ├── phase3_feature_engineering.py  # Phase 3: Feature engineering
│   ├── phase4_pattern_discovery.py    # Phase 4: Pattern discovery
│   ├── phase5_pattern_optimization.py # Phase 5: Pattern optimization
│   ├── phase6_validation.py          # Phase 6: Validation
│   ├── phase7_portfolio_construction.py  # Phase 7: Portfolio construction
│   ├── phase8_visualization.py       # Phase 8: Visualization
│   ├── phase9_realtime_detection.py  # Phase 9: Real-time detection
│   ├── phase10_final_report.py      # Phase 10: Final report
│   ├── integrated_system.py         # Main integrated system
│   ├── ml_pattern_discovery.py      # ML pattern discovery
│   ├── enhanced_rule_based_patterns.py  # Rule-based patterns
│   ├── pattern_validation_framework.py   # Validation framework
│   ├── cross_validation_framework.py     # Cross-validation
│   ├── performance_optimizer.py          # Performance optimization
│   ├── real_time_monitor.py              # Real-time monitoring
│   ├── pattern_visualization.py          # Visualization
│   └── technical_indicators.py           # Technical indicators
├── docs/                   # Documentation
├── data/                   # Data files
├── reports/                # Reports and analysis
├── charts/                 # Visualizations
├── tests/                  # Unit tests
├── .apm/                   # APM project management
├── config.yaml             # Configuration file
├── main.py                 # Main entry point
└── requirements.txt        # Python dependencies
```

## Documentation

- [Integration Guide](docs/integration_guide.md)
- [Performance Guide](docs/performance_guide.md)
- [Codebase Guide](docs/codebase_guide.md)
- [ML Pattern Discovery](docs/ml_pattern_discovery.md)
- [Rule-Based Patterns](docs/rule_based_patterns.md)
- [Validation Framework](docs/validation_framework.md)
- [Cross-Validation](docs/cross_validation.md)
- [Visualization Guide](docs/visualization_guide.md)
- [Monitoring Guide](docs/monitoring_guide.md)
- [Feature Engineering](docs/feature_engineering.md)
- [Technical Indicators](docs/technical_indicators.md)

## API Key Sources

| Service | Free Tier | Sign Up |
|---------|-----------|---------|
| Alpha Vantage | 25 requests/day | https://www.alphavantage.co/support/#api-key |
| Tiingo | 500 requests/day | https://api.tiingo.com/ |
| EODHD | 20 requests/day | https://eodhistoricaldata.com/ |
| Quandl | Limited | https://www.quandl.com/ |

## Configuration

Edit `config.yaml` to customize:
- Data sources and tickers
- Date ranges
- Validation thresholds
- Pattern discovery parameters
- Portfolio construction settings

## Output

- **Data**: `data/` directory contains verified OHLCV data
- **Reports**: `reports/` directory contains analysis reports
- **Charts**: `charts/` directory contains visualizations
- **Logs**: `pipeline.log` contains execution logs

## Notes

- The system uses Yahoo Finance as the primary data source (no API key required)
- Additional data sources require API keys configured in `.env`
- Phase 4 (Pattern Discovery) is the most computationally intensive phase
- Monitor progress in `pipeline.log` during execution

## License

MIT License