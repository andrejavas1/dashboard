# Data Acquisition Pipeline Documentation

## Overview

The enhanced data acquisition pipeline is designed to collect, validate, and process financial market data from multiple sources to ensure high-quality, reliable data for pattern discovery and analysis. This document provides detailed information about the pipeline's architecture, features, and usage.

## Key Features

### Enhanced Error Handling
- **Retry Mechanisms**: Automatic retry with exponential backoff for failed API calls
- **Circuit Breaker Pattern**: Prevents cascading failures by temporarily disabling problematic data sources
- **Graceful Degradation**: Continues operation even when some data sources are unavailable

### Data Quality Assurance
- **Cross-Source Validation**: Compares data from multiple sources to identify discrepancies
- **Consensus-Based Verification**: Uses median values when multiple sources agree within thresholds
- **Outlier Detection**: Identifies and flags anomalous price movements
- **Gap Analysis**: Detects missing data periods and reports them

### Multiple Data Sources
- **Primary Sources**: Yahoo Finance, Alpha Vantage, Tiingo, EODHD
- **Alternative Data**: Quandl datasets

### Caching Mechanism
- **Local Caching**: Stores previously collected data to reduce API calls
- **Cache Invalidation**: Automatically refreshes stale data based on date ranges

## Architecture

### DataAcquisition Class

The main class responsible for orchestrating the data acquisition process.

#### Key Methods

- `collect_all_sources()`: Collects data from all available sources
- `verify_data_quality()`: Validates and cross-checks data from multiple sources
- `save_verified_data()`: Saves the final verified dataset and validation report

#### Configuration

The pipeline is configured through `config.yaml` with the following sections:

```yaml
data_sources:
  ticker: "XOM"
  start_date: "2010-01-01"
  end_date: "current"
  # API keys for various data sources
  alpha_vantage:
    api_key: ""
  tiingo:
    api_key: ""
  eodhd:
    api_key: ""
  quandl:
    api_key: ""

validation:
  ohlc_discrepancy_threshold: 0.5  # % difference to flag
  consensus_agreement_threshold: 0.2  # % for consensus
  volume_discrepancy_threshold: 10  # % difference to flag
  max_consecutive_missing_days: 3
  min_confidence_score: 98  # % to proceed
```

### Circuit Breaker Pattern

Each data source is protected by a circuit breaker that:
1. Tracks consecutive failures
2. Opens the circuit after a threshold is reached
3. Automatically attempts recovery after a timeout period

### Retry Mechanism

API calls are automatically retried with exponential backoff:
- Initial delay: 1 second
- Backoff multiplier: 2x
- Maximum retries: 3 attempts

## Technical Indicators

The feature engineering module includes enhanced technical indicators:

### New Indicators Added
- **Williams %R**: Momentum indicator measuring overbought/oversold levels
- **CCI (Commodity Channel Index)**: Measures deviation from typical price
- **Chaikin Oscillator**: Measures momentum of the Accumulation Distribution Line

### Existing Indicators
- **Moving Averages**: Multiple periods (10, 20, 50, 100, 200)
- **RSI**: Relative Strength Index (7, 14, 21, 28 periods)
- **MACD**: Moving Average Convergence Divergence
- **Stochastic Oscillator**: Momentum indicator
- **ATR**: Average True Range for volatility
- **Bollinger Bands**: Volatility bands around moving average

## Data Validation Process

1. **Collection**: Gather data from all available sources
2. **Standardization**: Normalize column names and data formats
3. **Cross-Verification**: Compare values across sources
4. **Consensus Building**: Use median values when sources agree
5. **Discrepancy Reporting**: Flag significant differences for review
6. **Quality Scoring**: Calculate confidence score based on data completeness
7. **Outlier Detection**: Identify anomalous price movements
8. **Gap Analysis**: Detect and report missing data periods

## Usage Examples

### Basic Usage

```python
from src.data_acquisition import DataAcquisition

# Initialize the data acquisition system
da = DataAcquisition()

# Run the complete pipeline
verified_data, report = da.run_phase1()

# Access results
print(f"Confidence Score: {report['confidence_score']}%")
print(f"Verified Records: {len(verified_data)}")
```

### Custom Configuration

```python
# Using a custom configuration file
da = DataAcquisition(config_path="custom_config.yaml")

# Collect data from specific sources only
da.collect_yahoo_finance()
da.collect_alpha_vantage()
```

### Accessing Validation Report

```python
# After running the pipeline
report = da.verification_report

# Key metrics
print(f"Confidence Score: {report['confidence_score']}%")
print(f"Discrepancies Found: {len(report['discrepancies'])}")
print(f"Missing Data Gaps: {len(report['missing_gaps'])}")
```

## API Keys Configuration

To use premium data sources, set API keys in environment variables or in `config.yaml`:

```bash
export ALPHA_VANTAGE_API_KEY="your_key_here"
export TIINGO_API_KEY="your_key_here"
export EODHD_API_KEY="your_key_here"
export QUANDL_API_KEY="your_key_here"
```

## Data Output

### Verified Data
- **Format**: CSV file with OHLCV data
- **Location**: `data/{ticker}_verified_ohlcv.csv`
- **Content**: Cleaned, validated price data

### Validation Report
- **Format**: YAML file with detailed metrics
- **Location**: `data/{ticker}_verification_report.yaml`
- **Content**: 
  - Confidence score
  - Discrepancy analysis
  - Missing data gaps
  - Outlier detection results
  - Sources used

## Error Handling

### Common Issues and Solutions

1. **API Rate Limits**
   - Automatic retry with exponential backoff
   - Circuit breaker prevents repeated failures

2. **Network Connectivity**
   - Graceful handling of connection timeouts
   - Fallback to cached data when available

3. **Data Quality Issues**
   - Cross-source validation identifies discrepancies
   - Manual review dates flagged for inspection

### Logging

The pipeline uses Python's logging module with the following levels:
- **INFO**: General progress and summary information
- **WARNING**: Non-critical issues that don't stop execution
- **ERROR**: Critical issues that prevent data collection

## Performance Considerations

### Caching
- Reduces API calls by storing previously collected data
- Cache files stored in the `cache/` directory
- Automatic cache invalidation based on date ranges

### Memory Usage
- Processes data in chunks to manage memory efficiently
- Temporary data structures cleaned up after use

### Execution Time
- First run may take longer due to API calls
- Subsequent runs benefit from caching
- Parallel processing of independent data sources

## Extending the Pipeline

### Adding New Data Sources

1. Add the new source to the `collect_all_sources()` method
2. Implement a specific collection method (e.g., `collect_new_source()`)
3. Add circuit breaker protection
4. Implement retry logic using the `@retry_on_failure` decorator
5. Add caching support

### Adding New Validation Rules

1. Modify the `verify_data_quality()` method
2. Add new validation functions as needed
3. Update the validation report structure
4. Adjust confidence scoring algorithm

## Troubleshooting

### No Data Collected
- Verify API keys are correctly configured
- Check network connectivity
- Ensure the ticker symbol is valid
- Confirm date ranges are appropriate

### Low Confidence Score
- Check for data source discrepancies
- Review validation thresholds in config
- Examine the validation report for details

### Performance Issues
- Check for network latency
- Review API rate limits
- Consider reducing the number of data sources
- Enable caching for frequently accessed data

## Best Practices

1. **Regular Updates**: Run the pipeline regularly to keep data current
2. **Monitor Reports**: Review validation reports for data quality issues
3. **API Key Management**: Rotate keys periodically and monitor usage
4. **Backup Configuration**: Keep copies of working configuration files
5. **Data Validation**: Always verify confidence scores before using data

## Dependencies

The enhanced pipeline requires the following Python packages:
- yfinance
- alpha-vantage (optional)
- tiingo (optional)
- eodhistoricaldata (optional)
- quandl (optional)
- pandas
- numpy
- requests
- tqdm
- pyyaml

Install optional dependencies as needed:
```bash
pip install alpha-vantage tiingo quandl
```

Note: EODHD may require manual installation as it's not available on PyPI.