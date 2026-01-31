"""
Phase 1: Data Acquisition & Validation Module
Collects historical OHLCV data from multiple sources and validates data quality.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
from functools import wraps
import json
import hashlib
from pathlib import Path

# Data source libraries
import yfinance as yf

# Optional data sources (require API keys)
try:
    from alpha_vantage.timeseries import TimeSeries
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False
    TimeSeries = None

try:
    import tiingo
    TIINGO_AVAILABLE = True
except ImportError:
    TIINGO_AVAILABLE = False
    tiingo = None

# Additional data sources
try:
    import quandl
    QUANDL_AVAILABLE = True
except ImportError:
    QUANDL_AVAILABLE = False
    quandl = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breaker pattern implementation for data sources."""
    
    def __init__(self, failure_threshold=5, recovery_timeout=300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Call function with circuit breaker protection."""
        if self.state == "OPEN":
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e
    
    def on_success(self):
        """Reset failure count on successful call."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def on_failure(self):
        """Increment failure count and open circuit if threshold reached."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


def retry_on_failure(max_retries=3, delay=1, backoff=2):
    """Decorator to retry function calls with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise e
                    logger.warning(f"Function {func.__name__} failed, retrying in {current_delay} seconds: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
            return None
        return wrapper
    return decorator


class DataAcquisition:
    """
    Collects and validates historical price data from multiple sources.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the data acquisition system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.run_mode = self.config.get('run_mode', 'full')
        self.ticker = self.config['data_sources']['ticker']
        self.start_date = self._get_start_date()
        self.end_date = self._get_end_date()
        
        # Log run mode
        logger.info(f"Run Mode: {self.run_mode.upper()}")
        logger.info(f"Date Range: {self.start_date} to {self.end_date}")
        
        # API configurations
        self.av_api_key = os.getenv('ALPHA_VANTAGE_API_KEY', self.config['data_sources']['alpha_vantage']['api_key'])
        self.tiingo_api_key = os.getenv('TIINGO_API_KEY', self.config['data_sources']['tiingo']['api_key'])
        self.quandl_api_key = os.getenv('QUANDL_API_KEY', self.config['data_sources'].get('quandl', {}).get('api_key', ''))
        
        # Initialize data source clients
        self._init_clients()
        
        # Circuit breakers for each data source
        self.circuit_breakers = {
            'yahoo_finance': CircuitBreaker(),
            'alpha_vantage': CircuitBreaker(),
            'tiingo': CircuitBreaker(),
            'quandl': CircuitBreaker()
        }
        
        # Validation thresholds
        self.ohlc_threshold = self.config['validation']['ohlc_discrepancy_threshold']
        self.consensus_threshold = self.config['validation']['consensus_agreement_threshold']
        self.volume_threshold = self.config['validation']['volume_discrepancy_threshold']
        self.max_missing_days = self.config['validation']['max_consecutive_missing_days']
        self.min_confidence = self.config['validation']['min_confidence_score']
        
        # Storage for collected data
        self.data_sources = {}
        self.verified_data = None
        self.verification_report = {}
        
        # Caching
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
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
            'data_sources': {
                'ticker': 'XOM',
                'start_date': '2010-01-01',
                'end_date': 'current',
                'alpha_vantage': {'api_key': ''},
                'tiingo': {'api_key': ''},
                'eodhd': {'api_key': ''},
                'quandl': {'api_key': ''}
            },
            'validation': {
                'ohlc_discrepancy_threshold': 0.5,
                'consensus_agreement_threshold': 0.2,
                'volume_discrepancy_threshold': 10,
                'max_consecutive_missing_days': 3,
                'min_confidence_score': 98
            }
        }
    
    def _get_start_date(self) -> str:
        """Get start date based on run mode."""
        if self.run_mode == 'quick':
            # Quick mode: last 2 years of data
            end = datetime.now()
            start = end - timedelta(days=730)  # 2 years
            return start.strftime('%Y-%m-%d')
        else:
            # Full mode: use configured start date
            return self.config['data_sources']['start_date']
    
    def _get_end_date(self) -> str:
        """Get end date (current date if set to 'current')."""
        end_date = self.config['data_sources']['end_date']
        if end_date == 'current':
            return datetime.now().strftime('%Y-%m-%d')
        return end_date
    
    def _init_clients(self):
        """Initialize API clients for each data source."""
        # Alpha Vantage client
        if ALPHA_VANTAGE_AVAILABLE and self.av_api_key:
            self.av_client = TimeSeries(key=self.av_api_key, output_format='pandas')
            logger.info("Alpha Vantage client initialized")
        else:
            self.av_client = None
            if not ALPHA_VANTAGE_AVAILABLE:
                logger.warning("Alpha Vantage library not installed (optional)")
            else:
                logger.warning("Alpha Vantage API key not provided")
        
        # Tiingo client
        if TIINGO_AVAILABLE and self.tiingo_api_key:
            self.tiingo_client = tiingo.TiingoClient(
                {'api_key': self.tiingo_api_key, 'session': True}
            )
            logger.info("Tiingo client initialized")
        else:
            self.tiingo_client = None
            if not TIINGO_AVAILABLE:
                logger.warning("Tiingo library not installed (optional)")
            else:
                logger.warning("Tiingo API key not provided")
        
        # Quandl client
        if QUANDL_AVAILABLE and self.quandl_api_key:
            quandl.ApiConfig.api_key = self.quandl_api_key
            self.quandl_client = quandl
            logger.info("Quandl client initialized")
        else:
            self.quandl_client = None
            if not QUANDL_AVAILABLE:
                logger.warning("Quandl library not installed (optional)")
            else:
                logger.warning("Quandl API key not provided")
    
    def _get_cache_key(self, source_name: str, ticker: str, start_date: str, end_date: str) -> str:
        """Generate cache key for data source."""
        key_string = f"{source_name}_{ticker}_{start_date}_{end_date}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available."""
        cache_file = self.cache_dir / f"{cache_key}.csv"
        if cache_file.exists():
            try:
                data = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
                logger.info(f"Loaded data from cache: {cache_file}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """Save data to cache."""
        if not data.empty:
            cache_file = self.cache_dir / f"{cache_key}.csv"
            try:
                data.to_csv(cache_file)
                logger.info(f"Saved data to cache: {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save to cache: {e}")
    
    @retry_on_failure(max_retries=3, delay=1)
    def collect_yahoo_finance(self) -> pd.DataFrame:
        """
        Collect data from Yahoo Finance using yfinance.
        
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Collecting data from Yahoo Finance for {self.ticker}")
        
        # Check cache first
        cache_key = self._get_cache_key('yahoo_finance', self.ticker, self.start_date, self.end_date)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            # Normalize timezone for cached data
            if cached_data.index.tz is not None:
                cached_data.index = cached_data.index.tz_convert('UTC').tz_localize(None)
            else:
                # Ensure consistent date format without timezone
                cached_data.index = cached_data.index.normalize()
            return cached_data
        
        try:
            ticker_obj = yf.Ticker(self.ticker)
            data = ticker_obj.history(
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,  # Adjust for splits and dividends
                actions=False
            )
            
            # Standardize column names
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            data.index.name = 'Date'
            
            # Normalize timezone to UTC and remove timezone info to prevent duplicates
            if data.index.tz is not None:
                data.index = data.index.tz_convert('UTC').tz_localize(None)
            else:
                # Ensure consistent date format without timezone
                data.index = data.index.normalize()
            
            # Save to cache
            self._save_to_cache(cache_key, data)
            
            logger.info(f"Yahoo Finance: Collected {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Error collecting Yahoo Finance data: {e}")
            return pd.DataFrame()
    
    @retry_on_failure(max_retries=3, delay=1)
    def collect_alpha_vantage(self) -> pd.DataFrame:
        """
        Collect data from Alpha Vantage.
        
        Returns:
            DataFrame with OHLCV data
        """
        if not self.av_client:
            logger.warning("Alpha Vantage client not initialized, skipping")
            return pd.DataFrame()
        
        logger.info(f"Collecting data from Alpha Vantage for {self.ticker}")
        
        # Check cache first
        cache_key = self._get_cache_key('alpha_vantage', self.ticker, self.start_date, self.end_date)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            # Alpha Vantage has rate limits, so we need to be careful
            # Using get_daily (free endpoint) instead of get_daily_adjusted (premium)
            data, meta_data = self.av_client.get_daily(
                symbol=self.ticker,
                outputsize='full'
            )
            
            # Standardize column names (get_daily returns: open, high, low, close, volume)
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            data.index.name = 'Date'
            
            # Filter by date range
            data = data.loc[self.start_date:self.end_date]
            
            # Save to cache
            self._save_to_cache(cache_key, data)
            
            logger.info(f"Alpha Vantage: Collected {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Error collecting Alpha Vantage data: {e}")
            return pd.DataFrame()
    
    @retry_on_failure(max_retries=3, delay=1)
    def collect_tiingo(self) -> pd.DataFrame:
        """
        Collect data from Tiingo.
        
        Returns:
            DataFrame with OHLCV data
        """
        if not self.tiingo_client:
            logger.warning("Tiingo client not initialized, skipping")
            return pd.DataFrame()
        
        logger.info(f"Collecting data from Tiingo for {self.ticker}")
        
        # Check cache first
        cache_key = self._get_cache_key('tiingo', self.ticker, self.start_date, self.end_date)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            # Normalize timezone for cached data
            if cached_data.index.tz is not None:
                cached_data.index = cached_data.index.tz_convert('UTC').tz_localize(None)
            else:
                # Ensure consistent date format without timezone
                cached_data.index = cached_data.index.normalize()
            return cached_data
        
        try:
            data = self.tiingo_client.get_ticker_price(
                self.ticker,
                startDate=self.start_date,
                endDate=self.end_date,
                frequency='daily'
            )
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.index.name = 'Date'
            
            # Normalize timezone to UTC and remove timezone info to prevent duplicates
            if df.index.tz is not None:
                df.index = df.index.tz_convert('UTC').tz_localize(None)
            else:
                # Ensure consistent date format without timezone
                df.index = df.index.normalize()
            
            # Standardize column names
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'adjClose', 'adjHigh',
                          'adjLow', 'adjOpen', 'adjVolume', 'divCash', 'splitFactor']
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Save to cache
            self._save_to_cache(cache_key, df)
            
            logger.info(f"Tiingo: Collected {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting Tiingo data: {e}")
            return pd.DataFrame()
    
    @retry_on_failure(max_retries=3, delay=1)
    def collect_quandl(self) -> pd.DataFrame:
        """
        Collect data from Quandl.
        
        Returns:
            DataFrame with alternative data sources
        """
        if not self.quandl_client:
            logger.warning("Quandl client not initialized, skipping")
            return pd.DataFrame()
        
        logger.info("Collecting data from Quandl")
        
        try:
            # Example Quandl datasets
            datasets = [
                'FRED/GDP',  # US GDP
                'FRED/INFLATION_USA',  # US Inflation
                'YC/USA_10Y_INFLATION_EXPECTATION'  # 10-Year Inflation Expectation
            ]
            
            quandl_data = pd.DataFrame()
            for dataset in datasets:
                try:
                    data = self.quandl_client.get(dataset, start_date=self.start_date, end_date=self.end_date)
                    # Add prefix to column names to identify source
                    data.columns = [f"{dataset}_{col}" for col in data.columns]
                    if quandl_data.empty:
                        quandl_data = data
                    else:
                        quandl_data = quandl_data.join(data, how='outer')
                    logger.info(f"Collected Quandl dataset: {dataset}")
                except Exception as e:
                    logger.warning(f"Failed to collect Quandl dataset {dataset}: {e}")
                    continue
            
            return quandl_data
            
        except Exception as e:
            logger.error(f"Error collecting Quandl data: {e}")
            return pd.DataFrame()
    
    def collect_all_sources(self) -> Dict[str, pd.DataFrame]:
        """
        Collect data from all available sources.
        
        Returns:
            Dictionary mapping source names to DataFrames
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: Data Acquisition - Collecting from all sources")
        logger.info("=" * 60)
        
        sources = {
            'yahoo_finance': self.collect_yahoo_finance,
            'alpha_vantage': self.collect_alpha_vantage,
            'tiingo': self.collect_tiingo,
            'quandl': self.collect_quandl
        }
        
        # Add rate limiting for API calls
        for source_name, collector in sources.items():
            try:
                # Use circuit breaker
                self.data_sources[source_name] = self.circuit_breakers[source_name].call(collector)
                # Respect rate limits
                time.sleep(1)
            except Exception as e:
                logger.error(f"Failed to collect from {source_name}: {e}")
                self.data_sources[source_name] = pd.DataFrame()
        
        # Log summary
        logger.info("\nData Collection Summary:")
        for source, data in self.data_sources.items():
            if not data.empty:
                logger.info(f"  {source}: {len(data)} records from {data.index.min()} to {data.index.max()}")
            else:
                logger.info(f"  {source}: No data collected")
        
        return self.data_sources
    
    def verify_data_quality(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Verify data quality using Yahoo Finance as primary source.
        Other sources are used for validation only, not for combining data.
        
        Returns:
            Tuple of (verified DataFrame, verification report)
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: Data Validation - Using Yahoo Finance as primary source")
        logger.info("=" * 60)
        
        # Use Yahoo Finance as the primary data source
        if 'yahoo_finance' in self.data_sources and not self.data_sources['yahoo_finance'].empty:
            primary_data = self.data_sources['yahoo_finance'].copy()
            logger.info(f"Using Yahoo Finance as primary data source: {len(primary_data)} records")
        else:
            # Fallback to any available source if Yahoo Finance is not available
            available_sources = {k: v for k, v in self.data_sources.items() if not v.empty}
            if available_sources:
                primary_source = list(available_sources.keys())[0]
                primary_data = available_sources[primary_source].copy()
                logger.info(f"Using {primary_source} as primary data source: {len(primary_data)} records")
            else:
                logger.error("No valid data sources available")
                return pd.DataFrame(), {}
        
        # Validate data quality against other sources (for reporting only)
        validation_results = []
        other_sources = {k: v for k, v in self.data_sources.items() if not v.empty and k != 'yahoo_finance'}
        
        if other_sources:
            logger.info(f"Validating against {len(other_sources)} other sources for quality assessment")
            
            # For each other source, compare with primary data
            for source_name, source_data in other_sources.items():
                # Find common dates
                common_dates = primary_data.index.intersection(source_data.index)
                if len(common_dates) > 0:
                    discrepancies = []
                    for date in common_dates:
                        primary_row = primary_data.loc[date]
                        source_row = source_data.loc[date]
                        
                        # Calculate discrepancies for OHLC values
                        for col in ['Open', 'High', 'Low', 'Close']:
                            if pd.notna(primary_row[col]) and pd.notna(source_row[col]) and primary_row[col] > 0:
                                discrepancy = abs((source_row[col] - primary_row[col]) / primary_row[col]) * 100
                                if discrepancy > self.ohlc_threshold:
                                    discrepancies.append({
                                        'date': date,
                                        'column': col,
                                        'primary_value': primary_row[col],
                                        'source_value': source_row[col],
                                        'discrepancy_pct': discrepancy
                                    })
                    
                    validation_results.append({
                        'source': source_name,
                        'common_dates': len(common_dates),
                        'discrepancies': len(discrepancies),
                        'discrepancy_rate': len(discrepancies) / len(common_dates) * 100 if len(common_dates) > 0 else 0
                    })
                    
                    if discrepancies:
                        logger.info(f"  {source_name}: {len(discrepancies)} discrepancies in {len(common_dates)} common dates ({len(discrepancies)/len(common_dates)*100:.2f}%)")
                    else:
                        logger.info(f"  {source_name}: No significant discrepancies found")
        
        # Check for missing data gaps in primary data
        missing_gaps = self._check_missing_gaps(primary_data)
        
        # Check for outlier candles in primary data
        outliers = self._check_outliers(primary_data)
        
        # Calculate confidence score based on data quality metrics
        total_days = len(primary_data)
        verified_days = total_days - len(outliers)  # Subtract outliers from verified days
        confidence_score = (verified_days / total_days) * 100 if total_days > 0 else 0
        
        # Create verification report
        self.verification_report = {
            'confidence_score': round(confidence_score, 2),
            'verified_days': verified_days,
            'total_days': total_days,
            'validation_results': validation_results,
            'missing_gaps': missing_gaps,
            'outliers': outliers,
            'sources_used': list(self.data_sources.keys())
        }
        
        self.verified_data = primary_data
        
        # Log verification results
        logger.info("\nData Verification Results:")
        logger.info(f"  Confidence Score: {confidence_score:.2f}%")
        logger.info(f"  Verified Days: {verified_days} / {total_days}")
        logger.info(f"  Missing Data Gaps: {len(missing_gaps)}")
        logger.info(f"  Outlier Candles: {len(outliers)}")
        
        # Check if confidence meets minimum threshold
        if confidence_score >= self.min_confidence:
            logger.info(f"\n[PASS] Data quality PASSED (confidence >= {self.min_confidence}%)")
        else:
            logger.warning(f"\n[FAIL] Data quality FAILED (confidence < {self.min_confidence}%)")
        
        # Add data quality metrics to report
        self.verification_report['data_quality_metrics'] = {
            'completeness': round((verified_days / total_days) * 100, 2) if total_days > 0 else 0,
            'outlier_rate': round((len(outliers) / total_days) * 100, 2) if total_days > 0 else 0,
            'gap_rate': round((sum([gap['days'] for gap in missing_gaps]) / total_days) * 100, 2) if total_days > 0 and missing_gaps else 0
        }
        
        # Log data quality alerts
        if self.verification_report['data_quality_metrics']['outlier_rate'] > 2:
            logger.warning(f"High outlier rate: {self.verification_report['data_quality_metrics']['outlier_rate']}%")
        
        if self.verification_report['data_quality_metrics']['gap_rate'] > 1:
            logger.warning(f"High gap rate: {self.verification_report['data_quality_metrics']['gap_rate']}%")
        
        return self.verified_data, self.verification_report
    
    def _check_ohlc_discrepancies(self, source_values: Dict) -> Dict:
        """
        Check OHLC discrepancies across sources.
        
        Args:
            source_values: Dictionary of source name to values
            
        Returns:
            Dictionary with discrepancy metrics
        """
        discrepancies = []
        max_discrepancy = 0
        consensus_agreement = False
        
        for col in ['Open', 'High', 'Low', 'Close']:
            values = [v[col] for v in source_values.values() if pd.notna(v[col])]
            if len(values) >= 2:
                min_val = min(values)
                max_val = max(values)
                if min_val > 0:
                    discrepancy = (max_val - min_val) / min_val * 100
                    discrepancies.append(discrepancy)
                    max_discrepancy = max(max_discrepancy, discrepancy)
        
        # Check if 3+ sources agree within consensus threshold
        if len(source_values) >= 3:
            for col in ['Open', 'High', 'Low', 'Close']:
                values = [v[col] for v in source_values.values() if pd.notna(v[col])]
                if len(values) >= 3:
                    sorted_values = sorted(values)
                    # Check if middle values are close
                    if len(sorted_values) >= 3:
                        middle_range = (sorted_values[-1] - sorted_values[0]) / sorted_values[0] * 100
                        if middle_range <= self.consensus_threshold:
                            consensus_agreement = True
                            break
        
        return {
            'discrepancies': discrepancies,
            'max_discrepancy': max_discrepancy,
            'consensus_agreement': consensus_agreement
        }
    
    def _check_missing_gaps(self, data: pd.DataFrame) -> List[Dict]:
        """
        Check for gaps in data (consecutive missing days).
        
        Args:
            data: DataFrame with datetime index
            
        Returns:
            List of gap information
        """
        gaps = []
        if len(data) < 2:
            return gaps
        
        # Normalize timezone for date range calculation
        start_date = data.index.min()
        end_date = data.index.max()
        
        # Handle timezone-aware data
        if start_date.tz is not None:
            # Use UTC for date range
            start_date = start_date.tz_convert('UTC')
            end_date = end_date.tz_convert('UTC')
            date_range = pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC')
        else:
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        missing_dates = date_range.difference(data.index)
        
        # Find consecutive gaps
        if len(missing_dates) > 0:
            consecutive_count = 1
            gap_start = missing_dates[0]
            
            for i in range(1, len(missing_dates)):
                if (missing_dates[i] - missing_dates[i-1]).days == 1:
                    consecutive_count += 1
                else:
                    if consecutive_count > self.max_missing_days:
                        gaps.append({
                            'start_date': gap_start,
                            'end_date': missing_dates[i-1],
                            'days': consecutive_count
                        })
                    consecutive_count = 1
                    gap_start = missing_dates[i]
            
            # Check last gap
            if consecutive_count > self.max_missing_days:
                gaps.append({
                    'start_date': gap_start,
                    'end_date': missing_dates[-1],
                    'days': consecutive_count
                })
        
        return gaps
    
    def _check_outliers(self, data: pd.DataFrame) -> List[Dict]:
        """
        Check for outlier candles (moves >3 standard deviations from 20-day average).
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            List of outlier information
        """
        outliers = []
        
        if len(data) < 20:
            return outliers
        
        # Calculate 20-day average daily range
        data['daily_range'] = (data['High'] - data['Low']) / data['Close'] * 100
        data['avg_range_20'] = data['daily_range'].rolling(20).mean()
        data['std_range_20'] = data['daily_range'].rolling(20).std()
        
        # Find outliers
        outlier_mask = data['daily_range'] > (data['avg_range_20'] + 3 * data['std_range_20'])
        outlier_dates = data[outlier_mask].index
        
        for date in outlier_dates:
            outliers.append({
                'date': date,
                'daily_range': data.loc[date, 'daily_range'],
                'avg_range': data.loc[date, 'avg_range_20'],
                'std_range': data.loc[date, 'std_range_20']
            })
        
        return outliers
    
    def save_verified_data(self, output_dir: str = "data"):
        """
        Save verified data and verification report to files.
        
        Args:
            output_dir: Directory to save output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save verified data
        if self.verified_data is not None and not self.verified_data.empty:
            data_path = os.path.join(output_dir, f"{self.ticker}_verified_ohlcv.csv")
            self.verified_data.to_csv(data_path, index_label='Date')
            logger.info(f"Verified data saved to {data_path}")
        
        # Save verification report
        if self.verification_report:
            report_path = os.path.join(output_dir, f"{self.ticker}_verification_report.yaml")
            with open(report_path, 'w') as f:
                yaml.dump(self.verification_report, f, default_flow_style=False)
            logger.info(f"Verification report saved to {report_path}")
    
    def run_phase1(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Run complete Phase 1: Data Acquisition & Validation.
        
        Returns:
            Tuple of (verified DataFrame, verification report)
        """
        logger.info("\n" + "=" * 60)
        logger.info("STARTING PHASE 1: DATA ACQUISITION & VALIDATION")
        logger.info("=" * 60)
        
        # Collect data from all sources
        self.collect_all_sources()
        
        # Verify data quality
        verified_data, report = self.verify_data_quality()
        
        # Save results
        self.save_verified_data()
        
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1 COMPLETE")
        logger.info("=" * 60)
        
        return verified_data, report


if __name__ == "__main__":
    # Run Phase 1
    da = DataAcquisition()
    verified_data, report = da.run_phase1()
    
    print(f"\nFinal Results:")
    print(f"  Confidence Score: {report.get('confidence_score', 0):.2f}%")
    print(f"  Verified Days: {report.get('verified_days', 0)}")
    print(f"  Data Range: {verified_data.index.min()} to {verified_data.index.max()}")
    print(f"  Total Records: {len(verified_data)}")