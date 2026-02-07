"""Multi-Ticker Orchestrator - Manages pattern discovery across multiple tickers.

This module coordinates running the full pipeline for multiple tickers,
managing per-ticker data directories and aggregating results.
"""
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import yaml

logger = logging.getLogger(__name__)


class MultiTickerOrchestrator:
    """Orchestrates pattern discovery for multiple tickers."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Get tickers list
        self.tickers = self._get_tickers()
        self.base_data_dir = Path('data')
        
        logger.info(f"MultiTickerOrchestrator initialized with {len(self.tickers)} tickers: {self.tickers}")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _get_tickers(self) -> List[str]:
        """Get list of tickers from config."""
        streaming_config = self.config.get('streaming', {})
        
        # Try new multi-ticker format first
        tickers = streaming_config.get('tickers', [])
        if tickers:
            return tickers
        
        # Fall back to single ticker
        single_ticker = streaming_config.get('ticker', 'XOM')
        return [single_ticker]
    
    def setup_ticker_directories(self) -> Dict[str, Path]:
        """Create per-ticker data directories.
        
        Returns:
            Dict mapping ticker to its data directory
        """
        ticker_dirs = {}
        
        for ticker in self.tickers:
            # Create ticker-specific directory
            ticker_dir = self.base_data_dir / 'tickers' / ticker
            ticker_dir.mkdir(parents=True, exist_ok=True)
            ticker_dirs[ticker] = ticker_dir
            
            # Create subdirectories
            (ticker_dir / 'occurrences').mkdir(exist_ok=True)
            (ticker_dir / 'backups').mkdir(exist_ok=True)
            
            logger.info(f"Setup directory for {ticker}: {ticker_dir}")
        
        return ticker_dirs
    
    def run_pipeline_for_ticker(self, ticker: str) -> Dict:
        """Run the full pipeline for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Pipeline result dict
        """
        import subprocess
        import sys
        
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING PIPELINE FOR {ticker}")
        logger.info(f"{'='*60}\n")
        
        # Update config for this ticker
        self._set_active_ticker(ticker)
        
        # Verify config was updated
        import yaml
        with open(self.config_path, 'r') as f:
            verify_config = yaml.safe_load(f)
        actual_ticker = verify_config.get('data_sources', {}).get('ticker', 'NOT_FOUND')
        logger.info(f"[DEBUG] Config file now has ticker: {actual_ticker}")
        if actual_ticker != ticker:
            logger.error(f"[ERROR] Config mismatch! Expected {ticker}, got {actual_ticker}")
        
        # Run pipeline
        try:
            result = subprocess.run(
                [sys.executable, 'main.py'],
                capture_output=True,
                text=True,
                timeout=3600,
                cwd=str(Path.cwd())
            )
            
            success = result.returncode == 0
            
            if success:
                logger.info(f"✓ Pipeline completed for {ticker}")
            else:
                logger.error(f"✗ Pipeline failed for {ticker}")
                logger.error(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            
            return {
                'ticker': ticker,
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,
                'stderr': result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"Pipeline timeout for {ticker}")
            return {'ticker': ticker, 'success': False, 'error': 'Timeout'}
        except Exception as e:
            logger.error(f"Pipeline error for {ticker}: {e}")
            return {'ticker': ticker, 'success': False, 'error': str(e)}
    
    def _set_active_ticker(self, ticker: str):
        """Temporarily set the active ticker in config."""
        # Modify config to use single ticker for this run
        # DataAcquisition reads from data_sources.ticker
        self.config['data_sources']['ticker'] = ticker
        self.config['streaming']['ticker'] = ticker
        
        # Save temporary config
        temp_config_path = self.config_path.with_suffix('.temp.yaml')
        with open(temp_config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        # Replace original with temp
        shutil.move(temp_config_path, self.config_path)
        
        logger.info(f"Set active ticker to {ticker}")
    
    def move_ticker_data(self, ticker: str, ticker_dir: Path):
        """Move generated data to ticker-specific directory.
        
        Args:
            ticker: Ticker symbol
            ticker_dir: Target directory for this ticker
        """
        files_to_move = [
            'patterns.json',
            'features_matrix.csv',
            'ohlcv.json',
            'ohlcv_full_history.json',
            'validated_patterns.json',
            'discovered_patterns.json',
            'optimized_patterns.json',
            'ranked_patterns.json',
            'classified_patterns.json',
            'final_portfolio.json',
            'portfolio_summary.json',
            'pattern_regime_analysis.json',
            'pattern_correlation_matrix.csv',
            'movement_database.csv',
            'movement_labeled_data.csv',
            'test_status.json',
            'enhanced_patterns.json',
            'validation_results.json'
        ]
        
        moved = []
        for filename in files_to_move:
            src = self.base_data_dir / filename
            if src.exists():
                dst = ticker_dir / filename
                shutil.move(str(src), str(dst))
                moved.append(filename)
        
        # Move occurrence files
        for occ_file in self.base_data_dir.glob('pattern_*_occurrences.json'):
            dst = ticker_dir / 'occurrences' / occ_file.name
            shutil.move(str(occ_file), str(dst))
            moved.append(occ_file.name)
        
        logger.info(f"Moved {len(moved)} files to {ticker_dir}")
    
    def run_all_tickers(self) -> Dict:
        """Run pipeline for all tickers.
        
        Returns:
            Summary of results for all tickers
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"MULTI-TICKER PIPELINE START")
        logger.info(f"Tickers: {self.tickers}")
        logger.info(f"{'='*60}\n")
        
        # Setup directories
        ticker_dirs = self.setup_ticker_directories()
        
        # Backup original config
        original_config = self.config.copy()
        
        results = {
            'started_at': datetime.now().isoformat(),
            'tickers': self.tickers,
            'results': {},
            'successful': [],
            'failed': []
        }
        
        # Run pipeline for each ticker
        for ticker in self.tickers:
            # Run pipeline
            result = self.run_pipeline_for_ticker(ticker)
            results['results'][ticker] = result
            
            if result['success']:
                results['successful'].append(ticker)
                # Move data to ticker directory
                self.move_ticker_data(ticker, ticker_dirs[ticker])
            else:
                results['failed'].append(ticker)
        
        # Restore original config (with tickers list)
        self.config = original_config
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        results['completed_at'] = datetime.now().isoformat()
        
        # Save summary
        summary_path = self.base_data_dir / 'multi_ticker_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info(f"MULTI-TICKER PIPELINE COMPLETE")
        logger.info(f"Successful: {len(results['successful'])}/{len(self.tickers)}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"{'='*60}\n")
        
        return results
    
    def get_ticker_summary(self, ticker: str) -> Optional[Dict]:
        """Get summary for a specific ticker."""
        ticker_dir = self.base_data_dir / 'tickers' / ticker
        patterns_file = ticker_dir / 'patterns.json'
        
        if not patterns_file.exists():
            return None
        
        try:
            with open(patterns_file, 'r') as f:
                patterns = json.load(f)
            
            return {
                'ticker': ticker,
                'pattern_count': len(patterns),
                'data_dir': str(ticker_dir)
            }
        except Exception as e:
            logger.error(f"Error loading summary for {ticker}: {e}")
            return None
    
    def get_all_summaries(self) -> List[Dict]:
        """Get summaries for all tickers."""
        summaries = []
        for ticker in self.tickers:
            summary = self.get_ticker_summary(ticker)
            if summary:
                summaries.append(summary)
        return summaries


def main():
    """Run multi-ticker orchestrator."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    orchestrator = MultiTickerOrchestrator()
    results = orchestrator.run_all_tickers()
    
    print("\n" + "="*60)
    print("MULTI-TICKER RESULTS")
    print("="*60)
    for ticker, result in results['results'].items():
        status = "✓" if result['success'] else "✗"
        print(f"{status} {ticker}: {'Success' if result['success'] else 'Failed'}")
    print("="*60)


if __name__ == '__main__':
    main()
