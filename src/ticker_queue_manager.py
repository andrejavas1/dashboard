"""
Ticker Queue Manager
Manages persistent queue of tickers to be processed by the pipeline.
Discovers existing tickers and tracks processing status.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TickerQueueManager:
    """Manages ticker processing queue and portfolio."""
    
    QUEUE_FILE = "ticker_queue.json"
    DEFAULT_QUEUE = {
        "portfolio": [],  # Completed tickers with data
        "queued": [],     # Waiting to be processed
        "current": None,  # Currently processing
        "failed": {},     # Failed tickers with error info
        "settings": {
            "auto_start": True,
            "max_parallel": 1
        }
    }
    
    def __init__(self):
        """Initialize the queue manager."""
        self.queue_file = Path(self.QUEUE_FILE)
        self.queue = self._load_queue()
        self._discover_existing_tickers()
    
    def _load_queue(self) -> Dict:
        """Load queue from file or create default."""
        if self.queue_file.exists():
            try:
                with open(self.queue_file, 'r') as f:
                    queue = json.load(f)
                logger.info(f"Loaded queue with {len(queue.get('portfolio', []))} portfolio tickers")
                return queue
            except Exception as e:
                logger.error(f"Error loading queue: {e}, using default")
                return self.DEFAULT_QUEUE.copy()
        else:
            logger.info("No queue file found, creating new")
            return self.DEFAULT_QUEUE.copy()
    
    def _save_queue(self):
        """Save queue to file."""
        try:
            with open(self.queue_file, 'w') as f:
                json.dump(self.queue, f, indent=2)
            logger.debug("Queue saved")
        except Exception as e:
            logger.error(f"Error saving queue: {e}")
    
    def _discover_existing_tickers(self):
        """Scan data/tickers/ and add any new tickers to portfolio."""
        tickers_dir = Path("data/tickers")
        if not tickers_dir.exists():
            return
        
        existing_tickers = set([d.name for d in tickers_dir.iterdir() if d.is_dir()])
        portfolio_tickers = set(self.queue.get('portfolio', []))
        
        new_tickers = existing_tickers - portfolio_tickers
        for ticker in new_tickers:
            # Verify it has required files
            ticker_dir = tickers_dir / ticker
            if (ticker_dir / "final_portfolio.json").exists():
                logger.info(f"Discovered existing ticker: {ticker}")
                self.queue['portfolio'].append(ticker)
        
        if new_tickers:
            self._save_queue()
    
    def add_ticker(self, ticker: str) -> bool:
        """
        Add ticker to processing queue.
        
        Args:
            ticker: Ticker symbol (e.g., 'AAPL')
            
        Returns:
            True if added, False if already exists
        """
        ticker = ticker.upper().strip()
        
        # Check if already in portfolio or queue
        if ticker in self.queue['portfolio']:
            logger.warning(f"Ticker {ticker} already in portfolio")
            return False
        
        if ticker in self.queue['queued']:
            logger.warning(f"Ticker {ticker} already queued")
            return False
        
        if ticker == self.queue.get('current'):
            logger.warning(f"Ticker {ticker} already processing")
            return False
        
        # Validate ticker symbol
        if not (3 <= len(ticker) <= 5 and ticker.isalpha()):
            logger.error(f"Invalid ticker symbol: {ticker}")
            return False
        
        logger.info(f"Adding ticker to queue: {ticker}")
        self.queue['queued'].append(ticker)
        self._save_queue()
        return True
    
    def get_next_ticker(self) -> Optional[str]:
        """
        Get next ticker from queue to process.
        
        Returns:
            Next ticker symbol or None if queue is empty
        """
        # IMPORTANT: Reload queue from file to see updates from dashboard
        self.queue = self._load_queue()
        
        if self.queue['queued']:
            ticker = self.queue['queued'].pop(0)
            self.queue['current'] = ticker
            self._save_queue()
            logger.info(f"Starting processing: {ticker}")
            return ticker
        return None
    
    def mark_complete(self, ticker: str):
        """Mark ticker as complete and ready for portfolio."""
        if self.queue['current'] == ticker:
            self.queue['current'] = None
            if ticker not in self.queue['portfolio']:
                self.queue['portfolio'].append(ticker)
            # Remove from failed if it was there
            self.queue['failed'].pop(ticker, None)
            self._save_queue()
            logger.info(f"Ticker complete: {ticker}")
    
    def mark_failed(self, ticker: str, error: str):
        """Mark ticker as failed with error info."""
        if self.queue['current'] == ticker:
            self.queue['current'] = None
            self.queue['failed'][ticker] = {
                'error': error,
                'timestamp': datetime.now().isoformat()
            }
            self._save_queue()
            logger.error(f"Ticker failed: {ticker} - {error}")
    
    def retry_failed(self, ticker: str) -> bool:
        """Retry a failed ticker."""
        if ticker in self.queue['failed']:
            self.queue['failed'].pop(ticker)
            self.queue['queued'].append(ticker)
            self._save_queue()
            logger.info(f"Retrying ticker: {ticker}")
            return True
        return False
    
    def cancel_queued(self, ticker: str) -> bool:
        """Cancel a queued ticker."""
        if ticker in self.queue['queued']:
            self.queue['queued'].remove(ticker)
            self._save_queue()
            logger.info(f"Cancelled ticker: {ticker}")
            return True
        return False
    
    def remove_ticker(self, ticker: str) -> bool:
        """
        Remove ticker from portfolio and delete its data.
        
        Args:
            ticker: Ticker symbol to remove
            
        Returns:
            True if removed, False if not found
        """
        ticker = ticker.upper().strip()
        removed = False
        
        # Remove from portfolio
        if ticker in self.queue['portfolio']:
            self.queue['portfolio'].remove(ticker)
            removed = True
            logger.info(f"Removed {ticker} from portfolio")
        
        # Remove from queued
        if ticker in self.queue['queued']:
            self.queue['queued'].remove(ticker)
            removed = True
            logger.info(f"Removed {ticker} from queue")
        
        # Remove from failed
        if ticker in self.queue['failed']:
            self.queue['failed'].pop(ticker)
            removed = True
            logger.info(f"Removed {ticker} from failed")
        
        # Clear current if it's the one being removed
        if self.queue.get('current') == ticker:
            self.queue['current'] = None
            removed = True
            logger.info(f"Cleared current processing ticker {ticker}")
        
        if removed:
            self._save_queue()
            
            # Delete ticker data directory
            import shutil
            ticker_dir = Path(f"data/tickers/{ticker}")
            if ticker_dir.exists():
                shutil.rmtree(ticker_dir)
                logger.info(f"Deleted ticker directory: {ticker_dir}")
        
        return removed
    
    def get_status(self) -> Dict:
        """Get current queue status."""
        return {
            'portfolio': self.queue['portfolio'].copy(),
            'queued': self.queue['queued'].copy(),
            'current': self.queue['current'],
            'failed': self.queue['failed'].copy(),
            'total_completed': len(self.queue['portfolio']),
            'total_queued': len(self.queue['queued']),
            'is_processing': self.queue['current'] is not None
        }
    
    def get_portfolio(self) -> List[str]:
        """Get list of completed portfolio tickers."""
        return self.queue['portfolio'].copy()
    
    def default_ticker(self) -> Optional[str]:
        """Get first ticker in portfolio (default to display)."""
        if self.queue['portfolio']:
            return self.queue['portfolio'][0]
        return None


if __name__ == "__main__":
    # Test the queue manager
    manager = TickerQueueManager()
    
    print("\nCurrent Status:")
    status = manager.get_status()
    print(f"Portfolio: {status['portfolio']}")
    print(f"Queued: {status['queued']}")
    print(f"Processing: {status['current']}")
    
    # Test adding a ticker
    print("\nAdding MSFT to queue...")
    manager.add_ticker("MSFT")
    
    print("\nUpdated Status:")
    status = manager.get_status()
    print(f"Queued: {status['queued']}")
