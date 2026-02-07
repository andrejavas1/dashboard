"""
Pipeline Worker
Runs as a separate process and processes ticker queue in the background.
Monitors progress and updates status for dashboard.
"""

import os
import sys
import json
import logging
import subprocess
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ticker_queue_manager import TickerQueueManager

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_worker.log'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)


class PipelineWorker:
    """Processes ticker queue in background."""
    
    PROGRESS_FILE = "ticker_progress.json"
    DEFAULT_PROGRESS = {
        "ticker": None,
        "current_phase": 0,
        "total_phases": 10,
        "phase_name": "None",
        "progress_pct": 0,
        "status": "idle",  # idle, processing, complete, failed
        "start_time": None,
        "elapsed_seconds": 0,
        "error": None
    }
    
    def __init__(self):
        """Initialize the pipeline worker."""
        self.queue_manager = TickerQueueManager()
        self.progress_file = Path(self.PROGRESS_FILE)
        self.progress = self._load_progress()
        self.is_running = False
        
        logger.info("Pipeline Worker initialized")
    
    def _load_progress(self) -> Dict:
        """Load progress from file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading progress: {e}")
        return self.DEFAULT_PROGRESS.copy()
    
    def _save_progress(self):
        """Save progress to file."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    def _update_progress(self, ticker: str, phase: int, phase_name: str, pct: int = None):
        """Update progress for current ticker."""
        self.progress['ticker'] = ticker
        self.progress['current_phase'] = phase
        self.progress['phase_name'] = phase_name
        self.progress['status'] = 'processing'
        
        if pct is not None:
            self.progress['progress_pct'] = min(100, max(0, pct))
        else:
            # Estimate based on phase
            self.progress['progress_pct'] = int((phase / self.progress['total_phases']) * 100)
        
        elapsed = time.time() - self.progress['start_time'] if self.progress['start_time'] else 0
        self.progress['elapsed_seconds'] = int(elapsed)
        
        self._save_progress()
        logger.info(f"{ticker}: Phase {phase}/10 - {phase_name} ({self.progress['progress_pct']}%)")
    
    def _parse_main_output(self, line: str) -> Optional[Dict]:
        """
        Parse main.py output to extract phase info.
        
        Looks for patterns like:
        "PHASE 3: FEATURE ENGINEERING"
        "Phase 3: Feature Engineering"
        """
        patterns = [
            r'PHASE\s+(\d+):\s+(.+)',
            r'Phase\s+(\d+):\s+(.+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                phase_num = int(match.group(1))
                phase_name = match.group(2).strip()
                return {'phase': phase_num, 'name': phase_name}
        
        return None
    
    def _run_pipeline(self, ticker: str) -> bool:
        """
        Run main.py for given ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Starting pipeline for {ticker}")
        
        # Update config.yaml with ticker
        try:
            with open('config.yaml', 'r') as f:
                config_content = f.read()
            
            # Replace ticker line
            import re
            config_content = re.sub(
                r'ticker:\s*["\']?\w+["\']?',
                f'ticker: "{ticker}"',
                config_content
            )
            
            with open('config.yaml', 'w') as f:
                f.write(config_content)
            logger.info(f"Updated config.yaml with ticker: {ticker}")
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return False
        
        # Set progress
        self.progress['start_time'] = time.time()
        self.progress['status'] = 'processing'
        self._update_progress(ticker, 0, "Initializing", 0)
        
        try:
            # Run main.py
            process = subprocess.Popen(
                [sys.executable, 'main.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Monitor output
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                logger.debug(line)
                
                # Parse phase info
                phase_info = self._parse_main_output(line)
                if phase_info:
                    self._update_progress(
                        ticker,
                        phase_info['phase'],
                        phase_info['name']
                    )
            
            # Wait for completion
            exit_code = process.wait()
            
            if exit_code == 0:
                logger.info(f"Pipeline completed successfully for {ticker}")
                self.progress['status'] = 'complete'
                self.progress['progress_pct'] = 100
                self._save_progress()
                return True
            else:
                logger.error(f"Pipeline failed for {ticker} with exit code {exit_code}")
                self.progress['status'] = 'failed'
                self.progress['error'] = f"Pipeline exited with code {exit_code}"
                self._save_progress()
                return False
        
        except Exception as e:
            logger.error(f"Error running pipeline: {e}")
            self.progress['status'] = 'failed'
            self.progress['error'] = str(e)
            self._save_progress()
            return False
    
    def process_queue(self):
        """Process ticker queue until empty."""
        logger.info("=== STARTING PIPELINE WORKER ===")
        self.is_running = True
        check_count = 0
        
        try:
            while self.is_running:
                check_count += 1
                
                # Get next ticker (reloads queue from file each time)
                ticker = self.queue_manager.get_next_ticker()
                
                if not ticker:
                    # Queue is empty
                    status = self.queue_manager.get_status()
                    if check_count % 6 == 0:  # Log every 30 seconds (6 * 5 second checks)
                        logger.info(f"Queue check: Empty. Portfolio: {len(status['portfolio'])} | Queued: {len(status['queued'])} | Current: {status['current']}")
                    self.progress['status'] = 'idle'
                    self.progress['ticker'] = None
                    self._save_progress()
                    time.sleep(5)  # Check every 5 seconds
                    continue
                
                # Run pipeline for this ticker
                logger.info(f"★ Processing ticker: {ticker}")
                success = self._run_pipeline(ticker)
                
                if success:
                    self.queue_manager.mark_complete(ticker)
                    logger.info(f"✓ {ticker} completed and added to portfolio")
                else:
                    self.queue_manager.mark_failed(ticker, self.progress.get('error', 'Unknown error'))
                    logger.error(f"✗ {ticker} failed")
        
        except KeyboardInterrupt:
            logger.info("Pipeline worker stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error in worker: {e}", exc_info=True)
            raise
        finally:
            logger.info("=== PIPELINE WORKER STOPPED ===")
            self.is_running = False
    
    def stop(self):
        """Stop the worker gracefully."""
        logger.info("Stopping pipeline worker...")
        self.is_running = False


def run_worker():
    """Entry point for running the worker as a separate process."""
    logger.info("=" * 80)
    logger.info("PIPELINE WORKER STARTING")
    logger.info("=" * 80)
    
    worker = PipelineWorker()
    
    try:
        logger.info(f"Checking initial queue state...")
        logger.info(f"Portfolio: {worker.queue_manager.queue.get('portfolio', [])}")
        logger.info(f"Queued: {worker.queue_manager.queue.get('queued', [])}")
        logger.info(f"Current: {worker.queue_manager.queue.get('current')}")
        logger.info(f"Starting queue processor...")
        worker.process_queue()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_worker()
