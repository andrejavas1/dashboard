#!/usr/bin/env python3
"""
Start Dashboard with Background Pipeline Worker

Launches:
1. Flask dashboard server on port 5001
2. Pipeline worker process (runs main.py for queued tickers)

Both run concurrently.
"""

import os
import sys
import subprocess
import time
import logging
import threading
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_worker_output(process):
    """Read worker output to prevent buffer deadlock."""
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[WORKER] {line.rstrip()}")
    except:
        pass


def start_dashboard_and_worker():
    """Start dashboard server and pipeline worker."""
    
    logger.info("=" * 60)
    logger.info("STARTING DASHBOARD WITH PIPELINE WORKER")
    logger.info("=" * 60)
    
    # Start pipeline worker in background
    logger.info("Starting pipeline worker...")
    worker_process = subprocess.Popen(
        [sys.executable, 'src/pipeline_worker.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    logger.info(f"✓ Pipeline worker started (PID: {worker_process.pid})")
    
    # Start thread to read worker output (prevents buffer deadlock)
    output_thread = threading.Thread(target=read_worker_output, args=(worker_process,), daemon=True)
    output_thread.start()
    
    # Give worker time to initialize
    time.sleep(2)
    
    # Verify worker is running
    if worker_process.poll() is not None:
        logger.error("✗ Worker process exited immediately! Check pipeline_worker.log")
        return
    
    logger.info("✓ Worker is running")
    
    # Start Flask dashboard server
    logger.info("Starting dashboard server...")
    try:
        logger.info("\n" + "=" * 60)
        logger.info("DASHBOARD SERVER STARTING")
        logger.info("=" * 60)
        logger.info("Dashboard available at: http://localhost:5001")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60 + "\n")
        
        # Run Flask app
        os.system(f"{sys.executable} pattern_dashboard_server.py")
    
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    
    finally:
        # Stop pipeline worker
        logger.info("Stopping pipeline worker...")
        worker_process.terminate()
        try:
            worker_process.wait(timeout=5)
            logger.info("✓ Pipeline worker stopped")
        except subprocess.TimeoutExpired:
            logger.warning("Force killing pipeline worker...")
            worker_process.kill()
            worker_process.wait()
            logger.info("✓ Pipeline worker killed")
        
        logger.info("=" * 60)
        logger.info("DASHBOARD STOPPED")
        logger.info("=" * 60)


if __name__ == "__main__":
    try:
        start_dashboard_and_worker()
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
