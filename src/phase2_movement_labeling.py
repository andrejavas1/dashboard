"""
Phase 2: Forward-Looking Price Movement Labeling Module
Calculates future price movements and labels historical candles.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MovementLabeling:
    """
    Labels historical candles with forward-looking price movement information.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the movement labeling system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Movement parameters
        self.time_windows = self.config['movement']['time_windows']
        self.thresholds = self.config['movement']['thresholds']
        
        # Data storage
        self.data = None
        self.movement_data = None
        
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
            'movement': {
                'time_windows': [3, 5, 10, 20, 30],
                'thresholds': [1, 2, 3, 5, 7, 10]
            }
        }
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load verified OHLCV data from file.
        
        Args:
            data_path: Path to verified data CSV file
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Loading data from {data_path}")
        self.data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
        logger.info(f"Loaded {len(self.data)} records")
        return self.data
    
    def calculate_forward_movements(self) -> pd.DataFrame:
        """
        Calculate forward-looking price movements for each time window.
        
        Returns:
            DataFrame with movement information for each candle
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: Forward-Looking Price Movement Labeling")
        logger.info("=" * 60)
        
        if self.data is None or self.data.empty:
            logger.error("No data loaded. Call load_data() first.")
            return pd.DataFrame()
        
        # Initialize movement DataFrame
        movement_data = self.data.copy()
        
        # Calculate forward movements for each time window
        for window in self.time_windows:
            logger.info(f"Calculating {window}-day forward movements...")
            
            # Initialize columns for this window
            max_up_col = f'Max_Up_{window}d'
            max_down_col = f'Max_Down_{window}d'
            time_to_max_up_col = f'Time_To_Max_Up_{window}d'
            time_to_max_down_col = f'Time_To_Max_Down_{window}d'
            
            movement_data[max_up_col] = 0.0
            movement_data[max_down_col] = 0.0
            movement_data[time_to_max_up_col] = np.nan
            movement_data[time_to_max_down_col] = np.nan
            
            # Calculate for each candle
            for i in tqdm(range(len(movement_data)), desc=f"{window}d window"):
                current_idx = movement_data.index[i]
                current_close = movement_data.loc[current_idx, 'Close']
                
                # Get future data within window
                future_start = i + 1
                future_end = min(i + window, len(movement_data))
                
                if future_start >= len(movement_data):
                    # No future data available
                    continue
                
                future_data = movement_data.iloc[future_start:future_end]
                
                if len(future_data) == 0:
                    continue
                
                # Calculate maximum upward movement
                max_high = future_data['High'].max()
                max_up_pct = (max_high - current_close) / current_close * 100
                
                # Find time to reach maximum upward movement
                max_up_idx = future_data['High'].idxmax()
                time_to_max_up = (max_up_idx - current_idx).days
                
                # Calculate maximum downward movement
                min_low = future_data['Low'].min()
                max_down_pct = (current_close - min_low) / current_close * 100
                
                # Find time to reach maximum downward movement
                max_down_idx = future_data['Low'].idxmin()
                time_to_max_down = (max_down_idx - current_idx).days
                
                # Store values
                movement_data.loc[current_idx, max_up_col] = max_up_pct
                movement_data.loc[current_idx, max_down_col] = max_down_pct
                movement_data.loc[current_idx, time_to_max_up_col] = time_to_max_up
                movement_data.loc[current_idx, time_to_max_down_col] = time_to_max_down
        
        # Create labels for each threshold
        for threshold in self.thresholds:
            for window in self.time_windows:
                label_col = f'Label_{threshold}pct_{window}d'
                movement_data[label_col] = 'NEUTRAL'
                
                max_up_col = f'Max_Up_{window}d'
                max_down_col = f'Max_Down_{window}d'
                
                # Label based on threshold
                up_mask = movement_data[max_up_col] >= threshold
                down_mask = movement_data[max_down_col] >= threshold
                
                movement_data.loc[up_mask, label_col] = 'STRONG_UP'
                movement_data.loc[down_mask, label_col] = 'STRONG_DOWN'
                
                # If both up and down thresholds are met, label based on which happens first
                both_mask = up_mask & down_mask
                if both_mask.any():
                    for idx in movement_data[both_mask].index:
                        time_up = movement_data.loc[idx, f'Time_To_Max_Up_{window}d']
                        time_down = movement_data.loc[idx, f'Time_To_Max_Down_{window}d']
                        if time_up <= time_down:
                            movement_data.loc[idx, label_col] = 'STRONG_UP'
                        else:
                            movement_data.loc[idx, label_col] = 'STRONG_DOWN'
        
        # Calculate summary statistics
        self._calculate_label_statistics(movement_data)
        
        self.movement_data = movement_data
        
        logger.info(f"Movement labeling complete. Shape: {movement_data.shape}")
        return movement_data
    
    def _calculate_label_statistics(self, data: pd.DataFrame):
        """
        Calculate and log label statistics.
        
        Args:
            data: DataFrame with movement labels
        """
        logger.info("\nLabel Statistics:")
        
        for threshold in self.thresholds:
            for window in self.time_windows:
                label_col = f'Label_{threshold}pct_{window}d'
                
                label_counts = data[label_col].value_counts()
                total = len(data[label_col].dropna())
                
                logger.info(f"\n  {threshold}% threshold, {window}d window:")
                for label in ['STRONG_UP', 'STRONG_DOWN', 'NEUTRAL']:
                    if label in label_counts:
                        count = label_counts[label]
                        pct = count / total * 100 if total > 0 else 0
                        logger.info(f"    {label}: {count} ({pct:.2f}%)")
    
    def create_movement_database(self) -> pd.DataFrame:
        """
        Create the final movement database with all relevant columns.
        
        Returns:
            DataFrame with movement database
        """
        if self.movement_data is None:
            logger.error("Movement data not calculated. Run calculate_forward_movements() first.")
            return pd.DataFrame()
        
        # Select relevant columns
        base_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Movement columns
        move_cols = []
        for window in self.time_windows:
            move_cols.extend([
                f'Max_Up_{window}d',
                f'Max_Down_{window}d',
                f'Time_To_Max_Up_{window}d',
                f'Time_To_Max_Down_{window}d'
            ])
        
        # Label columns
        label_cols = []
        for threshold in self.thresholds:
            for window in self.time_windows:
                label_cols.append(f'Label_{threshold}pct_{window}d')
        
        # Create movement database
        movement_db = self.movement_data[base_cols + move_cols + label_cols].copy()
        
        return movement_db
    
    def save_movement_data(self, output_dir: str = "data"):
        """
        Save movement data to files.
        
        Args:
            output_dir: Directory to save output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.movement_data is None:
            logger.error("No movement data to save")
            return
        
        # Save full movement data
        data_path = os.path.join(output_dir, "movement_labeled_data.csv")
        self.movement_data.to_csv(data_path)
        logger.info(f"Movement data saved to {data_path}")
        
        # Save movement database
        movement_db = self.create_movement_database()
        db_path = os.path.join(output_dir, "movement_database.csv")
        movement_db.to_csv(db_path)
        logger.info(f"Movement database saved to {db_path}")
    
    def get_movement_summary(self) -> Dict:
        """
        Get summary statistics of movement data.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.movement_data is None:
            return {}
        
        summary = {
            'total_records': len(self.movement_data),
            'date_range': {
                'start': str(self.movement_data.index.min()),
                'end': str(self.movement_data.index.max())
            },
            'time_windows': self.time_windows,
            'thresholds': self.thresholds
        }
        
        # Add label distributions
        label_distributions = {}
        for threshold in self.thresholds:
            for window in self.time_windows:
                label_col = f'Label_{threshold}pct_{window}d'
                label_counts = self.movement_data[label_col].value_counts().to_dict()
                label_distributions[f'{threshold}pct_{window}d'] = label_counts
        
        summary['label_distributions'] = label_distributions
        
        return summary
    
    def run_phase2(self, data_path: str = None) -> pd.DataFrame:
        """
        Run complete Phase 2: Forward-Looking Price Movement Labeling.
        
        Args:
            data_path: Path to verified data CSV file
            
        Returns:
            DataFrame with movement labels
        """
        logger.info("\n" + "=" * 60)
        logger.info("STARTING PHASE 2: MOVEMENT LABELING")
        logger.info("=" * 60)
        
        # Load data
        if data_path:
            self.load_data(data_path)
        elif self.data is None:
            # Try to find the verified data file based on config ticker
            ticker = self.config.get('data_sources', {}).get('ticker', 'XOM')
            default_path = os.path.join("data", f"{ticker}_verified_ohlcv.csv")
            logger.info(f"[DEBUG Phase2] Looking for data file: {default_path}")
            if os.path.exists(default_path):
                self.load_data(default_path)
            else:
                logger.error(f"No data path provided and default file not found: {default_path}")
                return pd.DataFrame()
        
        # Calculate forward movements
        self.calculate_forward_movements()
        
        # Save results
        self.save_movement_data()
        
        # Print summary
        summary = self.get_movement_summary()
        logger.info(f"\nPhase 2 Summary:")
        logger.info(f"  Total Records: {summary['total_records']}")
        logger.info(f"  Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2 COMPLETE")
        logger.info("=" * 60)
        
        return self.movement_data


if __name__ == "__main__":
    # Run Phase 2
    ml = MovementLabeling()
    movement_data = ml.run_phase2()
    
    print(f"\nFinal Results:")
    print(f"  Total Records: {len(movement_data)}")
    print(f"  Columns: {len(movement_data.columns)}")
    print(f"\nSample Movement Columns:")
    for col in movement_data.columns:
        if 'Max_Up' in col or 'Max_Down' in col:
            print(f"  {col}")