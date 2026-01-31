"""
Phase 4: Pattern Discovery & Condition Identification Module
Discovers specific combinations of features that reliably predict price movements.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import combinations
import json

# Machine Learning
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PatternDiscovery:
    """
    Discovers patterns that predict price movements using multiple methods.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the pattern discovery system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.run_mode = self.config.get('run_mode', 'full')
        
        # Pattern discovery parameters
        self.min_occurrences = self.config['pattern_discovery']['min_occurrences']
        self.min_success_rate = self.config['pattern_discovery']['min_success_rate']
        self.high_confidence_rate = self.config['pattern_discovery']['high_confidence_rate']
        self.high_confidence_occurrences = self.config['pattern_discovery']['high_confidence_occurrences']
        self.p_value_threshold = self.config['pattern_discovery']['p_value_threshold']
        self.max_features = self.config['pattern_discovery']['max_features_per_pattern']
        self.test_combinations = self.config['pattern_discovery']['test_combinations']
        
        # Adjust parameters for quick mode
        if self.run_mode == 'quick':
            self.min_occurrences = 5  # Reduced from 15
            self.min_success_rate = 50  # Reduced from 53
            self.high_confidence_rate = 60  # Reduced from 75
            self.high_confidence_occurrences = 10  # Reduced from 30
            self.test_combinations = [1, 2, 3]  # Only test 1-3 features
            logger.info("Quick Mode: Reduced pattern discovery parameters")
        
        logger.info(f"Run Mode: {self.run_mode.upper()}")
        
        # Regime detection parameters
        self.market_regimes = self.config.get('market_regimes', {})
        self.enable_regime_detection = self.market_regimes.get('enable_regime_detection', False)
        self.discover_by_regime = self.config['pattern_discovery'].get('discover_by_regime', False)
        self.min_regime_samples = self.market_regimes.get('min_regime_samples', 100)
        
        # Movement parameters
        self.time_windows = self.config['movement']['time_windows']
        self.thresholds = self.config['movement']['thresholds']
        
        # Data storage
        self.data = None
        self.numeric_features = None
        self.discovered_patterns = []
        self.high_confidence_patterns = []
        self.medium_confidence_patterns = []
        self.low_confidence_patterns = []
        self.rejected_patterns = []
        
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
            'pattern_discovery': {
                'min_occurrences': 30,
                'min_success_rate': 65,
                'high_confidence_rate': 75,
                'high_confidence_occurrences': 50,
                'p_value_threshold': 0.05,
                'max_features_per_pattern': 5,
                'test_combinations': [2, 3, 4, 5]
            },
            'movement': {
                'time_windows': [3, 5, 10, 20, 30],
                'thresholds': [1, 2, 3, 5, 7, 10]
            }
        }
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load features matrix from file.
        
        Args:
            data_path: Path to features matrix CSV file
            
        Returns:
            DataFrame with features and labels
        """
        logger.info(f"Loading data from {data_path}")
        self.data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
        logger.info(f"Loaded {len(self.data)} records with {len(self.data.columns)} columns")
        return self.data
    
    def _get_numeric_features(self) -> List[str]:
        """Get list of numeric feature columns."""
        if self.data is None:
            return []
        
        # Exclude non-numeric and label columns
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        exclude_cols.extend([f'Max_Up_{w}d' for w in self.time_windows])
        exclude_cols.extend([f'Max_Down_{w}d' for w in self.time_windows])
        exclude_cols.extend([f'Time_To_Max_Up_{w}d' for w in self.time_windows])
        exclude_cols.extend([f'Time_To_Max_Down_{w}d' for w in self.time_windows])
        exclude_cols.extend([f'Label_{t}pct_{w}d' for t in self.thresholds for w in self.time_windows])
        exclude_cols.extend(['Day_of_Week_Name', 'Month_Name', 'Trend_Dir_14', 'Trend_Dir_20', 
                           'Vol_Regime', 'Trend_Regime', 'Vol_Regime_Level'])
        
        numeric_cols = []
        for col in self.data.columns:
            if col not in exclude_cols and self.data[col].dtype in [np.float64, np.int64]:
                numeric_cols.append(col)
        
        self.numeric_features = numeric_cols
        logger.info(f"Found {len(numeric_cols)} numeric features for pattern discovery")
        return numeric_cols
    
    def detect_volatility_regime(self, data: pd.DataFrame = None) -> str:
        """
        Detect the current volatility regime based on ATR percentage.
        
        Args:
            data: DataFrame to analyze (uses self.data if None)
            
        Returns:
            Regime name: 'Low Volatility', 'Normal Volatility', or 'High Volatility'
        """
        if data is None:
            data = self.data
        
        # Get ATR as percentage of price
        atr_pct_cols = [col for col in data.columns if 'ATR_14_Pct' in col]
        if not atr_pct_cols:
            # Fallback to ATR_10_Pct
            atr_pct_cols = [col for col in data.columns if 'ATR_10_Pct' in col]
        
        if not atr_pct_cols:
            return 'Normal Volatility'
        
        atr_pct_col = atr_pct_cols[0]
        avg_atr_pct = data[atr_pct_col].mean()
        
        volatility_regimes = self.market_regimes.get('volatility', [])
        
        for regime in volatility_regimes:
            name = regime['name']
            if 'atr_pct_threshold' in regime:
                threshold = regime['atr_pct_threshold']
                if 'Low' in name and avg_atr_pct < threshold:
                    return name
                elif 'High' in name and avg_atr_pct > threshold:
                    return name
            elif 'atr_pct_min' in regime and 'atr_pct_max' in regime:
                min_val = regime['atr_pct_min']
                max_val = regime['atr_pct_max']
                if min_val <= avg_atr_pct <= max_val:
                    return name
        
        return 'Normal Volatility'
    
    def detect_trend_regime(self, data: pd.DataFrame = None) -> str:
        """
        Detect the current trend regime based on moving averages.
        
        Args:
            data: DataFrame to analyze (uses self.data if None)
            
        Returns:
            Regime name describing the trend state
        """
        if data is None:
            data = self.data
        
        # Check if required columns exist
        if 'MA_50' not in data.columns or 'MA_200' not in data.columns:
            return 'Sideways'
        
        sma50 = data['MA_50'].iloc[-1]
        sma200 = data['MA_200'].iloc[-1]
        price = data['Close'].iloc[-1]
        
        trend_regimes = self.market_regimes.get('trend', [])
        
        for regime in trend_regimes:
            name = regime['name']
            conditions_met = True
            
            if regime.get('sma50_above_sma200'):
                if not (sma50 > sma200):
                    conditions_met = False
            
            if regime.get('sma50_below_sma200'):
                if not (sma50 < sma200):
                    conditions_met = False
            
            if regime.get('price_above_sma50'):
                if not (price > sma50):
                    conditions_met = False
            
            if regime.get('price_below_sma50'):
                if not (price < sma50):
                    conditions_met = False
            
            if regime.get('price_above_sma200'):
                if not (price > sma200):
                    conditions_met = False
            
            if regime.get('price_below_sma200'):
                if not (price < sma200):
                    conditions_met = False
            
            if regime.get('sma50_near_sma200'):
                pct_diff = abs(sma50 - sma200) / sma200 * 100
                max_diff = regime.get('sma50_sma200_pct_diff', 2.0)
                if not (pct_diff <= max_diff):
                    conditions_met = False
            
            if conditions_met:
                return name
        
        return 'Sideways'
    
    def filter_data_by_regime(self, volatility_regime: str = None, trend_regime: str = None) -> pd.DataFrame:
        """
        Filter data to only include rows matching the specified regimes.
        
        Args:
            volatility_regime: Volatility regime name (None to skip)
            trend_regime: Trend regime name (None to skip)
            
        Returns:
            Filtered DataFrame
        """
        mask = pd.Series(True, index=self.data.index)
        
        if volatility_regime:
            # Filter by volatility regime
            atr_pct_cols = [col for col in self.data.columns if 'ATR_14_Pct' in col]
            if not atr_pct_cols:
                atr_pct_cols = [col for col in self.data.columns if 'ATR_10_Pct' in col]
            
            if atr_pct_cols:
                atr_pct_col = atr_pct_cols[0]
                volatility_regimes = self.market_regimes.get('volatility', [])
                
                for regime in volatility_regimes:
                    if regime['name'] == volatility_regime:
                        if 'atr_pct_threshold' in regime:
                            threshold = regime['atr_pct_threshold']
                            if 'Low' in volatility_regime:
                                mask &= (self.data[atr_pct_col] < threshold)
                            elif 'High' in volatility_regime:
                                mask &= (self.data[atr_pct_col] > threshold)
                        elif 'atr_pct_min' in regime and 'atr_pct_max' in regime:
                            min_val = regime['atr_pct_min']
                            max_val = regime['atr_pct_max']
                            mask &= (self.data[atr_pct_col] >= min_val) & (self.data[atr_pct_col] <= max_val)
                        break
        
        if trend_regime:
            # Filter by trend regime
            if 'MA_50' in self.data.columns and 'MA_200' in self.data.columns:
                trend_regimes = self.market_regimes.get('trend', [])
                
                for regime in trend_regimes:
                    if regime['name'] == trend_regime:
                        if regime.get('sma50_above_sma200'):
                            mask &= (self.data['MA_50'] > self.data['MA_200'])
                        if regime.get('sma50_below_sma200'):
                            mask &= (self.data['MA_50'] < self.data['MA_200'])
                        if regime.get('price_above_sma50'):
                            mask &= (self.data['Close'] > self.data['MA_50'])
                        if regime.get('price_below_sma50'):
                            mask &= (self.data['Close'] < self.data['MA_50'])
                        if regime.get('price_above_sma200'):
                            mask &= (self.data['Close'] > self.data['MA_200'])
                        if regime.get('price_below_sma200'):
                            mask &= (self.data['Close'] < self.data['MA_200'])
                        if regime.get('sma50_near_sma200'):
                            pct_diff = abs(self.data['MA_50'] - self.data['MA_200']) / self.data['MA_200'] * 100
                            max_diff = regime.get('sma50_sma200_pct_diff', 2.0)
                            mask &= (pct_diff <= max_diff)
                        break
        
        return self.data[mask]
    
    def get_all_regime_combinations(self) -> List[Tuple[str, str]]:
        """
        Get all possible combinations of volatility and trend regimes.
        
        Returns:
            List of (volatility_regime, trend_regime) tuples
        """
        volatility_regimes = [r['name'] for r in self.market_regimes.get('volatility', [])]
        trend_regimes = [r['name'] for r in self.market_regimes.get('trend', [])]
        
        combinations = []
        for vol in volatility_regimes:
            for trend in trend_regimes:
                combinations.append((vol, trend))
        
        return combinations
    
    def rule_based_discovery(self, label_col: str, direction: str = 'long') -> List[Dict]:
        """
        Method 1: Rule-based pattern discovery.
        Tests combinations of features to find high-success patterns.
        
        Args:
            label_col: Label column to predict (e.g., 'Label_3pct_5d')
            direction: 'long' for STRONG_UP, 'short' for STRONG_DOWN
            
        Returns:
            List of discovered patterns
        """
        logger.info(f"Rule-based discovery for {label_col} ({direction})...")
        
        patterns = []
        
        # Get numeric features
        if self.numeric_features is None:
            self._get_numeric_features()
        
        # Limit features for computational efficiency
        # Use top features by variance
        if self.run_mode == 'quick':
            # Quick mode: use fewer features and combinations
            feature_variances = {f: self.data[f].var() for f in self.numeric_features}
            top_features = sorted(feature_variances.items(), key=lambda x: x[1], reverse=True)[:20]
            top_feature_names = [f[0] for f in top_features]
        else:
            # Full mode: use more features
            feature_variances = {f: self.data[f].var() for f in self.numeric_features}
            top_features = sorted(feature_variances.items(), key=lambda x: x[1], reverse=True)[:100]
            top_feature_names = [f[0] for f in top_features]
        
        # Test combinations based on config
        for n_features in self.test_combinations:
            logger.info(f"  Testing {n_features}-feature combinations...")
            
            # Use islice for memory efficiency with large combination counts
            from itertools import islice
            
            # Limit combinations for efficiency
            if self.run_mode == 'quick':
                max_combos = {1: 500, 2: 200, 3: 50}.get(n_features, 20)
            else:
                max_combos = {1: 5000, 2: 5000, 3: 5000, 4: 2000, 5: 500, 6: 50}.get(n_features, 100)
            
            # Use islice to avoid creating full list in memory
            feature_combos = islice(combinations(top_feature_names, n_features), max_combos)
            logger.info(f"    Testing up to {max_combos} combinations")
            
            for features in tqdm(feature_combos, desc=f"{n_features}-feature patterns"):
                # Generate threshold combinations
                pattern = self._find_best_thresholds(features, label_col, direction)
                if pattern and pattern['success_rate'] >= self.min_success_rate:
                    patterns.append(pattern)
        
        # Remove duplicates and keep best patterns
        patterns = self._deduplicate_patterns(patterns)
        
        logger.info(f"  Found {len(patterns)} patterns from rule-based discovery")
        return patterns
    
    def _find_best_thresholds(self, features: Tuple[str, ...], label_col: str, direction: str = 'long') -> Optional[Dict]:
        """
        Find optimal thresholds for a given feature combination.
        
        Args:
            features: Tuple of feature names
            label_col: Label column to predict
            direction: 'long' for STRONG_UP, 'short' for STRONG_DOWN
            
        Returns:
            Pattern dictionary or None
        """
        # Get quantiles for threshold testing
        thresholds_to_test = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        best_pattern = None
        best_score = 0
        
        # Generate threshold combinations
        for thresholds in combinations(thresholds_to_test, len(features)):
            conditions = {}
            for i, feature in enumerate(features):
                # Determine direction based on correlation with success
                corr = self._calculate_feature_success_correlation(feature, label_col, direction)
                
                if corr > 0:
                    # Higher values correlate with success
                    threshold_val = self.data[feature].quantile(thresholds[i])
                    conditions[feature] = {'operator': '>=', 'value': threshold_val}
                else:
                    # Lower values correlate with success
                    threshold_val = self.data[feature].quantile(thresholds[i])
                    conditions[feature] = {'operator': '<=', 'value': threshold_val}
            
            # Evaluate pattern
            pattern = self._evaluate_pattern(conditions, label_col, direction)
            
            if pattern and pattern['success_rate'] > best_score:
                best_score = pattern['success_rate']
                best_pattern = pattern
        
        return best_pattern
    
    def _calculate_feature_success_correlation(self, feature: str, label_col: str, direction: str = 'long') -> float:
        """
        Calculate correlation between feature and success.
        
        Args:
            feature: Feature name
            label_col: Label column name
            direction: 'long' for STRONG_UP, 'short' for STRONG_DOWN
            
        Returns:
            Correlation coefficient
        """
        # Create binary success indicator based on direction
        if direction == 'long':
            success = (self.data[label_col] == 'STRONG_UP').astype(float)
        else:  # short
            success = (self.data[label_col] == 'STRONG_DOWN').astype(float)
        
        # Calculate correlation
        corr = self.data[feature].corr(success)
        
        return corr if not np.isnan(corr) else 0
    
    def _evaluate_pattern(self, conditions: Dict, label_col: str, direction: str = 'long') -> Optional[Dict]:
        """
        Evaluate a pattern's performance.
        
        Args:
            conditions: Dictionary of feature conditions
            label_col: Label column to predict
            direction: 'long' for STRONG_UP, 'short' for STRONG_DOWN
            
        Returns:
            Pattern dictionary with performance metrics
        """
        # Build condition mask
        mask = pd.Series(True, index=self.data.index)
        
        for feature, condition in conditions.items():
            if condition['operator'] == '>=':
                mask &= (self.data[feature] >= condition['value'])
            elif condition['operator'] == '<=':
                mask &= (self.data[feature] <= condition['value'])
            elif condition['operator'] == '>':
                mask &= (self.data[feature] > condition['value'])
            elif condition['operator'] == '<':
                mask &= (self.data[feature] < condition['value'])
            elif condition['operator'] == 'range':
                # Explicit range operator with documented tolerance
                lower = condition.get('lower', condition['center'] * 0.9)
                upper = condition.get('upper', condition['center'] * 1.1)
                mask &= (self.data[feature] >= lower) & (self.data[feature] <= upper)
            elif condition['operator'] == '~':
                # Legacy operator - convert to range for backward compatibility
                mask &= (abs(self.data[feature] - condition['value']) < condition['value'] * 0.1)
        
        # Get occurrences
        occurrences = self.data[mask]
        
        if len(occurrences) < self.min_occurrences:
            return None
        
        # Calculate success rate based on direction
        if direction == 'long':
            success_mask = occurrences[label_col] == 'STRONG_UP'
            opposite_mask = occurrences[label_col] == 'STRONG_DOWN'
        else:  # short
            success_mask = occurrences[label_col] == 'STRONG_DOWN'
            opposite_mask = occurrences[label_col] == 'STRONG_UP'
        
        success_count = success_mask.sum()
        success_rate = success_count / len(occurrences) * 100
        
        # Calculate average move and time
        if success_count > 0:
            # Extract window and threshold from label_col
            parts = label_col.split('_')
            threshold = float(parts[1].replace('pct', ''))
            window = int(parts[2].replace('d', ''))
            
            if direction == 'long':
                max_move_col = f'Max_Up_{window}d'
                time_col = f'Time_To_Max_Up_{window}d'
            else:  # short
                max_move_col = f'Max_Down_{window}d'
                time_col = f'Time_To_Max_Down_{window}d'
            
            avg_move = occurrences.loc[success_mask, max_move_col].mean()
            avg_time = occurrences.loc[success_mask, time_col].mean()
        else:
            avg_move = 0
            avg_time = 0
        
        # Calculate false positive rate (opposite direction)
        false_positive_rate = opposite_mask.sum() / len(occurrences) * 100
        
        # Statistical significance (binomial test)
        if len(occurrences) > 0:
            p_value = stats.binomtest(success_count, len(occurrences), p=0.5, alternative='greater').pvalue
        else:
            p_value = 1.0
        
        pattern = {
            'conditions': conditions,
            'occurrences': len(occurrences),
            'success_count': success_count,
            'success_rate': success_rate,
            'avg_move': avg_move,
            'avg_time': avg_time,
            'false_positive_rate': false_positive_rate,
            'p_value': p_value,
            'label_col': label_col,
            'direction': direction,  # 'long' or 'short'
            'method': 'rule_based'
        }
        
        return pattern
    
    def decision_tree_discovery(self, label_col: str, direction: str = 'long') -> List[Dict]:
        """
        Method 2: Decision tree mining.
        Extracts rules from decision tree branches.
        
        Args:
            label_col: Label column to predict
            direction: 'long' for STRONG_UP, 'short' for STRONG_DOWN
            
        Returns:
            List of discovered patterns
        """
        logger.info(f"Decision tree discovery for {label_col} ({direction})...")
        
        patterns = []
        
        if self.numeric_features is None:
            self._get_numeric_features()
        
        # Prepare data
        X = self.data[self.numeric_features].fillna(0)
        if direction == 'long':
            y = (self.data[label_col] == 'STRONG_UP').astype(int)
        else:  # short
            y = (self.data[label_col] == 'STRONG_DOWN').astype(int)
        
        # Train decision tree
        dt = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=self.min_occurrences,
            random_state=42
        )
        dt.fit(X, y)
        
        # Extract rules from tree
        tree_rules = export_text(dt, feature_names=self.numeric_features)
        
        # Parse rules into patterns
        patterns = self._parse_tree_rules(tree_rules, label_col, direction)
        
        logger.info(f"  Found {len(patterns)} patterns from decision tree")
        return patterns
    
    def _parse_tree_rules(self, tree_rules: str, label_col: str, direction: str = 'long') -> List[Dict]:
        """
        Parse decision tree rules into patterns.
        
        Args:
            tree_rules: Text representation of decision tree
            label_col: Label column name
            direction: 'long' for STRONG_UP, 'short' for STRONG_DOWN
            
        Returns:
            List of patterns
        """
        patterns = []
        lines = tree_rules.split('\n')
        
        # This is a simplified parser - in production, you'd want more sophisticated parsing
        current_conditions = {}
        current_samples = 0
        current_value = 0
        
        for line in lines:
            if '|---' in line:
                # Leaf node or branch
                if 'class:' in line:
                    # Leaf node - extract value
                    parts = line.split('class:')
                    if len(parts) > 1:
                        current_value = float(parts[1].strip())
                        
                        # Calculate success rate
                        success_rate = current_value * 100
                        
                        if success_rate >= self.min_success_rate and current_samples >= self.min_occurrences:
                            pattern = {
                                'conditions': current_conditions.copy(),
                                'occurrences': current_samples,
                                'success_rate': success_rate,
                                'avg_move': 0,  # Would need to calculate from data
                                'avg_time': 0,
                                'false_positive_rate': (1 - current_value) * 100,
                                'p_value': 0,  # Would need to calculate
                                'label_col': label_col,
                                'direction': direction,
                                'method': 'decision_tree'
                            }
                            patterns.append(pattern)
                elif '<=' in line or '>' in line:
                    # Branch - extract condition
                    if '<=' in line:
                        feature, value = line.split('<=')
                        operator = '<='
                    else:
                        feature, value = line.split('>')
                        operator = '>'
                    
                    feature = feature.strip().split('|---')[-1].strip()
                    value = float(value.strip())
                    
                    current_conditions[feature] = {'operator': operator, 'value': value}
            elif 'samples' in line:
                # Extract sample count
                parts = line.split('samples=')
                if len(parts) > 1:
                    current_samples = int(parts[1].strip())
        
        return patterns
    
    def clustering_discovery(self, label_col: str, direction: str = 'long') -> List[Dict]:
        """
        Method 3: Clustering analysis.
        Identifies clusters with consistent directional moves.
        
        Args:
            label_col: Label column to predict
            direction: 'long' for STRONG_UP, 'short' for STRONG_DOWN
            
        Returns:
            List of discovered patterns
        """
        logger.info(f"Clustering discovery for {label_col} ({direction})...")
        
        patterns = []
        
        if self.numeric_features is None:
            self._get_numeric_features()
        
        # Prepare data
        X = self.data[self.numeric_features].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Try different numbers of clusters
        for n_clusters in [5, 10, 15]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Analyze each cluster
            for cluster_id in range(n_clusters):
                cluster_mask = clusters == cluster_id
                cluster_data = self.data[cluster_mask]
                
                if len(cluster_data) < self.min_occurrences:
                    continue
                
                # Calculate success rate based on direction
                if direction == 'long':
                    success_mask = cluster_data[label_col] == 'STRONG_UP'
                else:  # short
                    success_mask = cluster_data[label_col] == 'STRONG_DOWN'
                
                success_count = success_mask.sum()
                success_rate = success_count / len(cluster_data) * 100
                
                if success_rate >= self.min_success_rate:
                    # Get cluster characteristics (centroid)
                    centroid = kmeans.cluster_centers_[cluster_id]
                    
                    # Convert to conditions with explicit range operators
                    conditions = {}
                    for i, feature in enumerate(self.numeric_features):
                        # Use centroid as threshold with explicit range
                        orig_value = scaler.inverse_transform([centroid])[0][i]
                        # Use ±10% range explicitly documented
                        lower_bound = orig_value * 0.9
                        upper_bound = orig_value * 1.1
                        conditions[feature] = {
                            'operator': 'range',
                            'lower': lower_bound,
                            'upper': upper_bound,
                            'center': orig_value,
                            'tolerance_pct': 10.0
                        }
                    
                    # Calculate average move and time (same as rule-based)
                    if success_count > 0:
                        # Extract window and threshold from label_col
                        parts = label_col.split('_')
                        threshold = float(parts[1].replace('pct', ''))
                        window = int(parts[2].replace('d', ''))
                        
                        if direction == 'long':
                            max_move_col = f'Max_Up_{window}d'
                            time_col = f'Time_To_Max_Up_{window}d'
                        else:  # short
                            max_move_col = f'Max_Down_{window}d'
                            time_col = f'Time_To_Max_Down_{window}d'
                        
                        avg_move = cluster_data.loc[success_mask, max_move_col].mean()
                        avg_time = cluster_data.loc[success_mask, time_col].mean()
                    else:
                        avg_move = 0
                        avg_time = 0
                    
                    pattern = {
                        'conditions': conditions,
                        'occurrences': len(cluster_data),
                        'success_count': success_count,
                        'success_rate': success_rate,
                        'avg_move': avg_move,
                        'avg_time': avg_time,
                        'false_positive_rate': (1 - success_rate),
                        'p_value': 0,
                        'label_col': label_col,
                        'direction': direction,
                        'method': 'clustering',
                        'cluster_id': cluster_id
                    }
                    patterns.append(pattern)
        
        logger.info(f"  Found {len(patterns)} patterns from clustering")
        return patterns
    
    def sequential_pattern_discovery(self, label_col: str, direction: str = 'long') -> List[Dict]:
        """
        Method 4: Sequential pattern mining.
        Looks for multi-day sequences that precede big moves.
        
        Args:
            label_col: Label column to predict
            direction: 'long' for STRONG_UP, 'short' for STRONG_DOWN
            
        Returns:
            List of discovered patterns
        """
        logger.info(f"Sequential pattern discovery for {label_col} ({direction})...")
        
        patterns = []
        
        # Skip sequential discovery as it requires features not in the current dataset
        # This method would need additional feature engineering for sequential patterns
        logger.info(f"  Skipping sequential discovery - requires sequential features")
        
        return patterns
    
    def _deduplicate_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """
        Remove duplicate patterns, keeping the best one.
        
        Args:
            patterns: List of patterns
            
        Returns:
            Deduplicated patterns
        """
        if not patterns:
            return []
        
        # Sort by success rate
        patterns.sort(key=lambda x: x['success_rate'], reverse=True)
        
        unique_patterns = []
        seen_conditions = set()
        
        for pattern in patterns:
            # Create hash of conditions
            conditions_str = json.dumps(pattern['conditions'], sort_keys=True)
            conditions_hash = hash(conditions_str)
            
            if conditions_hash not in seen_conditions:
                seen_conditions.add(conditions_hash)
                unique_patterns.append(pattern)
        
        return unique_patterns
    
    def classify_patterns(self, patterns: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Classify patterns by confidence level.
        
        Args:
            patterns: List of patterns to classify
            
        Returns:
            Dictionary with classified patterns
        """
        high_confidence = []
        medium_confidence = []
        low_confidence = []
        rejected = []
        
        for pattern in patterns:
            success_rate = pattern['success_rate']
            occurrences = pattern['occurrences']
            p_value = pattern.get('p_value', 1.0)
            
            # Check statistical significance
            is_significant = p_value < self.p_value_threshold
            
            if success_rate >= self.high_confidence_rate and occurrences >= self.high_confidence_occurrences:
                if is_significant:
                    high_confidence.append(pattern)
                else:
                    medium_confidence.append(pattern)
            elif success_rate >= self.min_success_rate and occurrences >= self.min_occurrences:
                if is_significant:
                    medium_confidence.append(pattern)
                else:
                    low_confidence.append(pattern)
            else:
                rejected.append(pattern)
        
        self.high_confidence_patterns = high_confidence
        self.medium_confidence_patterns = medium_confidence
        self.low_confidence_patterns = low_confidence
        self.rejected_patterns = rejected
        
        logger.info(f"\nPattern Classification:")
        logger.info(f"  HIGH CONFIDENCE: {len(high_confidence)} patterns")
        logger.info(f"  MEDIUM CONFIDENCE: {len(medium_confidence)} patterns")
        logger.info(f"  LOW CONFIDENCE: {len(low_confidence)} patterns")
        logger.info(f"  REJECTED: {len(rejected)} patterns")
        
        return {
            'high': high_confidence,
            'medium': medium_confidence,
            'low': low_confidence,
            'rejected': rejected
        }
    
    def run_phase4(self, data_path: str = None) -> Dict[str, List[Dict]]:
        """
        Run complete Phase 4: Pattern Discovery & Condition Identification.
        
        Args:
            data_path: Path to features matrix CSV file
            
        Returns:
            Dictionary with classified patterns
        """
        logger.info("\n" + "=" * 60)
        logger.info("STARTING PHASE 4: PATTERN DISCOVERY")
        logger.info("=" * 60)
        
        # Load data
        if data_path:
            self.load_data(data_path)
        elif self.data is None:
            # Try default path
            default_path = os.path.join("data", "features_matrix.csv")
            if os.path.exists(default_path):
                self.load_data(default_path)
            else:
                logger.error("No data path provided and default file not found")
                return {}
        
        # Get numeric features
        self._get_numeric_features()
        
        # Detect current market regimes
        current_vol_regime = self.detect_volatility_regime()
        current_trend_regime = self.detect_trend_regime()
        logger.info(f"\nCurrent Market Regimes:")
        logger.info(f"  Volatility: {current_vol_regime}")
        logger.info(f"  Trend: {current_trend_regime}")
        
        # Discover patterns for each label combination and direction
        all_patterns = []
        
        # Determine if discovering by regime
        if self.enable_regime_detection and self.discover_by_regime:
            logger.info("\nRegime-based pattern discovery enabled")
            regime_combinations = self.get_all_regime_combinations()
            logger.info(f"Testing {len(regime_combinations)} regime combinations")
            
            for vol_regime, trend_regime in regime_combinations:
                logger.info(f"\n--- Regime: {vol_regime} + {trend_regime} ---")
                
                # Filter data for this regime
                regime_data = self.filter_data_by_regime(vol_regime, trend_regime)
                
                if len(regime_data) < self.min_regime_samples:
                    logger.info(f"  Skipping: Only {len(regime_data)} samples (minimum: {self.min_regime_samples})")
                    continue
                
                logger.info(f"  Using {len(regime_data)} samples for regime-based discovery")
                
                # Temporarily replace data with regime data
                original_data = self.data
                self.data = regime_data
                
                # Discover patterns for this regime
                regime_patterns = self._discover_patterns_for_data(vol_regime, trend_regime)
                all_patterns.extend(regime_patterns)
                
                # Restore original data
                self.data = original_data
        else:
            logger.info("\nStandard pattern discovery (not regime-based)")
            all_patterns = self._discover_patterns_for_data()
        
        # Deduplicate all patterns
        self.discovered_patterns = self._deduplicate_patterns(all_patterns)
        
        logger.info(f"\nTotal unique patterns discovered: {len(self.discovered_patterns)}")
        
        # Classify patterns
        classified = self.classify_patterns(self.discovered_patterns)
        
        # Save results
        self.save_patterns()
        
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 4 COMPLETE")
        logger.info("=" * 60)
        
        return classified
    
    def _discover_patterns_for_data(self, volatility_regime: str = 'All', trend_regime: str = 'All') -> List[Dict]:
        """
        Discover patterns for the current data in self.data.
        
        Args:
            volatility_regime: Volatility regime name for tagging
            trend_regime: Trend regime name for tagging
            
        Returns:
            List of discovered patterns
        """
        patterns = []
        
        # Use thresholds and windows from config
        # In quick mode, use fewer threshold/window combinations
        if self.run_mode == 'quick':
            key_thresholds = self.thresholds[:2]  # Only first 2 thresholds
            key_windows = self.time_windows[:2]  # Only first 2 windows
            logger.info(f"Quick Mode: Testing thresholds {key_thresholds} and windows {key_windows}")
        else:
            key_thresholds = self.thresholds
            key_windows = self.time_windows
        
        for threshold in key_thresholds:
            for window in key_windows:
                label_col = f'Label_{threshold}pct_{window}d'
                
                if label_col not in self.data.columns:
                    continue
                
                logger.info(f"\nDiscovering patterns for {label_col}...")
                
                # Discover patterns for both long and short directions
                for direction in ['long', 'short']:
                    logger.info(f"  Direction: {direction}")
                    
                    # Method 1: Rule-based
                    rule_patterns = self.rule_based_discovery(label_col, direction)
                    # Tag patterns with regime information
                    for p in rule_patterns:
                        p['volatility_regime'] = volatility_regime
                        p['trend_regime'] = trend_regime
                    patterns.extend(rule_patterns)
                    
                    # Method 2: Decision tree
                    dt_patterns = self.decision_tree_discovery(label_col, direction)
                    for p in dt_patterns:
                        p['volatility_regime'] = volatility_regime
                        p['trend_regime'] = trend_regime
                    patterns.extend(dt_patterns)
                    
                    # Method 3: Clustering (skip in quick mode for speed)
                    if self.run_mode != 'quick':
                        cluster_patterns = self.clustering_discovery(label_col, direction)
                        for p in cluster_patterns:
                            p['volatility_regime'] = volatility_regime
                            p['trend_regime'] = trend_regime
                        patterns.extend(cluster_patterns)
                    
                    # Method 4: Sequential (skip in quick mode)
                    if self.run_mode != 'quick' and len(patterns) < 200:
                        seq_patterns = self.sequential_pattern_discovery(label_col, direction)
                        for p in seq_patterns:
                            p['volatility_regime'] = volatility_regime
                            p['trend_regime'] = trend_regime
                        patterns.extend(seq_patterns)
        
        return patterns
    
    def save_patterns(self, output_dir: str = "data"):
        """
        Save discovered patterns to files.
        
        Args:
            output_dir: Directory to save output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all patterns
        patterns_path = os.path.join(output_dir, "discovered_patterns.json")
        with open(patterns_path, 'w') as f:
            json.dump(self.discovered_patterns, f, indent=2, default=str)
        logger.info(f"All patterns saved to {patterns_path}")
        
        # Save classified patterns
        classified = {
            'high_confidence': self.high_confidence_patterns,
            'medium_confidence': self.medium_confidence_patterns,
            'low_confidence': self.low_confidence_patterns,
            'rejected': self.rejected_patterns
        }
        
        classified_path = os.path.join(output_dir, "classified_patterns.json")
        with open(classified_path, 'w') as f:
            json.dump(classified, f, indent=2, default=str)
        logger.info(f"Classified patterns saved to {classified_path}")


if __name__ == "__main__":
    # Run Phase 4
    pd_obj = PatternDiscovery()
    classified = pd_obj.run_phase4()
    
    print(f"\nFinal Results:")
    print(f"  High Confidence: {len(classified['high'])} patterns")
    print(f"  Medium Confidence: {len(classified['medium'])} patterns")
    print(f"  Low Confidence: {len(classified['low'])} patterns")
    print(f"  Rejected: {len(classified['rejected'])} patterns")