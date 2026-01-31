"""
Machine Learning Pattern Discovery Module
Implements ML-based pattern discovery algorithms for high success rate trading patterns.
"""

import os
import logging
import json
import yaml
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split, TimeSeriesSplit, cross_val_score, 
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, 
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, IsolationForest
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN, KMeans
from scipy import stats
from scipy.signal import find_peaks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PatternMetrics:
    """Metrics for evaluating pattern discovery performance."""
    pattern_id: int
    pattern_name: str
    success_rate: float
    total_occurrences: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    avg_profit_loss: float
    avg_holding_period: float
    sharpe_ratio: float
    max_drawdown: float
    annual_occurrences: float
    meets_criteria: bool  # >80% success rate and 12+ occurrences/year


@dataclass
class MLModelConfig:
    """Configuration for ML model training."""
    model_type: str
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    cv_folds: int = 5
    scoring_metric: str = 'f1'
    n_iter_search: int = 50  # for RandomizedSearchCV
    min_success_rate: float = 0.80
    min_annual_occurrences: int = 12


class MLPatternDiscovery:
    """
    Machine Learning-based pattern discovery system for identifying
    high success rate trading patterns.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the ML pattern discovery system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.model_config = MLModelConfig(model_type='random_forest')
        
        # Data storage
        self.features = None
        self.target = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        
        # Model storage
        self.models = {}
        self.best_model = None
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        
        # Pattern storage
        self.discovered_patterns = []
        self.pattern_metrics = []
        
        # Results storage
        self.training_history = []
        self.evaluation_results = {}
        
        # Hyperparameter grids for different models
        self._setup_hyperparameter_grids()
        
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
            'ml_patterns': {
                'min_success_rate': 0.80,
                'min_annual_occurrences': 12,
                'max_patterns': 50,
                'feature_selection_method': 'mutual_info',
                'n_features_to_select': 50,
                'models': ['random_forest', 'gradient_boosting', 'logistic_regression']
            }
        }
    
    def _setup_hyperparameter_grids(self):
        """Setup hyperparameter grids for model tuning."""
        self.hyperparameter_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced', None]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0]
            },
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['saga', 'liblinear'],
                'class_weight': ['balanced', None],
                'max_iter': [1000, 2000]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'class_weight': ['balanced', None]
            },
            'decision_tree': {
                'max_depth': [3, 5, 7, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced', None]
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 10, 15],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                'p': [1, 2]
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh', 'logistic'],
                'solver': ['adam', 'sgd'],
                'learning_rate': ['constant', 'adaptive'],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'max_iter': [500, 1000],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
    
    def load_features(self, features_path: str = "data/features_matrix.csv") -> pd.DataFrame:
        """
        Load pre-calculated features from file.
        
        Args:
            features_path: Path to features CSV file
            
        Returns:
            DataFrame with features
        """
        logger.info(f"Loading features from {features_path}")
        self.features = pd.read_csv(features_path, index_col='Date', parse_dates=True)
        logger.info(f"Loaded {len(self.features)} records with {len(self.features.columns)} features")
        return self.features
    
    def create_target_variable(self, 
                                profit_threshold: float = 2.0,
                                loss_threshold: float = -1.0,
                                holding_period: int = 5) -> pd.Series:
        """
        Create target variable for pattern discovery based on future returns.
        
        Args:
            profit_threshold: Minimum profit percentage to consider successful
            loss_threshold: Maximum loss percentage to consider unsuccessful
            holding_period: Number of days to hold position
            
        Returns:
            Series with target labels (1=success, 0=failure)
        """
        logger.info("Creating target variable for pattern discovery...")
        
        if self.features is None:
            raise ValueError("No features loaded. Call load_features() first.")
        
        # Calculate future returns
        future_return = self.features['Close'].pct_change(holding_period).shift(-holding_period) * 100
        
        # Create target labels
        target = pd.Series(0, index=self.features.index)
        target[future_return >= profit_threshold] = 1  # Success
        target[future_return <= loss_threshold] = 0   # Failure
        target[(future_return > loss_threshold) & (future_return < profit_threshold)] = -1  # Neutral
        
        # Remove neutral cases (we only want clear success/failure patterns)
        target = target[target != -1]
        
        self.target = target
        
        # Align features with target
        self.features = self.features.loc[target.index]
        
        logger.info(f"Target variable created: {sum(target == 1)} successes, {sum(target == 0)} failures")
        
        return target
    
    def prepare_features_for_ml(self, 
                                 exclude_cols: List[str] = None,
                                 handle_categorical: bool = True) -> pd.DataFrame:
        """
        Prepare features for ML training by handling categorical variables and selecting numeric columns.
        
        Args:
            exclude_cols: Columns to exclude from features
            handle_categorical: Whether to handle categorical variables
            
        Returns:
            DataFrame with prepared features
        """
        logger.info("Preparing features for ML training...")
        
        if self.features is None:
            raise ValueError("No features loaded. Call load_features() first.")
        
        # Default columns to exclude
        if exclude_cols is None:
            exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'TR']
        
        # Get feature columns
        feature_cols = [col for col in self.features.columns 
                       if col not in exclude_cols and not col.endswith('_Name')]
        
        X = self.features[feature_cols].copy()
        
        # Handle categorical variables
        if handle_categorical:
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                logger.info(f"Encoding categorical columns: {categorical_cols}")
                X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Ensure all columns are numeric
        X = X.select_dtypes(include=[np.number])
        
        logger.info(f"Prepared {len(X.columns)} features for ML training")
        
        return X
    
    def feature_selection(self, 
                          X: pd.DataFrame, 
                          y: pd.Series,
                          method: str = 'mutual_info',
                          k: int = 50) -> Tuple[List[str], Any]:
        """
        Perform feature selection using specified method.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Feature selection method ('variance', 'kbest', 'mutual_info', 'rfe', 'model_based')
            k: Number of features to select
            
        Returns:
            Tuple of (selected feature names, fitted selector)
        """
        logger.info(f"Performing feature selection using {method} method...")
        
        # Remove constant features first
        variance_threshold = VarianceThreshold(threshold=0.01)
        X_var = variance_threshold.fit_transform(X)
        var_selected_cols = X.columns[variance_threshold.get_support()]
        
        if len(var_selected_cols) <= k:
            logger.info(f"Variance threshold reduced features to {len(var_selected_cols)}")
            return var_selected_cols.tolist(), variance_threshold
        
        X_var = pd.DataFrame(X_var, columns=var_selected_cols, index=X.index)
        
        if method == 'variance':
            selected_features = var_selected_cols.tolist()
            selector = variance_threshold
            
        elif method == 'kbest':
            selector = SelectKBest(f_classif, k=min(k, len(var_selected_cols)))
            X_selected = selector.fit_transform(X_var, y)
            selected_features = var_selected_cols[selector.get_support()].tolist()
            
        elif method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=min(k, len(var_selected_cols)))
            X_selected = selector.fit_transform(X_var, y)
            selected_features = var_selected_cols[selector.get_support()].tolist()
            
        elif method == 'rfe':
            # Use Random Forest for RFE
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            selector = RFE(rf, n_features_to_select=min(k, len(var_selected_cols)))
            selector.fit(X_var, y)
            selected_features = var_selected_cols[selector.get_support()].tolist()
            
        elif method == 'model_based':
            # Use feature importance from Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_var, y)
            selector = SelectFromModel(rf, max_features=min(k, len(var_selected_cols)))
            selector.fit(X_var, y)
            selected_features = var_selected_cols[selector.get_support()].tolist()
            
        else:
            logger.warning(f"Unknown feature selection method: {method}, using mutual_info")
            selector = SelectKBest(mutual_info_classif, k=min(k, len(var_selected_cols)))
            X_selected = selector.fit_transform(X_var, y)
            selected_features = var_selected_cols[selector.get_support()].tolist()
        
        self.selected_features = selected_features
        self.feature_selector = selector
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features[:10]}...")
        
        return selected_features, selector
    
    def split_data(self, 
                   X: pd.DataFrame, 
                   y: pd.Series,
                   test_size: float = 0.2,
                   validation_size: float = 0.1,
                   time_series: bool = True) -> Tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for test set
            validation_size: Proportion of training data for validation
            time_series: Whether to use time-series split (no shuffling)
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Splitting data into train/validation/test sets...")
        
        if time_series:
            # Time series split - maintain chronological order
            n_samples = len(X)
            test_start = int(n_samples * (1 - test_size))
            val_start = int(test_start * (1 - validation_size))
            
            X_train = X.iloc[:val_start]
            X_val = X.iloc[val_start:test_start]
            X_test = X.iloc[test_start:]
            
            y_train = y.iloc[:val_start]
            y_val = y.iloc[val_start:test_start]
            y_test = y.iloc[test_start:]
        else:
            # Random split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.model_config.random_state, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=validation_size, 
                random_state=self.model_config.random_state, stratify=y_temp
            )
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Train success rate: {y_train.mean():.2%}, Val: {y_val.mean():.2%}, Test: {y_test.mean():.2%}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, 
                       X_train: pd.DataFrame, 
                       X_val: pd.DataFrame = None,
                       X_test: pd.DataFrame = None,
                       method: str = 'standard') -> Tuple:
        """
        Scale features using specified method.
        
        Args:
            X_train: Training features
            X_val: Validation features (optional)
            X_test: Test features (optional)
            method: Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            Tuple of scaled feature arrays
        """
        logger.info(f"Scaling features using {method} method...")
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaling method: {method}, using standard")
            self.scaler = StandardScaler()
        
        # Fit on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Transform validation and test data
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        X_test_scaled = self.scaler.transform(X_test) if X_test is not None else None
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def get_model(self, model_type: str, **kwargs) -> Any:
        """
        Get a model instance of specified type.
        
        Args:
            model_type: Type of model to create
            **kwargs: Additional parameters for model initialization
            
        Returns:
            Model instance
        """
        models = {
            'random_forest': RandomForestClassifier(
                random_state=self.model_config.random_state,
                n_jobs=-1,
                **kwargs
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=self.model_config.random_state,
                **kwargs
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.model_config.random_state,
                **kwargs
            ),
            'svm': SVC(
                random_state=self.model_config.random_state,
                probability=True,
                **kwargs
            ),
            'decision_tree': DecisionTreeClassifier(
                random_state=self.model_config.random_state,
                **kwargs
            ),
            'knn': KNeighborsClassifier(
                n_jobs=-1,
                **kwargs
            ),
            'mlp': MLPClassifier(
                random_state=self.model_config.random_state,
                **kwargs
            ),
            'ada_boost': AdaBoostClassifier(
                random_state=self.model_config.random_state,
                **kwargs
            ),
            'naive_bayes': GaussianNB(**kwargs)
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return models[model_type]
    
    def train_model(self, 
                    model_type: str = 'random_forest',
                    X_train: np.ndarray = None,
                    y_train: pd.Series = None,
                    hyperparameter_tuning: bool = True,
                    cv_folds: int = 5) -> Tuple[Any, Dict]:
        """
        Train a model with optional hyperparameter tuning.
        
        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training labels
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            cv_folds: Number of cross-validation folds
            
        Returns:
            Tuple of (trained model, training history)
        """
        logger.info(f"Training {model_type} model...")
        
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
        
        # Get base model
        model = self.get_model(model_type)
        
        training_history = {
            'model_type': model_type,
            'training_time': None,
            'best_params': None,
            'cv_scores': None
        }
        
        start_time = datetime.now()
        
        if hyperparameter_tuning and model_type in self.hyperparameter_grids:
            logger.info("Performing hyperparameter tuning...")
            param_grid = self.hyperparameter_grids[model_type]
            
            # Use RandomizedSearchCV for efficiency
            search = RandomizedSearchCV(
                model,
                param_distributions=param_grid,
                n_iter=self.model_config.n_iter_search,
                cv=cv_folds,
                scoring=self.model_config.scoring_metric,
                random_state=self.model_config.random_state,
                n_jobs=-1,
                verbose=0
            )
            
            search.fit(X_train, y_train)
            model = search.best_estimator_
            training_history['best_params'] = search.best_params_
            training_history['cv_scores'] = {
                'mean': search.best_score_,
                'std': search.cv_results_['std_test_score'][search.best_index_]
            }
            
            logger.info(f"Best parameters: {search.best_params_}")
            logger.info(f"Best CV score: {search.best_score_:.4f}")
        else:
            model.fit(X_train, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        training_history['training_time'] = training_time
        
        logger.info(f"Model trained in {training_time:.2f} seconds")
        
        # Store model
        self.models[model_type] = model
        
        return model, training_history
    
    def evaluate_model(self, 
                       model: Any,
                       X_test: np.ndarray = None,
                       y_test: pd.Series = None) -> Dict:
        """
        Evaluate model performance on test set.
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model performance...")
        
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
        }
        
        if y_pred_proba is not None:
            metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = {
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1])
        }
        
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test Precision: {metrics['precision']:.4f}")
        logger.info(f"Test Recall: {metrics['recall']:.4f}")
        logger.info(f"Test F1: {metrics['f1']:.4f}")
        if 'auc' in metrics:
            logger.info(f"Test AUC: {metrics['auc']:.4f}")
        
        return metrics
    
    def discover_patterns_ml(self,
                            model_type: str = 'random_forest',
                            feature_selection_method: str = 'mutual_info',
                            n_features: int = 50,
                            min_occurrences: int = 20,
                            probability_threshold: float = 0.7) -> List[Dict]:
        """
        Discover patterns using ML approach.
        
        Args:
            model_type: Type of ML model to use
            feature_selection_method: Feature selection method
            n_features: Number of features to select
            min_occurrences: Minimum pattern occurrences
            probability_threshold: Threshold for pattern confidence
            
        Returns:
            List of discovered patterns
        """
        logger.info("=" * 60)
        logger.info("ML PATTERN DISCOVERY")
        logger.info("=" * 60)
        
        # Prepare features
        X = self.prepare_features_for_ml()
        
        # Feature selection
        selected_features, selector = self.feature_selection(
            X, self.target, method=feature_selection_method, k=n_features
        )
        X_selected = X[selected_features]
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            X_selected, self.target, time_series=True
        )
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test, method='standard'
        )
        
        # Train model
        model, training_history = self.train_model(
            model_type=model_type,
            X_train=X_train_scaled,
            y_train=y_train,
            hyperparameter_tuning=True
        )
        
        # Evaluate model
        metrics = self.evaluate_model(model, X_test_scaled, y_test)
        
        # Get feature importance
        feature_importance = self._get_feature_importance(model, selected_features)
        
        # Discover patterns based on model predictions
        patterns = self._extract_patterns_from_model(
            model, X_selected, self.target, selected_features,
            probability_threshold, min_occurrences
        )
        
        # Store results
        self.evaluation_results = {
            'model_type': model_type,
            'training_history': training_history,
            'test_metrics': metrics,
            'feature_importance': feature_importance,
            'num_patterns': len(patterns)
        }
        
        self.discovered_patterns = patterns
        
        logger.info(f"\nPattern Discovery Complete:")
        logger.info(f"  Model: {model_type}")
        logger.info(f"  Test F1 Score: {metrics['f1']:.4f}")
        logger.info(f"  Patterns Discovered: {len(patterns)}")
        
        return patterns
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """
        Extract feature importance from model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            logger.warning("Model does not have feature importance")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def _extract_patterns_from_model(self,
                                     model: Any,
                                     X: pd.DataFrame,
                                     y: pd.Series,
                                     feature_names: List[str],
                                     prob_threshold: float,
                                     min_occurrences: int) -> List[Dict]:
        """
        Extract patterns from model predictions.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            prob_threshold: Probability threshold for pattern confidence
            min_occurrences: Minimum pattern occurrences
            
        Returns:
            List of discovered patterns
        """
        logger.info("Extracting patterns from model predictions...")
        
        # Get predictions
        X_scaled = self.scaler.transform(X)
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_pred_proba >= prob_threshold).astype(int)
        
        patterns = []
        
        # Find high-confidence predictions
        high_conf_indices = np.where(y_pred_proba >= prob_threshold)[0]
        
        if len(high_conf_indices) == 0:
            logger.warning("No high-confidence predictions found")
            return patterns
        
        # Cluster similar patterns using DBSCAN
        X_high_conf = X.iloc[high_conf_indices]
        
        # Normalize features for clustering
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled_cluster = scaler.fit_transform(X_high_conf)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=2.0, min_samples=min_occurrences//4).fit(X_scaled_cluster)
        labels = clustering.labels_
        
        # Extract patterns for each cluster
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise points
        
        pattern_id = 0
        for label in unique_labels:
            cluster_indices = high_conf_indices[labels == label]
            
            if len(cluster_indices) < min_occurrences:
                continue
            
            # Get cluster characteristics
            cluster_features = X.iloc[cluster_indices]
            cluster_targets = y.iloc[cluster_indices]
            cluster_probs = y_pred_proba[cluster_indices]
            
            # Calculate pattern statistics
            success_rate = cluster_targets.mean()
            avg_prob = cluster_probs.mean()
            
            # Calculate annual occurrences (assuming ~252 trading days)
            data_years = (X.index[-1] - X.index[0]).days / 365.25
            annual_occurrences = len(cluster_indices) / data_years if data_years > 0 else 0
            
            # Check if pattern meets criteria
            meets_criteria = (success_rate >= self.model_config.min_success_rate and
                             annual_occurrences >= self.model_config.min_annual_occurrences)
            
            # Extract pattern conditions (feature ranges)
            conditions = {}
            for feat in feature_names[:20]:  # Top 20 features
                feat_mean = cluster_features[feat].mean()
                feat_std = cluster_features[feat].std()
                conditions[feat] = {
                    'mean': float(feat_mean),
                    'std': float(feat_std),
                    'min': float(cluster_features[feat].min()),
                    'max': float(cluster_features[feat].max())
                }
            
            pattern = {
                'pattern_id': pattern_id,
                'pattern_name': f"ML_Pattern_{pattern_id}_{model.__class__.__name__}",
                'model_type': model.__class__.__name__,
                'success_rate': float(success_rate),
                'total_occurrences': len(cluster_indices),
                'annual_occurrences': float(annual_occurrences),
                'avg_confidence': float(avg_prob),
                'meets_criteria': meets_criteria,
                'conditions': conditions,
                'cluster_label': int(label),
                'occurrence_dates': cluster_indices.tolist()
            }
            
            patterns.append(pattern)
            pattern_id += 1
        
        logger.info(f"Extracted {len(patterns)} patterns from {len(high_conf_indices)} high-confidence predictions")
        
        return patterns
    
    def calculate_pattern_metrics(self, 
                                  patterns: List[Dict],
                                  X: pd.DataFrame,
                                  y: pd.Series) -> List[PatternMetrics]:
        """
        Calculate comprehensive metrics for discovered patterns.
        
        Args:
            patterns: List of discovered patterns
            X: Feature matrix
            y: Target variable
            
        Returns:
            List of PatternMetrics objects
        """
        logger.info("Calculating pattern metrics...")
        
        metrics_list = []
        
        for pattern in patterns:
            pattern_id = pattern['pattern_id']
            pattern_name = pattern['pattern_name']
            
            # Get occurrences
            occurrence_indices = pattern['occurrence_dates']
            
            # Calculate confusion matrix components
            y_true = y.iloc[occurrence_indices]
            y_pred = np.ones(len(y_true))  # Assume pattern predicts success
            
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            
            # Calculate metrics
            success_rate = pattern['success_rate']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate profit/loss metrics
            returns = self.features['Close'].pct_change(5).shift(-5) * 100
            pattern_returns = returns.iloc[occurrence_indices]
            avg_profit_loss = float(pattern_returns.mean())
            
            # Calculate Sharpe ratio
            if len(pattern_returns) > 1:
                sharpe_ratio = float(pattern_returns.mean() / pattern_returns.std() * np.sqrt(252))
            else:
                sharpe_ratio = 0.0
            
            # Calculate max drawdown
            cumulative_returns = (1 + pattern_returns / 100).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = float(drawdown.min())
            
            # Calculate annual occurrences
            data_years = (X.index[-1] - X.index[0]).days / 365.25
            annual_occurrences = pattern['total_occurrences'] / data_years if data_years > 0 else 0
            
            # Check if meets criteria
            meets_criteria = (success_rate >= self.model_config.min_success_rate and
                             annual_occurrences >= self.model_config.min_annual_occurrences)
            
            metrics = PatternMetrics(
                pattern_id=pattern_id,
                pattern_name=pattern_name,
                success_rate=success_rate,
                total_occurrences=pattern['total_occurrences'],
                true_positives=tp,
                false_positives=fp,
                true_negatives=tn,
                false_negatives=fn,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_score=pattern.get('avg_confidence', 0),
                avg_profit_loss=avg_profit_loss,
                avg_holding_period=5.0,  # Default 5-day holding period
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                annual_occurrences=annual_occurrences,
                meets_criteria=meets_criteria
            )
            
            metrics_list.append(metrics)
        
        self.pattern_metrics = metrics_list
        
        logger.info(f"Calculated metrics for {len(metrics_list)} patterns")
        
        return metrics_list
    
    def ensemble_models(self, 
                       model_types: List[str] = None,
                       voting: str = 'soft') -> Tuple[Any, Dict]:
        """
        Create an ensemble of multiple models.
        
        Args:
            model_types: List of model types to include in ensemble
            voting: Voting method ('soft' or 'hard')
            
        Returns:
            Tuple of (ensemble model, evaluation results)
        """
        from sklearn.ensemble import VotingClassifier
        
        if model_types is None:
            model_types = ['random_forest', 'gradient_boosting', 'logistic_regression']
        
        logger.info(f"Creating ensemble model with {voting} voting...")
        
        # Prepare data
        X = self.prepare_features_for_ml()
        selected_features, _ = self.feature_selection(X, self.target, k=50)
        X_selected = X[selected_features]
        
        # Split and scale
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_selected, self.target)
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(X_train, X_val, X_test)
        
        # Train individual models
        estimators = []
        for model_type in model_types:
            model, _ = self.train_model(model_type, X_train_scaled, y_train, hyperparameter_tuning=False)
            estimators.append((model_type, model))
        
        # Create ensemble
        ensemble = VotingClassifier(estimators=estimators, voting=voting)
        ensemble.fit(X_train_scaled, y_train)
        
        # Evaluate
        metrics = self.evaluate_model(ensemble, X_test_scaled, y_test)
        
        self.models['ensemble'] = ensemble
        
        logger.info(f"Ensemble model created with {len(estimators)} base models")
        
        return ensemble, metrics
    
    def cross_validate_model(self,
                            model_type: str = 'random_forest',
                            n_splits: int = 5) -> Dict:
        """
        Perform time series cross-validation on model.
        
        Args:
            model_type: Type of model to validate
            n_splits: Number of cross-validation splits
            
        Returns:
            Dictionary of cross-validation results
        """
        logger.info(f"Performing {n_splits}-fold time series cross-validation...")
        
        # Prepare data
        X = self.prepare_features_for_ml()
        selected_features, _ = self.feature_selection(X, self.target, k=50)
        X_selected = X[selected_features]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        model = self.get_model(model_type)
        
        cv_scores = cross_val_score(
            model, X_scaled, self.target,
            cv=tscv,
            scoring=self.model_config.scoring_metric,
            n_jobs=-1
        )
        
        results = {
            'cv_scores': cv_scores.tolist(),
            'mean_score': float(cv_scores.mean()),
            'std_score': float(cv_scores.std()),
            'min_score': float(cv_scores.min()),
            'max_score': float(cv_scores.max())
        }
        
        logger.info(f"CV Scores: {cv_scores}")
        logger.info(f"Mean: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
        
        return results
    
    def save_models(self, output_dir: str = "data"):
        """
        Save trained models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(output_dir, f"ml_model_{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved to {model_path}")
        
        # Save scaler
        if self.scaler is not None:
            scaler_path = os.path.join(output_dir, "ml_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Scaler saved to {scaler_path}")
        
        # Save feature selector
        if self.feature_selector is not None:
            selector_path = os.path.join(output_dir, "ml_feature_selector.pkl")
            with open(selector_path, 'wb') as f:
                pickle.dump(self.feature_selector, f)
            logger.info(f"Feature selector saved to {selector_path}")
        
        # Save selected features
        if self.selected_features is not None:
            features_path = os.path.join(output_dir, "ml_selected_features.json")
            with open(features_path, 'w') as f:
                json.dump(self.selected_features, f, indent=2)
            logger.info(f"Selected features saved to {features_path}")
    
    def save_patterns(self, output_dir: str = "data"):
        """
        Save discovered patterns to disk.
        
        Args:
            output_dir: Directory to save patterns
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save patterns
        patterns_path = os.path.join(output_dir, "ml_discovered_patterns.json")
        with open(patterns_path, 'w') as f:
            json.dump(self.discovered_patterns, f, indent=2, default=str)
        logger.info(f"Patterns saved to {patterns_path}")
        
        # Save pattern metrics
        metrics_path = os.path.join(output_dir, "ml_pattern_metrics.json")
        metrics_data = [asdict(m) for m in self.pattern_metrics]
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        logger.info(f"Pattern metrics saved to {metrics_path}")
        
        # Save evaluation results
        eval_path = os.path.join(output_dir, "ml_evaluation_results.json")
        with open(eval_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        logger.info(f"Evaluation results saved to {eval_path}")
    
    def generate_pattern_report(self) -> pd.DataFrame:
        """
        Generate a summary report of discovered patterns.
        
        Returns:
            DataFrame with pattern summary
        """
        if not self.pattern_metrics:
            logger.warning("No pattern metrics available")
            return pd.DataFrame()
        
        report_data = []
        for metrics in self.pattern_metrics:
            report_data.append({
                'Pattern ID': metrics.pattern_id,
                'Pattern Name': metrics.pattern_name,
                'Success Rate': f"{metrics.success_rate:.2%}",
                'Total Occurrences': metrics.total_occurrences,
                'Annual Occurrences': f"{metrics.annual_occurrences:.1f}",
                'Precision': f"{metrics.precision:.3f}",
                'Recall': f"{metrics.recall:.3f}",
                'F1 Score': f"{metrics.f1_score:.3f}",
                'Avg Profit/Loss': f"{metrics.avg_profit_loss:.2f}%",
                'Sharpe Ratio': f"{metrics.sharpe_ratio:.2f}",
                'Max Drawdown': f"{metrics.max_drawdown:.2%}",
                'Meets Criteria': 'Yes' if metrics.meets_criteria else 'No'
            })
        
        report_df = pd.DataFrame(report_data)
        
        # Sort by success rate and annual occurrences
        report_df['Success Rate Num'] = report_df['Success Rate'].str.rstrip('%').astype(float)
        report_df['Annual Occ Num'] = report_df['Annual Occurrences'].str.extract(r'([\d.]+)').astype(float)
        report_df = report_df.sort_values(
            ['Meets Criteria', 'Success Rate Num', 'Annual Occ Num'],
            ascending=[False, False, False]
        )
        report_df = report_df.drop(['Success Rate Num', 'Annual Occ Num'], axis=1)
        
        return report_df
    
    def run_full_pipeline(self,
                         model_types: List[str] = None,
                         feature_selection_method: str = 'mutual_info',
                         n_features: int = 50) -> Dict:
        """
        Run the complete ML pattern discovery pipeline.
        
        Args:
            model_types: List of model types to train
            feature_selection_method: Feature selection method
            n_features: Number of features to select
            
        Returns:
            Dictionary with all results
        """
        logger.info("=" * 60)
        logger.info("RUNNING FULL ML PATTERN DISCOVERY PIPELINE")
        logger.info("=" * 60)
        
        if model_types is None:
            model_types = ['random_forest', 'gradient_boosting', 'logistic_regression']
        
        # Load features
        self.load_features()
        
        # Create target variable
        self.create_target_variable()
        
        # Prepare features
        X = self.prepare_features_for_ml()
        
        # Feature selection
        selected_features, _ = self.feature_selection(
            X, self.target, method=feature_selection_method, k=n_features
        )
        X_selected = X[selected_features]
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            X_selected, self.target, time_series=True
        )
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test, method='standard'
        )
        
        # Train multiple models
        all_patterns = []
        all_metrics = []
        
        for model_type in model_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model_type} model")
            logger.info(f"{'='*60}")
            
            # Train model
            model, training_history = self.train_model(
                model_type=model_type,
                X_train=X_train_scaled,
                y_train=y_train,
                hyperparameter_tuning=True
            )
            
            # Evaluate model
            metrics = self.evaluate_model(model, X_test_scaled, y_test)
            
            # Discover patterns
            patterns = self._extract_patterns_from_model(
                model, X_selected, self.target, selected_features,
                prob_threshold=0.7, min_occurrences=20
            )
            
            # Calculate pattern metrics
            pattern_metrics = self.calculate_pattern_metrics(patterns, X_selected, self.target)
            
            all_patterns.extend(patterns)
            all_metrics.extend(pattern_metrics)
        
        # Create ensemble
        if len(model_types) > 1:
            logger.info(f"\n{'='*60}")
            logger.info("Creating ensemble model")
            logger.info(f"{'='*60}")
            ensemble, ensemble_metrics = self.ensemble_models(model_types)
            self.evaluation_results['ensemble'] = ensemble_metrics
        
        # Save results
        self.save_models()
        self.save_patterns()
        
        # Generate report
        report = self.generate_pattern_report()
        
        # Save report
        report_path = os.path.join("data", "ml_pattern_report.csv")
        report.to_csv(report_path, index=False)
        logger.info(f"Pattern report saved to {report_path}")
        
        results = {
            'model_types': model_types,
            'num_patterns': len(all_patterns),
            'num_meeting_criteria': sum(1 for m in all_metrics if m.meets_criteria),
            'patterns': all_patterns,
            'metrics': all_metrics,
            'report': report.to_dict('records'),
            'evaluation_results': self.evaluation_results
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("PIPELINE COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total Patterns Discovered: {len(all_patterns)}")
        logger.info(f"Patterns Meeting Criteria: {results['num_meeting_criteria']}")
        
        return results


def main():
    """Main function to run ML pattern discovery."""
    logger.info("Starting ML Pattern Discovery...")
    
    # Initialize discovery system
    ml_discovery = MLPatternDiscovery()
    
    # Run full pipeline
    results = ml_discovery.run_full_pipeline(
        model_types=['random_forest', 'gradient_boosting'],
        feature_selection_method='mutual_info',
        n_features=50
    )
    
    # Print summary
    print("\n" + "="*60)
    print("ML PATTERN DISCOVERY SUMMARY")
    print("="*60)
    print(f"Total Patterns: {results['num_patterns']}")
    print(f"Meeting Criteria: {results['num_meeting_criteria']}")
    print("\nTop Patterns:")
    print(ml_discovery.generate_pattern_report().to_string(index=False))
    
    return ml_discovery, results


if __name__ == "__main__":
    ml_discovery, results = main()