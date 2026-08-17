#!/usr/bin/env python3
"""
Unified Training Pipeline for Gold Price Prediction

This script consolidates all training approaches into a single, efficient pipeline:
1. Eliminates duplicate code across multiple training scripts
2. Implements advanced ML techniques for maximum accuracy
3. Provides comprehensive model evaluation and comparison
4. Supports both traditional ML and deep learning approaches
5. Includes ensemble methods and hyperparameter optimization

Author: AI Assistant
Date: 2025-01-24
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
from typing import Dict, Any, Tuple, List, Optional
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')

# Import project modules
from ml.models import GoldPricePredictor
from ml.data_processor import DataProcessor
from ml.feature_engineering import FeatureEngineer
from ml.deep_learning_models import TimeSeriesDeepLearning
from ml.ensemble_methods import EnsembleManager
from data.data_collector import GoldDataCollector

# Import ML libraries
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import optuna
from prophet import Prophet

class UnifiedTrainingPipeline:
    """
    Unified training pipeline that consolidates all ML approaches
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the unified training pipeline
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config or self._get_default_config()
        self.data_collector = GoldDataCollector()
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.predictor = GoldPricePredictor()
        self.deep_learning = TimeSeriesDeepLearning()
        self.ensemble_manager = EnsembleManager()
        
        # Results storage
        self.results = {
            'traditional_ml': {},
            'deep_learning': {},
            'ensemble': {},
            'prophet': {},
            'best_models': {},
            'hyperparameters': {},
            'feature_importance': {},
            'validation_scores': {}
        }
        
        # Create results directory
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for training pipeline
        """
        return {
            'data': {
                'use_real_data': True,
                'force_refresh': False,
                'period': '5y',
                'interval': '1d'
            },
            'features': {
                'lag_features': [1, 2, 3, 5, 10, 20, 30],
                'rolling_windows': [5, 10, 20, 50, 100],
                'use_technical_indicators': True,
                'use_economic_indicators': True,
                'feature_selection_k': 50,
                'pca_components': 0.95
            },
            'validation': {
                'cv_folds': 5,
                'test_size': 0.2,
                'walk_forward_steps': 10,
                'validation_method': 'time_series_split'
            },
            'optimization': {
                'n_trials': 100,
                'timeout': 3600,  # 1 hour
                'use_optuna': True,
                'optimize_ensembles': True
            },
            'models': {
                'traditional_ml': True,
                'deep_learning': True,
                'ensemble': True,
                'prophet': True
            },
            'deep_learning': {
                'sequence_length': 30,
                'epochs': 100,
                'batch_size': 32,
                'early_stopping_patience': 10
            }
        }
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare data for training
        
        Returns:
            Tuple of (features, target)
        """
        print("Loading and preparing data...")
        
        if self.config['data']['use_real_data']:
            # Load real market data
            raw_data = self.data_collector.fetch_gold_prices_yahoo(
                period=self.config['data']['period'],
                interval=self.config['data']['interval']
            )
            
            if raw_data is None or raw_data.empty:
                print("Failed to load real data, using sample data...")
                raw_data = self.data_collector.generate_enhanced_sample_data()
        else:
            # Generate sample data
            raw_data = self.data_collector.generate_enhanced_sample_data()
        
        # Process data
        processed_data = self.data_processor.prepare_training_data()
        
        # Engineer features
        print("Engineering features...")
        
        # Create lag features
        for lag in self.config['features']['lag_features']:
            processed_data[f'price_lag_{lag}'] = processed_data['price'].shift(lag)
        
        # Create rolling features
        for window in self.config['features']['rolling_windows']:
            processed_data[f'price_rolling_mean_{window}'] = processed_data['price'].rolling(window).mean()
            processed_data[f'price_rolling_std_{window}'] = processed_data['price'].rolling(window).std()
            processed_data[f'price_rolling_min_{window}'] = processed_data['price'].rolling(window).min()
            processed_data[f'price_rolling_max_{window}'] = processed_data['price'].rolling(window).max()
        
        # Add technical indicators if enabled
        if self.config['features']['use_technical_indicators']:
            processed_data = self._add_technical_indicators(processed_data)
        
        # Add economic indicators if enabled
        if self.config['features']['use_economic_indicators']:
            processed_data = self._add_economic_indicators(processed_data)
        
        # Prepare features and target
        feature_columns = [col for col in processed_data.columns 
                          if col not in ['date', 'price'] and not col.startswith('target')]
        
        X = processed_data[feature_columns].copy()
        y = processed_data['price'].copy()
        
        # Remove rows with NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        print(f"Data shape: {X.shape}, Target shape: {y.shape}")
        print(f"Features: {list(X.columns)}")
        
        return X, y
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataset
        """
        data_copy = data.copy()
        
        # RSI
        delta = data_copy['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data_copy['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data_copy['price'].ewm(span=12).mean()
        exp2 = data_copy['price'].ewm(span=26).mean()
        data_copy['macd'] = exp1 - exp2
        data_copy['macd_signal'] = data_copy['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        rolling_mean = data_copy['price'].rolling(window=20).mean()
        rolling_std = data_copy['price'].rolling(window=20).std()
        data_copy['bb_upper'] = rolling_mean + (rolling_std * 2)
        data_copy['bb_lower'] = rolling_mean - (rolling_std * 2)
        data_copy['bb_width'] = data_copy['bb_upper'] - data_copy['bb_lower']
        
        # Stochastic Oscillator
        low_min = data_copy['price'].rolling(window=14).min()
        high_max = data_copy['price'].rolling(window=14).max()
        data_copy['stoch_k'] = 100 * (data_copy['price'] - low_min) / (high_max - low_min)
        data_copy['stoch_d'] = data_copy['stoch_k'].rolling(window=3).mean()
        
        return data_copy
    
    def _add_economic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add economic indicators (placeholder for now)
        """
        # This would fetch real economic data in production
        # For now, we'll add some synthetic indicators
        data_copy = data.copy()
        
        # Synthetic economic indicators
        np.random.seed(42)
        data_copy['interest_rate'] = np.random.normal(2.5, 0.5, len(data_copy))
        data_copy['inflation_rate'] = np.random.normal(2.0, 0.3, len(data_copy))
        data_copy['dollar_index'] = np.random.normal(100, 5, len(data_copy))
        
        return data_copy
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Optimize hyperparameters using Optuna
        
        Returns:
            Dictionary of optimized hyperparameters for each model
        """
        if not self.config['optimization']['use_optuna']:
            return self._get_default_hyperparameters()
        
        print("Optimizing hyperparameters with Optuna...")
        
        optimized_params = {}
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=self.config['validation']['cv_folds'])
        
        # Models to optimize
        models_to_optimize = {
            'random_forest': self._optimize_random_forest,
            'gradient_boosting': self._optimize_gradient_boosting,
            'xgboost': self._optimize_xgboost,
            'lightgbm': self._optimize_lightgbm,
            'svr': self._optimize_svr,
            'ridge': self._optimize_ridge,
            'lasso': self._optimize_lasso,
            'elastic_net': self._optimize_elastic_net
        }
        
        for model_name, optimize_func in models_to_optimize.items():
            print(f"Optimizing {model_name}...")
            
            study = optuna.create_study(direction='minimize')
            study.optimize(
                lambda trial: optimize_func(trial, X, y, tscv),
                n_trials=self.config['optimization']['n_trials'],
                timeout=self.config['optimization']['timeout'] // len(models_to_optimize)
            )
            
            optimized_params[model_name] = study.best_params
            print(f"Best {model_name} params: {study.best_params}")
        
        self.results['hyperparameters'] = optimized_params
        return optimized_params
    
    def _optimize_random_forest(self, trial, X, y, tscv):
        """Optimize Random Forest hyperparameters"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }
        
        model = RandomForestRegressor(**params, random_state=42)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        return -scores.mean()
    
    def _optimize_gradient_boosting(self, trial, X, y, tscv):
        """Optimize Gradient Boosting hyperparameters"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
        
        model = GradientBoostingRegressor(**params, random_state=42)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        return -scores.mean()
    
    def _optimize_xgboost(self, trial, X, y, tscv):
        """Optimize XGBoost hyperparameters"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
        }
        
        model = xgb.XGBRegressor(**params, random_state=42)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        return -scores.mean()
    
    def _optimize_lightgbm(self, trial, X, y, tscv):
        """Optimize LightGBM hyperparameters"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
        }
        
        model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        return -scores.mean()
    
    def _optimize_svr(self, trial, X, y, tscv):
        """Optimize SVR hyperparameters"""
        params = {
            'C': trial.suggest_float('C', 0.1, 100, log=True),
            'epsilon': trial.suggest_float('epsilon', 0.01, 1.0),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
        }
        
        model = SVR(**params)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        return -scores.mean()
    
    def _optimize_ridge(self, trial, X, y, tscv):
        """Optimize Ridge regression hyperparameters"""
        params = {
            'alpha': trial.suggest_float('alpha', 0.1, 100, log=True)
        }
        
        model = Ridge(**params)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        return -scores.mean()
    
    def _optimize_lasso(self, trial, X, y, tscv):
        """Optimize Lasso regression hyperparameters"""
        params = {
            'alpha': trial.suggest_float('alpha', 0.001, 10, log=True)
        }
        
        model = Lasso(**params)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        return -scores.mean()
    
    def _optimize_elastic_net(self, trial, X, y, tscv):
        """Optimize Elastic Net hyperparameters"""
        params = {
            'alpha': trial.suggest_float('alpha', 0.001, 10, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9)
        }
        
        model = ElasticNet(**params)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        return -scores.mean()
    
    def _get_default_hyperparameters(self) -> Dict[str, Dict[str, Any]]:
        """Get default hyperparameters if optimization is disabled"""
        return {
            'random_forest': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
            'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'random_state': 42},
            'xgboost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'random_state': 42},
            'lightgbm': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'random_state': 42, 'verbose': -1},
            'svr': {'C': 1.0, 'epsilon': 0.1, 'gamma': 'scale'},
            'ridge': {'alpha': 1.0},
            'lasso': {'alpha': 1.0},
            'elastic_net': {'alpha': 1.0, 'l1_ratio': 0.5}
        }
    
    def train_traditional_ml_models(self, X: pd.DataFrame, y: pd.Series, 
                                   optimized_params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train traditional ML models with optimized hyperparameters
        
        Returns:
            Dictionary of trained models and their performance metrics
        """
        print("Training traditional ML models...")
        
        models = {
            'random_forest': RandomForestRegressor(**optimized_params['random_forest']),
            'gradient_boosting': GradientBoostingRegressor(**optimized_params['gradient_boosting']),
            'xgboost': xgb.XGBRegressor(**optimized_params['xgboost']),
            'lightgbm': lgb.LGBMRegressor(**optimized_params['lightgbm']),
            'svr': SVR(**optimized_params['svr']),
            'ridge': Ridge(**optimized_params['ridge']),
            'lasso': Lasso(**optimized_params['lasso']),
            'elastic_net': ElasticNet(**optimized_params['elastic_net']),
            'linear_regression': LinearRegression()
        }
        
        # Time series split for evaluation
        tscv = TimeSeriesSplit(n_splits=self.config['validation']['cv_folds'])
        
        trained_models = {}
        model_results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
            
            # Train on full dataset
            model.fit(X, y)
            
            # Predictions
            y_pred = model.predict(X)
            
            # Calculate metrics
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            trained_models[name] = model
            model_results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_score_mean': -cv_scores.mean(),
                'cv_score_std': cv_scores.std()
            }
            
            print(f"{name} - R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
        self.results['traditional_ml'] = {
            'models': trained_models,
            'results': model_results
        }
        
        return trained_models
    
    def train_deep_learning_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train deep learning models
        
        Returns:
            Dictionary of trained deep learning models and results
        """
        if not self.config['models']['deep_learning']:
            return {}
        
        print("Training deep learning models...")
        
        # Prepare data for deep learning
        X_scaled = StandardScaler().fit_transform(X)
        y_scaled = StandardScaler().fit_transform(y.values.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_seq, y_seq = self.deep_learning.prepare_sequences(X_scaled, y_scaled)
        
        # Split data
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Train models
        dl_results = self.deep_learning.train_all_models(
            X_train, y_train, X_val, y_val,
            epochs=self.config['deep_learning']['epochs'],
            batch_size=self.config['deep_learning']['batch_size']
        )
        
        self.results['deep_learning'] = dl_results
        return dl_results
    
    def train_ensemble_models(self, X: pd.DataFrame, y: pd.Series, 
                            base_models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train ensemble models
        
        Returns:
            Dictionary of trained ensemble models and results
        """
        if not self.config['models']['ensemble']:
            return {}
        
        print("Training ensemble models...")
        
        # Create ensemble models
        ensembles = self.ensemble_manager.create_ensembles()
        
        # Evaluate ensembles
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        ensemble_results = self.ensemble_manager.evaluate_ensembles(
            X_train.values, y_train.values, X_test.values, y_test.values
        )
        
        self.results['ensemble'] = ensemble_results
        return ensemble_results
    
    def train_prophet_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train Prophet model for time series forecasting
        
        Returns:
            Dictionary with Prophet model and results
        """
        if not self.config['models']['prophet']:
            return {}
        
        print("Training Prophet model...")
        
        try:
            # Prepare data for Prophet
            prophet_data = data[['date', 'price']].copy()
            prophet_data.columns = ['ds', 'y']
            prophet_data = prophet_data.dropna()
            
            # Create and train Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            model.fit(prophet_data)
            
            # Make predictions
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            
            # Calculate metrics on training data
            y_true = prophet_data['y'].values
            y_pred = forecast['yhat'][:len(y_true)].values
            
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            prophet_results = {
                'model': model,
                'forecast': forecast,
                'metrics': {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
            }
            
            self.results['prophet'] = prophet_results
            print(f"Prophet - R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
            
            return prophet_results
            
        except Exception as e:
            print(f"Error training Prophet model: {e}")
            return {}
    
    def identify_best_models(self) -> Dict[str, Any]:
        """
        Identify the best performing models across all categories
        
        Returns:
            Dictionary with best models and their metrics
        """
        print("Identifying best models...")
        
        best_models = {
            'best_traditional_ml': None,
            'best_deep_learning': None,
            'best_ensemble': None,
            'best_overall': None
        }
        
        # Find best traditional ML model
        if 'traditional_ml' in self.results and self.results['traditional_ml']:
            ml_results = self.results['traditional_ml']['results']
            best_ml_name = max(ml_results.keys(), key=lambda k: ml_results[k]['r2'])
            best_models['best_traditional_ml'] = {
                'name': best_ml_name,
                'metrics': ml_results[best_ml_name],
                'model': self.results['traditional_ml']['models'][best_ml_name]
            }
        
        # Find best deep learning model
        if 'deep_learning' in self.results and self.results['deep_learning']:
            dl_results = self.results['deep_learning']
            if dl_results:
                best_dl_name, best_dl_metrics = self.deep_learning.get_best_model(dl_results)
                best_models['best_deep_learning'] = {
                    'name': best_dl_name,
                    'metrics': best_dl_metrics
                }
        
        # Find best ensemble model
        if 'ensemble' in self.results and self.results['ensemble']:
            ensemble_results = self.results['ensemble']
            if ensemble_results:
                best_ensemble_name, best_ensemble_metrics = self.ensemble_manager.get_best_ensemble(ensemble_results)
                best_models['best_ensemble'] = {
                    'name': best_ensemble_name,
                    'metrics': best_ensemble_metrics
                }
        
        # Find overall best model
        all_models = []
        
        if best_models['best_traditional_ml']:
            all_models.append(('traditional_ml', best_models['best_traditional_ml']))
        
        if best_models['best_deep_learning']:
            all_models.append(('deep_learning', best_models['best_deep_learning']))
        
        if best_models['best_ensemble']:
            all_models.append(('ensemble', best_models['best_ensemble']))
        
        if 'prophet' in self.results and self.results['prophet']:
            all_models.append(('prophet', {
                'name': 'prophet',
                'metrics': self.results['prophet']['metrics']
            }))
        
        if all_models:
            best_overall = max(all_models, key=lambda x: x[1]['metrics'].get('r2', -np.inf))
            best_models['best_overall'] = {
                'category': best_overall[0],
                'name': best_overall[1]['name'],
                'metrics': best_overall[1]['metrics']
            }
        
        self.results['best_models'] = best_models
        return best_models
    
    def save_models_and_results(self, timestamp: str) -> None:
        """
        Save trained models and results
        
        Args:
            timestamp: Timestamp string for file naming
        """
        print("Saving models and results...")
        
        # Save traditional ML models
        if 'traditional_ml' in self.results and self.results['traditional_ml']:
            models_dir = Path('models')
            models_dir.mkdir(exist_ok=True)
            
            for name, model in self.results['traditional_ml']['models'].items():
                model_path = models_dir / f"{name}_model_{timestamp}.joblib"
                joblib.dump(model, model_path)
        
        # Save deep learning models
        if 'deep_learning' in self.results and self.results['deep_learning']:
            self.deep_learning.save_models(f"models/deep_learning_{timestamp}")
        
        # Save results
        results_file = self.results_dir / f"unified_training_results_{timestamp}.json"
        
        # Convert non-serializable objects
        serializable_results = self._make_serializable(self.results.copy())
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"Results saved to {results_file}")
    
    def _make_serializable(self, obj):
        """
        Convert non-serializable objects to serializable format
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items() 
                   if k not in ['models', 'model']}  # Skip model objects
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def print_summary(self) -> None:
        """
        Print a comprehensive summary of training results
        """
        print("\n" + "="*80)
        print("UNIFIED TRAINING PIPELINE SUMMARY")
        print("="*80)
        
        # Traditional ML results
        if 'traditional_ml' in self.results and self.results['traditional_ml']:
            print("\nTraditional ML Models:")
            print("-" * 40)
            for name, metrics in self.results['traditional_ml']['results'].items():
                print(f"{name:20} | R²: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f}")
        
        # Deep learning results
        if 'deep_learning' in self.results and self.results['deep_learning']:
            print("\nDeep Learning Models:")
            print("-" * 40)
            for name, metrics in self.results['deep_learning'].items():
                if isinstance(metrics, dict) and 'test_r2' in metrics:
                    print(f"{name:20} | R²: {metrics['test_r2']:.4f} | RMSE: {metrics['test_rmse']:.2f}")
        
        # Ensemble results
        if 'ensemble' in self.results and self.results['ensemble']:
            print("\nEnsemble Models:")
            print("-" * 40)
            for name, metrics in self.results['ensemble'].items():
                if isinstance(metrics, dict) and 'r2' in metrics:
                    print(f"{name:20} | R²: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.2f}")
        
        # Prophet results
        if 'prophet' in self.results and self.results['prophet']:
            print("\nProphet Model:")
            print("-" * 40)
            metrics = self.results['prophet']['metrics']
            print(f"{'Prophet':20} | R²: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f}")
        
        # Best models
        if 'best_models' in self.results and self.results['best_models']:
            print("\nBest Models:")
            print("-" * 40)
            best_models = self.results['best_models']
            
            if best_models['best_overall']:
                best = best_models['best_overall']
                print(f"Overall Best: {best['name']} ({best['category']}) | R²: {best['metrics']['r2']:.4f}")
        
        print("\n" + "="*80)
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete unified training pipeline
        
        Returns:
            Complete results dictionary
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"Starting Unified Training Pipeline - {timestamp}")
        print("="*80)
        
        try:
            # Load and prepare data
            X, y = self.load_and_prepare_data()
            
            # Optimize hyperparameters
            optimized_params = self.optimize_hyperparameters(X, y)
            
            # Train traditional ML models
            traditional_models = self.train_traditional_ml_models(X, y, optimized_params)
            
            # Train deep learning models
            if self.config['models']['deep_learning']:
                self.train_deep_learning_models(X, y)
            
            # Train ensemble models
            if self.config['models']['ensemble']:
                self.train_ensemble_models(X, y, traditional_models)
            
            # Train Prophet model
            if self.config['models']['prophet']:
                # Need original data with date column for Prophet
                raw_data = self.data_collector.fetch_gold_prices_yahoo() if self.config['data']['use_real_data'] else self.data_collector.generate_enhanced_sample_data()
                self.train_prophet_model(raw_data)
            
            # Identify best models
            self.identify_best_models()
            
            # Save models and results
            self.save_models_and_results(timestamp)
            
            # Print summary
            self.print_summary()
            
            print(f"\nPipeline completed successfully! Results saved with timestamp: {timestamp}")
            
            return self.results
            
        except Exception as e:
            print(f"Error in training pipeline: {e}")
            import traceback
            traceback.print_exc()
            return {}

def main():
    """
    Main function to run the unified training pipeline
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Gold Price Prediction Training Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--real-data', action='store_true', help='Use real market data')
    parser.add_argument('--no-optimization', action='store_true', help='Skip hyperparameter optimization')
    parser.add_argument('--no-deep-learning', action='store_true', help='Skip deep learning models')
    parser.add_argument('--no-ensemble', action='store_true', help='Skip ensemble models')
    parser.add_argument('--no-prophet', action='store_true', help='Skip Prophet model')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Create pipeline
    pipeline = UnifiedTrainingPipeline(config)
    
    # Update config based on command line arguments
    if args.real_data:
        pipeline.config['data']['use_real_data'] = True
    
    if args.no_optimization:
        pipeline.config['optimization']['use_optuna'] = False
    
    if args.no_deep_learning:
        pipeline.config['models']['deep_learning'] = False
    
    if args.no_ensemble:
        pipeline.config['models']['ensemble'] = False
    
    if args.no_prophet:
        pipeline.config['models']['prophet'] = False
    
    # Run pipeline
    results = pipeline.run_complete_pipeline()
    
    if results:
        print("\nTraining completed successfully!")
        return 0
    else:
        print("\nTraining failed!")
        return 1

if __name__ == '__main__':
    exit(main())