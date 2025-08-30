#!/usr/bin/env python3
"""
Ensemble Methods for Gold Price Prediction

This module implements various ensemble methods including:
- Stacking with proper time series validation
- Blending with temporal constraints
- Meta-learning approaches
- Dynamic ensemble weighting

All methods are designed to prevent data leakage in time series contexts.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesStackingRegressor(BaseEstimator, RegressorMixin):
    """
    Stacking regressor designed for time series data.
    Uses proper temporal validation to prevent data leakage.
    """
    
    def __init__(self, base_models: List[BaseEstimator], 
                 meta_model: BaseEstimator = None,
                 cv_folds: int = 5,
                 use_features_in_meta: bool = True,
                 validation_size: float = 0.2):
        """
        Initialize the stacking regressor.
        
        Args:
            base_models: List of base models to use for stacking
            meta_model: Meta-learner model (default: Ridge regression)
            cv_folds: Number of CV folds for generating meta-features
            use_features_in_meta: Whether to include original features in meta-model
            validation_size: Fraction of data to use for validation
        """
        self.base_models = base_models
        self.meta_model = meta_model if meta_model else Ridge(alpha=1.0)
        self.cv_folds = cv_folds
        self.use_features_in_meta = use_features_in_meta
        self.validation_size = validation_size
        self.fitted_base_models_ = []
        self.meta_model_ = None
        self.feature_scaler_ = StandardScaler()
        
    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate meta-features using time series cross-validation.
        """
        n_samples = X.shape[0]
        meta_features = np.zeros((n_samples, len(self.base_models)))
        
        # Use TimeSeriesSplit to maintain temporal order
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold = y[train_idx]
            
            # Train each base model on this fold
            for model_idx, base_model in enumerate(self.base_models):
                model_clone = clone(base_model)
                model_clone.fit(X_train_fold, y_train_fold)
                
                # Generate predictions for validation set
                val_predictions = model_clone.predict(X_val_fold)
                meta_features[val_idx, model_idx] = val_predictions
        
        return meta_features
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TimeSeriesStackingRegressor':
        """
        Fit the stacking regressor.
        """
        X = np.array(X)
        y = np.array(y)
        
        # Split data temporally for meta-model training
        split_idx = int(len(X) * (1 - self.validation_size))
        X_base, X_meta = X[:split_idx], X[split_idx:]
        y_base, y_meta = y[:split_idx], y[split_idx:]
        
        # Generate meta-features for base data
        meta_features_base = self._generate_meta_features(X_base, y_base)
        
        # Train final base models on all base data
        self.fitted_base_models_ = []
        for base_model in self.base_models:
            model_clone = clone(base_model)
            model_clone.fit(X_base, y_base)
            self.fitted_base_models_.append(model_clone)
        
        # Generate meta-features for meta data using fitted base models
        meta_features_meta = np.zeros((len(X_meta), len(self.base_models)))
        for model_idx, fitted_model in enumerate(self.fitted_base_models_):
            meta_features_meta[:, model_idx] = fitted_model.predict(X_meta)
        
        # Prepare meta-model input
        if self.use_features_in_meta:
            # Scale original features
            X_meta_scaled = self.feature_scaler_.fit_transform(X_meta)
            meta_input = np.hstack([meta_features_meta, X_meta_scaled])
        else:
            meta_input = meta_features_meta
        
        # Train meta-model
        self.meta_model_ = clone(self.meta_model)
        self.meta_model_.fit(meta_input, y_meta)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the stacking regressor.
        """
        X = np.array(X)
        
        # Generate base model predictions
        base_predictions = np.zeros((len(X), len(self.fitted_base_models_)))
        for model_idx, fitted_model in enumerate(self.fitted_base_models_):
            base_predictions[:, model_idx] = fitted_model.predict(X)
        
        # Prepare meta-model input
        if self.use_features_in_meta:
            X_scaled = self.feature_scaler_.transform(X)
            meta_input = np.hstack([base_predictions, X_scaled])
        else:
            meta_input = base_predictions
        
        # Make final prediction
        return self.meta_model_.predict(meta_input)


class TimeSeriesBlendingRegressor(BaseEstimator, RegressorMixin):
    """
    Blending regressor for time series data.
    Uses holdout validation to learn optimal weights.
    """
    
    def __init__(self, base_models: List[BaseEstimator],
                 validation_size: float = 0.2,
                 weight_method: str = 'linear'):
        """
        Initialize the blending regressor.
        
        Args:
            base_models: List of base models to blend
            validation_size: Fraction of data to use for learning weights
            weight_method: Method for learning weights ('linear', 'ridge', 'lasso')
        """
        self.base_models = base_models
        self.validation_size = validation_size
        self.weight_method = weight_method
        self.fitted_base_models_ = []
        self.weights_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TimeSeriesBlendingRegressor':
        """
        Fit the blending regressor.
        """
        X = np.array(X)
        y = np.array(y)
        
        # Split data temporally
        split_idx = int(len(X) * (1 - self.validation_size))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train base models
        self.fitted_base_models_ = []
        val_predictions = []
        
        for base_model in self.base_models:
            model_clone = clone(base_model)
            model_clone.fit(X_train, y_train)
            self.fitted_base_models_.append(model_clone)
            
            # Get validation predictions
            val_pred = model_clone.predict(X_val)
            val_predictions.append(val_pred)
        
        # Learn optimal weights
        val_predictions = np.column_stack(val_predictions)
        self._learn_weights(val_predictions, y_val)
        
        return self
    
    def _learn_weights(self, predictions: np.ndarray, y_true: np.ndarray):
        """
        Learn optimal blending weights.
        """
        if self.weight_method == 'linear':
            # Simple linear regression without intercept
            model = LinearRegression(fit_intercept=False, positive=True)
        elif self.weight_method == 'ridge':
            model = Ridge(alpha=1.0, fit_intercept=False, positive=True)
        elif self.weight_method == 'lasso':
            model = Lasso(alpha=0.1, fit_intercept=False, positive=True)
        else:
            raise ValueError(f"Unknown weight method: {self.weight_method}")
        
        model.fit(predictions, y_true)
        self.weights_ = model.coef_
        
        # Normalize weights to sum to 1
        self.weights_ = self.weights_ / np.sum(self.weights_)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the blending regressor.
        """
        X = np.array(X)
        
        # Get predictions from all base models
        predictions = []
        for fitted_model in self.fitted_base_models_:
            pred = fitted_model.predict(X)
            predictions.append(pred)
        
        predictions = np.column_stack(predictions)
        
        # Weighted average
        return np.dot(predictions, self.weights_)


class DynamicEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Dynamic ensemble that adapts weights based on recent performance.
    """
    
    def __init__(self, base_models: List[BaseEstimator],
                 window_size: int = 50,
                 adaptation_rate: float = 0.1):
        """
        Initialize the dynamic ensemble.
        
        Args:
            base_models: List of base models
            window_size: Size of window for performance evaluation
            adaptation_rate: Rate of weight adaptation
        """
        self.base_models = base_models
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        self.fitted_base_models_ = []
        self.weights_ = None
        self.performance_history_ = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DynamicEnsembleRegressor':
        """
        Fit the dynamic ensemble.
        """
        X = np.array(X)
        y = np.array(y)
        
        # Train base models
        self.fitted_base_models_ = []
        for base_model in self.base_models:
            model_clone = clone(base_model)
            model_clone.fit(X, y)
            self.fitted_base_models_.append(model_clone)
        
        # Initialize equal weights
        self.weights_ = np.ones(len(self.base_models)) / len(self.base_models)
        self.performance_history_ = []
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions and update weights if ground truth is available.
        """
        X = np.array(X)
        
        # Get predictions from all base models
        predictions = []
        for fitted_model in self.fitted_base_models_:
            pred = fitted_model.predict(X)
            predictions.append(pred)
        
        predictions = np.column_stack(predictions)
        
        # Weighted average
        return np.dot(predictions, self.weights_)
    
    def update_weights(self, X: np.ndarray, y_true: np.ndarray):
        """
        Update weights based on recent performance.
        """
        # Get individual model predictions
        predictions = []
        for fitted_model in self.fitted_base_models_:
            pred = fitted_model.predict(X)
            predictions.append(pred)
        
        # Calculate individual model errors
        errors = []
        for pred in predictions:
            error = mean_squared_error(y_true, pred)
            errors.append(error)
        
        # Update performance history
        self.performance_history_.append(errors)
        
        # Keep only recent history
        if len(self.performance_history_) > self.window_size:
            self.performance_history_ = self.performance_history_[-self.window_size:]
        
        # Calculate average recent performance
        if len(self.performance_history_) > 1:
            avg_errors = np.mean(self.performance_history_, axis=0)
            
            # Convert errors to weights (inverse relationship)
            inv_errors = 1.0 / (avg_errors + 1e-8)
            new_weights = inv_errors / np.sum(inv_errors)
            
            # Smooth weight update
            self.weights_ = (1 - self.adaptation_rate) * self.weights_ + \
                           self.adaptation_rate * new_weights


class EnsembleManager:
    """
    Manager class for creating and evaluating different ensemble methods.
    """
    
    def __init__(self):
        self.base_models = self._create_base_models()
        self.ensemble_models = {}
        
    def _create_base_models(self) -> List[BaseEstimator]:
        """
        Create a diverse set of base models.
        """
        return [
            LinearRegression(),
            Ridge(alpha=1.0),
            Lasso(alpha=0.1),
            RandomForestRegressor(n_estimators=100, random_state=42),
            GradientBoostingRegressor(n_estimators=100, random_state=42),
            xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1),
            SVR(kernel='rbf', C=1.0)
        ]
    
    def create_ensembles(self) -> Dict[str, BaseEstimator]:
        """
        Create different ensemble models.
        """
        ensembles = {
            'stacking': TimeSeriesStackingRegressor(
                base_models=self.base_models[:6],  # Exclude SVR for speed
                meta_model=Ridge(alpha=1.0),
                cv_folds=3
            ),
            'blending_linear': TimeSeriesBlendingRegressor(
                base_models=self.base_models[:6],
                weight_method='linear'
            ),
            'blending_ridge': TimeSeriesBlendingRegressor(
                base_models=self.base_models[:6],
                weight_method='ridge'
            ),
            'dynamic_ensemble': DynamicEnsembleRegressor(
                base_models=self.base_models[:5],  # Use fewer models for speed
                window_size=30
            )
        }
        
        self.ensemble_models = ensembles
        return ensembles
    
    def evaluate_ensembles(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all ensemble methods.
        """
        if not self.ensemble_models:
            self.create_ensembles()
        
        results = {}
        
        for name, ensemble in self.ensemble_models.items():
            print(f"Training {name}...")
            
            try:
                # Train ensemble
                ensemble.fit(X_train, y_train)
                
                # Make predictions
                y_pred = ensemble.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                results[name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': rmse
                }
                
                print(f"{name} - MSE: {mse:.2f}, R²: {r2:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                results[name] = {
                    'mse': float('inf'),
                    'mae': float('inf'),
                    'r2': -float('inf'),
                    'rmse': float('inf'),
                    'error': str(e)
                }
        
        return results
    
    def get_best_ensemble(self, results: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
        """
        Get the best performing ensemble based on R² score.
        """
        best_name = None
        best_r2 = -float('inf')
        best_results = None
        
        for name, metrics in results.items():
            if 'error' not in metrics and metrics['r2'] > best_r2:
                best_r2 = metrics['r2']
                best_name = name
                best_results = metrics
        
        return best_name, best_results


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    # Generate sample time series data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    
    # Split data temporally (not randomly)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create and evaluate ensembles
    manager = EnsembleManager()
    results = manager.evaluate_ensembles(X_train, y_train, X_test, y_test)
    
    # Print results
    print("\nEnsemble Results:")
    for name, metrics in results.items():
        if 'error' not in metrics:
            print(f"{name}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.2f}")
        else:
            print(f"{name}: Error - {metrics['error']}")
    
    # Get best ensemble
    best_name, best_metrics = manager.get_best_ensemble(results)
    print(f"\nBest ensemble: {best_name} with R² = {best_metrics['r2']:.4f}")