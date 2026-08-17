#!/usr/bin/env python3
"""
Hyperparameter Optimization Module for Gold Price Prediction

This module implements Bayesian optimization using Optuna for:
1. Individual model hyperparameter tuning
2. Ensemble method optimization
3. Feature selection optimization
4. Time series specific validation
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

class HyperparameterOptimizer:
    """
    Bayesian hyperparameter optimization for time series models
    """
    
    def __init__(self, n_trials: int = 100, cv_folds: int = 5, random_state: int = 42):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.study = None
        self.best_params = {}
        self.optimization_history = []
        
    def _time_series_cv_score(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """
        Perform time series cross-validation and return mean R² score
        """
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train and predict
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            
            # Calculate R² score
            score = r2_score(y_val, y_pred)
            scores.append(score)
            
        return np.mean(scores)
    
    def optimize_random_forest(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Optimize Random Forest hyperparameters
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            model = RandomForestRegressor(**params)
            score = self._time_series_cv_score(model, X, y)
            return score
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params
    
    def optimize_xgboost(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            model = xgb.XGBRegressor(**params)
            score = self._time_series_cv_score(model, X, y)
            return score
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params
    
    def optimize_lightgbm(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Optimize LightGBM hyperparameters
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            score = self._time_series_cv_score(model, X, y)
            return score
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params
    
    def optimize_gradient_boosting(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Optimize Gradient Boosting hyperparameters
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.random_state
            }
            
            model = GradientBoostingRegressor(**params)
            score = self._time_series_cv_score(model, X, y)
            return score
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params
    
    def optimize_ridge(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Optimize Ridge regression hyperparameters
        """
        def objective(trial):
            params = {
                'alpha': trial.suggest_float('alpha', 0.001, 100.0, log=True),
                'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
            }
            
            model = Ridge(**params)
            score = self._time_series_cv_score(model, X, y)
            return score
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params
    
    def optimize_lasso(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Optimize Lasso regression hyperparameters
        """
        def objective(trial):
            params = {
                'alpha': trial.suggest_float('alpha', 0.001, 10.0, log=True),
                'max_iter': trial.suggest_int('max_iter', 1000, 5000),
                'selection': trial.suggest_categorical('selection', ['cyclic', 'random'])
            }
            
            model = Lasso(**params)
            score = self._time_series_cv_score(model, X, y)
            return score
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params
    
    def optimize_elastic_net(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Optimize Elastic Net hyperparameters
        """
        def objective(trial):
            params = {
                'alpha': trial.suggest_float('alpha', 0.001, 10.0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9),
                'max_iter': trial.suggest_int('max_iter', 1000, 5000),
                'selection': trial.suggest_categorical('selection', ['cyclic', 'random'])
            }
            
            model = ElasticNet(**params)
            score = self._time_series_cv_score(model, X, y)
            return score
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params
    
    def optimize_svr(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Optimize SVR hyperparameters
        """
        def objective(trial):
            # Choose gamma strategy
            gamma_type = trial.suggest_categorical('gamma_type', ['preset', 'custom'])
            
            params = {
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'epsilon': trial.suggest_float('epsilon', 0.01, 1.0),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
            }
            
            # Handle gamma parameter properly
            if gamma_type == 'preset':
                params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
            else:
                params['gamma'] = trial.suggest_float('gamma', 0.001, 1.0, log=True)
            
            model = SVR(**params)
            score = self._time_series_cv_score(model, X, y)
            return score
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials)
        
        # Clean up the best params to remove gamma_type
        best_params = study.best_params.copy()
        best_params.pop('gamma_type', None)
        
        return best_params
    
    def optimize_feature_selection(self, X: np.ndarray, y: np.ndarray, model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Optimize feature selection parameters
        """
        def objective(trial):
            # Feature selection
            k_features = trial.suggest_int('k_features', 10, min(100, X.shape[1]))
            selector = SelectKBest(score_func=f_regression, k=k_features)
            X_selected = selector.fit_transform(X, y)
            
            # Use a simple model for feature selection optimization
            if model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
            elif model_type == 'xgboost':
                model = xgb.XGBRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
            else:
                model = Ridge(alpha=1.0)
            
            score = self._time_series_cv_score(model, X_selected, y)
            return score
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=min(50, self.n_trials))
        
        return study.best_params
    
    def optimize_all_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Optimize hyperparameters for all models
        """
        print("Starting comprehensive hyperparameter optimization...")
        
        optimizers = {
            'random_forest': self.optimize_random_forest,
            'xgboost': self.optimize_xgboost,
            'lightgbm': self.optimize_lightgbm,
            'gradient_boosting': self.optimize_gradient_boosting,
            'ridge': self.optimize_ridge,
            'lasso': self.optimize_lasso,
            'elastic_net': self.optimize_elastic_net,
            'svr': self.optimize_svr
        }
        
        optimized_params = {}
        
        for model_name, optimizer_func in optimizers.items():
            print(f"\nOptimizing {model_name}...")
            try:
                best_params = optimizer_func(X, y)
                optimized_params[model_name] = best_params
                print(f"Best params for {model_name}: {best_params}")
            except Exception as e:
                print(f"Error optimizing {model_name}: {e}")
                optimized_params[model_name] = {}
        
        # Optimize feature selection
        print("\nOptimizing feature selection...")
        try:
            feature_params = self.optimize_feature_selection(X, y)
            optimized_params['feature_selection'] = feature_params
            print(f"Best feature selection params: {feature_params}")
        except Exception as e:
            print(f"Error optimizing feature selection: {e}")
            optimized_params['feature_selection'] = {'k_features': min(50, X.shape[1])}
        
        self.best_params = optimized_params
        return optimized_params
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get summary of optimization results
        """
        return {
            'n_trials': self.n_trials,
            'cv_folds': self.cv_folds,
            'best_params': self.best_params,
            'optimization_completed': len(self.best_params) > 0
        }