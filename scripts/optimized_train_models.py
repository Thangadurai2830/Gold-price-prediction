#!/usr/bin/env python3
"""
Optimized Gold Price Prediction Training Script

This script combines:
1. Robust feature engineering with temporal constraints
2. Bayesian hyperparameter optimization using Optuna
3. Ensemble methods with optimized parameters
4. Comprehensive model evaluation
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import json
import warnings
from typing import Dict, Any, Tuple
warnings.filterwarnings('ignore')

# Import project modules
from ml.robust_feature_engineering import RobustFeatureEngineer
from ml.hyperparameter_optimization import HyperparameterOptimizer
from ml.ensemble_methods import EnsembleManager
from data.data_collector import GoldDataCollector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import joblib

class OptimizedTrainingPipeline:
    """
    Complete training pipeline with hyperparameter optimization
    """
    
    def __init__(self, optimization_trials: int = 100, cv_folds: int = 5):
        self.feature_engineer = RobustFeatureEngineer()
        self.optimizer = HyperparameterOptimizer(n_trials=optimization_trials, cv_folds=cv_folds)
        self.ensemble_manager = EnsembleManager()
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.optimized_models = {}
        self.results = {}
        
    def load_data(self, use_real_data: bool = True, force_refresh: bool = False) -> pd.DataFrame:
        """
        Load and prepare data
        """
        if use_real_data:
            print("Loading real market data...")
            collector = GoldDataCollector()
            data = collector.fetch_gold_prices_yahoo()
            
            # Standardize column names
            data = data.rename(columns={
                'date': 'Date',
                'open': 'Open', 
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'price': 'Close'
            })
        else:
            print("Generating sample data...")
            collector = GoldDataCollector()
            data = collector.generate_enhanced_sample_data()
            
        # Ensure Date column is datetime
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            
        print(f"Loaded {len(data)} data points")
        return data
    
    def engineer_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Engineer features with temporal constraints
        """
        print("Engineering features with temporal constraints...")
        X, y = self.feature_engineer.prepare_training_data(data, price_col='Close', date_col='Date')
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        return X, y
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Optimize hyperparameters for all models
        """
        print("\n=== Hyperparameter Optimization Phase ===")
        
        # Convert to numpy arrays for optimization
        X_array = X.values
        y_array = y.values
        
        # Optimize all models
        optimized_params = self.optimizer.optimize_all_models(X_array, y_array)
        
        print("\nOptimization completed!")
        return optimized_params
    
    def create_optimized_models(self, optimized_params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create models with optimized hyperparameters
        """
        models = {}
        
        # Linear models
        if 'ridge' in optimized_params and optimized_params['ridge']:
            models['ridge'] = Ridge(**optimized_params['ridge'])
        else:
            models['ridge'] = Ridge(alpha=1.0)
            
        if 'lasso' in optimized_params and optimized_params['lasso']:
            models['lasso'] = Lasso(**optimized_params['lasso'])
        else:
            models['lasso'] = Lasso(alpha=0.1)
            
        if 'elastic_net' in optimized_params and optimized_params['elastic_net']:
            models['elastic_net'] = ElasticNet(**optimized_params['elastic_net'])
        else:
            models['elastic_net'] = ElasticNet(alpha=0.1, l1_ratio=0.5)
        
        # Tree-based models
        if 'random_forest' in optimized_params and optimized_params['random_forest']:
            models['random_forest'] = RandomForestRegressor(**optimized_params['random_forest'])
        else:
            models['random_forest'] = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
        if 'gradient_boosting' in optimized_params and optimized_params['gradient_boosting']:
            models['gradient_boosting'] = GradientBoostingRegressor(**optimized_params['gradient_boosting'])
        else:
            models['gradient_boosting'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        # XGBoost
        if 'xgboost' in optimized_params and optimized_params['xgboost']:
            models['xgboost'] = xgb.XGBRegressor(**optimized_params['xgboost'])
        else:
            models['xgboost'] = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        # LightGBM
        if 'lightgbm' in optimized_params and optimized_params['lightgbm']:
            models['lightgbm'] = lgb.LGBMRegressor(**optimized_params['lightgbm'])
        else:
            models['lightgbm'] = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
        
        # SVR
        if 'svr' in optimized_params and optimized_params['svr']:
            models['svr'] = SVR(**optimized_params['svr'])
        else:
            models['svr'] = SVR(kernel='rbf', C=1.0, gamma='scale')
        
        # Baseline
        models['linear_regression'] = LinearRegression()
        
        return models
    
    def evaluate_models(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluate models with time series cross-validation
        """
        print("\n=== Model Evaluation Phase ===")
        
        # Feature selection
        k_features = 50  # Default
        if hasattr(self.optimizer, 'best_params') and 'feature_selection' in self.optimizer.best_params:
            k_features = self.optimizer.best_params['feature_selection'].get('k_features', 50)
        
        self.feature_selector = SelectKBest(score_func=f_regression, k=min(k_features, X.shape[1]))
        X_selected = pd.DataFrame(
            self.feature_selector.fit_transform(X, y),
            columns=X.columns[self.feature_selector.get_support()],
            index=X.index
        )
        
        print(f"Selected {X_selected.shape[1]} features out of {X.shape[1]}")
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_selected),
            columns=X_selected.columns,
            index=X_selected.index
        )
        
        # Time series split
        train_size = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled.iloc[:train_size], X_scaled.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        results = {}
        
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            
            try:
                # Cross-validation
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = []
                
                for train_idx, val_idx in tscv.split(X_train):
                    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    model.fit(X_cv_train, y_cv_train)
                    y_cv_pred = model.predict(X_cv_val)
                    cv_score = r2_score(y_cv_val, y_cv_pred)
                    cv_scores.append(cv_score)
                
                # Final model training and testing
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                results[name] = {
                    'cv_score': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores),
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': rmse
                }
                
                print(f"  CV R²: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
                print(f"  Test R²: {r2:.4f}")
                print(f"  Test RMSE: {rmse:.2f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                results[name] = {
                    'cv_score': -np.inf,
                    'cv_std': 0,
                    'mse': np.inf,
                    'mae': np.inf,
                    'r2': -np.inf,
                    'rmse': np.inf
                }
        
        return results
    
    def train_optimized_ensembles(self, X: pd.DataFrame, y: pd.Series, base_models: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Train ensemble methods with optimized base models
        """
        print("\n=== Optimized Ensemble Training Phase ===")
        
        # Use the same feature selection and scaling as in model evaluation
        k_features = 50  # Default
        if hasattr(self.optimizer, 'best_params') and 'feature_selection' in self.optimizer.best_params:
            k_features = self.optimizer.best_params['feature_selection'].get('k_features', 50)
        
        # Apply feature selection
        feature_selector = SelectKBest(score_func=f_regression, k=min(k_features, X.shape[1]))
        X_selected = pd.DataFrame(
            feature_selector.fit_transform(X, y),
            columns=X.columns[feature_selector.get_support()],
            index=X.index
        )
        
        # Apply scaling
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_selected),
            columns=X_selected.columns,
            index=X_selected.index
        )
        
        # Convert to numpy arrays
        X_array = X_scaled.values
        y_array = y.values
        
        # Train ensembles using EnsembleManager
        train_size = int(len(X_array) * 0.8)
        X_train, X_test = X_array[:train_size], X_array[train_size:]
        y_train, y_test = y_array[:train_size], y_array[train_size:]
        
        ensemble_results = self.ensemble_manager.evaluate_ensembles(
            X_train, y_train, X_test, y_test
        )
        
        return ensemble_results
    
    def save_results(self, results: Dict[str, Any], optimized_params: Dict[str, Dict[str, Any]], 
                    ensemble_results: Dict[str, Dict[str, float]], timestamp: str):
        """
        Save all results to files
        """
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Combine all results
        combined_results = {
            'timestamp': timestamp,
            'optimization_trials': self.optimizer.n_trials,
            'cv_folds': self.optimizer.cv_folds,
            'optimized_hyperparameters': optimized_params,
            'model_results': results,
            'ensemble_results': ensemble_results,
            'best_individual_model': max(results.items(), key=lambda x: x[1]['r2']),
            'best_ensemble_model': max(ensemble_results.items(), key=lambda x: x[1]['r2']) if ensemble_results else None
        }
        
        # Save to JSON
        results_file = f'results/optimized_training_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(combined_results, f, indent=2, default=str)
        
        # Save models
        models_file = f'results/optimized_models_{timestamp}.joblib'
        joblib.dump({
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_engineer': self.feature_engineer,
            'optimized_params': optimized_params
        }, models_file)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Models saved to: {models_file}")
    
    def run_complete_pipeline(self, use_real_data: bool = True, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Run the complete optimized training pipeline
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print("=== Optimized Gold Price Prediction Training Pipeline ===")
        print(f"Timestamp: {timestamp}")
        print(f"Optimization trials: {self.optimizer.n_trials}")
        print(f"CV folds: {self.optimizer.cv_folds}")
        
        try:
            # 1. Load data
            data = self.load_data(use_real_data, force_refresh)
            
            # 2. Engineer features
            X, y = self.engineer_features(data)
            
            # 3. Optimize hyperparameters
            optimized_params = self.optimize_hyperparameters(X, y)
            
            # 4. Create optimized models
            models = self.create_optimized_models(optimized_params)
            
            # 5. Evaluate models
            results = self.evaluate_models(models, X, y)
            
            # 6. Train optimized ensembles
            ensemble_results = self.train_optimized_ensembles(X, y, models)
            
            # 7. Save results
            self.save_results(results, optimized_params, ensemble_results, timestamp)
            
            # 8. Print summary
            self.print_summary(results, ensemble_results)
            
            return {
                'individual_models': results,
                'ensemble_models': ensemble_results,
                'optimized_params': optimized_params
            }
            
        except Exception as e:
            print(f"\nError in pipeline: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def print_summary(self, results: Dict[str, Dict[str, float]], ensemble_results: Dict[str, Dict[str, float]]):
        """
        Print training summary
        """
        print("\n" + "="*60)
        print("OPTIMIZED TRAINING SUMMARY")
        print("="*60)
        
        # Best individual model
        if results:
            best_individual = max(results.items(), key=lambda x: x[1]['r2'])
            print(f"\nBest Individual Model: {best_individual[0]}")
            print(f"  R²: {best_individual[1]['r2']:.4f}")
            print(f"  RMSE: {best_individual[1]['rmse']:.2f}")
            print(f"  CV R²: {best_individual[1]['cv_score']:.4f}")
        
        # Best ensemble model
        if ensemble_results:
            best_ensemble = max(ensemble_results.items(), key=lambda x: x[1]['r2'])
            print(f"\nBest Ensemble Model: {best_ensemble[0]}")
            print(f"  R²: {best_ensemble[1]['r2']:.4f}")
            print(f"  RMSE: {best_ensemble[1]['rmse']:.2f}")
        
        print("\n" + "="*60)
        print("Optimized training completed successfully!")
        print("="*60)

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Optimized Gold Price Prediction Training')
    parser.add_argument('--real-data', action='store_true', help='Use real market data')
    parser.add_argument('--force-refresh', action='store_true', help='Force refresh of data')
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials per model')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = OptimizedTrainingPipeline(
        optimization_trials=args.trials,
        cv_folds=args.cv_folds
    )
    
    # Run pipeline
    results = pipeline.run_complete_pipeline(
        use_real_data=args.real_data,
        force_refresh=args.force_refresh
    )
    
    return 0 if results else 1

if __name__ == '__main__':
    exit(main())