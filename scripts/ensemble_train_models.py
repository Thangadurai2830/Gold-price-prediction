#!/usr/bin/env python3
"""
Ensemble Training Script for Gold Price Prediction

This script integrates ensemble methods with robust feature engineering
and proper time series validation to create a comprehensive ML pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple

# Import our modules
from ml.robust_feature_engineering import RobustFeatureEngineer
from ml.ensemble_methods import EnsembleManager
from data.data_collector import GoldDataCollector

# ML imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR

import warnings
warnings.filterwarnings('ignore')


class EnsembleTrainingPipeline:
    """
    Complete ensemble training pipeline with robust feature engineering.
    """
    
    def __init__(self, test_size: float = 0.2, cv_folds: int = 5):
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.feature_engineer = RobustFeatureEngineer()
        self.ensemble_manager = EnsembleManager()
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_and_prepare_data(self, use_real_data: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load and prepare data for training.
        """
        collector = GoldDataCollector()
        
        if use_real_data:
            print("Fetching real gold price data...")
            try:
                data = collector.fetch_gold_prices_yahoo(period='5y')
                if data is None or len(data) < 100:
                    print("Failed to fetch real data, using enhanced sample data...")
                    data = collector.generate_enhanced_sample_data()
                else:
                    print(f"Successfully loaded {len(data)} real data points")
            except Exception as e:
                print(f"Error fetching real data: {e}. Using enhanced sample data...")
                data = collector.generate_enhanced_sample_data()
        else:
            print("Generating enhanced sample data...")
            data = collector.generate_enhanced_sample_data()
        
        # Standardize column names to match our feature engineering expectations
        if 'date' in data.columns:
            data = data.rename(columns={'date': 'Date'})
        if 'close' in data.columns:
            data = data.rename(columns={'close': 'Close'})
        if 'open' in data.columns:
            data = data.rename(columns={'open': 'Open'})
        if 'high' in data.columns:
            data = data.rename(columns={'high': 'High'})
        if 'low' in data.columns:
            data = data.rename(columns={'low': 'Low'})
        if 'volume' in data.columns:
            data = data.rename(columns={'volume': 'Volume'})
        
        # Ensure we have the required columns
        required_columns = ['Date', 'Close']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}. Available: {list(data.columns)}")
        
        # Ensure Date column is datetime
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Sort by date and reset index
        data = data.sort_values('Date').reset_index(drop=True)
        
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        print(f"Available columns: {list(data.columns)}")
        
        return data, data['Close'].values
    
    def engineer_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Engineer features using robust temporal constraints.
        """
        print("Engineering features with temporal constraints...")
        
        # Use the prepare_training_data method which handles feature engineering and target creation
        X, y = self.feature_engineer.prepare_training_data(data, price_col='Close', date_col='Date')
        
        print(f"Features shape after engineering: {X.shape}")
        print(f"Feature columns: {list(X.columns)}")
        
        # Convert to numpy arrays
        X_values = X.values
        y_values = y.values
        
        return X_values, y_values
    
    def temporal_train_test_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data temporally (not randomly) to maintain time series structure.
        """
        split_idx = int(len(X) * (1 - self.test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using training data statistics only.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def evaluate_baseline_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate baseline individual models for comparison.
        """
        print("\nEvaluating baseline models...")
        
        baseline_models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
        }
        
        baseline_results = {}
        
        for name, model in baseline_models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                baseline_results[name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': rmse
                }
                
                print(f"{name}: R² = {r2:.4f}, RMSE = {rmse:.2f}")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                baseline_results[name] = {
                    'mse': float('inf'),
                    'mae': float('inf'),
                    'r2': -float('inf'),
                    'rmse': float('inf'),
                    'error': str(e)
                }
        
        return baseline_results
    
    def cross_validate_ensembles(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Perform time series cross-validation on ensemble methods.
        """
        print("\nPerforming time series cross-validation on ensembles...")
        
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        ensemble_models = self.ensemble_manager.create_ensembles()
        
        cv_results = {}
        
        for name, ensemble in ensemble_models.items():
            cv_scores = []
            
            print(f"Cross-validating {name}...")
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                try:
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                    
                    # Scale features for this fold
                    scaler_fold = StandardScaler()
                    X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold)
                    X_val_fold_scaled = scaler_fold.transform(X_val_fold)
                    
                    # Train ensemble
                    ensemble_clone = type(ensemble)(**ensemble.get_params())
                    ensemble_clone.fit(X_train_fold_scaled, y_train_fold)
                    
                    # Predict and score
                    y_pred = ensemble_clone.predict(X_val_fold_scaled)
                    score = r2_score(y_val_fold, y_pred)
                    cv_scores.append(score)
                    
                    print(f"  Fold {fold + 1}: R² = {score:.4f}")
                    
                except Exception as e:
                    print(f"  Fold {fold + 1}: Error - {str(e)}")
                    cv_scores.append(-float('inf'))
            
            cv_results[name] = np.mean(cv_scores) if cv_scores else -float('inf')
            print(f"{name} CV R²: {cv_results[name]:.4f}")
        
        return cv_results
    
    def train_and_evaluate(self, use_real_data: bool = False, 
                          perform_cv: bool = True) -> Dict[str, Any]:
        """
        Complete training and evaluation pipeline.
        """
        print("Starting ensemble training pipeline...")
        
        # Load and prepare data
        data, target = self.load_and_prepare_data(use_real_data)
        
        # Engineer features
        X, y = self.engineer_features(data)
        
        # Split data temporally
        X_train, X_test, y_train, y_test = self.temporal_train_test_split(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Evaluate baseline models
        baseline_results = self.evaluate_baseline_models(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        # Cross-validate ensembles if requested
        cv_results = {}
        if perform_cv:
            cv_results = self.cross_validate_ensembles(X_train_scaled, y_train)
        
        # Train and evaluate ensemble methods
        print("\nTraining ensemble methods...")
        ensemble_results = self.ensemble_manager.evaluate_ensembles(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        # Find best models
        best_baseline = max(baseline_results.items(), 
                           key=lambda x: x[1].get('r2', -float('inf')))
        best_ensemble = self.ensemble_manager.get_best_ensemble(ensemble_results)
        
        # Compile results
        results = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'data_info': {
                'use_real_data': use_real_data,
                'total_samples': len(X),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'n_features': X.shape[1]
            },
            'baseline_models': baseline_results,
            'ensemble_models': ensemble_results,
            'cross_validation': cv_results,
            'best_baseline': {
                'name': best_baseline[0],
                'metrics': best_baseline[1]
            },
            'best_ensemble': {
                'name': best_ensemble[0],
                'metrics': best_ensemble[1]
            }
        }
        
        self.results = results
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """
        Save results to JSON file.
        """
        if filename is None:
            timestamp = results['timestamp']
            filename = f"ensemble_training_results_{timestamp}.json"
        
        filepath = os.path.join('results', filename)
        os.makedirs('results', exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filepath}")
    
    def print_summary(self, results: Dict[str, Any]):
        """
        Print a summary of the training results.
        """
        print("\n" + "="*60)
        print("ENSEMBLE TRAINING SUMMARY")
        print("="*60)
        
        # Data info
        data_info = results['data_info']
        print(f"Data: {'Real' if data_info['use_real_data'] else 'Synthetic'}")
        print(f"Total samples: {data_info['total_samples']}")
        print(f"Features: {data_info['n_features']}")
        print(f"Train/Test split: {data_info['train_samples']}/{data_info['test_samples']}")
        
        # Best baseline
        best_baseline = results['best_baseline']
        print(f"\nBest Baseline Model: {best_baseline['name']}")
        print(f"  R²: {best_baseline['metrics']['r2']:.4f}")
        print(f"  RMSE: {best_baseline['metrics']['rmse']:.2f}")
        
        # Best ensemble
        best_ensemble = results['best_ensemble']
        print(f"\nBest Ensemble Model: {best_ensemble['name']}")
        print(f"  R²: {best_ensemble['metrics']['r2']:.4f}")
        print(f"  RMSE: {best_ensemble['metrics']['rmse']:.2f}")
        
        # Improvement
        improvement = best_ensemble['metrics']['r2'] - best_baseline['metrics']['r2']
        print(f"\nEnsemble Improvement: {improvement:+.4f} R² points")
        
        # Cross-validation results
        if results['cross_validation']:
            print("\nCross-Validation R² Scores:")
            for name, score in results['cross_validation'].items():
                print(f"  {name}: {score:.4f}")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Train ensemble models for gold price prediction')
    parser.add_argument('--real-data', action='store_true', 
                       help='Use real gold price data instead of synthetic')
    parser.add_argument('--no-cv', action='store_true',
                       help='Skip cross-validation (faster)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--cv-folds', type=int, default=3,
                       help='Number of cross-validation folds')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = EnsembleTrainingPipeline(
        test_size=args.test_size,
        cv_folds=args.cv_folds
    )
    
    # Train and evaluate
    results = pipeline.train_and_evaluate(
        use_real_data=args.real_data,
        perform_cv=not args.no_cv
    )
    
    # Print summary
    pipeline.print_summary(results)
    
    # Save results
    pipeline.save_results(results)
    
    print("\nEnsemble training completed successfully!")


if __name__ == "__main__":
    main()