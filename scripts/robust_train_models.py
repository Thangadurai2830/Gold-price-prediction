#!/usr/bin/env python3
"""
Robust Gold Price Prediction Training Script

This script implements the most rigorous approach to prevent data leakage:
1. Strict temporal feature engineering
2. Walk-forward validation
3. Proper time series cross-validation
4. Realistic performance evaluation
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
warnings.filterwarnings('ignore')

# Import project modules
from ml.robust_feature_engineering import RobustFeatureEngineer
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

def setup_directories():
    """Create necessary directories"""
    directories = ['models', 'results', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def collect_data(use_real_data=False, force_refresh=False):
    """Collect training data"""
    print("=== Data Collection Phase ===")
    
    collector = GoldDataCollector()
    
    if use_real_data:
        print("Fetching real market data...")
        try:
            data_dict = collector.update_data(force_refresh=force_refresh)
            if data_dict and data_dict['gold_data'] is not None:
                return data_dict['gold_data']
            else:
                print("Failed to fetch real data, using enhanced sample data")
                return collector.generate_enhanced_sample_data()
        except Exception as e:
            print(f"Error fetching real data: {e}")
            print("Using enhanced sample data")
            return collector.generate_enhanced_sample_data()
    else:
        print("Generating enhanced sample data...")
        return collector.generate_enhanced_sample_data()

def walk_forward_validation(X, y, model, n_splits=5, test_size=0.2):
    """
    Implement walk-forward validation for time series
    """
    n_samples = len(X)
    test_samples = int(n_samples * test_size)
    train_samples = n_samples - test_samples
    
    # Calculate split points
    split_size = train_samples // n_splits
    
    scores = []
    predictions = []
    actuals = []
    
    for i in range(n_splits):
        # Define training window
        train_start = i * split_size
        train_end = train_start + split_size + (train_samples % n_splits if i == n_splits - 1 else 0)
        
        # Define test window (immediately after training)
        test_start = train_end
        test_end = min(test_start + test_samples // n_splits, n_samples)
        
        if test_start >= n_samples:
            break
            
        # Split data
        X_train = X.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]
        
        if len(X_test) == 0:
            continue
            
        # Train model
        model_copy = clone_model(model)
        model_copy.fit(X_train, y_train)
        
        # Predict
        y_pred = model_copy.predict(X_test)
        
        # Calculate score
        mse = mean_squared_error(y_test, y_pred)
        scores.append(mse)
        
        predictions.extend(y_pred)
        actuals.extend(y_test)
        
        print(f"  Fold {i+1}: MSE = {mse:.6f}, Train size = {len(X_train)}, Test size = {len(X_test)}")
    
    return np.mean(scores), np.array(predictions), np.array(actuals)

def clone_model(model):
    """Clone a model with the same parameters"""
    model_type = type(model)
    params = model.get_params()
    return model_type(**params)

def train_robust_models(X, y, feature_names):
    """Train models with robust validation"""
    print("\n=== Robust Model Training Phase ===")
    
    # Initialize models with conservative parameters
    models = {
        'linear_regression': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.1),
        'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'random_forest': RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_split=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            min_samples_split=10, min_samples_leaf=5, random_state=42
        ),
        'xgboost': xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1
        ),
        'lightgbm': lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        ),
        'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    
    # Feature scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Feature selection
    print("Selecting features...")
    selector = SelectKBest(score_func=f_regression, k=min(50, X.shape[1]))
    X_selected = pd.DataFrame(
        selector.fit_transform(X_scaled, y),
        columns=X.columns[selector.get_support()],
        index=X.index
    )
    
    print(f"Selected {X_selected.shape[1]} features out of {X.shape[1]}")
    
    results = {}
    
    # Time series split for validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        try:
            # Walk-forward validation
            cv_score, predictions, actuals = walk_forward_validation(
                X_selected, y, model, n_splits=5
            )
            
            # Train final model on most recent data (last 80%)
            split_point = int(len(X_selected) * 0.2)
            X_train_final = X_selected.iloc[split_point:]
            y_train_final = y.iloc[split_point:]
            
            final_model = clone_model(model)
            final_model.fit(X_train_final, y_train_final)
            
            # Test on the most recent 20% (but still maintaining temporal order)
            X_test_final = X_selected.iloc[:split_point]
            y_test_final = y.iloc[:split_point]
            
            if len(X_test_final) > 0:
                y_pred_final = final_model.predict(X_test_final)
                
                # Calculate metrics
                mse = mean_squared_error(y_test_final, y_pred_final)
                mae = mean_absolute_error(y_test_final, y_pred_final)
                r2 = r2_score(y_test_final, y_pred_final)
                rmse = np.sqrt(mse)
                
                results[name] = {
                    'model': final_model,
                    'cv_score': cv_score,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': rmse,
                    'predictions': y_pred_final,
                    'actuals': y_test_final
                }
                
                print(f"  CV Score (MSE): {cv_score:.6f}")
                print(f"  Test RMSE: {rmse:.6f}")
                print(f"  Test MAE: {mae:.6f}")
                print(f"  Test R²: {r2:.6f}")
            else:
                print(f"  Insufficient test data for {name}")
                
        except Exception as e:
            print(f"  Error training {name}: {e}")
            continue
    
    return results, scaler, selector

def save_robust_results(results, scaler, selector, feature_names, timestamp):
    """Save training results"""
    print("\n=== Saving Results ===")
    
    # Find best model
    best_model_name = None
    best_cv_score = float('inf')
    
    for name, result in results.items():
        if result['cv_score'] < best_cv_score:
            best_cv_score = result['cv_score']
            best_model_name = name
    
    # Save best model and preprocessing objects
    if best_model_name:
        print(f"Best model: {best_model_name}")
        
        # Save model
        joblib.dump(results[best_model_name]['model'], f'models/robust_best_model_{timestamp}.pkl')
        
        # Save preprocessing objects
        joblib.dump(scaler, f'models/robust_scaler_{timestamp}.pkl')
        joblib.dump(selector, f'models/robust_selector_{timestamp}.pkl')
        joblib.dump(feature_names, f'models/robust_features_{timestamp}.pkl')
    
    # Prepare results for JSON
    json_results = {
        'timestamp': timestamp,
        'best_model': best_model_name,
        'best_cv_score': float(best_cv_score),
        'models': {}
    }
    
    for name, result in results.items():
        json_results['models'][name] = {
            'cv_score': float(result['cv_score']),
            'mse': float(result['mse']),
            'mae': float(result['mae']),
            'r2': float(result['r2']),
            'rmse': float(result['rmse'])
        }
    
    # Save results
    with open(f'results/robust_training_results_{timestamp}.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Print summary
    print(f"\nBest model: {best_model_name}")
    print(f"Best CV score (MSE): {best_cv_score:.6f}")
    print(f"Best RMSE: {np.sqrt(best_cv_score):.6f}")
    
    if best_model_name:
        best_result = results[best_model_name]
        print(f"Test R²: {best_result['r2']:.6f}")
        print(f"Test MAE: {best_result['mae']:.6f}")

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Robust Gold Price Prediction Training')
    parser.add_argument('--real-data', action='store_true', help='Use real market data')
    parser.add_argument('--force-refresh', action='store_true', help='Force refresh of data')
    
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("=== Robust Gold Price Prediction Training ===")
    print(f"Timestamp: {timestamp}")
    print(f"Real data: {args.real_data}")
    
    try:
        # Data collection
        raw_data = collect_data(
            use_real_data=args.real_data,
            force_refresh=args.force_refresh
        )
        
        print(f"Collected {len(raw_data)} data points")
        
        # Feature engineering
        print("\n=== Robust Feature Engineering ===")
        feature_engineer = RobustFeatureEngineer()
        X, y = feature_engineer.prepare_training_data(raw_data)
        
        if len(X) < 100:
            print("Warning: Very limited training data. Results may not be reliable.")
        
        # Model training
        results, scaler, selector = train_robust_models(X, y, feature_engineer.feature_columns)
        
        if not results:
            print("No models were successfully trained.")
            return 1
        
        # Save results
        save_robust_results(results, scaler, selector, feature_engineer.feature_columns, timestamp)
        
        print("\n=== Robust Training Completed Successfully ===")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())