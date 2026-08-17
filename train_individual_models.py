#!/usr/bin/env python3
"""
Train individual ML models and save them with correct naming convention
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_collector import GoldDataCollector
from ml.data_processor import DataProcessor

def train_individual_models():
    """
    Train individual ML models and save them with correct naming convention
    """
    print("Starting individual model training...")
    
    # Initialize components
    data_collector = GoldDataCollector()
    processor = DataProcessor()
    
    # Generate and prepare data
    print("Generating sample data...")
    data = data_collector.generate_enhanced_sample_data(days=500)
    
    print("Processing data...")
    processor.raw_data = data
    processed_data = processor.prepare_training_data()
    
    if len(processed_data) < 10:
        print(f"Warning: Only {len(processed_data)} records available for training")
        return
    
    print(f"Training data shape: {processed_data.shape}")
    
    # Prepare features and target
    feature_columns = [col for col in processed_data.columns if col not in ['price', 'date']]
    X = processed_data[feature_columns]
    y = processed_data['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models to train
    models = {
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'extra_trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'linear_regression': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=1.0),
        'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5),
        'bayesian_ridge': BayesianRidge(),
        'huber': HuberRegressor(),
        'svr_rbf': SVR(kernel='rbf', C=1.0),
        'svr_linear': SVR(kernel='linear', C=1.0),
        'svr_poly': SVR(kernel='poly', C=1.0, degree=3)
    }
    
    # Train and save models
    os.makedirs('models', exist_ok=True)
    results = {}
    
    for name, model in models.items():
        try:
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Save model
            model_path = os.path.join('models', f'{name}_model.pkl')
            joblib.dump(model, model_path)
            
            results[name] = {
                'mse': mse,
                'r2': r2,
                'model_path': model_path
            }
            
            print(f"✓ {name}: MSE={mse:.4f}, R2={r2:.4f}")
            
        except Exception as e:
            print(f"✗ Failed to train {name}: {e}")
            continue
    
    # Save feature columns
    feature_path = os.path.join('models', 'feature_columns.joblib')
    joblib.dump(feature_columns, feature_path)
    print(f"✓ Feature columns saved to {feature_path}")
    
    # Try to import and train additional models
    try:
        import xgboost as xgb
        print("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        model_path = os.path.join('models', 'xgboost_model.pkl')
        joblib.dump(xgb_model, model_path)
        results['xgboost'] = {'mse': mse, 'r2': r2, 'model_path': model_path}
        print(f"✓ xgboost: MSE={mse:.4f}, R2={r2:.4f}")
    except ImportError:
        print("⚠ XGBoost not available")
    except Exception as e:
        print(f"✗ Failed to train XGBoost: {e}")
    
    try:
        import lightgbm as lgb
        print("Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        lgb_model.fit(X_train, y_train)
        y_pred = lgb_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        model_path = os.path.join('models', 'lightgbm_model.pkl')
        joblib.dump(lgb_model, model_path)
        results['lightgbm'] = {'mse': mse, 'r2': r2, 'model_path': model_path}
        print(f"✓ lightgbm: MSE={mse:.4f}, R2={r2:.4f}")
    except ImportError:
        print("⚠ LightGBM not available")
    except Exception as e:
        print(f"✗ Failed to train LightGBM: {e}")
    
    try:
        import catboost as cb
        print("Training CatBoost...")
        cb_model = cb.CatBoostRegressor(iterations=100, random_state=42, verbose=False)
        cb_model.fit(X_train, y_train)
        y_pred = cb_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        model_path = os.path.join('models', 'catboost_model.pkl')
        joblib.dump(cb_model, model_path)
        results['catboost'] = {'mse': mse, 'r2': r2, 'model_path': model_path}
        print(f"✓ catboost: MSE={mse:.4f}, R2={r2:.4f}")
    except ImportError:
        print("⚠ CatBoost not available")
    except Exception as e:
        print(f"✗ Failed to train CatBoost: {e}")
    
    print(f"\n=== Training Complete ===")
    print(f"Successfully trained {len(results)} models")
    print(f"Models saved to: models/")
    
    return results

if __name__ == "__main__":
    results = train_individual_models()
    print("Individual model training completed!")