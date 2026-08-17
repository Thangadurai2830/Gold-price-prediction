#!/usr/bin/env python3
"""
Enhanced Gold Price Prediction Model Training Script

This script implements state-of-the-art ML techniques for maximum accuracy:
1. Proper time series validation to prevent data leakage
2. Advanced feature engineering with temporal constraints
3. Hyperparameter optimization using Optuna
4. Deep learning models (LSTM, GRU)
5. Ensemble methods with proper validation
6. Walk-forward analysis
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
from ml.models import GoldPricePredictor
from ml.data_processor import DataProcessor
from ml.feature_engineering import FeatureEngineer
from data.data_collector import GoldDataCollector

def setup_directories():
    """Create necessary directories for models and results"""
    directories = ['models', 'results', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def collect_enhanced_data(use_real_data=True, force_refresh=False):
    """Collect and prepare enhanced training data with external factors"""
    print("=== Enhanced Data Collection Phase ===")
    
    collector = GoldDataCollector()
    
    if use_real_data:
        print("Fetching real market data with external factors...")
        try:
            # Update data from real sources
            data_dict = collector.update_data(force_refresh=force_refresh)
            
            if data_dict and data_dict['gold_data'] is not None:
                gold_data = data_dict['gold_data']
                economic_data = data_dict.get('economic_data')
                
                print(f"Collected {len(gold_data)} gold price records")
                if economic_data is not None:
                    print(f"Collected economic indicators: {len(economic_data)} records")
                
                return gold_data, economic_data
            else:
                print("Failed to fetch real data, falling back to enhanced sample data")
                return collector.generate_enhanced_sample_data(), None
                
        except Exception as e:
            print(f"Error fetching real data: {e}")
            print("Falling back to enhanced sample data")
            return collector.generate_enhanced_sample_data(), None
    else:
        print("Generating enhanced sample data for training...")
        return collector.generate_enhanced_sample_data(), None

def preprocess_enhanced_data(gold_data, economic_data=None):
    """Preprocess and engineer features with advanced techniques"""
    print("\n=== Enhanced Data Preprocessing Phase ===")
    
    # Initialize data processor
    processor = DataProcessor()
    processor.raw_data = gold_data
    
    if economic_data is not None:
        processor.external_factors = economic_data
    
    # Prepare training data with external factors
    processed_data = processor.prepare_training_data(use_external_factors=(economic_data is not None))
    
    print(f"Preprocessed data shape: {processed_data.shape}")
    
    # Advanced feature engineering
    print("\n=== Advanced Feature Engineering Phase ===")
    feature_engineer = FeatureEngineer()
    
    # Apply comprehensive feature engineering
    engineered_data = feature_engineer.engineer_all_features(
        processed_data, 
        target_column='price',
        use_pca=False,
        n_features=100  # Increased feature count for better accuracy
    )
    
    print(f"Final feature set shape: {engineered_data.shape}")
    print(f"Selected features: {len(feature_engineer.selected_features)}")
    
    return engineered_data

def train_enhanced_models(data, optimize_hyperparams=True, use_deep_learning=True):
    """Train models with enhanced techniques for maximum accuracy"""
    print("\n=== Enhanced Model Training Phase ===")
    
    # Initialize predictor
    predictor = GoldPricePredictor()
    
    # Hyperparameter optimization for key models
    if optimize_hyperparams:
        print("\nOptimizing hyperparameters...")
        
        # Prepare data for optimization
        data_clean = predictor.prepare_features(data, is_training=True)
        data_clean = data_clean.dropna()
        
        feature_columns = [col for col in data_clean.columns if col != 'price' and col != 'date']
        X = data_clean[feature_columns]
        y = data_clean['price']
        
        # Split for optimization
        split_idx = int(len(data_clean) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        
        # Scale features for optimization
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        # Optimize key models
        models_to_optimize = ['random_forest', 'xgboost', 'lightgbm']
        
        for model_name in models_to_optimize:
            print(f"Optimizing {model_name}...")
            try:
                best_params = predictor.optimize_hyperparameters(
                    X_train_scaled, y_train, model_name, n_trials=30
                )
                print(f"Best parameters for {model_name}: {best_params}")
                
                # Update model with best parameters
                if model_name == 'random_forest':
                    from sklearn.ensemble import RandomForestRegressor
                    predictor.models[model_name] = RandomForestRegressor(**best_params, n_jobs=-1)
                elif model_name == 'xgboost':
                    import xgboost as xgb
                    predictor.models[model_name] = xgb.XGBRegressor(**best_params)
                elif model_name == 'lightgbm':
                    import lightgbm as lgb
                    predictor.models[model_name] = lgb.LGBMRegressor(**best_params)
                    
            except Exception as e:
                print(f"Error optimizing {model_name}: {e}")
    
    # Train all models with enhanced validation
    results = predictor.train_models(
        data, 
        validation_split=0.2, 
        feature_selection_k=75,  # Increased feature selection
        use_walk_forward=True
    )
    
    return predictor, results

def save_enhanced_results(predictor, results, timestamp):
    """Save enhanced training results and models"""
    print("\n=== Saving Enhanced Results ===")
    
    # Save models
    import joblib
    
    # Save best model
    if predictor.best_model is not None:
        model_path = f'models/best_model_{timestamp}.pkl'
        joblib.dump(predictor.best_model, model_path)
        print(f"Best model saved to {model_path}")
    
    # Save feature scaler and selector
    if hasattr(predictor, 'scaler'):
        joblib.dump(predictor.scaler, f'models/feature_scaler_{timestamp}.pkl')
    
    if hasattr(predictor, 'feature_selector'):
        joblib.dump(predictor.feature_selector, f'models/feature_selector_{timestamp}.pkl')
    
    # Save feature columns
    if hasattr(predictor, 'feature_columns'):
        joblib.dump(predictor.feature_columns, f'models/feature_columns_{timestamp}.pkl')
    
    # Prepare results for JSON serialization
    json_results = {
        'timestamp': timestamp,
        'data_shape': [len(predictor.feature_columns), len(predictor.feature_columns)],
        'models': {},
        'best_model': None,
        'best_score': float('inf')
    }
    
    best_score = float('inf')
    best_model_name = None
    
    for name, result in results.items():
        # Skip deep learning models for JSON serialization
        if name in ['lstm', 'gru']:
            continue
            
        json_results['models'][name] = {
            'rmse': float(result['rmse']),
            'mae': float(result['mae']),
            'r2': float(result['r2']),
            'mse': float(result['mse']),
            'cv_score': float(result['cv_score'])
        }
        
        if result['cv_score'] < best_score:
            best_score = result['cv_score']
            best_model_name = name
    
    json_results['best_model'] = best_model_name
    json_results['best_score'] = float(best_score)
    
    # Save results
    results_path = f'results/enhanced_training_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Enhanced results saved to {results_path}")
    
    # Print summary
    print("\n=== Training Summary ===")
    print(f"Best model: {best_model_name}")
    print(f"Best CV score (MSE): {best_score:.6f}")
    print(f"Best RMSE: {np.sqrt(best_score):.6f}")
    
    if best_model_name and best_model_name in results:
        best_result = results[best_model_name]
        print(f"Test R²: {best_result['r2']:.6f}")
        print(f"Test MAE: {best_result['mae']:.6f}")

def main():
    """Main training pipeline with enhanced techniques"""
    parser = argparse.ArgumentParser(description='Enhanced Gold Price Prediction Training')
    parser.add_argument('--real-data', action='store_true', help='Use real market data')
    parser.add_argument('--force-refresh', action='store_true', help='Force refresh of data')
    parser.add_argument('--optimize-hyperparams', action='store_true', default=True, help='Optimize hyperparameters')
    parser.add_argument('--use-deep-learning', action='store_true', default=True, help='Include deep learning models')
    
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("=== Enhanced Gold Price Prediction Training ===")
    print(f"Timestamp: {timestamp}")
    print(f"Real data: {args.real_data}")
    print(f"Hyperparameter optimization: {args.optimize_hyperparams}")
    print(f"Deep learning: {args.use_deep_learning}")
    
    try:
        # Data collection
        gold_data, economic_data = collect_enhanced_data(
            use_real_data=args.real_data, 
            force_refresh=args.force_refresh
        )
        
        # Data preprocessing
        processed_data = preprocess_enhanced_data(gold_data, economic_data)
        
        # Model training
        predictor, results = train_enhanced_models(
            processed_data,
            optimize_hyperparams=args.optimize_hyperparams,
            use_deep_learning=args.use_deep_learning
        )
        
        # Save results
        save_enhanced_results(predictor, results, timestamp)
        
        print("\n=== Enhanced Training Completed Successfully ===")
        
    except Exception as e:
        print(f"\nError during enhanced training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())