#!/usr/bin/env python3
"""
Gold Price Prediction Model Training Script

This script handles the complete training pipeline:
1. Data collection and preprocessing
2. Feature engineering
3. Model training and evaluation
4. Model saving and performance reporting
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

def collect_data(use_real_data=False, force_refresh=False):
    """Collect and prepare training data"""
    print("=== Data Collection Phase ===")
    
    collector = GoldDataCollector()
    
    if use_real_data:
        print("Fetching real market data...")
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
                print("Failed to fetch real data, falling back to sample data")
                return collector.generate_sample_data(), None
                
        except Exception as e:
            print(f"Error fetching real data: {e}")
            print("Falling back to sample data")
            return collector.generate_sample_data(), None
    else:
        print("Generating sample data for training...")
        return collector.generate_sample_data(), None

def preprocess_data(gold_data, economic_data=None):
    """Preprocess and engineer features"""
    print("\n=== Data Preprocessing Phase ===")
    
    # Initialize data processor
    processor = DataProcessor()
    processor.raw_data = gold_data
    
    if economic_data is not None:
        processor.external_factors = economic_data
    
    # Prepare training data
    processed_data = processor.prepare_training_data(use_external_factors=(economic_data is not None))
    
    print(f"Preprocessed data shape: {processed_data.shape}")
    
    # Feature engineering
    print("\n=== Feature Engineering Phase ===")
    feature_engineer = FeatureEngineer()
    
    # Apply comprehensive feature engineering
    engineered_data = feature_engineer.engineer_all_features(
        processed_data, 
        target_column='price',
        use_pca=False,
        n_features=50
    )
    
    print(f"Final feature set shape: {engineered_data.shape}")
    print(f"Selected features: {len(feature_engineer.selected_features)}")
    
    return engineered_data, feature_engineer

def train_models(data, feature_engineer):
    """Train all machine learning models"""
    print("\n=== Model Training Phase ===")
    
    # Initialize predictor
    predictor = GoldPricePredictor()
    
    # Set feature columns from the feature engineer to avoid double processing
    feature_columns = [col for col in data.columns if col not in ['date', 'price']]
    predictor.feature_columns = feature_columns
    
    # Train traditional ML models with pre-engineered data
    print("Training traditional ML models...")
    ml_results = predictor.train_models_with_features(data, feature_columns)
    
    # Train Prophet model for time series forecasting
    print("\nTraining Prophet time series model...")
    try:
        prophet_model = predictor.train_prophet_model(data)
        print("Prophet model trained successfully")
    except Exception as e:
        print(f"Error training Prophet model: {e}")
        prophet_model = None
    
    return predictor, ml_results

def evaluate_models(predictor, ml_results, data):
    """Evaluate model performance"""
    print("\n=== Model Evaluation Phase ===")
    
    evaluation_results = {
        'timestamp': datetime.now().isoformat(),
        'data_shape': data.shape,
        'models': {}
    }
    
    # Evaluate each model
    for model_name, results in ml_results.items():
        print(f"\n{model_name.upper()} Results:")
        print(f"  RMSE: {results['rmse']:.2f}")
        print(f"  MAE: {results['mae']:.2f}")
        print(f"  R²: {results['r2']:.4f}")
        
        evaluation_results['models'][model_name] = {
            'rmse': float(results['rmse']),
            'mae': float(results['mae']),
            'r2': float(results['r2']),
            'mse': float(results['mse'])
        }
    
    # Find best model
    best_model = min(ml_results.keys(), key=lambda x: ml_results[x]['rmse'])
    print(f"\nBest performing model: {best_model} (RMSE: {ml_results[best_model]['rmse']:.2f})")
    
    evaluation_results['best_model'] = best_model
    evaluation_results['best_rmse'] = float(ml_results[best_model]['rmse'])
    
    return evaluation_results

def save_models_and_results(predictor, feature_engineer, evaluation_results):
    """Save trained models and evaluation results"""
    print("\n=== Saving Models and Results ===")
    
    # Save models
    predictor.save_models('models')
    
    # Save feature engineering components
    if feature_engineer.scaler:
        import joblib
        joblib.dump(feature_engineer.scaler, 'models/feature_scaler.pkl')
        print("Feature scaler saved")
    
    if feature_engineer.feature_selector:
        import joblib
        joblib.dump(feature_engineer.feature_selector, 'models/feature_selector.pkl')
        print("Feature selector saved")
    
    # Save evaluation results
    results_file = f"results/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"Evaluation results saved to {results_file}")
    
    # Save feature importance if available
    try:
        feature_importance = feature_engineer.get_feature_importance_scores()
        if feature_importance is not None:
            importance_file = f"results/feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            feature_importance.to_csv(importance_file, index=False)
            print(f"Feature importance saved to {importance_file}")
    except Exception as e:
        print(f"Could not save feature importance: {e}")

def generate_predictions(predictor, data, days_ahead=30):
    """Generate sample predictions"""
    print("\n=== Generating Sample Predictions ===")
    
    try:
        # Get latest data for prediction
        latest_data = data.tail(1)
        feature_columns = [col for col in data.columns if col not in ['date', 'price']]
        latest_features = latest_data[feature_columns].values[0]
        
        # Make ensemble prediction
        ensemble_pred = predictor.predict_ensemble(latest_features)
        print(f"Ensemble prediction for next period: ${ensemble_pred:.2f}")
        
        # Individual model predictions
        print("\nIndividual model predictions:")
        for model_name in predictor.trained_models.keys():
            try:
                pred = predictor.predict_single(model_name, latest_features)
                print(f"  {model_name}: ${pred:.2f}")
            except Exception as e:
                print(f"  {model_name}: Error - {e}")
        
        # Prophet future predictions
        if predictor.prophet_model:
            try:
                future_predictions = predictor.predict_future_prophet(days=days_ahead)
                print(f"\nProphet predictions for next {days_ahead} days:")
                print(f"  Average predicted price: ${future_predictions['yhat'].mean():.2f}")
                print(f"  Price range: ${future_predictions['yhat_lower'].mean():.2f} - ${future_predictions['yhat_upper'].mean():.2f}")
            except Exception as e:
                print(f"Error generating Prophet predictions: {e}")
        
    except Exception as e:
        print(f"Error generating predictions: {e}")

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train Gold Price Prediction Models')
    parser.add_argument('--real-data', action='store_true', help='Use real market data instead of sample data')
    parser.add_argument('--force-refresh', action='store_true', help='Force refresh of market data')
    parser.add_argument('--no-predictions', action='store_true', help='Skip generating sample predictions')
    
    args = parser.parse_args()
    
    print("Gold Price Prediction Model Training")
    print("====================================")
    print(f"Start time: {datetime.now()}")
    
    # Setup
    setup_directories()
    
    try:
        # Step 1: Data Collection
        gold_data, economic_data = collect_data(args.real_data, args.force_refresh)
        
        if gold_data is None or len(gold_data) < 100:
            print("Insufficient data for training. Need at least 100 records.")
            return
        
        # Step 2: Data Preprocessing and Feature Engineering
        processed_data, feature_engineer = preprocess_data(gold_data, economic_data)
        
        if len(processed_data) < 50:
            print("Insufficient processed data for training. Need at least 50 records after preprocessing.")
            return
        
        # Step 3: Model Training
        predictor, ml_results = train_models(processed_data, feature_engineer)
        
        # Step 4: Model Evaluation
        evaluation_results = evaluate_models(predictor, ml_results, processed_data)
        
        # Step 5: Save Models and Results
        save_models_and_results(predictor, feature_engineer, evaluation_results)
        
        # Step 6: Generate Sample Predictions
        if not args.no_predictions:
            generate_predictions(predictor, processed_data)
        
        print(f"\n=== Training Complete ===")
        print(f"End time: {datetime.now()}")
        print(f"Best model: {evaluation_results['best_model']} (RMSE: {evaluation_results['best_rmse']:.2f})")
        print("Models saved to 'models/' directory")
        print("Results saved to 'results/' directory")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)