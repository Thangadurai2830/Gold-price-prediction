#!/usr/bin/env python3
"""
Quick Gold Price Prediction Model Training Script

This script handles a simplified training pipeline without deep learning models:
1. Data collection and preprocessing
2. Feature engineering
3. Core ML model training and evaluation
4. Model saving and performance reporting
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
                print(f"Successfully collected {len(data_dict['gold_data'])} gold price records")
                return collector.generate_sample_data()
            else:
                print("Failed to collect real data, falling back to sample data")
                return collector.generate_sample_data()
        except Exception as e:
            print(f"Error collecting real data: {e}")
            print("Falling back to sample data")
            return collector.generate_sample_data()
    else:
        print("Using sample data for training...")
        return collector.generate_sample_data()

def preprocess_data(data):
    """Preprocess the collected data"""
    print("\n=== Data Preprocessing Phase ===")
    
    processor = DataProcessor()
    
    # Clean and validate data
    cleaned_data = processor.clean_data(data)
    print(f"Data shape after cleaning: {cleaned_data.shape}")
    
    # Add technical indicators
    processed_data = processor.add_technical_indicators(cleaned_data)
    print(f"Data shape after adding technical indicators: {processed_data.shape}")
    
    # Remove any remaining NaN values
    processed_data = processed_data.dropna()
    print(f"Data shape after removing NaN values: {processed_data.shape}")
    
    return processed_data

def engineer_features(data):
    """Engineer features for model training"""
    print("\n=== Feature Engineering Phase ===")
    
    engineer = FeatureEngineer()
    
    # Create features using comprehensive feature engineering
    featured_data = engineer.engineer_all_features(data, target_column='price', use_pca=False, n_features=30)
    print(f"Data shape after feature engineering: {featured_data.shape}")
    print(f"Number of features created: {len(featured_data.columns) - 2}")  # -2 for date and target columns
    
    return featured_data, engineer

def train_models(data, feature_engineer):
    """Train machine learning models"""
    print("\n=== Model Training Phase ===")
    
    # Initialize predictor with simplified models (no deep learning)
    predictor = GoldPricePredictor()
    
    # Remove deep learning models to speed up training
    models_to_remove = ['lstm', 'gru', 'transformer', 'cnn_lstm']
    for model_name in models_to_remove:
        if model_name in predictor.models:
            del predictor.models[model_name]
    
    print(f"Training {len(predictor.models)} models: {list(predictor.models.keys())}")
    
    # Set feature columns from pre-engineered data
    predictor.feature_columns = [col for col in data.columns if col != 'price']
    
    # Get feature columns (exclude target and date columns)
    feature_columns = [col for col in data.columns if col not in ['price', 'date']]
    
    # Train models with pre-engineered features
    results = predictor.train_models_with_features(data, feature_columns, validation_split=0.2)
    
    return predictor, results

def save_results(predictor, results, feature_engineer):
    """Save trained models and results"""
    print("\n=== Saving Results ===")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save models
    model_path = f"models/gold_predictor_{timestamp}.joblib"
    predictor.save_models(model_path)
    print(f"Models saved to: {model_path}")
    
    # Save feature engineer components
    feature_path = f"models/feature_engineer_{timestamp}.joblib"
    import joblib
    
    # Save feature engineer components separately
    feature_components = {
        'scaler': feature_engineer.scaler,
        'feature_selector': feature_engineer.feature_selector,
        'selected_features': feature_engineer.selected_features
    }
    joblib.dump(feature_components, feature_path)
    print(f"Feature engineer components saved to: {feature_path}")
    
    # Save results
    results_path = f"results/training_results_{timestamp}.json"
    
    # Convert results to JSON-serializable format
    json_results = {}
    for model_name, metrics in results.items():
        if isinstance(metrics, dict) and 'model' in metrics:
            json_results[model_name] = {
                'mse': float(metrics['mse']),
                'mae': float(metrics['mae']),
                'r2': float(metrics['r2']),
                'rmse': float(metrics['rmse']),
                'cv_score': float(metrics['cv_score'])
            }
    
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'model_results': json_results,
            'best_model': min(json_results.keys(), key=lambda x: json_results[x]['mse']),
            'feature_count': len(predictor.feature_columns)
        }, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    return model_path, feature_path, results_path

def print_summary(results):
    """Print training summary"""
    print("\n=== Training Summary ===")
    print(f"{'Model':<20} {'MSE':<10} {'MAE':<10} {'R2':<10} {'RMSE':<10}")
    print("-" * 70)
    
    for model_name, metrics in results.items():
        if isinstance(metrics, dict) and 'mse' in metrics:
            print(f"{model_name:<20} {metrics['mse']:<10.4f} {metrics['mae']:<10.4f} {metrics['r2']:<10.4f} {metrics['rmse']:<10.4f}")
    
    # Find best model
    best_model = min(results.keys(), key=lambda x: results[x]['mse'] if isinstance(results[x], dict) and 'mse' in results[x] else float('inf'))
    print(f"\nBest performing model: {best_model}")
    print(f"Best MSE: {results[best_model]['mse']:.4f}")

def main():
    """Main training pipeline"""
    print("Starting Gold Price Prediction Model Training (Quick Version)")
    print("=" * 60)
    
    # Setup
    setup_directories()
    
    # Collect data
    raw_data = collect_data(use_real_data=False)  # Use sample data for quick training
    
    # Preprocess data
    processed_data = preprocess_data(raw_data)
    
    # Engineer features
    featured_data, feature_engineer = engineer_features(processed_data)
    
    # Train models
    predictor, results = train_models(featured_data, feature_engineer)
    
    # Save results
    model_path, feature_path, results_path = save_results(predictor, results, feature_engineer)
    
    # Print summary
    print_summary(results)
    
    print("\n=== Training Complete ===")
    print(f"Models saved to: {model_path}")
    print(f"Feature engineer saved to: {feature_path}")
    print(f"Results saved to: {results_path}")
    
    return predictor, results

if __name__ == "__main__":
    try:
        predictor, results = main()
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)