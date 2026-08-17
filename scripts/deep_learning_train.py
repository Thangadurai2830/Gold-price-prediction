#!/usr/bin/env python3
"""
Deep Learning Training Script for Gold Price Prediction

This script trains LSTM, GRU, and Transformer models for gold price prediction
using proper time series validation and feature engineering.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_collector import GoldDataCollector
from ml.robust_feature_engineering import RobustFeatureEngineer
from ml.deep_learning_models import TimeSeriesDeepLearning

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train deep learning models for gold price prediction')
    parser.add_argument('--real-data', action='store_true', help='Use real data instead of synthetic')
    parser.add_argument('--sequence-length', type=int, default=30, help='Sequence length for time series models')
    parser.add_argument('--prediction-horizon', type=int, default=1, help='Prediction horizon (days ahead)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--cv-folds', type=int, default=3, help='Number of cross-validation folds')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size ratio')
    return parser.parse_args()

def prepare_deep_learning_data(df: pd.DataFrame, feature_engineer: RobustFeatureEngineer,
                              sequence_length: int, prediction_horizon: int,
                              test_size: float = 0.2):
    """
    Prepare data for deep learning models.
    
    Args:
        df: Input dataframe
        feature_engineer: Feature engineering instance
        sequence_length: Length of input sequences
        prediction_horizon: Number of steps ahead to predict
        test_size: Proportion of data for testing
        
    Returns:
        Tuple of prepared data splits
    """
    print("Preparing features for deep learning...")
    
    # Prepare training data with proper temporal constraints
    X, y = feature_engineer.prepare_training_data(
        df, 
        price_col='close',
        prediction_horizon=prediction_horizon
    )
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Time series split
    split_idx = int(len(X) * (1 - test_size))
    X_temp, X_test = X[:split_idx], X[split_idx:]
    y_temp, y_test = y[:split_idx], y[split_idx:]
    
    # Further split temp into train/val
    val_size = 0.2
    val_split_idx = int(len(X_temp) * (1 - val_size))
    X_train, X_val = X_temp[:val_split_idx], X_temp[val_split_idx:]
    y_train, y_val = y_temp[:val_split_idx], y_temp[val_split_idx:]
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_sequences_from_features(X: np.ndarray, y: np.ndarray, sequence_length: int):
    """
    Create sequences from feature matrix for deep learning.

    Args:
        X: Feature matrix
        y: Target values
        sequence_length: Length of sequences
        
    Returns:
        Tuple of (X_sequences, y_sequences)
    """
    # Convert y to numpy array if it's a pandas Series
    if hasattr(y, 'values'):
        y = y.values
    
    X_sequences = []
    y_sequences = []
    
    for i in range(sequence_length, len(X)):
        X_sequences.append(X[i-sequence_length:i])
        y_sequences.append(y[i])
    
    return np.array(X_sequences), np.array(y_sequences)

def evaluate_with_time_series_cv(dl_model: TimeSeriesDeepLearning, X: np.ndarray, y: np.ndarray,
                                cv_folds: int = 3, epochs: int = 50, batch_size: int = 32):
    """
    Evaluate deep learning models using time series cross-validation.
    
    Args:
        dl_model: Deep learning model instance
        X: Feature sequences
        y: Target values
        cv_folds: Number of CV folds
        epochs: Training epochs per fold
        batch_size: Training batch size
        
    Returns:
        Dictionary with CV results
    """
    print(f"\nPerforming {cv_folds}-fold time series cross-validation...")
    
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    cv_results = {'lstm': [], 'gru': [], 'transformer': []}
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\nFold {fold + 1}/{cv_folds}")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Create a new model instance for each fold
        fold_model = TimeSeriesDeepLearning(
            sequence_length=dl_model.sequence_length,
            prediction_horizon=dl_model.prediction_horizon
        )
        
        # Train models for this fold
        fold_results = fold_model.train_all_models(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold,
            epochs=epochs, batch_size=batch_size
        )
        
        # Store results
        for model_name, results in fold_results.items():
            cv_results[model_name].append(results['r2'])
    
    # Calculate mean and std for each model
    cv_summary = {}
    for model_name, scores in cv_results.items():
        cv_summary[model_name] = {
            'cv_mean': np.mean(scores),
            'cv_std': np.std(scores),
            'cv_scores': scores
        }
    
    return cv_summary

def main():
    """Main training function."""
    args = parse_arguments()
    
    print("=" * 60)
    print("DEEP LEARNING TRAINING FOR GOLD PRICE PREDICTION")
    print("=" * 60)
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Prediction Horizon: {args.prediction_horizon} days")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"CV Folds: {args.cv_folds}")
    print(f"Using real data: {args.real_data}")
    
    # Initialize data collector and feature engineer
    collector = GoldDataCollector()
    feature_engineer = RobustFeatureEngineer()
    
    # Load data
    if args.real_data:
        print("\nFetching real gold price data...")
        df = collector.fetch_gold_prices_yahoo()
        print(f"Loaded {len(df)} days of real data")
    else:
        print("\nGenerating synthetic data...")
        df = collector.generate_synthetic_data(days=1000)
        print(f"Generated {len(df)} days of synthetic data")
    
    # Prepare data for deep learning
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_deep_learning_data(
        df, feature_engineer, args.sequence_length, args.prediction_horizon, args.test_size
    )
    
    # Create sequences for deep learning
    print("\nCreating sequences for deep learning models...")
    X_train_seq, y_train_seq = create_sequences_from_features(X_train, y_train, args.sequence_length)
    X_val_seq, y_val_seq = create_sequences_from_features(X_val, y_val, args.sequence_length)
    X_test_seq, y_test_seq = create_sequences_from_features(X_test, y_test, args.sequence_length)
    
    print(f"Training sequences shape: {X_train_seq.shape}")
    print(f"Validation sequences shape: {X_val_seq.shape}")
    print(f"Test sequences shape: {X_test_seq.shape}")
    
    # Initialize deep learning model
    dl_model = TimeSeriesDeepLearning(
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon
    )
    
    # Train all models
    print("\n" + "=" * 50)
    print("TRAINING DEEP LEARNING MODELS")
    print("=" * 50)
    
    results = dl_model.train_all_models(
        X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        epochs=args.epochs, batch_size=args.batch_size
    )
    
    # Evaluate on test set
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)
    
    test_results = {}
    for model_name in ['lstm', 'gru', 'transformer']:
        print(f"\nEvaluating {model_name.upper()}...")
        
        # Make predictions
        y_pred = dl_model.predict(model_name, X_test_seq)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_seq, y_pred)
        mae = mean_absolute_error(y_test_seq, y_pred)
        r2 = r2_score(y_test_seq, y_pred)
        rmse = np.sqrt(mse)
        
        test_results[model_name] = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': rmse
        }
        
        print(f"  Test R²: {r2:.4f}")
        print(f"  Test RMSE: {rmse:.2f}")
        print(f"  Test MAE: {mae:.2f}")
    
    # Cross-validation evaluation
    if args.cv_folds > 1:
        print("\n" + "=" * 50)
        print("CROSS-VALIDATION EVALUATION")
        print("=" * 50)
        
        # Combine train and val for CV
        X_cv = np.concatenate([X_train_seq, X_val_seq], axis=0)
        y_cv = np.concatenate([y_train_seq, y_val_seq], axis=0)
        
        cv_results = evaluate_with_time_series_cv(
            dl_model, X_cv, y_cv, args.cv_folds, 
            epochs=args.epochs//2, batch_size=args.batch_size
        )
        
        for model_name, cv_stats in cv_results.items():
            print(f"\n{model_name.upper()} CV Results:")
            print(f"  CV R²: {cv_stats['cv_mean']:.4f} (±{cv_stats['cv_std']:.4f})")
    
    # Find best model
    best_model_name, best_model_results = dl_model.get_best_model(results)
    print(f"\n" + "=" * 50)
    print("BEST MODEL SUMMARY")
    print("=" * 50)
    print(f"Best Model: {best_model_name.upper()}")
    print(f"Validation R²: {best_model_results['r2']:.4f}")
    print(f"Test R²: {test_results[best_model_name]['r2']:.4f}")
    print(f"Test RMSE: {test_results[best_model_name]['rmse']:.2f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare results for saving
    final_results = {
        'timestamp': timestamp,
        'parameters': {
            'sequence_length': args.sequence_length,
            'prediction_horizon': args.prediction_horizon,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'cv_folds': args.cv_folds,
            'real_data': args.real_data
        },
        'validation_results': results,
        'test_results': test_results,
        'best_model': {
            'name': best_model_name,
            'validation_metrics': best_model_results,
            'test_metrics': test_results[best_model_name]
        }
    }
    
    if args.cv_folds > 1:
        final_results['cv_results'] = cv_results
    
    # Save results
    results_file = os.path.join(results_dir, f"deep_learning_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # Save models
    models_file = os.path.join(results_dir, f"deep_learning_models_{timestamp}")
    dl_model.save_models(models_file)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Models saved to: {models_file}_*.h5")
    
    print("\n" + "=" * 60)
    print("DEEP LEARNING TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()